"""
SNNDashboard: TensorBoard 训练看板

记录 ~627 个可训练参数的权重/梯度动态，用于论文写作和 debug。

两级频率：
  - log_step()       每 log_interval 步：训练标量 + 参数 norm + 神经元动力学 + PonderNet E[K]
  - log_save_point() 每 save_interval 步：权重/梯度直方图 + 调制补偿因子

调用时机：optimizer.step() 之后、optimizer.zero_grad() 之前（梯度仍可用）。

用法：
    dashboard = SNNDashboard(log_dir='runs/pretrain', model=raw_model, rank=rank)

    # 边界步（log_interval）
    optimizer.step()
    dashboard.log_step(global_step, metrics_dict, raw_model, log_params=True)
    optimizer.zero_grad(set_to_none=True)

    # 保存点（save_interval）
    dashboard.log_save_point(global_step, raw_model)

    # 训练结束
    dashboard.close()
"""

import torch
import torch.nn.functional as F


class SNNDashboard:
    """SNN 训练看板（仅 rank 0 记录，其余为 no-op）。

    FSDP 兼容：调用 log_step/log_save_point 前需在训练循环中
    用 FSDP.summon_full_params() 聚合参数（集合操作，所有 rank 参与）。

    Args:
        log_dir: TensorBoard 日志目录，None 时完全禁用（零开销）
        model: FSDP 包装前的原始模型（raw_model）
        rank: 分布式 rank，仅 rank 0 启用记录
    """

    def __init__(self, log_dir, model, rank=0):
        self._enabled = (log_dir is not None) and (rank == 0)
        if not self._enabled:
            return

        # 延迟导入：仅启用时加载 tensorboard
        from torch.utils.tensorboard import SummaryWriter
        self._writer = SummaryWriter(log_dir=log_dir)
        self._registry = self._build_registry(model)
        self._neuron_semantics = self._build_neuron_semantics(model)

    # ====== 公开方法 ======

    def log_step(self, step, metrics_dict, model, log_params=True):
        """每 log_interval 步调用：训练标量 + 参数监控 + 神经元动力学。"""
        if not self._enabled:
            return

        self._log_training_scalars(step, metrics_dict)

        if log_params:
            self._log_param_norms(step, metrics_dict.get('lr', 1e-4))
            self._log_neuron_dynamics(step)
            self._log_dynamic_k(step, model)

    def log_save_point(self, step, model):
        """每 save_interval 步调用：直方图 + 补偿因子。"""
        if not self._enabled:
            return

        self._log_histograms(step)
        self._log_compensation_factors(step, model)

    def close(self):
        """关闭 TensorBoard writer，刷新缓冲。"""
        if not self._enabled:
            return
        self._writer.close()

    # ====== 注册表构建 ======

    def _build_registry(self, model):
        """构建 param name → (TensorBoard tag, param) 映射。

        命名规则：
          layers.5.snn_block.W_in.weight → layer_05/snn_block/W_in/weight
          embed_tokens.weight             → global/embed_tokens/weight
        """
        registry = {}
        for name, param in model.named_parameters():
            parts = name.split('.')
            if parts[0] == 'layers' and parts[1].isdigit():
                layer_idx = int(parts[1])
                tag = f"layer_{layer_idx:02d}/" + "/".join(parts[2:])
            else:
                tag = "global/" + "/".join(parts)
            registry[name] = (tag, param)
        return registry

    def _build_neuron_semantics(self, model):
        """识别需语义变换的神经元参数。

        Returns:
            list of (tag_prefix, param, transform_fn, metric_name)
        """
        semantics = []

        for i, layer_module in enumerate(model.layers):
            prefix = f"layer_{i:02d}"
            block = layer_module.snn_block
            ffn = layer_module.snn_ffn

            # 输入神经元: PLIFNode.w → sigmoid → beta
            semantics.append(
                (f"{prefix}/input1_beta", layer_module.input_neuron1.w,
                 torch.sigmoid, "beta"))
            semantics.append(
                (f"{prefix}/input2_beta", layer_module.input_neuron2.w,
                 torch.sigmoid, "beta"))

            # SNNBlock 调制偏置
            semantics.append(
                (f"{prefix}/block_beta_t", block.b_beta,
                 torch.sigmoid, "beta(t)"))
            semantics.append(
                (f"{prefix}/block_alpha_t", block.b_alpha,
                 F.softplus, "alpha(t)"))
            v_th_min = block.v_th_min
            semantics.append(
                (f"{prefix}/block_vth_t", block.b_th,
                 lambda x, m=v_th_min: m + torch.abs(x), "V_th(t)"))

            # FFN 神经元: PLIFNode.w → sigmoid → beta
            semantics.append(
                (f"{prefix}/ffn_gate_beta", ffn.gate_neuron.w,
                 torch.sigmoid, "beta"))
            semantics.append(
                (f"{prefix}/ffn_up_beta", ffn.up_neuron.w,
                 torch.sigmoid, "beta"))

        # 输出神经元
        semantics.append(
            ("global/output_beta", model.output_neuron.w,
             torch.sigmoid, "beta"))

        return semantics

    # ====== 轻量日志（每 log_interval 步） ======

    def _log_training_scalars(self, step, metrics):
        """记录训练标量：loss, ppl, lr, tps, tokens_seen, ponder_cost, memory。"""
        w = self._writer
        for key in ('loss', 'ppl', 'lr', 'tps', 'tokens_seen', 'ponder_cost'):
            if key in metrics:
                w.add_scalar(f"train/{key}", metrics[key], step)
        if 'memory_current_gb' in metrics:
            w.add_scalar("train/memory/current_gb", metrics['memory_current_gb'], step)
        if 'memory_peak_gb' in metrics:
            w.add_scalar("train/memory/peak_gb", metrics['memory_peak_gb'], step)

    def _log_param_norms(self, step, lr):
        """记录每个参数的 weight_norm, grad_norm, update_ratio。"""
        w = self._writer
        for name, (tag, param) in self._registry.items():
            if not param.requires_grad:
                continue

            weight_norm = param.data.norm().item()
            w.add_scalar(f"params/{tag}/weight_norm", weight_norm, step)

            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                w.add_scalar(f"params/{tag}/grad_norm", grad_norm, step)
                # update_ratio ≈ lr·‖grad‖ / ‖weight‖（Adam 实际步长不同，但趋势一致）
                update_ratio = (lr * grad_norm) / (weight_norm + 1e-8)
                w.add_scalar(f"params/{tag}/update_ratio", update_ratio, step)

    def _log_neuron_dynamics(self, step):
        """记录神经元参数的语义值（sigmoid/softplus 变换后）。"""
        w = self._writer
        for tag, param, transform_fn, metric_name in self._neuron_semantics:
            with torch.no_grad():
                val = transform_fn(param.data)
                w.add_scalar(f"neuron_dynamics/{tag}/mean", val.mean().item(), step)
                w.add_scalar(f"neuron_dynamics/{tag}/std", val.std().item(), step)

    def _log_dynamic_k(self, step, model):
        """记录每层 PonderNet E[K] 的极值范围 + halt 参数统计。"""
        w = self._writer
        for i, layer_module in enumerate(model.layers):
            ek_min = getattr(layer_module, '_ek_min', None)
            ek_max = getattr(layer_module, '_ek_max', None)
            if ek_min is not None:
                w.add_scalar(f"ponder/layer_{i:02d}/ek_min", ek_min, step)
                w.add_scalar(f"ponder/layer_{i:02d}/ek_max", ek_max, step)

            # halt 参数监控：权重 norm + 偏置值（诊断 halt 爆炸）
            with torch.no_grad():
                for name, halt in [('block_halt', layer_module.block_halt),
                                   ('ffn_halt', layer_module.ffn_halt)]:
                    w.add_scalar(f"halt/layer_{i:02d}/{name}/weight_norm",
                                 halt.weight.data.norm().item(), step)
                    w.add_scalar(f"halt/layer_{i:02d}/{name}/bias",
                                 halt.bias.data.item(), step)

            # SubLN Post-RMSNorm gain 监控（诊断深层梯度均衡）
            with torch.no_grad():
                w.add_scalar(f"post_norm/layer_{i:02d}/block_gain_mean",
                             layer_module.block_post_norm.weight.data.mean().item(), step)
                w.add_scalar(f"post_norm/layer_{i:02d}/ffn_gain_mean",
                             layer_module.ffn_post_norm.weight.data.mean().item(), step)

            # MPD-AGL 自适应 surrogate alpha 监控
            for attr, tag in [('_alpha_input1', 'input1'),
                              ('_alpha_hidden', 'hidden'),
                              ('_alpha_input2', 'input2'),
                              ('_alpha_ffn', 'ffn')]:
                val = getattr(layer_module, attr, None)
                if val is not None:
                    w.add_scalar(f"mpd_alpha/layer_{i:02d}/{tag}", val, step)

    # ====== 重量日志（每 save_interval 步） ======

    def _log_histograms(self, step):
        """记录所有参数权重/梯度分布（直方图）。"""
        w = self._writer
        for name, (tag, param) in self._registry.items():
            if not param.requires_grad:
                continue
            w.add_histogram(f"histograms/{tag}/weight", param.data, step)
            if param.grad is not None:
                w.add_histogram(f"histograms/{tag}/grad", param.grad, step)

    def _log_compensation_factors(self, step, model):
        """记录 sigmoid/softplus 导数均值（诊断梯度补偿效果）。"""
        w = self._writer
        for i, layer_module in enumerate(model.layers):
            block = layer_module.snn_block
            with torch.no_grad():
                # sigmoid 导数: β·(1-β), β = sigmoid(b_beta)
                beta = torch.sigmoid(block.b_beta.data)
                sigmoid_deriv = (beta * (1.0 - beta)).mean().item()
                w.add_scalar(
                    f"compensation/layer_{i:02d}/sigmoid_deriv_mean",
                    sigmoid_deriv, step)

                # softplus 导数: sigmoid(b_alpha)
                softplus_deriv = torch.sigmoid(block.b_alpha.data).mean().item()
                w.add_scalar(
                    f"compensation/layer_{i:02d}/softplus_deriv_mean",
                    softplus_deriv, step)
