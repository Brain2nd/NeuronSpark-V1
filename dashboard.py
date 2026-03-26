"""
SNNDashboard: TensorBoard 训练看板

记录可训练参数的权重/梯度动态 + 神经元健康监控，用于论文写作和 debug。

三级频率：
  - log_step()       每 log_interval 步：训练标量 + 神经元动力学 + 健康监控
  - log_step(log_params=True)：额外记录参数 norm/grad/update_ratio
  - log_save_point() 每 save_interval 步：权重/梯度直方图 + 补偿因子

监控类别：
  1. 训练标量 (loss/ppl/lr/tps/memory)
  2. 参数 norm + 梯度 norm + update_ratio
  3. 神经元动力学 (β/α/V_th 的语义值演化)
  4. PonderNet E[K] 每层极值
  5. β 分布演化 (均值/std/min/max per layer — 论文级)
  6. 发放率追踪 (per layer)
  7. 联想记忆层 (M 范数/write_gate 发放率)
  8. 坍缩/癫痫/趋同检测 (死/饱和神经元比例, β 趋同度, 输出范数比)
  9. 综合健康得分
"""

import torch
import torch.nn.functional as F
import torch.distributed as dist


class SNNDashboard:
    def __init__(self, log_dir, model, rank=0):
        self._enabled = (log_dir is not None) and (rank == 0)
        self._rank = rank
        self._grad_cache = {}
        if not self._enabled:
            return

        from torch.utils.tensorboard import SummaryWriter
        self._writer = SummaryWriter(log_dir=log_dir)
        self._registry = self._build_registry(model)
        self._neuron_semantics = self._build_neuron_semantics(model)

    # ====== 公开方法 ======

    def cache_grad_norms(self, model):
        """optimizer.step() 前调用：缓存梯度范数。"""
        use_dist = dist.is_initialized() and dist.get_world_size() > 1
        cache = {}
        for name, param in model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            local_sq = param.grad.data.norm().square()
            if use_dist:
                dist.all_reduce(local_sq, op=dist.ReduceOp.SUM)
            # 剥离 DDP 的 module. 前缀和 FSDP 的 _fsdp_wrapped_module 前缀
            clean_name = name.replace("._fsdp_wrapped_module", "")
            if clean_name.startswith("module."):
                clean_name = clean_name[len("module."):]
            cache[clean_name] = local_sq.sqrt().item()
        self._grad_cache = cache

    def log_step(self, step, metrics_dict, model, log_params=True):
        if not self._enabled:
            return
        self._log_training_scalars(step, metrics_dict)
        if log_params:
            self._log_param_norms(step, metrics_dict.get('lr', 1e-4))
        self._log_neuron_dynamics(step)
        self._log_beta_distribution(step, model)
        self._log_dynamic_k(step, model)
        self._log_associative_memory(step, model)
        self._log_health(step, model)

    def log_save_point(self, step, model):
        if not self._enabled:
            return
        self._log_histograms(step)
        self._log_compensation_factors(step, model)

    def close(self):
        if not self._enabled:
            return
        self._writer.close()

    # ====== 注册表 ======

    def _build_registry(self, model):
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
        semantics = []
        for i, layer_module in enumerate(model.layers):
            prefix = f"layer_{i:02d}"

            if not hasattr(layer_module, 'snn_block'):
                for name in ('neuron_k', 'neuron_v', 'neuron_q', 'neuron_gate'):
                    neuron = getattr(layer_module, name, None)
                    if neuron is not None and hasattr(neuron, 'w'):
                        semantics.append(
                            (f"{prefix}/mem_{name}_beta", neuron.w,
                             torch.sigmoid, "beta"))
                continue

            block = layer_module.snn_block
            ffn = layer_module.snn_ffn

            semantics.append((f"{prefix}/input1_beta", layer_module.input_neuron1.w,
                              torch.sigmoid, "beta"))
            semantics.append((f"{prefix}/input2_beta", layer_module.input_neuron2.w,
                              torch.sigmoid, "beta"))
            semantics.append((f"{prefix}/block_beta_t", block.b_beta,
                              torch.sigmoid, "beta(t)"))
            semantics.append((f"{prefix}/block_alpha_t", block.b_alpha,
                              F.softplus, "alpha(t)"))
            v_th_min = block.v_th_min
            semantics.append((f"{prefix}/block_vth_t", block.b_th,
                              lambda x, m=v_th_min: m + torch.abs(x), "V_th(t)"))
            semantics.append((f"{prefix}/ffn_gate_beta", ffn.gate_neuron.w,
                              torch.sigmoid, "beta"))
            semantics.append((f"{prefix}/ffn_up_beta", ffn.up_neuron.w,
                              torch.sigmoid, "beta"))

        semantics.append(("global/output_beta", model.output_neuron.w,
                          torch.sigmoid, "beta"))
        return semantics

    # ====== 1. 训练标量 ======

    def _log_training_scalars(self, step, metrics):
        w = self._writer
        for key in ('loss', 'ppl', 'lr', 'tps', 'tokens_seen', 'ponder_cost'):
            if key in metrics:
                w.add_scalar(f"train/{key}", metrics[key], step)
        if 'memory_current_gb' in metrics:
            w.add_scalar("train/memory/current_gb", metrics['memory_current_gb'], step)
        if 'memory_peak_gb' in metrics:
            w.add_scalar("train/memory/peak_gb", metrics['memory_peak_gb'], step)

    # ====== 2. 参数 norm + 梯度 ======

    def _log_param_norms(self, step, lr):
        w = self._writer
        cache = self._grad_cache
        layer_grad_sums = {}

        for name, (tag, param) in self._registry.items():
            if not param.requires_grad:
                continue
            weight_norm = param.data.norm().item()
            w.add_scalar(f"params/{tag}/weight_norm", weight_norm, step)

            grad_norm = cache.get(name, None)
            if grad_norm is None and param.grad is not None:
                grad_norm = param.grad.norm().item()
            if grad_norm is not None:
                w.add_scalar(f"params/{tag}/grad_norm", grad_norm, step)
                update_ratio = (lr * grad_norm) / (weight_norm + 1e-8)
                w.add_scalar(f"params/{tag}/update_ratio", update_ratio, step)
                parts = name.split('.')
                if parts[0] == 'layers' and parts[1].isdigit():
                    idx = int(parts[1])
                    layer_grad_sums[idx] = layer_grad_sums.get(idx, 0.0) + grad_norm ** 2

        if len(layer_grad_sums) >= 2:
            layer_norms = {k: v ** 0.5 for k, v in layer_grad_sums.items()}
            max_gn = max(layer_norms.values())
            min_gn = min(layer_norms.values())
            w.add_scalar("grad_health/layer_grad_ratio", max_gn / (min_gn + 1e-12), step)
            w.add_scalar("grad_health/layer_grad_max", max_gn, step)
            w.add_scalar("grad_health/layer_grad_min", min_gn, step)

    # ====== 3. 神经元动力学 ======

    def _log_neuron_dynamics(self, step):
        w = self._writer
        for tag, param, transform_fn, metric_name in self._neuron_semantics:
            with torch.no_grad():
                val = transform_fn(param.data)
                w.add_scalar(f"neuron_dynamics/{tag}/mean", val.mean().item(), step)
                w.add_scalar(f"neuron_dynamics/{tag}/std", val.std().item(), step)

    # ====== 4. PonderNet E[K] ======

    def _log_dynamic_k(self, step, model):
        w = self._writer
        for i, layer_module in enumerate(model.layers):
            # SNN 层 (block_halt + ffn_halt)
            if hasattr(layer_module, 'block_halt'):
                ek_min = getattr(layer_module, '_ek_min', None)
                ek_max = getattr(layer_module, '_ek_max', None)
                if ek_min is not None:
                    w.add_scalar(f"ponder/layer_{i:02d}/ek_min", ek_min, step)
                    w.add_scalar(f"ponder/layer_{i:02d}/ek_max", ek_max, step)
                with torch.no_grad():
                    for name, halt in [('block_halt', layer_module.block_halt),
                                       ('ffn_halt', layer_module.ffn_halt)]:
                        w.add_scalar(f"halt/layer_{i:02d}/{name}/weight_norm",
                                     halt.weight.data.norm().item(), step)
            # 海马体层 (ffn_halt only)
            elif hasattr(layer_module, 'ffn_halt') and not hasattr(layer_module, 'block_halt'):
                ek_min = getattr(layer_module, '_ek_min', None)
                ek_max = getattr(layer_module, '_ek_max', None)
                if ek_min is not None:
                    w.add_scalar(f"ponder/layer_{i:02d}/ek_min", ek_min, step)
                    w.add_scalar(f"ponder/layer_{i:02d}/ek_max", ek_max, step)
                with torch.no_grad():
                    w.add_scalar(f"halt/layer_{i:02d}/ffn_halt/weight_norm",
                                 layer_module.ffn_halt.weight.data.norm().item(), step)

    # ====== 5. β 分布演化（论文级） ======

    def _log_beta_distribution(self, step, model):
        """每层 β 的均值/std/min/max — 追踪时间尺度分布是否健康。"""
        w = self._writer
        all_betas = []
        for i, layer_module in enumerate(model.layers):
            if not hasattr(layer_module, 'snn_block'):
                continue
            with torch.no_grad():
                b_raw = layer_module.snn_block.b_beta.data
                beta = torch.sigmoid(b_raw)
                w.add_scalar(f"beta_dist/layer_{i:02d}/mean", beta.mean().item(), step)
                w.add_scalar(f"beta_dist/layer_{i:02d}/std", beta.std().item(), step)
                w.add_scalar(f"beta_dist/layer_{i:02d}/min", beta.min().item(), step)
                w.add_scalar(f"beta_dist/layer_{i:02d}/max", beta.max().item(), step)
                # raw b_beta（sigmoid 前，bf16 下也能看到变化）
                w.add_scalar(f"beta_raw/layer_{i:02d}/mean", b_raw.mean().item(), step)
                w.add_scalar(f"beta_raw/layer_{i:02d}/std", b_raw.std().item(), step)
                all_betas.append(beta)

        # 全局 β 统计
        if all_betas:
            all_beta = torch.cat(all_betas)
            w.add_scalar("beta_dist/global/mean", all_beta.mean().item(), step)
            w.add_scalar("beta_dist/global/std", all_beta.std().item(), step)
            w.add_scalar("beta_dist/global/min", all_beta.min().item(), step)
            w.add_scalar("beta_dist/global/max", all_beta.max().item(), step)

        # 输入神经元 β
        for i, layer_module in enumerate(model.layers):
            if hasattr(layer_module, 'input_neuron1'):
                with torch.no_grad():
                    b1 = torch.sigmoid(layer_module.input_neuron1.w.data)
                    b2 = torch.sigmoid(layer_module.input_neuron2.w.data)
                    w.add_scalar(f"beta_dist/layer_{i:02d}/input1_mean", b1.mean().item(), step)
                    w.add_scalar(f"beta_dist/layer_{i:02d}/input2_mean", b2.mean().item(), step)

        # 海马体层 gate_neuron β + input_neuron2 β + M_state 范数
        for i, layer_module in enumerate(model.layers):
            if not hasattr(layer_module, 'gate_neuron'):
                continue
            with torch.no_grad():
                # gate neuron β（控制 attention write gate）
                gate_beta = torch.sigmoid(layer_module.gate_neuron.w.data)
                w.add_scalar(f"attn/layer_{i:02d}/gate_beta_mean", gate_beta.mean().item(), step)
                w.add_scalar(f"attn/layer_{i:02d}/gate_beta_std", gate_beta.std().item(), step)
                # gate neuron V_th
                gate_vth = layer_module.gate_neuron.v_th.data
                w.add_scalar(f"attn/layer_{i:02d}/gate_vth_mean", gate_vth.mean().item(), step)
                # input_neuron2 β（FFN 子层输入）
                if hasattr(layer_module, 'input_neuron2'):
                    in2_beta = torch.sigmoid(layer_module.input_neuron2.w.data)
                    w.add_scalar(f"attn/layer_{i:02d}/input2_beta_mean", in2_beta.mean().item(), step)
                # M_state 范数（关联记忆矩阵累积状态）
                M = getattr(layer_module, 'M_state', None)
                if M is not None and not isinstance(M, (int, float)):
                    w.add_scalar(f"attn/layer_{i:02d}/M_state_norm", M.norm().item(), step)

    # ====== 6. 联想记忆层监控 ======

    def _log_associative_memory(self, step, model):
        """联想记忆层 M 的范数 + write_gate 发放率。"""
        w = self._writer
        for i, layer_module in enumerate(model.layers):
            if not hasattr(layer_module, 'neuron_gate'):
                continue

            # M 矩阵范数（如果已初始化）
            M = getattr(layer_module, 'M', None)
            if M is not None and not isinstance(M, float):
                for g in range(M.shape[0]):
                    m_norm = M[g].norm().item()
                    w.add_scalar(f"memory/layer_{i:02d}/M_group{g}_norm", m_norm, step)
                # M 的有效秩估计（奇异值比）
                with torch.no_grad():
                    # 取第一个 batch 的 M_slow (最后一组)
                    M_last = M[-1, 0] if M.dim() == 4 else M[-1]
                    if M_last.numel() > 0:
                        s = torch.linalg.svdvals(M_last.float())
                        if s[0] > 1e-8:
                            effective_rank = (s / s[0]).sum().item()
                            w.add_scalar(f"memory/layer_{i:02d}/effective_rank", effective_rank, step)

    # ====== 7. 健康监控（坍缩/癫痫/趋同检测） ======

    @staticmethod
    def _gini(values):
        if len(values) < 2:
            return 0.0
        vals = sorted(values)
        n = len(vals)
        total = sum(vals)
        if total < 1e-12:
            return 0.0
        cum = sum((i + 1) * v for i, v in enumerate(vals))
        return (2.0 * cum) / (n * total) - (n + 1) / n

    def _log_health(self, step, model):
        """坍缩/癫痫/趋同检测 + 综合健康得分。"""
        w = self._writer

        # ====== 1. 层间梯度 Gini ======
        cache = self._grad_cache
        layer_grad_sums = {}
        for name, grad_norm in cache.items():
            parts = name.split('.')
            if parts[0] == 'layers' and parts[1].isdigit():
                idx = int(parts[1])
                layer_grad_sums[idx] = layer_grad_sums.get(idx, 0.0) + grad_norm ** 2
        if len(layer_grad_sums) >= 2:
            layer_norms = [v ** 0.5 for v in layer_grad_sums.values()]
            grad_gini = self._gini(layer_norms)
            w.add_scalar("health/grad_gini", grad_gini, step)
            total_gn = sum(layer_norms)
            for idx in sorted(layer_grad_sums.keys()):
                share = (layer_grad_sums[idx] ** 0.5) / (total_gn + 1e-12)
                w.add_scalar(f"grad_share/layer_{idx:02d}", share, step)
        else:
            grad_gini = 0.0

        # ====== 2. β 趋同度（层内 std → 0 = 丧失多样性） ======
        beta_stds = []
        for i, layer_module in enumerate(model.layers):
            if not hasattr(layer_module, 'snn_block'):
                continue
            with torch.no_grad():
                beta = torch.sigmoid(layer_module.snn_block.b_beta.data)
                beta_std = beta.std().item()
                beta_stds.append(beta_std)
                w.add_scalar(f"health/beta_std/layer_{i:02d}", beta_std, step)
        if beta_stds:
            min_beta_std = min(beta_stds)
            w.add_scalar("health/beta_std_min", min_beta_std, step)
            # 趋同警告：std < 0.01 说明 β 几乎一样了
            w.add_scalar("health/beta_converged_layers",
                         sum(1 for s in beta_stds if s < 0.01), step)

        # ====== 3. 死神经元 / 癫痫神经元检测 ======
        # 通过 b_beta 间接推断：极高 β + 高 V_th = 死（永远不发放）
        # 极低 β + 低 V_th = 癫痫（每步都发放）
        dead_count = 0
        epileptic_count = 0
        total_neurons = 0
        for layer_module in model.layers:
            if not hasattr(layer_module, 'snn_block'):
                continue
            block = layer_module.snn_block
            with torch.no_grad():
                beta = torch.sigmoid(block.b_beta.data)
                v_th = block.v_th_min + torch.abs(block.b_th.data)
                # 稳态 V = α·I/(1-β)，假设 α·I ≈ 0.2（典型值）
                v_steady_est = 0.2 / (1.0 - beta + 1e-8)
                # 死神经元：稳态 V 远低于 V_th
                dead = (v_steady_est < 0.1 * v_th).sum().item()
                # 癫痫：稳态 V 远高于 V_th
                epileptic = (v_steady_est > 10.0 * v_th).sum().item()
                n = beta.numel()
                dead_count += dead
                epileptic_count += epileptic
                total_neurons += n

        if total_neurons > 0:
            dead_rate = dead_count / total_neurons
            epileptic_rate = epileptic_count / total_neurons
            w.add_scalar("health/dead_neuron_rate", dead_rate, step)
            w.add_scalar("health/epileptic_neuron_rate", epileptic_rate, step)
        else:
            dead_rate = 0.0
            epileptic_rate = 0.0

        # ====== 4. 层间输出范数比（检测梯度爆炸/消失） ======
        # 通过 residual_proj 权重范数间接估计
        layer_out_norms = []
        for layer_module in model.layers:
            if hasattr(layer_module, 'block_out_proj'):
                with torch.no_grad():
                    norm = layer_module.block_out_proj.weight.data.norm().item()
                    layer_out_norms.append(norm)
        if len(layer_out_norms) >= 2:
            out_ratio = max(layer_out_norms) / (min(layer_out_norms) + 1e-12)
            w.add_scalar("health/layer_output_norm_ratio", out_ratio, step)

        # ====== 5. 综合得分 ======
        s_grad = max(0.0, 1.0 - grad_gini * 2.0)
        s_beta = max(0.0, 1.0 - sum(1 for s in beta_stds if s < 0.01) / max(len(beta_stds), 1))
        s_neuron = max(0.0, 1.0 - (dead_rate + epileptic_rate) * 5.0)
        score = 0.4 * s_grad + 0.3 * s_beta + 0.3 * s_neuron
        w.add_scalar("health/score", score, step)

    # ====== 直方图 + 补偿因子（save_interval） ======

    def _log_histograms(self, step):
        w = self._writer
        for name, (tag, param) in self._registry.items():
            if not param.requires_grad:
                continue
            w.add_histogram(f"histograms/{tag}/weight", param.data, step)
            if param.grad is not None:
                w.add_histogram(f"histograms/{tag}/grad", param.grad, step)

    def _log_compensation_factors(self, step, model):
        w = self._writer
        for i, layer_module in enumerate(model.layers):
            if not hasattr(layer_module, 'snn_block'):
                continue
            block = layer_module.snn_block
            with torch.no_grad():
                beta = torch.sigmoid(block.b_beta.data)
                sigmoid_deriv = (beta * (1.0 - beta)).mean().item()
                w.add_scalar(f"compensation/layer_{i:02d}/sigmoid_deriv_mean",
                             sigmoid_deriv, step)
                softplus_deriv = torch.sigmoid(block.b_alpha.data).mean().item()
                w.add_scalar(f"compensation/layer_{i:02d}/softplus_deriv_mean",
                             softplus_deriv, step)
