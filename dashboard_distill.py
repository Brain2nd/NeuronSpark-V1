"""
蒸馏训练看板: Mamba → BioSSM (TensorBoard)

专为 FSDP 30B 模型优化:
  - 不使用 summon_full_params (30B 模型显存不允许全聚合)
  - 梯度范数通过 all_reduce 聚合 (FSDP 安全)
  - 前向状态直接从 raw_model 读取 (use_orig_params=True)

两级频率:
  - log_step()       每 log_interval 步: 训练标量 + 逐层 cosine/ponder/firing_rate + 梯度健康
  - log_save_point() 每 save_interval 步: 逐层梯度详情 + 补偿因子

用法:
    dashboard = DistillDashboard(log_dir='runs/distill_v3', model=raw_model, rank=rank)

    # 边界步
    dashboard.cache_grad_norms(model)     # step 前: 缓存梯度 (所有 rank 参与)
    optimizer.step()
    dashboard.log_step(global_step, metrics_dict, raw_model)
    optimizer.zero_grad(set_to_none=True)

    # 保存点
    dashboard.log_save_point(global_step, raw_model)

    # 训练结束
    dashboard.close()
"""

import torch
import torch.nn.functional as F
import torch.distributed as dist


class DistillDashboard:
    """蒸馏训练看板 (仅 rank 0 记录, 其余 no-op)。

    Args:
        log_dir: TensorBoard 日志目录, None 时禁用
        model: raw_model (DistillHybridModel, FSDP wrap 前)
        rank: 分布式 rank
    """

    def __init__(self, log_dir, model, rank=0):
        self._enabled = (log_dir is not None) and (rank == 0)
        self._rank = rank
        self._grad_cache = {}  # clean_name → grad_norm
        if not self._enabled:
            return

        from torch.utils.tensorboard import SummaryWriter
        self._writer = SummaryWriter(log_dir=log_dir)

        # 记录 Mamba 层索引 → 顺序 SSM 编号 (用于 TensorBoard tag)
        self._mamba_indices = list(model.mamba_indices)
        self._idx_to_ssm = {idx: i for i, idx in enumerate(self._mamba_indices)}

    # ====== 公开方法 ======

    def cache_grad_norms(self, fsdp_model):
        """optimizer.step() 前调用: 缓存 BioSSM 参数梯度范数 (FSDP 安全)。

        FSDP FULL_SHARD 下 param.grad 只在拥有该分片的 rank 上有值,
        不能对每个参数单独 all_reduce (各 rank 跳过不同参数 → 死锁)。
        改为: 收集所有可训练参数 local squared norm (无 grad 填 0),
        合并为一个 tensor 做单次 all_reduce(SUM)。
        """
        use_dist = dist.is_initialized() and dist.get_world_size() > 1

        # 固定顺序收集所有可训练参数 (所有 rank 一致)
        names = []
        local_sqs = []
        for name, param in fsdp_model.named_parameters():
            if not param.requires_grad:
                continue
            clean_name = name.replace("._fsdp_wrapped_module", "")
            names.append(clean_name)
            if param.grad is not None:
                local_sqs.append(param.grad.data.float().norm().square().item())
            else:
                local_sqs.append(0.0)

        if not names:
            self._grad_cache = {}
            return

        # 单次 all_reduce: 所有 rank 参与, tensor 大小一致
        device = next(fsdp_model.parameters()).device
        sq_tensor = torch.tensor(local_sqs, dtype=torch.float32, device=device)
        if use_dist:
            dist.all_reduce(sq_tensor, op=dist.ReduceOp.SUM)

        self._grad_cache = {
            name: sq_tensor[i].sqrt().item()
            for i, name in enumerate(names)
        }

    def log_step(self, step, metrics, raw_model):
        """轻量日志: 训练标量 + 逐层指标 + 梯度健康。"""
        if not self._enabled:
            return
        self._log_training_scalars(step, metrics)
        self._log_per_layer_metrics(step, raw_model)
        self._log_grad_health(step)

    def log_save_point(self, step, raw_model):
        """重量日志: 逐层梯度详情 + 补偿因子统计。"""
        if not self._enabled:
            return
        self._log_per_layer_grad_detail(step)

    def close(self):
        """关闭 TensorBoard writer。"""
        if not self._enabled:
            return
        self._writer.close()

    # ====== Stage 1/2 渐进蒸馏日志 ======

    def log_stage1_step(self, step, mamba_idx, metrics):
        """Stage 1 逐块对齐日志。"""
        if not self._enabled:
            return
        w = self._writer
        ssm_id = self._idx_to_ssm.get(mamba_idx, mamba_idx)
        tag = f"ssm_{ssm_id:02d}"

        for key in ['mse_loss', 'cos_loss', 'cos_sim', 'ponder_cost',
                     'ek_floor_cost', 'loss', 'lr', 'tps']:
            if key in metrics:
                w.add_scalar(f"stage1/{tag}/{key}", metrics[key], step)

        # 汇总
        if 'cos_sim' in metrics:
            w.add_scalar("stage1/active_cos_sim", metrics['cos_sim'], step)
        if 'mse_loss' in metrics:
            w.add_scalar("stage1/active_mse", metrics['mse_loss'], step)
        if 'memory_peak_gb' in metrics:
            w.add_scalar("stage1/memory_peak_gb", metrics['memory_peak_gb'], step)

    def log_stage2_step(self, step, phase, metrics):
        """Stage 2 端到端整合日志。"""
        if not self._enabled:
            return
        w = self._writer

        for key in ['kl_loss', 'ce_loss', 'mse_loss', 'ponder_cost',
                     'ek_floor_cost', 'loss', 'alpha_ce', 'alpha_kl',
                     'beta_mse', 'temperature', 'lr', 'tps']:
            if key in metrics:
                w.add_scalar(f"stage2/{key}", metrics[key], step)

        if 'memory_peak_gb' in metrics:
            w.add_scalar("stage2/memory_peak_gb", metrics['memory_peak_gb'], step)

        # 逐块 cosine
        per_block_cos = metrics.get('per_block_cosine', {})
        if per_block_cos:
            cos_vals = list(per_block_cos.values())
            w.add_scalar("stage2/cos_mean", sum(cos_vals) / len(cos_vals), step)
            w.add_scalar("stage2/cos_min", min(cos_vals), step)
            w.add_scalar("stage2/cos_max", max(cos_vals), step)

            for idx, cos_val in per_block_cos.items():
                ssm_id = self._idx_to_ssm.get(idx, idx)
                w.add_scalar(f"stage2/cos_per_block/ssm_{ssm_id:02d}", cos_val, step)

        w.add_scalar("stage2/phase", phase, step)

    def log_phase_transition(self, step, phase, active_blocks):
        """标记阶段切换。"""
        if not self._enabled:
            return
        w = self._writer
        w.add_scalar("stage2/phase_transition", phase, step)
        w.add_scalar("stage2/n_active_blocks", len(active_blocks), step)
        # 以文本形式记录活跃块
        block_str = ','.join(str(b) for b in sorted(active_blocks))
        w.add_text("stage2/active_blocks", f"Phase {phase}: [{block_str}]", step)

    # ====== 训练标量 ======

    def _log_training_scalars(self, step, metrics):
        """训练标量: loss 组件 + 蒸馏权重 + 系统指标。"""
        w = self._writer
        scalar_keys = [
            'loss', 'ce_loss', 'cosine_loss', 'ponder_cost', 'ek_floor_cost', 'ek_smooth_cost',
            'alpha_ce', 'beta_hidden',
            'lr', 'tps', 'tokens_seen', 'seq_len',
        ]
        for key in scalar_keys:
            if key in metrics:
                w.add_scalar(f"train/{key}", metrics[key], step)
        if 'memory_current_gb' in metrics:
            w.add_scalar("train/memory/current_gb", metrics['memory_current_gb'], step)
        if 'memory_peak_gb' in metrics:
            w.add_scalar("train/memory/peak_gb", metrics['memory_peak_gb'], step)

        # PPL (from CE loss)
        if 'ce_loss' in metrics and metrics['ce_loss'] is not None:
            import math
            ppl = math.exp(min(metrics['ce_loss'], 20.0))
            w.add_scalar("train/ppl", ppl, step)

    # ====== 逐层指标 (从 forward 状态读取, 无需 summon) ======

    def _log_per_layer_metrics(self, step, raw_model):
        """逐层 cosine alignment / PonderNet E[K] / 发放率。"""
        w = self._writer

        for idx in self._mamba_indices:
            ssm_id = self._idx_to_ssm[idx]
            tag = f"ssm_{ssm_id:02d}"
            bio_mixer = raw_model.bio_ssm_modules[str(idx)]

            # Cosine alignment distance (低 = 好)
            cos_val = raw_model._layer_cosine_dict.get(idx, None)
            if cos_val is not None:
                w.add_scalar(f"cosine_per_layer/{tag}", cos_val, step)

            # PonderNet E[K]
            pc = bio_mixer.ponder_cost
            if pc is not None:
                pc_val = pc.item() if torch.is_tensor(pc) else float(pc)
                w.add_scalar(f"ponder/{tag}/E_K", pc_val, step)

            # EK floor cost
            efc = bio_mixer.ek_floor_cost
            if efc is not None:
                efc_val = efc.item() if torch.is_tensor(efc) else float(efc)
                w.add_scalar(f"ponder/{tag}/ek_floor_cost", efc_val, step)

            # 隐层发放率
            fr = getattr(bio_mixer.bio_ssm.snn_block, '_firing_rate_hidden', None)
            if fr is not None:
                w.add_scalar(f"firing_rate/{tag}/hidden", fr, step)

        # 汇总统计
        cosine_vals = list(raw_model._layer_cosine_dict.values())
        if cosine_vals:
            w.add_scalar("cosine_summary/mean", sum(cosine_vals) / len(cosine_vals), step)
            w.add_scalar("cosine_summary/max", max(cosine_vals), step)
            w.add_scalar("cosine_summary/min", min(cosine_vals), step)

        # 发放率汇总
        all_frs = []
        for idx in self._mamba_indices:
            fr = getattr(raw_model.bio_ssm_modules[str(idx)].bio_ssm.snn_block,
                         '_firing_rate_hidden', None)
            if fr is not None:
                all_frs.append(fr)
        if all_frs:
            w.add_scalar("firing_rate_summary/mean", sum(all_frs) / len(all_frs), step)
            w.add_scalar("firing_rate_summary/min", min(all_frs), step)
            w.add_scalar("firing_rate_summary/max", max(all_frs), step)
            dead = sum(1 for f in all_frs if f < 0.001) / len(all_frs)
            saturated = sum(1 for f in all_frs if f > 0.8) / len(all_frs)
            w.add_scalar("health/fr_dead_ratio", dead, step)
            w.add_scalar("health/fr_saturated_ratio", saturated, step)

    # ====== 梯度健康 ======

    def _log_grad_health(self, step):
        """BioSSM 层间梯度均衡性。"""
        w = self._writer
        cache = self._grad_cache
        if not cache:
            return

        # 按 SSM 层聚合梯度范数
        layer_grad_sums = {}  # ssm_idx → sum of grad_norm²
        for name, grad_norm in cache.items():
            if 'bio_ssm_modules' not in name:
                continue
            # 提取 Mamba 层索引: bio_ssm_modules.0.bio_ssm...
            parts = name.split('.')
            try:
                bio_idx = parts.index('bio_ssm_modules')
                mamba_idx = int(parts[bio_idx + 1])
            except (ValueError, IndexError):
                continue
            layer_grad_sums[mamba_idx] = layer_grad_sums.get(mamba_idx, 0.0) + grad_norm ** 2

        if len(layer_grad_sums) < 2:
            return

        layer_norms = {k: v ** 0.5 for k, v in layer_grad_sums.items()}
        max_gn = max(layer_norms.values())
        min_gn = min(layer_norms.values())
        grad_ratio = max_gn / (min_gn + 1e-12)
        w.add_scalar("grad_health/bio_ssm_grad_ratio", grad_ratio, step)
        w.add_scalar("grad_health/bio_ssm_grad_max", max_gn, step)
        w.add_scalar("grad_health/bio_ssm_grad_min", min_gn, step)

        # Gini 系数
        vals = sorted(layer_norms.values())
        n = len(vals)
        total = sum(vals)
        if total > 1e-12:
            cum = sum((i + 1) * v for i, v in enumerate(vals))
            gini = (2.0 * cum) / (n * total) - (n + 1) / n
        else:
            gini = 0.0
        w.add_scalar("grad_health/bio_ssm_grad_gini", gini, step)

        # 逐层梯度份额
        for mamba_idx in sorted(layer_norms.keys()):
            if mamba_idx in self._idx_to_ssm:
                ssm_id = self._idx_to_ssm[mamba_idx]
                share = layer_norms[mamba_idx] / (sum(layer_norms.values()) + 1e-12)
                w.add_scalar(f"grad_share/ssm_{ssm_id:02d}", share, step)

        # 综合健康分 (0~1, 1=健康)
        all_frs = []
        for name, grad_norm in cache.items():
            if 'bio_ssm_modules' in name:
                all_frs.append(grad_norm)
        s_grad = max(0.0, 1.0 - gini * 2.0)
        s_ratio = max(0.0, 1.0 - (grad_ratio - 1.0) / 50.0)  # ratio=1→1.0, ratio=51→0.0
        score = 0.5 * s_grad + 0.5 * s_ratio
        w.add_scalar("health/score", score, step)

    # ====== 逐层梯度详情 (save_interval) ======

    def _log_per_layer_grad_detail(self, step):
        """每个 BioSSM 参数的梯度范数。"""
        w = self._writer
        cache = self._grad_cache
        for name, grad_norm in cache.items():
            if 'bio_ssm_modules' not in name:
                continue
            # 转换为可读 tag
            parts = name.split('.')
            try:
                bio_idx = parts.index('bio_ssm_modules')
                mamba_idx = int(parts[bio_idx + 1])
                if mamba_idx in self._idx_to_ssm:
                    ssm_id = self._idx_to_ssm[mamba_idx]
                    param_path = "/".join(parts[bio_idx + 2:])
                    w.add_scalar(f"grad_detail/ssm_{ssm_id:02d}/{param_path}", grad_norm, step)
            except (ValueError, IndexError):
                continue
