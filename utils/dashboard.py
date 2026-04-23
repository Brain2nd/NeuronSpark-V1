"""SNNDashboard: TensorBoard monitoring for NeuronSpark training.

Accepts either an HF `NeuronSparkForCausalLM` (extracts `.snn`) or the raw
`SNNLanguageModel`. Records weight/gradient dynamics + neuron health.

Three-level frequency:
  - log_step()              : every log_interval steps (train scalars + dynamics + health)
  - log_step(log_params=True): also logs param/grad norms + update ratios
  - log_save_point()        : every save_interval steps (histograms + compensation factors)

Categories:
  1. Training scalars (loss/ppl/lr/tps/memory)
  2. Param norm + grad norm + update_ratio
  3. Neuron dynamics (β/α/V_th semantic values)
  4. PonderNet E[K] per-layer extrema
  5. β distribution (mean/std/min/max per layer)
  6. Associative-memory layer (M norm, write_gate firing rate)
  7. Collapse/seizure/convergence health + composite score
  8. Histograms (save points)
  9. Gradient-compensation factors
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F


def _inner_snn(model):
    """Unwrap DDP/DeepSpeed + HF wrapper → raw SNNLanguageModel."""
    raw = model.module if hasattr(model, "module") else model
    if hasattr(raw, "snn"):
        return raw.snn
    return raw


class SNNDashboard:
    def __init__(self, log_dir, model, rank: int = 0):
        self._enabled = (log_dir is not None) and (rank == 0)
        self._rank = rank
        self._grad_cache: dict[str, float] = {}
        if not self._enabled:
            return

        from torch.utils.tensorboard import SummaryWriter
        self._writer = SummaryWriter(log_dir=log_dir)
        snn = _inner_snn(model)
        self._registry = self._build_registry(snn)
        self._neuron_semantics = self._build_neuron_semantics(snn)

    # ====== public ======

    def cache_grad_norms(self, model):
        """Before optimizer.step(), capture per-param grad norm.

        Multi-GPU note (ZeRO-2):
          Parameters are replicated across ranks but gradients are reduce-scatter
          SHARDED — each rank holds only 1/world_size of the full gradient.
          We all-reduce the squared norm across ranks to get the GLOBAL grad
          norm (mathematically correct: ||g||² = Σ_rank ||g_rank||²).
          This gives correct scalar norms but NOT full gradient tensors —
          histograms / per-element distributions are NOT reconstructible here
          without expensive all-gather (see `_log_histograms`, disabled under ZeRO).
        """
        use_dist = dist.is_initialized() and dist.get_world_size() > 1
        cache: dict[str, float] = {}
        for name, p in _inner_snn(model).named_parameters():
            if not p.requires_grad or p.grad is None:
                continue
            local_sq = p.grad.data.norm().square()
            if use_dist:
                dist.all_reduce(local_sq, op=dist.ReduceOp.SUM)
            clean = name.replace("._fsdp_wrapped_module", "")
            if clean.startswith("module."):
                clean = clean[len("module."):]
            cache[clean] = local_sq.sqrt().item()
        self._grad_cache = cache
        # Mark distributed mode for histogram gating
        self._is_distributed_sharded = use_dist

    def log_step(self, step, metrics_dict, model, log_params: bool = True):
        if not self._enabled:
            return
        snn = _inner_snn(model)
        self._log_training_scalars(step, metrics_dict)
        if log_params:
            self._log_param_norms(step, metrics_dict.get("lr", 1e-4))
        self._log_neuron_dynamics(step)
        self._log_beta_distribution(step, snn)
        self._log_dynamic_k(step, snn)
        self._log_associative_memory(step, snn)
        self._log_ponder_collapse(step, snn)       # v3: K collapse statistics
        self._log_ponder_gradients(step, snn)      # v3: k_predictor grad health
        self._log_health(step, snn)

    def log_save_point(self, step, model):
        if not self._enabled:
            return
        self._log_histograms(step)
        self._log_compensation_factors(step, _inner_snn(model))

    def close(self):
        if not self._enabled:
            return
        self._writer.close()

    # ====== registry ======

    def _build_registry(self, snn):
        registry = {}
        for name, p in snn.named_parameters():
            parts = name.split(".")
            if parts[0] == "layers" and parts[1].isdigit():
                layer_idx = int(parts[1])
                tag = f"layer_{layer_idx:02d}/" + "/".join(parts[2:])
            else:
                tag = "global/" + "/".join(parts)
            registry[name] = (tag, p)
        return registry

    def _build_neuron_semantics(self, snn):
        semantics = []
        for i, layer in enumerate(snn.layers):
            prefix = f"layer_{i:02d}"

            if not hasattr(layer, "snn_block"):
                # v3 SNNAttentionDecoderLayer: gate_neuron (write gate) + input_neuron2 (FFN)
                for name in ("gate_neuron", "input_neuron2"):
                    neuron = getattr(layer, name, None)
                    if neuron is not None and hasattr(neuron, "w"):
                        semantics.append(
                            (f"{prefix}/mem_{name}_beta", neuron.w, torch.sigmoid, "beta")
                        )
                # FFN PLIF neurons in memory layers (gate/up)
                if hasattr(layer, "snn_ffn"):
                    ffn = layer.snn_ffn
                    semantics.append((f"{prefix}/ffn_gate_beta", ffn.gate_neuron.w, torch.sigmoid, "beta"))
                    semantics.append((f"{prefix}/ffn_up_beta", ffn.up_neuron.w, torch.sigmoid, "beta"))
                continue

            block = layer.snn_block
            ffn = layer.snn_ffn
            semantics.append((f"{prefix}/input1_beta", layer.input_neuron1.w, torch.sigmoid, "beta"))
            semantics.append((f"{prefix}/input2_beta", layer.input_neuron2.w, torch.sigmoid, "beta"))
            semantics.append((f"{prefix}/block_beta_t", block.b_beta, torch.sigmoid, "beta(t)"))
            semantics.append((f"{prefix}/block_alpha_t", block.b_alpha, F.softplus, "alpha(t)"))
            v_th_min = block.v_th_min
            semantics.append(
                (f"{prefix}/block_vth_t", block.b_th,
                 lambda x, m=v_th_min: m + torch.abs(x), "V_th(t)")
            )
            semantics.append((f"{prefix}/ffn_gate_beta", ffn.gate_neuron.w, torch.sigmoid, "beta"))
            semantics.append((f"{prefix}/ffn_up_beta", ffn.up_neuron.w, torch.sigmoid, "beta"))

        semantics.append(("global/output_beta", snn.output_neuron.w, torch.sigmoid, "beta"))
        return semantics

    # ====== 1. training scalars ======

    def _log_training_scalars(self, step, metrics):
        w = self._writer
        for key in ("loss", "ppl", "lr", "tps", "tokens_seen", "ponder_cost"):
            if key in metrics:
                w.add_scalar(f"train/{key}", metrics[key], step)
        if "memory_current_gb" in metrics:
            w.add_scalar("train/memory/current_gb", metrics["memory_current_gb"], step)
        if "memory_peak_gb" in metrics:
            w.add_scalar("train/memory/peak_gb", metrics["memory_peak_gb"], step)

    # ====== 2. param norms + grad ======

    def _log_param_norms(self, step, lr):
        w = self._writer
        cache = self._grad_cache
        layer_grad_sums = {}

        for name, (tag, p) in self._registry.items():
            if not p.requires_grad:
                continue
            weight_norm = p.data.norm().item()
            w.add_scalar(f"params/{tag}/weight_norm", weight_norm, step)

            grad_norm = cache.get(name, None)
            if grad_norm is None and p.grad is not None:
                grad_norm = p.grad.norm().item()
            if grad_norm is not None:
                w.add_scalar(f"params/{tag}/grad_norm", grad_norm, step)
                update_ratio = (lr * grad_norm) / (weight_norm + 1e-8)
                w.add_scalar(f"params/{tag}/update_ratio", update_ratio, step)
                parts = name.split(".")
                if parts[0] == "layers" and parts[1].isdigit():
                    idx = int(parts[1])
                    layer_grad_sums[idx] = layer_grad_sums.get(idx, 0.0) + grad_norm ** 2

        if len(layer_grad_sums) >= 2:
            layer_norms = {k: v ** 0.5 for k, v in layer_grad_sums.items()}
            max_gn = max(layer_norms.values())
            min_gn = min(layer_norms.values())
            w.add_scalar("grad_health/layer_grad_ratio", max_gn / (min_gn + 1e-12), step)
            w.add_scalar("grad_health/layer_grad_max", max_gn, step)
            w.add_scalar("grad_health/layer_grad_min", min_gn, step)

    # ====== 3. neuron dynamics ======

    def _log_neuron_dynamics(self, step):
        w = self._writer
        for tag, p, transform, metric_name in self._neuron_semantics:
            with torch.no_grad():
                val = transform(p.data)
                w.add_scalar(f"neuron_dynamics/{tag}/mean", val.mean().item(), step)
                w.add_scalar(f"neuron_dynamics/{tag}/std", val.std().item(), step)

    # ====== 4. PonderNet E[K] ======

    def _log_dynamic_k(self, step, snn):
        """Per-layer E[k_t] extrema (set by each layer's forward as diagnostic).

        v3: `block_halt` / `ffn_halt` modules removed; detailed k_predictor weight
        monitoring lives in `_log_ponder_gradients`. This method keeps only the
        cheap _ek_min/_ek_max scalars set by `_ponder_aggregate_v3` for timeline.
        """
        w = self._writer
        for i, layer in enumerate(snn.layers):
            ek_min = getattr(layer, "_ek_min", None)
            ek_max = getattr(layer, "_ek_max", None)
            if ek_min is not None:
                w.add_scalar(f"ponder/layer_{i:02d}/ek_min", ek_min, step)
                w.add_scalar(f"ponder/layer_{i:02d}/ek_max", ek_max, step)

    # ====== 5. β distribution ======

    def _log_beta_distribution(self, step, snn):
        w = self._writer
        all_betas = []
        for i, layer in enumerate(snn.layers):
            if not hasattr(layer, "snn_block"):
                continue
            with torch.no_grad():
                b_raw = layer.snn_block.b_beta.data
                beta = torch.sigmoid(b_raw)
                w.add_scalar(f"beta_dist/layer_{i:02d}/mean", beta.mean().item(), step)
                w.add_scalar(f"beta_dist/layer_{i:02d}/std", beta.std().item(), step)
                w.add_scalar(f"beta_dist/layer_{i:02d}/min", beta.min().item(), step)
                w.add_scalar(f"beta_dist/layer_{i:02d}/max", beta.max().item(), step)
                w.add_scalar(f"beta_raw/layer_{i:02d}/mean", b_raw.mean().item(), step)
                w.add_scalar(f"beta_raw/layer_{i:02d}/std", b_raw.std().item(), step)
                all_betas.append(beta)

        if all_betas:
            all_beta = torch.cat(all_betas)
            w.add_scalar("beta_dist/global/mean", all_beta.mean().item(), step)
            w.add_scalar("beta_dist/global/std", all_beta.std().item(), step)
            w.add_scalar("beta_dist/global/min", all_beta.min().item(), step)
            w.add_scalar("beta_dist/global/max", all_beta.max().item(), step)

        for i, layer in enumerate(snn.layers):
            if hasattr(layer, "input_neuron1"):
                with torch.no_grad():
                    b1 = torch.sigmoid(layer.input_neuron1.w.data)
                    b2 = torch.sigmoid(layer.input_neuron2.w.data)
                    w.add_scalar(f"beta_dist/layer_{i:02d}/input1_mean", b1.mean().item(), step)
                    w.add_scalar(f"beta_dist/layer_{i:02d}/input2_mean", b2.mean().item(), step)

        for i, layer in enumerate(snn.layers):
            if not hasattr(layer, "gate_neuron"):
                continue
            with torch.no_grad():
                gate_beta = torch.sigmoid(layer.gate_neuron.w.data)
                w.add_scalar(f"attn/layer_{i:02d}/gate_beta_mean", gate_beta.mean().item(), step)
                w.add_scalar(f"attn/layer_{i:02d}/gate_beta_std", gate_beta.std().item(), step)
                gate_vth = layer.gate_neuron.v_th.data
                w.add_scalar(f"attn/layer_{i:02d}/gate_vth_mean", gate_vth.mean().item(), step)
                if hasattr(layer, "input_neuron2"):
                    in2_beta = torch.sigmoid(layer.input_neuron2.w.data)
                    w.add_scalar(f"attn/layer_{i:02d}/input2_beta_mean", in2_beta.mean().item(), step)
                M = getattr(layer, "M_state", None)
                if M is not None and not isinstance(M, (int, float)):
                    w.add_scalar(f"attn/layer_{i:02d}/M_state_norm", M.norm().item(), step)

    # ====== 6. associative memory ======

    def _log_associative_memory(self, step, snn):
        """v3 SNNAttentionDecoderLayer M_state monitoring.

        Each SNNAttentionDecoderLayer maintains `self.M_state` — a per-forward
        cumulative K-V associative memory. Dashboard tracks its norm and
        effective rank as diagnostics for long-range retrieval health.
        """
        w = self._writer
        for i, layer in enumerate(snn.layers):
            M = getattr(layer, "M_state", None)
            if M is None or isinstance(M, (int, float)):
                continue  # 未初始化 / 非联想记忆层
            with torch.no_grad():
                w.add_scalar(f"memory/layer_{i:02d}/M_state_norm", M.norm().item(), step)
                # 有效秩: if 2D+ matrix, compute σ-ratio-based effective rank
                if M.dim() >= 2 and M.numel() > 0:
                    M_last = M[-1] if M.dim() >= 3 else M
                    if M_last.dim() >= 2 and M_last.numel() > 0:
                        try:
                            s = torch.linalg.svdvals(M_last.float())
                            if s[0] > 1e-8:
                                eff_rank = (s / s[0]).sum().item()
                                w.add_scalar(
                                    f"memory/layer_{i:02d}/effective_rank", eff_rank, step,
                                )
                        except RuntimeError:
                            pass  # SVD may fail on tiny/degenerate M

    # ====== 7a. v3 PonderNet K-collapse detection ======

    def _log_ponder_collapse(self, step, snn):
        """Detect per-layer halt-distribution collapse (v3 PonderNet).

        Per KPredictor (`block_k_predictor` / `ffn_k_predictor`), we inspect the
        running usage EMA (batch-averaged soft-halt probabilities across tokens)
        and the gradient-free `bias` buffer:

          1. usage_entropy      = H(usage_ema) / log(K)         in [0, 1]
                                  0 = full collapse to single k
                                  1 = uniform across k
          2. usage_min / max    = smallest / largest bin's EMA share
          3. dead_k             = fraction of k bins with EMA < 0.01 (under-used)
          4. bias_range         = max(bias) - min(bias)
          5. bias_at_clamp      = count of bias components at ±5 clamp
          6. global summary     : fraction of layers in collapse state

        Plus per-step diagnostic:
          7. y_hard_entropy     = entropy of the current step's batch one-hot
                                  (high = diverse token picks, low = all pick same k)
        """
        import math
        w = self._writer
        collapse_count_block = 0
        collapse_count_ffn = 0
        total_block = total_ffn = 0

        for i, layer in enumerate(snn.layers):
            # Per-sublayer: block (only SNNDecoderLayer) + ffn (both layer types)
            for sublayer_name, attr in (("block", "block_k_predictor"),
                                         ("ffn", "ffn_k_predictor")):
                kp = getattr(layer, attr, None)
                if kp is None:
                    continue
                K = kp.K

                with torch.no_grad():
                    usage = kp._usage_ema.float()          # (K,)
                    bias = kp.bias.float()                  # (K,)

                    # Normalize usage (should already sum to ~1)
                    usage = usage / usage.sum().clamp(min=1e-8)

                    # 1. Entropy / log(K) ∈ [0, 1]
                    ent = -(usage * torch.log(usage.clamp(min=1e-8))).sum().item()
                    norm_ent = ent / math.log(K)

                    # 2. min / max bin share
                    usage_min = usage.min().item()
                    usage_max = usage.max().item()

                    # 3. dead k fraction (usage < 0.01 = less than 1% of mass)
                    dead_k = (usage < 0.01).float().mean().item()

                    # 4. bias range (imbalance)
                    bias_range = (bias.max() - bias.min()).item()
                    # 5. bias saturation (at ±5 clamp)
                    bias_at_clamp = ((bias.abs() > 4.9).float().sum().item())

                    tag = f"ponder/layer_{i:02d}/{sublayer_name}"
                    w.add_scalar(f"{tag}/usage_entropy_norm", norm_ent, step)
                    w.add_scalar(f"{tag}/usage_min", usage_min, step)
                    w.add_scalar(f"{tag}/usage_max", usage_max, step)
                    w.add_scalar(f"{tag}/dead_k_fraction", dead_k, step)
                    w.add_scalar(f"{tag}/bias_range", bias_range, step)
                    w.add_scalar(f"{tag}/bias_at_clamp_count", bias_at_clamp, step)

                    # Detailed usage histogram per bin
                    for kk in range(K):
                        w.add_scalar(f"{tag}/usage_k{kk:02d}", usage[kk].item(), step)

                    # 7. Current-step y_hard entropy (batch-averaged)
                    last_y_attr = f"_last_y_hard_{sublayer_name}"
                    last_y = getattr(layer, last_y_attr, None)
                    if last_y is not None and last_y.numel() > 0:
                        # last_y: (..., K) one-hot. Batch-mean gives per-k usage this step.
                        step_usage = last_y.float().reshape(-1, K).mean(dim=0)
                        step_usage = step_usage / step_usage.sum().clamp(min=1e-8)
                        step_ent = -(step_usage * torch.log(step_usage.clamp(min=1e-8))).sum().item()
                        w.add_scalar(f"{tag}/step_entropy_norm", step_ent / math.log(K), step)

                    # Collapse threshold: norm_entropy < 0.5 AND dead_k > 0.3
                    collapsed = (norm_ent < 0.5) and (dead_k > 0.3)
                    if sublayer_name == "block":
                        total_block += 1
                        if collapsed: collapse_count_block += 1
                    else:
                        total_ffn += 1
                        if collapsed: collapse_count_ffn += 1

        # Global summary
        if total_block > 0:
            w.add_scalar("ponder/global/collapsed_block_layers_frac",
                         collapse_count_block / total_block, step)
        if total_ffn > 0:
            w.add_scalar("ponder/global/collapsed_ffn_layers_frac",
                         collapse_count_ffn / total_ffn, step)
        total = total_block + total_ffn
        if total > 0:
            w.add_scalar("ponder/global/collapse_any_frac",
                         (collapse_count_block + collapse_count_ffn) / total, step)

    # ====== 7b. v3 PonderNet k_predictor gradient health ======

    def _log_ponder_gradients(self, step, snn):
        """Surface k_predictor-specific gradient + weight norms.

        Why separate from generic param monitoring:
          - If Gumbel-Softmax Straight-Through breaks, k_predictor.net gets
            ZERO gradient → silent collapse. Need explicit visibility.
          - Two layers of net (Linear + SiLU + Linear); either saturated →
            predictor freezes.
          - Multi-GPU ZeRO-2: grad cache is already all-reduced-norm-squared,
            so values here are GLOBAL norms (correct across ranks).

        Metrics per (layer, sublayer):
          - net0_grad_norm / net0_weight_norm / net0_update_ratio
          - net1_grad_norm / net1_weight_norm / net1_update_ratio
          - net0_grad_zero_flag (1 if grad norm < 1e-8 → broken)
          - net1_grad_zero_flag (same)
        """
        w = self._writer
        cache = self._grad_cache
        lr = 1e-4  # fallback if called outside log_step; typically dashboard has lr in metrics
        # Try to read last-seen lr from scalar log buffer if available
        # (For simplicity, use fixed sentinel — ratio only meaningful at display time.)

        broken_count = 0
        total_count = 0

        for i, layer in enumerate(snn.layers):
            for sublayer_name, attr in (("block", "block_k_predictor"),
                                         ("ffn", "ffn_k_predictor")):
                kp = getattr(layer, attr, None)
                if kp is None:
                    continue
                # net is Sequential(Linear, SiLU, Linear). Index 0 and 2 are Linear.
                # Full parameter names (relative to snn): layers.{i}.{attr}.net.0.weight etc.
                prefix = f"layers.{i}.{attr}.net"
                tag_prefix = f"ponder_grads/layer_{i:02d}/{sublayer_name}"

                for j, sub_idx in [(0, "0"), (1, "2")]:  # j used for display name, sub_idx = PyTorch Sequential index
                    weight_key = f"{prefix}.{sub_idx}.weight"
                    bias_key = f"{prefix}.{sub_idx}.bias"
                    w_norm = 0.0
                    g_norm = 0.0

                    # Find the Linear module to read its weight directly
                    try:
                        lin = kp.net[int(sub_idx)]
                    except (IndexError, TypeError):
                        continue
                    if not hasattr(lin, "weight"):
                        continue
                    with torch.no_grad():
                        w_norm = lin.weight.data.norm().item()
                    g_norm = cache.get(weight_key, 0.0)
                    g_bias_norm = cache.get(bias_key, 0.0)

                    w.add_scalar(f"{tag_prefix}/net{j}_weight_norm", w_norm, step)
                    w.add_scalar(f"{tag_prefix}/net{j}_grad_norm", g_norm, step)
                    w.add_scalar(f"{tag_prefix}/net{j}_grad_bias_norm", g_bias_norm, step)
                    if w_norm > 0:
                        w.add_scalar(f"{tag_prefix}/net{j}_update_ratio", g_norm / w_norm, step)

                    is_broken = g_norm < 1e-8 and g_bias_norm < 1e-8
                    w.add_scalar(f"{tag_prefix}/net{j}_grad_broken_flag",
                                 1.0 if is_broken else 0.0, step)
                    if is_broken:
                        broken_count += 1
                    total_count += 1

        if total_count > 0:
            w.add_scalar("ponder_grads/global/broken_net_fraction",
                         broken_count / total_count, step)

    # ====== 7. health ======

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

    def _log_health(self, step, snn):
        w = self._writer
        cache = self._grad_cache

        # Layer-gradient Gini + share
        layer_grad_sums = {}
        for name, grad_norm in cache.items():
            parts = name.split(".")
            if parts[0] == "layers" and parts[1].isdigit():
                idx = int(parts[1])
                layer_grad_sums[idx] = layer_grad_sums.get(idx, 0.0) + grad_norm ** 2
        if len(layer_grad_sums) >= 2:
            layer_norms = [v ** 0.5 for v in layer_grad_sums.values()]
            grad_gini = self._gini(layer_norms)
            w.add_scalar("health/grad_gini", grad_gini, step)
            total_gn = sum(layer_norms)
            for idx in sorted(layer_grad_sums):
                share = (layer_grad_sums[idx] ** 0.5) / (total_gn + 1e-12)
                w.add_scalar(f"grad_share/layer_{idx:02d}", share, step)
        else:
            grad_gini = 0.0

        # β convergence
        beta_stds = []
        for i, layer in enumerate(snn.layers):
            if not hasattr(layer, "snn_block"):
                continue
            with torch.no_grad():
                beta = torch.sigmoid(layer.snn_block.b_beta.data)
                bstd = beta.std().item()
                beta_stds.append(bstd)
                w.add_scalar(f"health/beta_std/layer_{i:02d}", bstd, step)
        if beta_stds:
            w.add_scalar("health/beta_std_min", min(beta_stds), step)
            w.add_scalar("health/beta_converged_layers",
                         sum(1 for s in beta_stds if s < 0.01), step)

        # Dead/seizure neurons (inferred from β, V_th)
        dead_count = epileptic_count = total_neurons = 0
        for layer in snn.layers:
            if not hasattr(layer, "snn_block"):
                continue
            block = layer.snn_block
            with torch.no_grad():
                beta = torch.sigmoid(block.b_beta.data)
                v_th = block.v_th_min + torch.abs(block.b_th.data)
                v_steady_est = 0.2 / (1.0 - beta + 1e-8)
                dead_count += (v_steady_est < 0.1 * v_th).sum().item()
                epileptic_count += (v_steady_est > 10.0 * v_th).sum().item()
                total_neurons += beta.numel()

        if total_neurons > 0:
            dead_rate = dead_count / total_neurons
            epi_rate = epileptic_count / total_neurons
            w.add_scalar("health/dead_neuron_rate", dead_rate, step)
            w.add_scalar("health/epileptic_neuron_rate", epi_rate, step)
        else:
            dead_rate = epi_rate = 0.0

        layer_out_norms = []
        for layer in snn.layers:
            if hasattr(layer, "block_out_proj"):
                with torch.no_grad():
                    layer_out_norms.append(layer.block_out_proj.weight.data.norm().item())
        if len(layer_out_norms) >= 2:
            w.add_scalar(
                "health/layer_output_norm_ratio",
                max(layer_out_norms) / (min(layer_out_norms) + 1e-12),
                step,
            )

        s_grad = max(0.0, 1.0 - grad_gini * 2.0)
        s_beta = max(0.0, 1.0 - sum(1 for s in beta_stds if s < 0.01) / max(len(beta_stds), 1))
        s_neuron = max(0.0, 1.0 - (dead_rate + epi_rate) * 5.0)
        w.add_scalar("health/score", 0.4 * s_grad + 0.3 * s_beta + 0.3 * s_neuron, step)

    # ====== 8. histograms ======

    def _log_histograms(self, step):
        """Weight histograms are always valid (parameters replicated in ZeRO-2).
        Gradient histograms are only meaningful when grad is NOT sharded:
          - Single GPU: grad is full, plot OK
          - ZeRO-2 multi-GPU: grad is sharded (each rank ~1/W of full grad),
            histograms would show misleading partial distributions → SKIP
        """
        w = self._writer
        is_sharded = getattr(self, "_is_distributed_sharded", False)
        for name, (tag, p) in self._registry.items():
            if not p.requires_grad:
                continue
            w.add_histogram(f"histograms/{tag}/weight", p.data, step)
            if p.grad is not None and not is_sharded:
                w.add_histogram(f"histograms/{tag}/grad", p.grad, step)

    # ====== 9. compensation factors ======

    def _log_compensation_factors(self, step, snn):
        w = self._writer
        for i, layer in enumerate(snn.layers):
            if not hasattr(layer, "snn_block"):
                continue
            block = layer.snn_block
            with torch.no_grad():
                beta = torch.sigmoid(block.b_beta.data)
                sigmoid_deriv = (beta * (1.0 - beta)).mean().item()
                w.add_scalar(f"compensation/layer_{i:02d}/sigmoid_deriv_mean", sigmoid_deriv, step)
                softplus_deriv = torch.sigmoid(block.b_alpha.data).mean().item()
                w.add_scalar(f"compensation/layer_{i:02d}/softplus_deriv_mean", softplus_deriv, step)
