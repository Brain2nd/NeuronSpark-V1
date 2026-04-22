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
        """Before optimizer.step(), capture per-param grad norm (all-reduced across ranks)."""
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
                # Associative-memory layer
                for name in ("neuron_k", "neuron_v", "neuron_q", "neuron_gate"):
                    neuron = getattr(layer, name, None)
                    if neuron is not None and hasattr(neuron, "w"):
                        semantics.append(
                            (f"{prefix}/mem_{name}_beta", neuron.w, torch.sigmoid, "beta")
                        )
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
        w = self._writer
        for i, layer in enumerate(snn.layers):
            if hasattr(layer, "block_halt"):
                ek_min = getattr(layer, "_ek_min", None)
                ek_max = getattr(layer, "_ek_max", None)
                if ek_min is not None:
                    w.add_scalar(f"ponder/layer_{i:02d}/ek_min", ek_min, step)
                    w.add_scalar(f"ponder/layer_{i:02d}/ek_max", ek_max, step)
                with torch.no_grad():
                    for name, halt in [("block_halt", layer.block_halt),
                                        ("ffn_halt", layer.ffn_halt)]:
                        w.add_scalar(f"halt/layer_{i:02d}/{name}/weight_norm",
                                     halt.weight.data.norm().item(), step)
            elif hasattr(layer, "ffn_halt"):
                ek_min = getattr(layer, "_ek_min", None)
                ek_max = getattr(layer, "_ek_max", None)
                if ek_min is not None:
                    w.add_scalar(f"ponder/layer_{i:02d}/ek_min", ek_min, step)
                    w.add_scalar(f"ponder/layer_{i:02d}/ek_max", ek_max, step)
                with torch.no_grad():
                    w.add_scalar(f"halt/layer_{i:02d}/ffn_halt/weight_norm",
                                 layer.ffn_halt.weight.data.norm().item(), step)

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
        w = self._writer
        for i, layer in enumerate(snn.layers):
            if not hasattr(layer, "neuron_gate"):
                continue
            M = getattr(layer, "M", None)
            if M is not None and not isinstance(M, float):
                for g in range(M.shape[0]):
                    w.add_scalar(f"memory/layer_{i:02d}/M_group{g}_norm", M[g].norm().item(), step)
                with torch.no_grad():
                    M_last = M[-1, 0] if M.dim() == 4 else M[-1]
                    if M_last.numel() > 0:
                        s = torch.linalg.svdvals(M_last.float())
                        if s[0] > 1e-8:
                            eff_rank = (s / s[0]).sum().item()
                            w.add_scalar(f"memory/layer_{i:02d}/effective_rank", eff_rank, step)

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
        w = self._writer
        for name, (tag, p) in self._registry.items():
            if not p.requires_grad:
                continue
            w.add_histogram(f"histograms/{tag}/weight", p.data, step)
            if p.grad is not None:
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
