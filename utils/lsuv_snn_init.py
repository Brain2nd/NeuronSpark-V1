"""LSUV-style 数据驱动 SNN 初始化 (quantal-aware) —— 解决 v4.1 quantal 模式下的级联衰减问题。

研究背景：
  - Esser et al. 2017 (IBM TrueNorth): SNN 的 v_th 应按观察到的 V_pre 分布的目标分位数设置,
    让初始化时每个 channel 的发放率就是目标 firing rate (~30-50%).
  - LSUV (Mishkin & Matas 2016): 前向 sample batch, 按观察到的输出方差校准参数.
  - 这里组合两者: 跑一个 sample forward, 对每个 PLIF/Selective 神经元
    (1) 重建每 channel 的 V_pre 分布, (2) 把 v_th 设为 (1 - target_p_fire) 分位数.

quantal 比 supra 更需要这个 init 因为:
  - supra: output = relu(V_pre - v_th), 输出方差跟 V_pre 方差同阶, 即使 v_th 小, output 也能传信号
  - quantal: output = v_th · 𝟙[V_pre > v_th], 输出幅度被 v_th 锁死. 如果 v_th 太小,
    output 微弱 → 下层 V_pre 微弱 → 发放更稀疏 → 信号逐层指数衰减 → 深层全死

DeepSpeed 兼容: 在 NeuronSparkForCausalLM 构造之后、deepspeed.initialize() 之前调用,
对 fp32 master copies 直接修改 .data, 不破坏 ZeRO 分片.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from neuronspark.modeling_neuronspark import (
    PLIFNode, SelectivePLIFNode,
    segmented_plif_rowparam, segmented_plif_selective,
    _ahp_row_of,
)


@torch.no_grad()
def lsuv_snn_init(
    model: nn.Module,
    sample_input_ids: torch.Tensor,
    target_p_fire: float = 0.4,
    layers_to_calibrate: tuple = ("hidden", "input", "ffn", "output"),
    n_passes: int = 2,
    verbose: bool = True,
):
    """LSUV-style v_th calibration for all PLIF / Selective neurons.

    Procedure:
        1. Monkey-patch `segmented_plif_rowparam` / `segmented_plif_selective` to capture
           V_pre stats (reconstructed from V_post + output + ahp·s_hard).
        2. Run model forward on `sample_input_ids` once → collect per-channel V_pre samples.
        3. For each neuron, set `v_th.data` to per-channel (1-target_p_fire) quantile of V_pre.
        4. (Optional) Re-run forward → verify firing rates near target. Refine if `n_passes > 1`.

    Args:
        model: NeuronSparkForCausalLM (or wrapper). Calibration touches `.snn.*` neurons.
        sample_input_ids: (B, T) int — one calibration batch.
        target_p_fire: target initial firing rate per channel (default 0.4).
        layers_to_calibrate: which neuron types to recalibrate.
        n_passes: refine iterations (1 = single pass, 2 = init + verify).
        verbose: print before/after firing rate stats.
    """
    # Locate inner SNN
    raw = model.module if hasattr(model, "module") else model
    snn = raw.snn if hasattr(raw, "snn") else raw
    dev = next(snn.parameters()).device

    if sample_input_ids.device != dev:
        sample_input_ids = sample_input_ids.to(dev)

    import neuronspark.modeling_neuronspark as nm
    orig_row = nm.segmented_plif_rowparam
    orig_sel = nm.segmented_plif_selective

    # call_stats[i] = (V_pre flat per-channel tensor on CPU, ahp_row or None, neuron_dim)
    call_stats: list = []
    capture_idx = [0]

    def _reconstruct_v_pre(output, V_post, ahp_row):
        """V_pre = V_post + output + ahp · s_hard,  s_hard = 𝟙[output > 0]."""
        s_hard = (output > 0).to(V_post.dtype)
        if ahp_row is not None:
            if ahp_row.ndim == 2:  # (b, H)
                ahp = ahp_row.unsqueeze(0)  # (1, b, H)
            else:
                ahp = ahp_row.view(1, 1, -1)
            ahp_term = ahp * s_hard
        else:
            ahp_term = 0.0
        return V_post + output + ahp_term

    def make_wrap(orig):
        def wrap(*args, **kwargs):
            out = orig(*args, **kwargs)
            output, V_post, v_carry = out
            ahp_row = kwargs.get("ahp_row", None)
            V_pre = _reconstruct_v_pre(output, V_post, ahp_row)
            # V_pre shape: (TK, batch, H) — flatten time/batch, keep channels
            v_flat = V_pre.float().reshape(-1, V_pre.shape[-1]).detach().cpu()
            call_stats.append(v_flat)
            capture_idx[0] += 1
            return out
        return wrap

    nm.segmented_plif_rowparam = make_wrap(orig_row)
    nm.segmented_plif_selective = make_wrap(orig_sel)

    def _enumerate_segmented_callers(snn_model):
        """Return list of (name, neuron, kind) in the SAME order forward_parallel calls them.

        Calling order per layer (forward_parallel):
            SNNDecoderLayer:        input_neuron1 → snn_block.hidden_neuron → input_neuron2 → snn_ffn (merged gate+up)
            SNNAttentionDecoderLayer: input_neuron2 → snn_ffn (merged gate+up)
              (gate_neuron is token-level, NOT segmented — skipped)
        Then model-level: output_neuron.
        """
        order = []
        for li, layer in enumerate(snn_model.layers):
            ltype = snn_model.layer_types[li]
            if ltype == "snn":
                order.append((f"layer{li}.input_neuron1", layer.input_neuron1, "rowparam"))
                order.append((f"layer{li}.snn_block.hidden_neuron", layer.snn_block.hidden_neuron, "selective"))
                order.append((f"layer{li}.input_neuron2", layer.input_neuron2, "rowparam"))
                # snn_ffn merges gate+up into one rowparam call; calibrate them together
                order.append((f"layer{li}.snn_ffn.gate+up", (layer.snn_ffn.gate_neuron, layer.snn_ffn.up_neuron), "rowparam_merged"))
            else:  # memory / attention layer
                order.append((f"layer{li}.input_neuron2", layer.input_neuron2, "rowparam"))
                order.append((f"layer{li}.snn_ffn.gate+up", (layer.snn_ffn.gate_neuron, layer.snn_ffn.up_neuron), "rowparam_merged"))
        order.append(("output_neuron", snn_model.output_neuron, "rowparam"))
        return order

    def _firing_rate(v_flat: torch.Tensor, v_th: torch.Tensor) -> torch.Tensor:
        """per-channel fraction of frames with V_pre > v_th."""
        return (v_flat > v_th.unsqueeze(0)).float().mean(dim=0)

    def _quantile_per_channel(v_flat: torch.Tensor, q: float) -> torch.Tensor:
        """per-channel q-quantile (q ∈ [0,1])."""
        # v_flat: (N, H); return (H,)
        return v_flat.quantile(q, dim=0)

    model_was_training = model.training
    model.eval()

    for pass_idx in range(n_passes):
        call_stats.clear()
        capture_idx[0] = 0
        # Reset all PLIF/Selective neuron memory states so the calibration pass starts fresh
        for m in snn.modules():
            if hasattr(m, "v"):
                m.v = 0.0
        # Forward (use outer model.forward → autocast matmul with fp32 neuron params + bf16 matrices)
        _ = model(input_ids=sample_input_ids)

        order = _enumerate_segmented_callers(snn)
        n_callers = len(order)
        n_captured = len(call_stats)
        if n_captured != n_callers:
            if verbose:
                print(f"[lsuv] WARN: expected {n_callers} segmented calls, captured {n_captured}; "
                      f"order mapping may be off")
            # Fall back: try to proceed with whatever we have, mapping by index
        if verbose:
            print(f"[lsuv] pass {pass_idx+1}/{n_passes}: captured {n_captured} segmented calls")

        for i, (name, neuron, kind) in enumerate(order[:n_captured]):
            v_flat = call_stats[i]  # (N_samples, H)
            # Decide whether to calibrate this neuron
            calibrate = False
            if "hidden" in layers_to_calibrate and "hidden_neuron" in name:
                calibrate = True
            if "input" in layers_to_calibrate and "input_neuron" in name:
                calibrate = True
            if "ffn" in layers_to_calibrate and "snn_ffn" in name:
                calibrate = True
            if "output" in layers_to_calibrate and name == "output_neuron":
                calibrate = True
            if not calibrate:
                continue

            # SelectivePLIFNode: v_th is data-dependent (v_th_min + |W_th_x·x + b_th|),
            # no static .v_th parameter to set. Skip for now; the input scale is what
            # matters and that's calibrated upstream via PLIFNode v_th.
            if isinstance(neuron, SelectivePLIFNode):
                if verbose:
                    fr = (v_flat > 0).float().mean(dim=0).mean().item()
                    print(f"[lsuv]   {name}: SelectivePLIF (data-dependent v_th) — skipping; "
                          f"P[V_pre>0]={fr:.3f}")
                continue

            if kind == "rowparam_merged":
                # gate_neuron + up_neuron, concatenated in segmented call
                gate, up = neuron
                H_total = v_flat.shape[1]
                H_each = H_total // 2
                # Set each half's v_th separately
                v_gate = v_flat[:, :H_each]
                v_up = v_flat[:, H_each:]
                fr_before = (_firing_rate(v_gate, gate.v_th.cpu()).mean().item(),
                             _firing_rate(v_up, up.v_th.cpu()).mean().item())
                new_vth_gate = _quantile_per_channel(v_gate, 1 - target_p_fire).clamp(min=1e-4)
                new_vth_up = _quantile_per_channel(v_up, 1 - target_p_fire).clamp(min=1e-4)
                gate.v_th.data.copy_(new_vth_gate.to(gate.v_th.device, dtype=gate.v_th.dtype))
                up.v_th.data.copy_(new_vth_up.to(up.v_th.device, dtype=up.v_th.dtype))
                if verbose:
                    fr_after = (_firing_rate(v_gate, gate.v_th.cpu()).mean().item(),
                                _firing_rate(v_up, up.v_th.cpu()).mean().item())
                    print(f"[lsuv]   {name}: gate fr {fr_before[0]:.3f}→{fr_after[0]:.3f}  "
                          f"up fr {fr_before[1]:.3f}→{fr_after[1]:.3f}  "
                          f"v_th mean: gate={gate.v_th.mean().item():.3f} up={up.v_th.mean().item():.3f}")
            else:
                # PLIFNode (rowparam) or SelectivePLIFNode (selective)
                fr_before = _firing_rate(v_flat, neuron.v_th.cpu()).mean().item()
                new_vth = _quantile_per_channel(v_flat, 1 - target_p_fire).clamp(min=1e-4)
                neuron.v_th.data.copy_(new_vth.to(neuron.v_th.device, dtype=neuron.v_th.dtype))
                if verbose:
                    fr_after = (_firing_rate(v_flat, neuron.v_th.cpu()).mean().item())
                    print(f"[lsuv]   {name}: fr {fr_before:.3f}→{fr_after:.3f}  "
                          f"v_th mean={neuron.v_th.mean().item():.3f}")

    # Cleanup: restore originals
    nm.segmented_plif_rowparam = orig_row
    nm.segmented_plif_selective = orig_sel
    if model_was_training:
        model.train()
    if verbose:
        print(f"[lsuv] DONE — calibrated v_th to target p_fire={target_p_fire}")
