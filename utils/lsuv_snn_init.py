"""LSUV-style 数据驱动 SNN 初始化 v2 (quantal-aware) —— 解决 v4.1 quantal 模式下深层 hidden 神经元发放率塌缩。

研究背景：
  - Esser et al. 2017 (IBM TrueNorth): SNN 的 v_th 应按观察到的 V_pre 分布的目标分位数设置，让 init 时发放率 = 目标。
  - LSUV (Mishkin & Matas 2016): 前向 sample batch，按观察到的激活方差校准参数（这里：校准 v_th + scale W_in）。

quantal 的问题（实测 §4090, D=512/12 层）：
  output = v_th·𝟙[V_pre>v_th] —— 输出幅度锁死在 v_th（~0.02-0.05），比 supra 的 relu(V_pre-v_th) 小约 4-8x。
  SNNBlock 的 hidden_neuron 输入 = W_in @ (input_neuron 的 quantal 输出 ~v_th·s)，太小 → hidden V_pre 远低于其 v_th
  → hidden 发放率 ~0.006（vs supra ~0.30）→ SNN block 在 init 就基本失效。
  （input_neuron / gate_neuron / output_neuron 因为读的是 RMSNorm 后的残差流 ~unit-var，发放率 init 就正常 ~0.3-0.45，
   只是 v_th 没对到最优 output 幅度的点。）

v2 的修复（顺序逐层）：
  1. 前向 2 batch，逐 SNNDecoderLayer:
     a. 观察 hidden_neuron 的 V_pre std + 有效 v_th = v_th_min + mean|W_th_x·x| 的均值；
        scale `W_in.weight` ×= s，使 std(hidden V_pre) ≈ effective_v_th / Φ⁻¹(1-p)  → hidden 在 init 发放 ~p（p≈0.3）。
        （注：quantal output=v_th·s 与 W_in 无关 → 不需补偿 W_out。）
     b. （FFN gate/up 同理：若发放率 < 0.5·p，scale gate_proj/up_proj。）
  2. 对所有 PLIFNode（input_neuron1/2, gate_neuron, up_neuron, output_neuron）: v_th.data ← 观察到的 V_pre 的 (1-p) 经验分位数 per-channel（clamp ≥ 1e-4）。
     —— quantal 下这给 ~p 发放率 + 接近最优的 output 幅度（E[output]=v_th·p 在 v_th≈Φ⁻¹(1-p)·σ 附近最大）。
  3. 再前向一次 verify。

DeepSpeed 兼容：在 NeuronSparkForCausalLM 构造之后、deepspeed.initialize() 之前调用；只改 .data，不破坏 ZeRO。
supra 模式：也可调（v_th 校准对 supra 也有意义），但 supra 的 hidden 本来就不塌，效果有限；默认对 supra/quantal 都跑（无害）。
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
from neuronspark.modeling_neuronspark import (
    PLIFNode, SelectivePLIFNode, SNNDecoderLayer, SNNAttentionDecoderLayer, functional,
)

try:
    from scipy.stats import norm as _scipy_norm
    def _ppf(q):  # inverse normal CDF
        return float(_scipy_norm.ppf(q))
except Exception:
    # 粗略近似 Φ⁻¹（够用）
    def _ppf(q):
        # rational approx (Acklam) — 简化版
        import math as _m
        if q <= 0: return -6.0
        if q >= 1: return 6.0
        # 用 erfinv via torch
        return float(torch.erfinv(torch.tensor(2 * q - 1)) * _m.sqrt(2))


@torch.no_grad()
def lsuv_snn_init(
    model: nn.Module,
    sample_input_ids: torch.Tensor,
    target_p_fire: float = 0.3,
    n_passes: int = 2,
    calibrate_w_in: bool = True,
    verbose: bool = True,
):
    """LSUV v2 — 校准 PLIFNode v_th 到 (1-p) 分位数 + scale SNNBlock W_in 让 hidden 发放 ~p.

    Args:
        model: NeuronSparkForCausalLM (或 wrapper)。改 .snn.* 的参数 .data。
        sample_input_ids: (B, T) int — 校准 batch。
        target_p_fire: 目标 init 发放率 (default 0.3)。
        n_passes: 迭代次数 (1 = 单遍, 2 = init + 校准 + verify)。
        calibrate_w_in: 是否 scale SNNBlock 的 W_in (quantal 必须；supra 也无害)。
        verbose: 打印前后发放率。
    """
    raw = model.module if hasattr(model, "module") else model
    snn = raw.snn if hasattr(raw, "snn") else raw
    z = _ppf(1.0 - target_p_fire)  # e.g. p=0.3 → z≈0.524

    import neuronspark.modeling_neuronspark as nm
    orig_row = nm.segmented_plif_rowparam
    orig_sel = nm.segmented_plif_selective

    # 每次 forward 抓所有 segmented PLIF call 的 V_pre (reconstruct from output+V_post+ahp·s)
    calls = []  # list of (V_pre flat (N, H) on CPU, ahp_row or None)

    def _recon_v_pre(output, V_post, ahp_row):
        s_hard = (output > 0).to(V_post.dtype)
        if ahp_row is not None:
            ahp = ahp_row.unsqueeze(0) if ahp_row.ndim == 2 else ahp_row.view(1, 1, -1)
            return V_post + output + ahp * s_hard
        return V_post + output

    def mk_wrap(orig):
        def wrap(*a, **kw):
            out = orig(*a, **kw)
            output, V_post, v_carry = out
            V_pre = _recon_v_pre(output, V_post, kw.get("ahp_row", None))
            calls.append(V_pre.float().reshape(-1, V_pre.shape[-1]).detach().cpu())
            return out
        return wrap

    def _segmented_caller_order():
        """返回 forward_parallel 调 segmented PLIF 的顺序: (name, kind, ref).
        SNNDecoderLayer:        input_neuron1(rowparam) → hidden_neuron(selective) → input_neuron2(rowparam) → snn_ffn gate+up(rowparam_merged)
        SNNAttentionDecoderLayer: input_neuron2(rowparam) → snn_ffn gate+up(rowparam_merged)  [gate_neuron 是 token 级非 segmented → 跳过]
        然后 model 级: output_neuron(rowparam)
        """
        order = []
        for li, layer in enumerate(snn.layers):
            if isinstance(layer, SNNDecoderLayer):
                order.append((f"L{li}.input_neuron1", "plif", layer.input_neuron1))
                order.append((f"L{li}.snn_block.hidden_neuron", "selective", (layer, layer.snn_block.hidden_neuron)))
                order.append((f"L{li}.input_neuron2", "plif", layer.input_neuron2))
                order.append((f"L{li}.snn_ffn.gate+up", "merged", (layer.snn_ffn.gate_neuron, layer.snn_ffn.up_neuron)))
            else:
                order.append((f"L{li}.input_neuron2", "plif", layer.input_neuron2))
                order.append((f"L{li}.snn_ffn.gate+up", "merged", (layer.snn_ffn.gate_neuron, layer.snn_ffn.up_neuron)))
        order.append(("output_neuron", "plif", snn.output_neuron))
        return order

    def _fire_rate(v_flat, v_th):  # v_flat (N,H), v_th (H,)
        return (v_flat > v_th.unsqueeze(0)).float().mean().item()

    was_training = model.training
    model.eval()
    nm.segmented_plif_rowparam = mk_wrap(orig_row)
    nm.segmented_plif_selective = mk_wrap(orig_sel)
    try:
        for p_idx in range(n_passes):
            calls.clear()
            functional.reset_net(snn)
            _ = model(input_ids=sample_input_ids)
            order = _segmented_caller_order()
            n = min(len(order), len(calls))
            if verbose and len(order) != len(calls):
                print(f"[lsuv] WARN: order {len(order)} vs captured {len(calls)} — 部分跳过")
            for i in range(n):
                name, kind, ref = order[i]
                v_flat = calls[i]  # (N, H)
                if kind == "plif":
                    neuron = ref
                    fr0 = _fire_rate(v_flat, neuron.v_th.detach().float().cpu())
                    new_vth = v_flat.quantile(1.0 - target_p_fire, dim=0).clamp(min=1e-4)
                    neuron.v_th.data.copy_(new_vth.to(neuron.v_th.device, dtype=neuron.v_th.dtype))
                    if verbose:
                        fr1 = _fire_rate(v_flat, neuron.v_th.detach().float().cpu())
                        print(f"[lsuv] {name:40s} plif: fr {fr0:.3f}→{fr1:.3f}  v_th μ={neuron.v_th.mean().item():.4f}")
                elif kind == "merged":
                    gate, up = ref
                    H = v_flat.shape[1] // 2
                    for sub, vf in ((gate, v_flat[:, :H]), (up, v_flat[:, H:])):
                        fr0 = _fire_rate(vf, sub.v_th.detach().float().cpu())
                        new_vth = vf.quantile(1.0 - target_p_fire, dim=0).clamp(min=1e-4)
                        sub.v_th.data.copy_(new_vth.to(sub.v_th.device, dtype=sub.v_th.dtype))
                        if verbose:
                            fr1 = _fire_rate(vf, sub.v_th.detach().float().cpu())
                            print(f"[lsuv] {name:40s} merged-{'gate' if sub is gate else 'up'}: fr {fr0:.3f}→{fr1:.3f}  v_th μ={sub.v_th.mean().item():.4f}")
                elif kind == "selective":
                    layer, hidden = ref
                    block = layer.snn_block
                    # hidden 的 v_th(t) = v_th_min + |W_th_x·x| 是数据相关、不受 W_in scale 影响。
                    # 用「当前发放率 fr0」估有效阈值相对 V_pre std 的位置: effective_v_th ≈ Φ⁻¹(1-fr0)·σ_Vpre。
                    # 要发放率 = target_p_fire → 需 σ_Vpre ↑ 到 effective_v_th / Φ⁻¹(1-target) → scale W_in ×= Φ⁻¹(1-fr0) / Φ⁻¹(1-target)。
                    # (W_in scale 同时放大 u_hidden=α·(W_in@x) → V_pre 同比例放大；output=v_th·s 与 V_pre 幅度无关 → 不影响下游幅度。)
                    fr0 = getattr(hidden, "_last_firing_rate", None)
                    v_pre_std = v_flat.std().item()
                    if calibrate_w_in and fr0 is not None:
                        fr0_c = min(max(float(fr0), 1e-4), 1.0 - 1e-4)
                        z0 = _ppf(1.0 - fr0_c)
                        s = z0 / z if z > 1e-6 else 1.0
                        s = float(max(0.1, min(50.0, s)))
                        block.W_in.weight.data.mul_(s)
                        if verbose:
                            print(f"[lsuv] {name:40s} selective: fr {fr0:.4f}→target {target_p_fire}; V_pre σ={v_pre_std:.4f}; W_in ×{s:.2f}")
                    elif verbose:
                        print(f"[lsuv] {name:40s} selective: fr {fr0}; V_pre σ={v_pre_std:.4f} (no W_in scale)")
        # final verify pass
        calls.clear()
        functional.reset_net(snn)
        _ = model(input_ids=sample_input_ids)
        if verbose:
            frs = {}
            for nm_, mod in snn.named_modules():
                fr = getattr(mod, "_last_firing_rate", None)
                if fr is not None:
                    key = ("hidden" if "hidden_neuron" in nm_ else
                           "input" if "input_neuron" in nm_ else
                           "ffn_gate" if "snn_ffn.gate_neuron" in nm_ else
                           "ffn_up" if "snn_ffn.up_neuron" in nm_ else
                           "gate" if nm_.endswith("gate_neuron") else
                           "output" if "output_neuron" in nm_ else "other")
                    frs.setdefault(key, []).append(fr)
            print("[lsuv] verify firing rates:", {k: f"{min(v):.3f}~{max(v):.3f}μ{sum(v)/len(v):.3f}" for k, v in frs.items()})
    finally:
        nm.segmented_plif_rowparam = orig_row
        nm.segmented_plif_selective = orig_sel
        functional.reset_net(snn)
        if was_training:
            model.train()
    if verbose:
        print(f"[lsuv] DONE — target p_fire={target_p_fire}")
