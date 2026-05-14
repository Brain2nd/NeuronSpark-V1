"""LongMamba (training-free Mamba long-context extension, ICLR 2025) eval on our trained mamba ckpt.

算法 (paper §3):
1. Calibration (offline, train-len 数据): 每层每 d_inner channel, 跑 train-len 序列, 算 cum-decay
     prod_t Ā_t = exp( A · sum_t Δ_t )  (Ā = exp(Δ ⊗ A), A 是 negative)
   per (d_inner, d_state) 一个值; 我们 collapse 到 per d_inner (取 d_state 维 mean), 标记 global if cum_decay > θ.
   θ 由分位 (paper: 10⁻³⁰ for Mamba-1.4B; 我们小模型按 calibration 数据自动选 75-percentile 的 cum_decay 作 cut, 大约前 25% channel 标 global).

2. Threshold g(S) per target length S: 用 calibration Δ 分布, 求 g 使 expected cum-decay 在 S 上 ≈ train-len 上 (即 filtered Δ 的累积量保持训练时水平).
   论文公式: prod_{i=1..S} Ā'_i ≈ prod_{i=1..L_train} Ā_i,
     其中 Ā'_i = Ā_i if Δ_i ≥ g else 1
   ⟹ 在 S 个 token 里 "通过 filter" 的期望数 ≈ L_train.
   实现: g(S) = quantile_q(calibrated Δ distribution), q = 1 - L_train / S.
   (e.g., S=1M, L_train=256 → q ≈ 0.9997 → g 是 Δ 分布的极高分位数 → 大多数 token 被 filter)

3. Inference (online): selective_scan_fn 调用前, 对 global channel 的 Δ_t < g(S) 位置, 把 Δ 置极小负
   (softplus 后 ≈0), 等价于 Ā=1, B̄=0, 即 "跳过状态更新".

实现要点:
- monkeypatch mamba_ssm 的 selective_scan_fn (在 _ssi 模块和 _msm 模块两处导出都要 patch)
- 每个 forward call 用全局 layer_idx 计数 (Mamba.forward 顺序调用每层一次)
- calibrate / filter / vanilla 三模式切换

用法:
  CUDA_VISIBLE_DEVICES=N python scripts/longmamba_eval.py \
    --load_ckpt /workspace/ih6_mamba.ckpt --eval_lens 131072,262144,524288,1048576 \
    --eval_examples 64 --calib_examples 32 --theta_quantile 0.75 \
    [--mode vanilla|longmamba]
"""
import sys, os, math, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--load_ckpt", required=True)
ap.add_argument("--mode", default="longmamba", choices=("vanilla", "longmamba"),
                help="vanilla=原 Mamba 不动; longmamba=施加 calibration+filter")
ap.add_argument("--eval_lens", default="131072,262144,524288,1048576")
ap.add_argument("--eval_examples", type=int, default=64)
ap.add_argument("--calib_examples", type=int, default=32, help="calibration 序列数 (在 train_len 上跑)")
ap.add_argument("--theta_quantile", type=float, default=0.75,
                help="cum_decay 取此分位作 global cut (越高 → 越少 channel 标 global)")
ap.add_argument("--seed_eval", type=int, default=999)
ap.add_argument("--seed_calib", type=int, default=42)
ap.add_argument("--out", default=None)
args = ap.parse_args()

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LOG = open(args.out, "a") if args.out else None
def log(s):
    print(s, flush=True)
    if LOG: LOG.write(s + "\n"); LOG.flush()


# ============================================================
# 1. monkeypatch selective_scan_fn
# ============================================================
import mamba_ssm.ops.selective_scan_interface as _ssi
import mamba_ssm.modules.mamba_simple as _msm

_orig_ssf = _ssi.selective_scan_fn

LM_STATE = {
    "mode": "vanilla",        # "vanilla" | "calibrate" | "filter"
    "layer_idx": 0,
    "n_layers": 0,
    "captured": None,         # list[n_layers] of dict {"A": (d_inner,d_state) | None, "dt_samples": list of (B,d_inner,T)}
    "global_mask": None,      # list[n_layers] of (d_inner,) bool tensor
    "threshold_g": None,      # scalar (per eval-S)
}

_LARGE_NEG = -50.0  # softplus(-50) ≈ 0 → effective delta ≈ 0 → Ā=1, B̄=0

def _patched_ssf(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                 delta_softplus=False, return_last_state=False):
    """Wrapper: in calibrate mode capture A & effective delta; in filter mode mask delta for global channels."""
    n_layers = LM_STATE["n_layers"]
    if n_layers == 0:
        return _orig_ssf(u, delta, A, B, C, D=D, z=z, delta_bias=delta_bias,
                         delta_softplus=delta_softplus, return_last_state=return_last_state)
    li = LM_STATE["layer_idx"] % n_layers

    # 计算 "effective delta" (softplus 后的, 即 selective_scan 实际用的 Δ)
    # delta shape: (B, d_inner, T) for Mamba1
    eff = delta.float()
    if delta_bias is not None:
        eff = eff + delta_bias.float().view(1, -1, 1) if eff.dim() == 3 else eff + delta_bias.float()
    if delta_softplus:
        eff = torch.nn.functional.softplus(eff)

    if LM_STATE["mode"] == "calibrate":
        cap = LM_STATE["captured"][li]
        if cap["A"] is None:
            cap["A"] = A.detach().float().cpu().clone()  # (d_inner, d_state)
        cap["dt_samples"].append(eff.detach().float().cpu())

    elif LM_STATE["mode"] == "filter":
        gmask = LM_STATE["global_mask"][li].to(delta.device).view(1, -1, 1)   # (1, d_inner, 1)
        g = LM_STATE["threshold_g"]
        low = (eff < g)                                                       # (B, d_inner, T)
        filter_mask = gmask & low
        # 把 raw delta 设成极小负, 这样 selective_scan 内部 softplus(delta+bias) ≈ 0
        # 注意 raw delta 可能是 bf16/fp32, 保持原 dtype
        delta = torch.where(filter_mask, torch.full_like(delta, _LARGE_NEG), delta)
        if delta_bias is not None:
            # 还要抵消 bias 不然 +bias 后可能就不极小负了
            # 已经在 fill 时设 -50, bias 通常 ~O(1), softplus(-50+1) 还是 ~0, OK
            pass

    LM_STATE["layer_idx"] += 1
    return _orig_ssf(u, delta, A, B, C, D=D, z=z, delta_bias=delta_bias,
                     delta_softplus=delta_softplus, return_last_state=return_last_state)

_ssi.selective_scan_fn = _patched_ssf
_msm.selective_scan_fn = _patched_ssf


# ============================================================
# 2. data: induction-heads (从 v4_induction_heads.py 复刻, 保证 eval 协议一致)
# ============================================================
COPY_PREFIX = None  # 从 ckpt 恢复
EMB_SIZE = None
def make_batch(B, input_seq_len, np_rng, vocab, induction_len=1, num_triggers=5):
    il = induction_len
    T = input_seq_len + 1 + il
    ids = np.empty((B, T), dtype=np.int64)
    for b in range(B):
        s = np_rng.integers(0, vocab, size=input_seq_len).tolist()
        s.append(vocab)  # COPY_PREFIX = vocab
        raw = np.sort(np_rng.integers(input_seq_len - (1 + il), size=max(num_triggers, 1)))
        pf = []
        for i, q in enumerate(raw):
            if i == 0 or q - pf[-1] > il:
                pf.append(int(q))
        tc = [s[pf[0] + 1 + i] for i in range(il)]
        for q in pf:
            s[q] = vocab
            for i in range(il):
                s[q + 1 + i] = tc[i]
        ids[b] = s + tc
    return torch.from_numpy(ids).to(DEV), T


# ============================================================
# 3. load ckpt
# ============================================================
log(f"=== LongMamba eval — mode={args.mode} ckpt={args.load_ckpt} ===")
ck = torch.load(args.load_ckpt, map_location="cpu", weights_only=False)
saved_args = ck.get("args", {})
m_d = saved_args.get("m_d", 128)
m_layers = saved_args.get("m_layers", 4)
vocab = saved_args.get("vocab", 16)
train_len = saved_args.get("train_len", 256)
induction_len = saved_args.get("induction_len", 1)
num_triggers = saved_args.get("num_triggers", 5)
COPY_PREFIX = vocab
EMB_SIZE = vocab + 1
log(f"saved args: m_d={m_d} m_layers={m_layers} vocab={vocab} train_len={train_len} il={induction_len} nt={num_triggers}")

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
cfg = MambaConfig(d_model=m_d, n_layer=m_layers, vocab_size=EMB_SIZE,
                  ssm_cfg={"layer": "Mamba1"},
                  rms_norm=True, fused_add_norm=False, residual_in_fp32=True, pad_vocab_size_multiple=1)
model = MambaLMHeadModel(cfg).to(DEV)
model.load_state_dict(ck["state_dict"], strict=True)
model.eval()
# 强制走 slow path 让 selective_scan_fn 被直接调用 (fast path 走 fused mamba_inner_fn 跳过 ssf patch)
_fastpath_disabled = 0
for blk in model.backbone.layers:
    mx = getattr(blk, "mixer", None)
    if mx is not None and hasattr(mx, "use_fast_path"):
        mx.use_fast_path = False; _fastpath_disabled += 1
LM_STATE["n_layers"] = m_layers
log(f"loaded {sum(p.numel() for p in model.parameters())/1e6:.3f}M params; n_layers={m_layers}; "
    f"disabled use_fast_path on {_fastpath_disabled} Mamba block(s)")


# ============================================================
# 4. calibration (only for longmamba mode)
# ============================================================
def fwd(ids):
    LM_STATE["layer_idx"] = 0  # reset per forward (每次 forward 内 ssf 被各层依次调用)
    with torch.no_grad():
        return model(ids).logits.float()

if args.mode == "longmamba":
    log(f"\n=== Calibration (mode=calibrate, {args.calib_examples} sequences @ train_len={train_len}) ===")
    LM_STATE["mode"] = "calibrate"
    LM_STATE["captured"] = [{"A": None, "dt_samples": []} for _ in range(m_layers)]
    np_calib = np.random.default_rng(args.seed_calib)
    for i in range(args.calib_examples):
        ids, _ = make_batch(1, train_len, np_calib, vocab, induction_len, num_triggers)
        _ = fwd(ids)
    LM_STATE["mode"] = "vanilla"

    # 计算每层 cum_decay per d_inner channel: 用 mean over d_state
    # ∏_t Ā_t,c,s = exp(A_c,s · ∑_t Δ_t,c)  → over (B, T): ∑_t Δ → (B, d_inner)
    global_masks = []
    for li in range(m_layers):
        cap = LM_STATE["captured"][li]
        A = cap["A"]  # (d_inner, d_state), values are A_log (log of negative A, paper uses A_log convention)
        # mamba_ssm: A = -exp(A_log) (A is negative; in ssf they pass A directly as negative real)
        # 这里 captured 的 A 应该已经是负实数 (selective_scan_fn 接受 negative A)
        # cum decay per (B, d_inner, d_state): exp(A_c,s · sum_t Δ_t,c)
        dt_all = torch.cat([d.flatten(0, 1) for d in cap["dt_samples"]], dim=0)  # (total_samples, d_inner, T) → flatten B,N
        # 实际 shape: 每个 capture (1, d_inner, T) → concat (N_calib, d_inner, T)
        dt_per_seq = torch.stack([s.squeeze(0) for s in cap["dt_samples"]], dim=0)  # (N, d_inner, T)
        sum_dt = dt_per_seq.sum(dim=-1)  # (N, d_inner)
        mean_sum_dt = sum_dt.mean(dim=0)  # (d_inner,)
        # cum_decay per (d_inner, d_state) = exp(A · mean_sum_dt[d_inner])
        cum_decay = torch.exp(A * mean_sum_dt.unsqueeze(-1))  # (d_inner, d_state)
        # per d_inner: mean over d_state (collapse)
        cum_decay_per_dinner = cum_decay.mean(dim=-1)  # (d_inner,)
        # global = cum_decay > θ; θ = θ_quantile of cum_decay_per_dinner
        theta = torch.quantile(cum_decay_per_dinner, args.theta_quantile).item()
        gmask = cum_decay_per_dinner > theta  # (d_inner,) bool
        global_masks.append(gmask)
        log(f"  layer {li}: cum_decay_per_dinner [min/med/max] = "
            f"{cum_decay_per_dinner.min().item():.3e} / "
            f"{cum_decay_per_dinner.median().item():.3e} / "
            f"{cum_decay_per_dinner.max().item():.3e}; "
            f"θ={theta:.3e} → {gmask.sum().item()}/{gmask.numel()} global channels")
    LM_STATE["global_mask"] = global_masks

    # per layer dt 分布 (用来算 g(S))
    # 简化: 所有 global channel 共享一个 g per S, 取所有 global channel 上所有 token 的 Δ 分布
    # g(S) = quantile_{1-L_train/S}( all_global_dt )
    all_global_dt = []
    for li in range(m_layers):
        cap = LM_STATE["captured"][li]
        gmask = global_masks[li]  # (d_inner,)
        dt_per_seq = torch.stack([s.squeeze(0) for s in cap["dt_samples"]], dim=0)  # (N, d_inner, T)
        # 只取 global channel 的 Δ 值
        dt_global = dt_per_seq[:, gmask, :].flatten()  # (N * n_global * T,)
        all_global_dt.append(dt_global)
    all_global_dt = torch.cat(all_global_dt)
    log(f"  global-channel Δ distribution: [min/q25/med/q75/max] = "
        f"{all_global_dt.min().item():.3e} / "
        f"{torch.quantile(all_global_dt, 0.25).item():.3e} / "
        f"{all_global_dt.median().item():.3e} / "
        f"{torch.quantile(all_global_dt, 0.75).item():.3e} / "
        f"{all_global_dt.max().item():.3e}; n={len(all_global_dt)}")

    # 算 g(S) per target S: 取 1 - L_train/S 分位
    g_lookup = {}
    for L in [int(x) for x in args.eval_lens.split(",")]:
        S = L + 1 + induction_len  # total seq length
        q = max(0.0, min(0.9999, 1.0 - train_len / S))
        g = torch.quantile(all_global_dt, q).item()
        g_lookup[L] = g
        log(f"  g(L={L}) at q={q:.6f} → {g:.4e}")
    LM_STATE["calibration_done"] = True
    LM_STATE["g_lookup"] = g_lookup


# ============================================================
# 5. eval
# ============================================================
def answer_acc(logits, ids, T, il):
    pred = logits[:, T - il - 1: T - 1, :].argmax(-1)
    tgt = ids[:, T - il:]
    return (pred == tgt).all(dim=1).sum().item(), tgt.shape[0]

log(f"\n=== Length-generalization eval (mode={args.mode}, eval_examples={args.eval_examples}) ===")
np_eval = np.random.default_rng(args.seed_eval)
for L in [int(x) for x in args.eval_lens.split(",")]:
    if args.mode == "longmamba":
        LM_STATE["mode"] = "filter"
        LM_STATE["threshold_g"] = g_lookup[L]
    else:
        LM_STATE["mode"] = "vanilla"

    # batch: 长 L 用 batch=1
    B = 1 if L > 65536 else max(2, 32 // max(1, L // 2048))
    n_iter = max(1, args.eval_examples // B)
    n_corr = n_tot = 0
    for _ in range(n_iter):
        ids, T = make_batch(B, L, np_eval, vocab, induction_len, num_triggers)
        logits = fwd(ids)
        c, t = answer_acc(logits, ids, T, induction_len)
        n_corr += c; n_tot += t
    tag = "VANILLA" if args.mode == "vanilla" else f"LONGMAMBA g={LM_STATE['threshold_g']:.3e}"
    log(f"  input_seq_len={L:8d}: acc={n_corr/n_tot:.4f}  ({n_corr}/{n_tot})  [{tag}]")

log("=== DONE ===")
