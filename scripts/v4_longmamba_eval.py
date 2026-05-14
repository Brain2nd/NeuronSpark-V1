"""v4 + LongMamba-style SNNBlock filter — 把 LongMamba (ICLR 2025) 思路迁移到 v4 的 SNNBlock 路径.

算法映射 (Mamba ↔ v4 SNNBlock):
  Mamba: Ā_t = exp(Δ_t·A) per channel; cum_decay = ∏Ā_t > θ → global; filter Δ_t < g → (Ā=1, B̄=0)
  v4   : β_t = sigmoid(W_β·x) per token per channel; cum_decay = ∏β_t > θ → global;
         filter β_t > g_β → (β=1, u=0)  ((β=1 ⇒ u=(1-β)·I=0 自动; 但模型代码先算 u 后 expand β,
         所以 patch 时 β 和 u 都要同步设)

实施:
  - monkeypatch neuronspark.modeling_neuronspark.segmented_plif_selective
    入参顺序: (beta, u, v_th, v_init, k_t, K, training, ahp_row=None, alpha=4.0, spike_mode='supra')
    我们在调用前把 beta/u 张量上 global×high-β 位置改写
  - calibration mode: 捕获 beta (T*K, b, H) 张量, 按 token 折叠 (K 帧 β 是 replicated 的, 取每 K 帧第 0 个即可)
  - filter mode: 应用 mask
  - 仅触及 SNNDecoderLayer 那几层 (SNNAttentionDecoderLayer 不调 segmented_plif_selective, 自然跳过)

注意:
  - v4 还有 SNNAttention 主载长程 → 本实验只动 SNNBlock 这条 (Mamba 类比). SNNAttention 的 PE-ext (PI/NTK/YaRN)
    本实验不叠加 (隔离测 SNNBlock filter 单独效果). 后续如果有效, 再做 v4 双路增程组合.
  - 计算 cum_decay = ∏ β_t. β ∈ (0, 1) 且 train_len=256, 通常 cum_decay 在 ~β^256 ≈ 极小. 跟 Mamba 类似.
  - per-d_inner global 判定: 用 mean β over train_len 替代逐 token cum_decay 的 mean (数值稳定)

用法:
  python scripts/v4_longmamba_eval.py --load_ckpt /tmp/ih6_v4.ckpt --mode longmamba \\
    --eval_lens 131072,262144,524288,1048576 --eval_examples 64 --calib_examples 32 --theta_quantile 0.75
  (--mode vanilla 复现裸 v4 作对照, 应得到 §4c.3 Tab.2 v4-none 行: 0.44/0.53/0.50/0.42)
"""
import sys, os, math, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--load_ckpt", required=True)
ap.add_argument("--mode", default="longmamba", choices=("vanilla", "longmamba"))
ap.add_argument("--eval_lens", default="131072,262144,524288,1048576")
ap.add_argument("--eval_examples", type=int, default=64)
ap.add_argument("--calib_examples", type=int, default=32)
ap.add_argument("--theta_quantile", type=float, default=0.75,
                help="cum_decay 取此分位作 global cut (越高 → 越少 channel 标 global). 0.0 = 全 channel 都 global")
ap.add_argument("--g_beta_override", type=float, default=None,
                help="直接指定 filter 阈值 g_β (β>g_β 的 token 被 filter), 覆盖 calibration 的 quantile 公式; 用来 sweep 调参")
ap.add_argument("--print_filter_stats", action="store_true",
                help="每个 eval cell 打印实际 filter 触发的 token 数 (diagnostic)")
ap.add_argument("--rope_eval", default="none", choices=("none", "pi", "ntk", "yarn"),
                help="同 v4_induction_heads.py 的 rope_eval: 对 SNNAttention RoPE 叠加 context-length extension. 与 --mode=longmamba 正交, 可组合: vanilla+yarn / longmamba+none / longmamba+yarn")
ap.add_argument("--yarn_beta_fast", type=float, default=32.0)
ap.add_argument("--yarn_beta_slow", type=float, default=1.0)
ap.add_argument("--rope_base", type=float, default=10000.0)
ap.add_argument("--seed_eval", type=int, default=999)
ap.add_argument("--seed_calib", type=int, default=42)
ap.add_argument("--out", default=None)
args = ap.parse_args()

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LOG = open(args.out, "a") if args.out else None
def log(s):
    print(s, flush=True)
    if LOG: LOG.write(s + "\n"); LOG.flush()


# =========================================================
# 1. monkeypatch segmented_plif_selective
# =========================================================
import neuronspark.modeling_neuronspark as nm
_orig_segplif = nm.segmented_plif_selective

LM_STATE = {
    "mode": "vanilla",
    "layer_idx": 0,
    "n_snnblock_layers": 0,           # 通常 v4 num_layers / 2 (memory_layer_interval=2 时 SNNBlock 占一半)
    "captured": None,                 # list of {"beta_per_token": list[(T, b, H)]}
    "global_mask": None,              # list of (H,) bool
    "g_beta": None,                   # scalar per current eval-S
}

def _patched_segplif(beta, u, v_th, v_init, k_t, K, training=True, ahp_row=None, alpha=4.0, spike_mode="supra"):
    """beta/u shape: (T*K, b, H). beta 在 K 维上是复制的 (segmented PLIF 内同 token 共享 β).
    Mode:
      vanilla   : passthrough
      calibrate : 捕获 per-token β 分布
      filter    : 对 global channel 在 β > g_β 的 token 上 patch β←1, u←0
    """
    n = LM_STATE["n_snnblock_layers"]
    if n == 0:
        return _orig_segplif(beta, u, v_th, v_init, k_t, K, training, ahp_row=ahp_row, alpha=alpha, spike_mode=spike_mode)
    li = LM_STATE["layer_idx"] % n

    TK, b, H = beta.shape
    T = TK // K
    if LM_STATE["mode"] == "calibrate":
        # 取每个 token 的第 0 帧 β (K 帧内是复制的, 取一帧就够)
        beta_per_token = beta.view(T, K, b, H)[:, 0]   # (T, b, H)
        LM_STATE["captured"][li]["beta_per_token"].append(beta_per_token.detach().float().cpu())

    elif LM_STATE["mode"] == "filter":
        # OOM 防护: 用 masked_fill_ 原地修改 + 不 expand mask (用 broadcast). 老 torch.where 版本会创建 ~50GB 临时.
        gmask = LM_STATE["global_mask"][li].to(beta.device).view(1, 1, 1, H)   # (1,1,1,H)
        g = LM_STATE["g_beta"]
        beta_4d = beta.view(T, K, b, H)             # view, 不复制
        u_4d = u.view(T, K, b, H)
        beta_t = beta_4d[:, 0:1]                    # (T, 1, b, H) view
        low_contribution = (beta_t > g)
        filter_mask = gmask & low_contribution      # (T, 1, b, H) bool, ~MB 量级 @ T=1M
        if args.print_filter_stats and LM_STATE.get("_stat_print_done", 0) < n:
            nfilt = filter_mask.sum().item()
            ntot = T * b * H
            LM_STATE["_stat_print_done"] = LM_STATE.get("_stat_print_done", 0) + 1
            log(f"    [filter layer_{li}: {nfilt}/{ntot} = {100*nfilt/max(1,ntot):.2f}% (token-level, K帧广播) g_β={g:.3e}]")
        # masked_fill_ 接 broadcastable mask, 不实例化 (T,K,b,H), 完全原地: 零临时张量
        beta_4d.masked_fill_(filter_mask, 1.0)
        u_4d.masked_fill_(filter_mask, 0.0)

    LM_STATE["layer_idx"] += 1
    return _orig_segplif(beta, u, v_th, v_init, k_t, K, training, ahp_row=ahp_row, alpha=alpha, spike_mode=spike_mode)

nm.segmented_plif_selective = _patched_segplif


# =========================================================
# 1.5 SNNAttention RoPE 增程 (与 v4_induction_heads.py 同源)
# =========================================================
def _ntk_base_adjust(base, dim, scale):
    return base * (scale ** (dim / max(1, dim - 2)))

def _yarn_inv_freq(dim, base, scale, train_total_len, beta_fast, beta_slow, device, dtype):
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim))
    inv_interp = inv / scale
    r = train_total_len * inv / (2 * math.pi)
    gamma = ((r - beta_slow) / max(1e-6, (beta_fast - beta_slow))).clamp(0.0, 1.0)
    return inv * gamma + inv_interp * (1 - gamma)

def _effective_inv_freq(dim, base, method, scale, train_total_len, device, dtype):
    if method == "none" or scale <= 1.0:
        return 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim))
    if method == "pi":
        return (1.0 / scale) * 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim))
    if method == "ntk":
        base_eff = _ntk_base_adjust(base, dim, scale)
        return 1.0 / (base_eff ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim))
    if method == "yarn":
        return _yarn_inv_freq(dim, base, scale, train_total_len, args.yarn_beta_fast, args.yarn_beta_slow, device, dtype)
    raise ValueError(f"unknown rope_eval method: {method}")

def _rebuild_v4_rope_for_eval(model, method, eval_total_len, train_total_len, rope_base):
    from neuronspark.modeling_neuronspark import SNNAttentionDecoderLayer
    scale = max(1.0, eval_total_len / float(train_total_len)) if train_total_len > 0 else 1.0
    for layer in model.snn.layers:
        if not isinstance(layer, SNNAttentionDecoderLayer):
            continue
        dim = layer.D_key
        dev = layer.rope_cos.device; ddt = layer.rope_cos.dtype
        inv = _effective_inv_freq(dim, rope_base, method, scale, train_total_len, dev, torch.float32)
        max_len = max(int(eval_total_len) + 8, 8192)
        t = torch.arange(max_len, device=dev, dtype=torch.float32)
        ang = torch.outer(t, inv)
        cos = ang.cos().to(ddt); sin = ang.sin().to(ddt)
        layer.register_buffer('rope_cos', cos, persistent=False)
        layer.register_buffer('rope_sin', sin, persistent=False)


# =========================================================
# 2. data
# =========================================================
def make_batch(B, input_seq_len, np_rng, vocab, induction_len=1, num_triggers=5):
    il = induction_len
    T = input_seq_len + 1 + il
    ids = np.empty((B, T), dtype=np.int64)
    for b in range(B):
        s = np_rng.integers(0, vocab, size=input_seq_len).tolist()
        s.append(vocab)  # COPY_PREFIX
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


# =========================================================
# 3. load ckpt + 数清 SNNBlock 层数
# =========================================================
log(f"=== v4-LongMamba eval — mode={args.mode} ckpt={args.load_ckpt} ===")
ck = torch.load(args.load_ckpt, map_location="cpu", weights_only=False)
saved_args = ck.get("args", {})
vocab = saved_args.get("vocab", 16)
train_len = saved_args.get("train_len", 256)
induction_len = saved_args.get("induction_len", 1)
num_triggers = saved_args.get("num_triggers", 5)
no_xpos = saved_args.get("no_xpos", False)
log(f"saved args: vocab={vocab} train_len={train_len} il={induction_len} nt={num_triggers} no_xpos={no_xpos}")

from neuronspark import NeuronSparkConfig, NeuronSparkForCausalLM
cfg = NeuronSparkConfig(vocab_size=vocab + 1,
                        D=saved_args.get("D", 64), N=saved_args.get("N", 8),
                        K=saved_args.get("K", 8), num_layers=saved_args.get("num_layers", 4),
                        D_ff=saved_args.get("D_ff", 128),
                        memory_layer_interval=saved_args.get("memory_layer_interval", 2),
                        D_key=saved_args.get("D_key", 16), D_value=saved_args.get("D_value", 16),
                        spike_mode="quantal", use_ahp=False)
model = NeuronSparkForCausalLM(cfg).to(DEV)
for n, p in model.named_parameters():
    p.data = p.data.to(torch.bfloat16)
model.load_state_dict(ck["state_dict"], strict=True)
model.eval()

# 数 SNNDecoderLayer 数 (segmented_plif_selective 每层调一次)
from neuronspark.modeling_neuronspark import SNNDecoderLayer
n_snn = sum(1 for ly in model.snn.layers if isinstance(ly, SNNDecoderLayer))
LM_STATE["n_snnblock_layers"] = n_snn
log(f"loaded {sum(p.numel() for p in model.parameters())/1e6:.3f}M params; SNNDecoderLayer count={n_snn}")

from neuronspark.modeling_neuronspark import functional
def fwd(ids):
    LM_STATE["layer_idx"] = 0
    with torch.no_grad():
        functional.reset_net(model.snn)
        with torch.amp.autocast(DEV, dtype=torch.bfloat16):
            out = model(input_ids=ids)
    return out.logits.float()


# =========================================================
# 4. calibration
# =========================================================
if args.mode == "longmamba":
    log(f"\n=== Calibration (mode=calibrate, {args.calib_examples} sequences @ train_len={train_len}) ===")
    LM_STATE["mode"] = "calibrate"
    LM_STATE["captured"] = [{"beta_per_token": []} for _ in range(n_snn)]
    np_calib = np.random.default_rng(args.seed_calib)
    for i in range(args.calib_examples):
        ids, _ = make_batch(1, train_len, np_calib, vocab, induction_len, num_triggers)
        _ = fwd(ids)
    LM_STATE["mode"] = "vanilla"

    # 每层: cum_decay_c = prod_t β_t,c, mean over calibration samples; 取 75% 分位标 global
    global_masks = []
    for li in range(n_snn):
        beta_list = LM_STATE["captured"][li]["beta_per_token"]
        beta_stack = torch.stack([s.squeeze(1) for s in beta_list], dim=0)   # (N, T, H) — batch=1 squeezed
        # cum_decay per (N, H) = prod_t β_t,c
        log_beta = torch.log(beta_stack.clamp(min=1e-30))
        sum_log_beta = log_beta.sum(dim=1)         # (N, H)
        cum_decay = torch.exp(sum_log_beta).mean(dim=0)   # (H,) mean over calibration samples
        theta = torch.quantile(cum_decay, args.theta_quantile).item()
        gmask = cum_decay > theta
        global_masks.append(gmask)
        log(f"  layer {li} (SNNBlock): cum_decay_per_channel [min/med/max] = "
            f"{cum_decay.min().item():.3e} / {cum_decay.median().item():.3e} / {cum_decay.max().item():.3e}; "
            f"θ={theta:.3e} → {gmask.sum().item()}/{gmask.numel()} global channels")
    LM_STATE["global_mask"] = global_masks

    # g_β(S): 在 global channel 的 β 分布上取 quantile q = L_train/S (越长 S 越低分位 → 更严苛过滤更多 token)
    all_global_beta = []
    for li in range(n_snn):
        beta_list = LM_STATE["captured"][li]["beta_per_token"]
        beta_stack = torch.stack([s.squeeze(1) for s in beta_list], dim=0)  # (N, T, H)
        gmask = global_masks[li]
        b_glob = beta_stack[..., gmask].flatten()
        all_global_beta.append(b_glob)
    all_global_beta = torch.cat(all_global_beta)
    log(f"  global-channel β distribution: [min/q25/med/q75/max] = "
        f"{all_global_beta.min().item():.3e} / "
        f"{torch.quantile(all_global_beta, 0.25).item():.3e} / "
        f"{all_global_beta.median().item():.3e} / "
        f"{torch.quantile(all_global_beta, 0.75).item():.3e} / "
        f"{all_global_beta.max().item():.3e}; n={len(all_global_beta)}")

    g_lookup = {}
    for L in [int(x) for x in args.eval_lens.split(",")]:
        S = L + 1 + induction_len
        q = max(0.0001, min(0.9999, train_len / S))   # 期望保留 L_train/S 比例的 token (高 β = 被 filter)
        # filter: β > g_β 被 filter, 即 β ≤ g_β 保留. 保留比例 = CDF_β(g_β) = q → g_β = quantile_q(β)
        g_b = torch.quantile(all_global_beta, q).item()
        g_lookup[L] = g_b
        log(f"  g_β(L={L}) at q={q:.6f} → {g_b:.4e}  (β>g_β 的 token 被 filter; 期望保留率 {q:.3%})")
    LM_STATE["g_lookup"] = g_lookup


# =========================================================
# 5. eval
# =========================================================
def answer_acc(logits, ids, T, il):
    pred = logits[:, T - il - 1: T - 1, :].argmax(-1)
    tgt = ids[:, T - il:]
    return (pred == tgt).all(dim=1).sum().item(), tgt.shape[0]

train_total_len = train_len + 1 + induction_len
log(f"\n=== Length-generalization eval (mode={args.mode}, rope_eval={args.rope_eval}, eval_examples={args.eval_examples}) ===")
np_eval = np.random.default_rng(args.seed_eval)
for L in [int(x) for x in args.eval_lens.split(",")]:
    if args.mode == "longmamba":
        LM_STATE["mode"] = "filter"
        LM_STATE["g_beta"] = g_lookup[L]
    else:
        LM_STATE["mode"] = "vanilla"
    # SNNAttention RoPE 增程: 每 (rope_eval method, L) 重建 cos/sin
    if args.rope_eval != "none":
        L_total = L + 1 + induction_len
        _rebuild_v4_rope_for_eval(model, args.rope_eval, L_total, train_total_len, args.rope_base)

    # 长 L 用 batch=1
    B = 1 if L > 65536 else max(2, 32 // max(1, L // 2048))
    n_iter = max(1, args.eval_examples // B)
    n_corr = n_tot = 0
    for _ in range(n_iter):
        ids, T = make_batch(B, L, np_eval, vocab, induction_len, num_triggers)
        logits = fwd(ids)
        c, t = answer_acc(logits, ids, T, induction_len)
        n_corr += c; n_tot += t
    base_tag = "VANILLA" if args.mode == "vanilla" else f"LM g_β={LM_STATE['g_beta']:.2e}"
    rope_tag = "" if args.rope_eval == "none" else f" +{args.rope_eval.upper()}"
    log(f"  input_seq_len={L:8d}: acc={n_corr/n_tot:.4f}  ({n_corr}/{n_tot})  [{base_tag}{rope_tag}]")

log("=== DONE ===")
