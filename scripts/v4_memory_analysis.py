"""V4.1 记忆归因分析 —— 数学严谨地测量「历史信息在 SNNBlock vs SNNAttention 之间如何分配」。

三个实验（每个都给可复现的数学定义）：

E1  β 时间常数谱（design-level，data-independent + data-conditioned）
    每个 PLIF 神经元: β = sigmoid(w)，记忆时间常数 τ = -1 / ln(β)（膜递推 ∂V_t/∂V_{t-1} = β → 影响按 β^Δ 衰减 → 1/e 寿命 = -1/ln β）。
    SNNBlock hidden_neuron (selective): β(t) = sigmoid(b_beta + W_beta_x·x)，基准 β_base = sigmoid(b_beta)。
    SNNAttention: 线性注意力 + RoPE → 位置 s 对位置 t 输出的有效权重 ∝ Σ_j cos((t-s)·θ_j)，θ_j = base^(-2j/d_head)。
      "记忆波长" λ_j = 2π/θ_j ∈ [2π ≈ 6 tokens（最高频）, 2π·base ≈ 63000 tokens（最低频）] → 跨多个数量级。
    输出: 每层每神经元 τ 的 min/median/max + 分位数 + 是否多峰（长短期分化）；跨层比较；SNNBlock τ-range vs SNNAttention λ-range。

E2  经验状态自相关 → 经验记忆时间常数
    对一条长 val 序列做 forward，逐 token 抓 SNNBlock hidden_neuron 携带的膜电位 V_post[k_t]（= 跨 token 递归状态）。
    ρ(Δ) = mean over channels of pearson_corr_t( X_t[c], X_{t-Δ}[c] )。拟合 ρ(Δ) ≈ exp(-Δ/τ_emp)（或报对数间隔 Δ 的 ρ）。
    （注: SNNAttention 的 M_state 是 cumsum 累加器，原始自相关 ≈ 1 无意义；改报 RoPE 谱 + 注意力质心，见下。）
    SNNAttention 的「有效回看」: 对序列末尾的 query q_T，逐位置 s 的注意力权重 a_s ∝ |q_T · (RoPE-rotated k_s) · gate_s|（归一化）。
      报「注意力质心距离」E[T-s | weight a_s] 和「90% 质量距离」，跨层比较。

E3  因果扰动影响（causal perturbation）
    取真序列，扰动位置 s = T - Δ 的 embedding 一个小 ε，测末位 logits 变化 ‖Δlogits_T‖ / ‖ε‖ = influence(Δ)。
    influence(Δ) 的衰减速率 = 整模型的有效记忆视野。（component 归因需 component-ablation，留 TODO。）

用法: python scripts/v4_memory_analysis.py [--ckpt path]  (无 ckpt → 分析新初始化模型，只显示 design-level 结构)
"""
import sys, os, argparse, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from neuronspark import NeuronSparkConfig, NeuronSparkForCausalLM
from neuronspark.modeling_neuronspark import (
    PLIFNode, SelectivePLIFNode, SNNDecoderLayer, SNNAttentionDecoderLayer, functional,
)


def _fwd(model, ids):
    """每次 forward 前 reset 所有递归状态 —— 否则多次 forward 之间 hidden_neuron.v / M_state 泄漏，测量无意义。"""
    functional.reset_net(model.snn)
    return model(input_ids=ids)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=None, help="训练好的 model ckpt (torch.save 的 {state_dict, config})；无则用新初始化模型")
ap.add_argument("--seq", type=int, default=512)
ap.add_argument("--batch", type=int, default=2)
ap.add_argument("--rope_base", type=float, default=10000.0)
args = ap.parse_args()

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    if args.ckpt and os.path.exists(args.ckpt):
        ck = torch.load(args.ckpt, map_location="cpu")
        cfg = NeuronSparkConfig(**ck["config"])
        m = NeuronSparkForCausalLM(cfg)
        m.load_state_dict(ck["state_dict"])
        print(f"loaded ckpt {args.ckpt}: name={ck.get('name')}, final_loss={ck.get('final_loss')}, step={ck.get('step')}")
    else:
        torch.manual_seed(42)
        cfg = NeuronSparkConfig(vocab_size=128387, D=512, N=16, K=12, num_layers=12, D_ff=1024,
                                memory_layer_interval=4, spike_mode="quantal", use_ahp=False, ahp_init=0.02)
        m = NeuronSparkForCausalLM(cfg)
        print("no ckpt → fresh-init model (design-level analysis only)")
    for nm, p in m.named_parameters():
        if nm.endswith(('.w', '.v_th', '.b_beta', '.b_alpha', '.b_th', '.ahp')):
            p.data = p.data.float()
        else:
            p.data = p.data.to(torch.bfloat16)
    return m.to(DEV).eval(), cfg


def get_val_batch(seq, batch, vocab):
    db = os.path.join(ROOT, "data/sft_think_binned_2048")
    if os.path.isdir(db):
        bins = sorted([f for f in os.listdir(db) if f.endswith(".bin") and ".mask" not in f])
        arr = np.fromfile(os.path.join(db, bins[0]), dtype=np.uint32).reshape(-1, 2048)
        arr = arr[1000:1000 + batch, :seq].astype(np.int64)  # 用第 1000+ 行 (避开训练前段)
        return torch.from_numpy(arr).to(DEV)
    return torch.randint(0, vocab, (batch, seq), device=DEV)


def pct(x, ps=(1, 25, 50, 75, 99)):
    x = np.asarray(x)
    return {p: float(np.percentile(x, p)) for p in ps}


def tau_of_beta(beta):
    """记忆时间常数 τ = -1/ln(β)，β∈(0,1)。β→1 → τ→∞。"""
    beta = np.clip(np.asarray(beta, dtype=np.float64), 1e-6, 1 - 1e-9)
    return -1.0 / np.log(beta)


# ============================================================
# E1: β / RoPE 谱
# ============================================================
def e1_beta_spectrum(model, cfg):
    print("\n" + "=" * 78)
    print("E1  β 时间常数谱 (τ = -1/ln β，单位 = K-frame 步；β = sigmoid(w) 或 sigmoid(b_beta))")
    print("=" * 78)
    snn = model.snn
    # PLIFNode 们 (input_neuron, gate_neuron, up_neuron, output_neuron)
    print("\n[PLIFNode β=sigmoid(w)] 每模块 τ 分位 (1/25/50/75/99%) + multimodal?")
    for name, mod in snn.named_modules():
        if isinstance(mod, PLIFNode):
            beta = torch.sigmoid(mod.w.detach().float()).cpu().numpy()
            tau = tau_of_beta(beta)
            q = pct(tau)
            # 多峰检测: 把 τ 分到 [<3, 3-10, 10-30, 30-100, >100] 桶，看分布
            buckets = np.histogram(np.clip(tau, 0, 200), bins=[0, 3, 10, 30, 100, 1e9])[0]
            bf = buckets / buckets.sum()
            print(f"  {name:50s} n={len(beta):5d}  τ[1/25/50/75/99%]={q[1]:.1f}/{q[25]:.1f}/{q[50]:.1f}/{q[75]:.1f}/{q[99]:.1f}  "
                  f"buckets[<3,3-10,10-30,30-100,>100]={['%.0f%%'%(100*b) for b in bf]}")
    # SelectivePLIF hidden_neuron: β_base = sigmoid(b_beta)
    print("\n[SNNBlock hidden_neuron (selective) β_base=sigmoid(b_beta), per-layer]")
    for li, layer in enumerate(snn.layers):
        if isinstance(layer, SNNDecoderLayer):
            bb = layer.snn_block.b_beta.detach().float().cpu().numpy()
            beta_base = 1 / (1 + np.exp(-bb))
            tau = tau_of_beta(beta_base)
            q = pct(tau)
            buckets = np.histogram(np.clip(tau, 0, 200), bins=[0, 3, 10, 30, 100, 1e9])[0]
            bf = buckets / buckets.sum()
            print(f"  layer{li:2d}.snn_block  DN={len(bb):5d}  τ_base[1/25/50/75/99%]={q[1]:.1f}/{q[25]:.1f}/{q[50]:.1f}/{q[75]:.1f}/{q[99]:.1f}  "
                  f"buckets={['%.0f%%'%(100*b) for b in bf]}  β_base[min,max]=[{beta_base.min():.3f},{beta_base.max():.3f}]")
    # RoPE 谱 for SNNAttention
    print("\n[SNNAttention RoPE 记忆波长 λ_j = 2π/θ_j, θ_j = base^(-2j/d_head)]  (单位 = token)")
    d_head = cfg.D_key  # head dim for RoPE
    base = args.rope_base
    js = np.arange(0, d_head // 2)
    theta = base ** (-2.0 * js / d_head)
    lam = 2 * np.pi / theta
    print(f"  d_head={d_head}, rope_base={base:.0f}  →  λ ∈ [{lam.min():.1f}, {lam.max():.0f}] tokens  "
          f"(分位 25/50/75% = {np.percentile(lam,25):.1f} / {np.percentile(lam,50):.1f} / {np.percentile(lam,75):.0f})")
    print(f"  对比: SNNBlock 的 τ 最大 ~{tau_of_beta(0.99):.0f} K-frame 步（跨 token 约同量级）；SNNAttention 最长 λ ~{lam.max():.0f} tokens")
    print(f"  → 设计层面: SNNAttention 的记忆视野跨度（~6 到 ~{lam.max():.0f} tokens）远超 SNNBlock（~5 到 ~100 步）。")


# ============================================================
# E2: 经验状态自相关 + 注意力质心
# ============================================================
def _capture_states(model, ids):
    """forward 一次，逐 token 抓 (a) 每 SNNDecoderLayer 的 hidden_neuron 携带膜电位 V_post[k_t]，(b) 每 SNNAttn 的注意力权重 over positions."""
    snn = model.snn
    captured = {"block_carry": {}, "attn_weights": {}}
    import neuronspark.modeling_neuronspark as nm
    orig_sel = nm.segmented_plif_selective
    sel_call_layers = [li for li, l in enumerate(snn.layers) if isinstance(l, SNNDecoderLayer)]
    sel_idx = [0]

    def wrap_sel(*a, **kw):
        out = orig_sel(*a, **kw)
        output, V_post, v_carry = out
        K = kw.get("K", a[5] if len(a) > 5 else None)
        # V_post: (TK, b, H); k_t: (T, b)
        # reconstruct per-token carried V_post = V_post[t*K + k_t[t]]
        # we need k_t — it's a[4] or kw['k_t']
        k_t = kw.get("k_t", a[4])
        TK, b, H = V_post.shape
        T = TK // K
        idx = k_t.view(T, 1, b, 1).expand(T, 1, b, H)
        carried = V_post.view(T, K, b, H).gather(1, idx).squeeze(1)  # (T, b, H)
        layer_i = sel_call_layers[sel_idx[0] % len(sel_call_layers)] if sel_idx[0] < len(sel_call_layers) else sel_idx[0]
        captured["block_carry"][layer_i] = carried[:, 0, :].float().detach().cpu().numpy()  # (T, H), batch 0
        sel_idx[0] += 1
        return out

    nm.segmented_plif_selective = wrap_sel
    handles = []

    # SNNAttention: hook to capture q, k, gate → compute per-position attention weight from last query
    attn_capture = {}
    def make_attn_hook(li):
        def hook(module, inp, out):
            attn_capture[li] = {}
        return hook
    # 简化: 我们直接在 forward 后从模块状态拿；但 q/k 是内部变量。改用另一种: hook attn_out_proj 的输入。太复杂 → 改报 RoPE 谱 (E1 已做)。
    # 这里只抓 block_carry; attn 的经验质心需要更深 hook，先跳过，靠 E1 的 RoPE 谱 + E3 的 perturbation。

    try:
        functional.reset_net(model.snn)
        _ = model(input_ids=ids)
    finally:
        nm.segmented_plif_selective = orig_sel
        for h in handles:
            h.remove()
    return captured


def autocorr_profile(X, deltas):
    """X: (T, C). 返回 ρ(Δ) = mean_c pearson(X[:-Δ,c], X[Δ:,c]) for Δ in deltas."""
    T, C = X.shape
    X = X - X.mean(axis=0, keepdims=True)
    std = X.std(axis=0) + 1e-12
    out = {}
    for d in deltas:
        if d >= T:
            out[d] = float("nan"); continue
        a = X[:-d]; bb = X[d:]
        num = (a * bb).mean(axis=0)
        den = std * std
        rho = num / den
        out[d] = float(np.nanmean(rho))
    return out


def e2_state_autocorr(model, ids):
    print("\n" + "=" * 78)
    print("E2  经验状态自相关 ρ(Δ) = mean_c pearson(state_t[c], state_{t-Δ}[c])")
    print("    (SNNBlock hidden_neuron 跨 token 携带膜电位 V_post[k_t]；衰减越慢 → 经验记忆越长)")
    print("=" * 78)
    cap = _capture_states(model, ids)
    deltas = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    for li in sorted(cap["block_carry"]):
        X = cap["block_carry"][li]  # (T, H)
        prof = autocorr_profile(X, deltas)
        # 经验 τ: 找 ρ 降到 1/e ≈ 0.368 的 Δ（线性插值）
        ds = [d for d in deltas if not math.isnan(prof[d])]
        rhos = [prof[d] for d in ds]
        tau_emp = float("inf")
        for i in range(len(ds) - 1):
            if rhos[i] >= 1/math.e >= rhos[i+1]:
                # linear interp in Δ
                t = (rhos[i] - 1/math.e) / (rhos[i] - rhos[i+1] + 1e-12)
                tau_emp = ds[i] + t * (ds[i+1] - ds[i]); break
        if rhos[0] < 1/math.e:
            tau_emp = ds[0]  # decays within 1 step
        rho_str = "  ".join(f"Δ{d}={prof[d]:+.3f}" for d in deltas)
        print(f"  layer{li:2d}.snn_block  τ_emp(1/e)≈{tau_emp if tau_emp!=float('inf') else '>'+str(deltas[-1])}  {rho_str}")
    print("  注: 训练步数少 (1500) 时模型基本没学到长程结构，ρ 主要反映 β 设计 (见 E1)；长训练后再测会更有意义。")


# ============================================================
# E3: 因果扰动影响
# ============================================================
@torch.no_grad()
def _influence_curve(model, ids, deltas, label, n_eps=16, eps_scale=0.05):
    """influence(Δ) = mean over n_eps 随机方向 ε of ‖logits_T(扰动 pos T-1-Δ) - logits_T(原)‖ / ‖ε‖.
    多 ε 平均消掉「单个 ε 碰巧对齐敏感方向」的噪声。返回 dict Δ→influence。"""
    snn = model.snn
    T = ids.shape[1]
    embed = snn.embed_tokens
    base_logits_T = _fwd(model, ids).logits[:, -1, :].float()
    orig_fwd = embed.forward
    out = {}
    for d in deltas:
        pos = T - 1 - d
        if pos < 0:
            out[d] = float("nan"); continue
        g = torch.Generator(device=DEV).manual_seed(1000 + d)
        accum = 0.0
        for _ in range(n_eps):
            noise = torch.randn(snn.D, device=DEV, dtype=torch.float32, generator=g) * eps_scale
            eps_norm = noise.norm().item()
            def patched(input_ids, _orig=orig_fwd, _pos=pos, _noise=noise):
                e = _orig(input_ids).clone()
                e[:, _pos, :] = e[:, _pos, :] + _noise.to(e.dtype)
                return e
            embed.forward = patched
            try:
                dl = (_fwd(model, ids).logits[:, -1, :].float() - base_logits_T).norm(dim=-1).mean().item()
            finally:
                embed.forward = orig_fwd
            accum += dl / eps_norm
        out[d] = accum / n_eps
    return out


import types
import torch.nn.functional as F
from neuronspark.modeling_neuronspark import _apply_rope, _ponder_v3_pick


def _attn_forward_no_xpos(self, h):
    """SNNAttentionDecoderLayer.forward_parallel 的副本，唯一改动:
    M_all = kv_gated（不做 cumsum, 每位置只看自己；M_state 也不加）→ 注意力子层变成纯 per-position，
    切断所有经 SNNAttention 的跨位置信息。其余（RoPE/gate/FFN/PonderNet）原样。"""
    seq_len, batch, D = h.shape
    K = self.K
    steps_vec = torch.arange(1, K + 1, device=h.device, dtype=h.dtype)
    # 子层 1: SNN-Attention，但跨位置混合关掉
    h_normed = self.attn_norm(h)
    flat = h_normed.reshape(seq_len * batch, D)
    qkv = self.qkv_proj(flat)
    q, k, v = qkv.split([self.D_key, self.D_key, self.D_value], dim=-1)
    q = q.reshape(seq_len, batch, self.D_key)
    k = k.reshape(seq_len, batch, self.D_key)
    v = v.reshape(seq_len, batch, self.D_value)
    pos = self.pos_offset
    rope_cos = self.rope_cos[pos:pos + seq_len].unsqueeze(1).to(q.dtype)
    rope_sin = self.rope_sin[pos:pos + seq_len].unsqueeze(1).to(q.dtype)
    q = _apply_rope(q, rope_cos, rope_sin)
    k = _apply_rope(k, rope_cos, rope_sin)
    self.pos_offset = pos + seq_len
    gate = self._gate_neuron_parallel(h_normed)
    k = F.normalize(k, dim=-1)
    kv_outer = k.unsqueeze(-1) * v.unsqueeze(-2)
    kv_gated = gate.unsqueeze(-1) * kv_outer
    M_all = kv_gated  # <<< 改动：不 cumsum，每位置只保留自己的 kv（M_state 也不加 → 切断跨位置）
    attn_out = torch.einsum('sbk,sbkv->sbv', q, M_all)
    attn_out = self.attn_out_norm(attn_out)
    res_attn = self.attn_out_proj(attn_out.reshape(seq_len * batch, self.D_value)).reshape(seq_len, batch, D)
    res_attn = res_attn - res_attn.mean(dim=-1, keepdim=True)
    h = h + res_attn
    # 子层 2: SNNFFN (原样)
    k_logits_ffn = self.ffn_k_predictor(h)
    y_st_ffn, y_hard_ffn = _ponder_v3_pick(k_logits_ffn, training=self.training,
                                           temperature=self.ponder_T, eps_explore=self.eps_explore)
    k_t_ffn = y_hard_ffn.argmax(dim=-1)
    Ka_f = K if self.training else (int(k_t_ffn.max().item()) + 1)
    h_normed_k2 = self.ffn_norm(h).repeat_interleave(Ka_f, dim=0)
    v_in2 = self._input_neuron_parallel(self.input_neuron2, h_normed_k2, k_t_ffn, Ka_f, self.training)
    cont_ffn = self.snn_ffn.forward_parallel(v_in2, k_t_ffn, Ka_f, self.training)
    frames_ffn = cont_ffn.view(seq_len, Ka_f, batch, D)
    combined_ffn = (y_st_ffn[..., :Ka_f].permute(0, 2, 1).unsqueeze(-1) * frames_ffn).sum(dim=1)
    res_ffn = self.ffn_out_proj(combined_ffn)
    res_ffn = res_ffn - res_ffn.mean(dim=-1, keepdim=True)
    h = h + res_ffn
    ek_ffn = (y_st_ffn.detach() * steps_vec[None, None, :]).sum(-1)
    self._last_y_hard_ffn = y_hard_ffn
    return h, ek_ffn.mean()


@torch.no_grad()
def e3_perturbation_influence(model, ids):
    print("\n" + "=" * 78)
    print("E3  因果扰动影响 influence(Δ) = ‖Δlogits_T‖ / ‖ε‖  (扰动 emb at pos T-1-Δ → 末位 logits 变化)")
    print("    component 归因: (a) full / (b) SNNAttention 跨位置混合关掉 (M_all=kv_gated, 直接 override forward_parallel)")
    print("    → (b) 下跨位置信息只剩 SNNBlock segmented-PLIF carry (快通道~1-2 tokens, 慢通道~3% ~100 tokens). influence(Δ) 在长 Δ 塌掉 ⇒ 长程是 SNNAttention 扛的。")
    print("=" * 78)
    T = ids.shape[1]
    deltas = [d for d in [1, 2, 4, 8, 16, 32, 64, 128, 256] if d < T - 1]
    inf_full = _influence_curve(model, ids, deltas, "full")
    # (b) override 所有 SNNAttentionDecoderLayer 的 forward_parallel
    attn_layers = [l for l in model.snn.layers if isinstance(l, SNNAttentionDecoderLayer)]
    orig_fps = [l.forward_parallel for l in attn_layers]
    for l in attn_layers:
        l.forward_parallel = types.MethodType(_attn_forward_no_xpos, l)
    try:
        inf_noattn = _influence_curve(model, ids, deltas, "no-attn-mix")
    finally:
        for l, fp in zip(attn_layers, orig_fps):
            l.forward_parallel = fp
    print(f"  {'Δ':>5s} | {'influence(full)':>16s} | {'influence(no-attn-mix)':>22s} | ratio noattn/full")
    for d in deltas:
        f, n = inf_full[d], inf_noattn[d]
        print(f"  {d:5d} | {f:16.4e} | {n:22.4e} | {n/f if f>0 else float('nan'):.4f}")
    print("  解读: ratio≈1 → 该距离的信息不靠 SNNAttention (近距离, SNNBlock/残差就够); ratio≈0 → 该距离全靠 SNNAttention 扛 (长距离)。")


def main():
    model, cfg = load_model()
    ids = get_val_batch(args.seq, args.batch, cfg.vocab_size)
    print(f"\nmodel: D={cfg.D}, N={cfg.N}, K={cfg.K}, layers={cfg.num_layers} (mem interval {cfg.memory_layer_interval}), spike_mode={getattr(cfg,'spike_mode','supra')}")
    print(f"val batch: {tuple(ids.shape)}")
    with torch.amp.autocast(DEV, dtype=torch.bfloat16):
        e1_beta_spectrum(model, cfg)
        e2_state_autocorr(model, ids)
        e3_perturbation_influence(model, ids)
    print("\n=== ANALYSIS DONE ===")


if __name__ == "__main__":
    main()
