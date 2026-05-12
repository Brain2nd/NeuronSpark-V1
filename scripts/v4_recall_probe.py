"""V4.1 记忆归因 — 合成关联召回探针（Step 1，干净的因果实验，替代 v4_memory_analysis.py 噪声主导的 E3）。

任务（induction-head 风格）：序列长度 T，随机选 KEY K、VALUE V（content vocab，K≠V）：
    pos 0 .. p-1   : 随机 filler（≠ K）
    pos p          : K          ← "定义"
    pos p+1        : V
    pos p+2 .. T-2 : 随机 filler（≠ K）
    pos T-1        : K          ← "查询"
    目标：位置 T-1 的下一 token 预测 = V。  p = T-1-Δ-1（→ V 在 T-1 之前 Δ 个 token 处）。
模型必须 in-context 召回（K/V 每例随机，不能背）。训练时 Δ 在 [Δ_min, T-4] 均匀采样；评测时固定 Δ ∈ Δ_evals。

跑：train 一个小 V4.1（MAL + LSUV）~train_steps 步学会召回，然后对 {full, no-SNNAttention(cumsum→identity)}
两个配置评测 recall-acc(Δ)。
  - full 应在所有 Δ 高（SNNAttention 的 cumsum + 低频 RoPE 覆盖到 ~万 tokens）
  - no-SNNAttention 应在小 Δ 高（SNNBlock segmented-PLIF carry + 皮层侧向连接），在 Δ > ~100 掉
  → no-SNNAttention 还能召回的最大 Δ = SNNBlock 的「reach」；full 与 no-SNNAttention 在大 Δ 的 gap = SNNAttention 独有贡献。
论文图：recall-acc vs Δ，两条线。
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from neuronspark import NeuronSparkConfig, NeuronSparkForCausalLM
from neuronspark.modeling_neuronspark import functional, SNNAttentionDecoderLayer

ap = argparse.ArgumentParser()
ap.add_argument("--vocab", type=int, default=256)
ap.add_argument("--T", type=int, default=1280, help="序列长度（→ 最大可测 Δ ≈ T-4）")
ap.add_argument("--D", type=int, default=256)
ap.add_argument("--N", type=int, default=8)
ap.add_argument("--K_max", type=int, default=8)
ap.add_argument("--layers", type=int, default=6)
ap.add_argument("--mem_interval", type=int, default=3)
ap.add_argument("--spike_mode", default="quantal", choices=("supra", "quantal"))
ap.add_argument("--train_steps", type=int, default=3000)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--muon_lr", type=float, default=0.005)
ap.add_argument("--lion_lr", type=float, default=1e-4)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--delta_min", type=int, default=8)
ap.add_argument("--delta_evals", default="16,64,256,1024")
ap.add_argument("--eval_samples", type=int, default=256)
ap.add_argument("--no_lsuv", action="store_true")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=None)
args = ap.parse_args()

DEV = "cuda"
torch.manual_seed(args.seed)
rng = np.random.RandomState(args.seed)
DELTA_EVALS = [int(x) for x in args.delta_evals.split(",")]

# token layout: 0..3 unused/special pool start; content tokens = [4, vocab). K and V drawn from content.
CONTENT_LO = 4


def make_batch(B, T, delta=None, gen=None):
    """返回 (ids (B,T) int64, answer_pos (B,) = T-1, answer_tok (B,))。delta=None → 每例随机采样 Δ。"""
    gen = gen or rng
    ids = gen.randint(CONTENT_LO, args.vocab, size=(B, T)).astype(np.int64)
    ans = np.empty(B, dtype=np.int64)
    for b in range(B):
        d = delta if delta is not None else int(gen.randint(args.delta_min, T - 3))
        K = int(gen.randint(CONTENT_LO, args.vocab))
        V = int(gen.randint(CONTENT_LO, args.vocab))
        while V == K:
            V = int(gen.randint(CONTENT_LO, args.vocab))
        p = T - 1 - d - 1  # K 在 p, V 在 p+1, 查询 K 在 T-1; V 在 T-1 之前 d 个 token
        # 把序列里其它位置的 K 替换掉（保证 K 只出现在 p 和 T-1）
        mask = ids[b] == K
        ids[b][mask] = (K + 1) if (K + 1) < args.vocab else CONTENT_LO
        # 重新检查（替换值也可能 == K? K+1 != K, 安全）
        ids[b, p] = K
        ids[b, p + 1] = V
        ids[b, T - 1] = K
        ans[b] = V
    return torch.from_numpy(ids).to(DEV), torch.full((B,), T - 1, device=DEV), torch.from_numpy(ans).to(DEV)


# --- no-SNNAttention forward (cumsum→identity, per-position) — 复用 memory_analysis 的思路 ---
def _attn_forward_no_xpos(self, h):
    seq_len, batch, D = h.shape
    K = self.K
    steps_vec = torch.arange(1, K + 1, device=h.device, dtype=h.dtype)
    h_normed = self.attn_norm(h)
    flat = h_normed.reshape(seq_len * batch, D)
    qkv = self.qkv_proj(flat)
    q, k, v = qkv.split([self.D_key, self.D_key, self.D_value], dim=-1)
    q = q.reshape(seq_len, batch, self.D_key); k = k.reshape(seq_len, batch, self.D_key); v = v.reshape(seq_len, batch, self.D_value)
    pos = self.pos_offset
    from neuronspark.modeling_neuronspark import _apply_rope
    rope_cos = self.rope_cos[pos:pos + seq_len].unsqueeze(1).to(q.dtype); rope_sin = self.rope_sin[pos:pos + seq_len].unsqueeze(1).to(q.dtype)
    q = _apply_rope(q, rope_cos, rope_sin); k = _apply_rope(k, rope_cos, rope_sin)
    self.pos_offset = pos + seq_len
    gate = self._gate_neuron_parallel(h_normed)
    k = F.normalize(k, dim=-1)
    kv_gated = gate.unsqueeze(-1) * (k.unsqueeze(-1) * v.unsqueeze(-2))
    M_all = kv_gated  # <<< 不 cumsum: 每位置只看自己 → 切断经 SNNAttention 的跨位置
    attn_out = torch.einsum('sbk,sbkv->sbv', q, M_all)
    attn_out = self.attn_out_norm(attn_out)
    res_attn = self.attn_out_proj(attn_out.reshape(seq_len * batch, self.D_value)).reshape(seq_len, batch, D)
    res_attn = res_attn - res_attn.mean(dim=-1, keepdim=True)
    h = h + res_attn
    from neuronspark.modeling_neuronspark import _ponder_v3_pick
    k_logits_ffn = self.ffn_k_predictor(h)
    y_st_ffn, y_hard_ffn = _ponder_v3_pick(k_logits_ffn, training=self.training, temperature=self.ponder_T, eps_explore=self.eps_explore)
    k_t_ffn = y_hard_ffn.argmax(dim=-1)
    Ka_f = K if self.training else (int(k_t_ffn.max().item()) + 1)
    h_normed_k2 = self.ffn_norm(h).repeat_interleave(Ka_f, dim=0)
    v_in2 = self._input_neuron_parallel(self.input_neuron2, h_normed_k2, k_t_ffn, Ka_f, self.training)
    cont_ffn = self.snn_ffn.forward_parallel(v_in2, k_t_ffn, Ka_f, self.training)
    frames_ffn = cont_ffn.view(seq_len, Ka_f, batch, D)
    combined_ffn = (y_st_ffn[..., :Ka_f].permute(0, 2, 1).unsqueeze(-1) * frames_ffn).sum(dim=1)
    res_ffn = self.ffn_out_proj(combined_ffn); res_ffn = res_ffn - res_ffn.mean(dim=-1, keepdim=True)
    h = h + res_ffn
    self._last_y_hard_ffn = y_hard_ffn
    return h, (y_st_ffn.detach() * steps_vec[None, None, :]).sum(-1).mean()


@torch.no_grad()
def eval_recall(model, n_samples, label, no_attn=False):
    model.eval()
    attn_layers = [l for l in model.snn.layers if isinstance(l, SNNAttentionDecoderLayer)]
    orig = [l.forward_parallel for l in attn_layers]
    import types
    if no_attn:
        for l in attn_layers:
            l.forward_parallel = types.MethodType(_attn_forward_no_xpos, l)
    try:
        accs = {}
        for d in DELTA_EVALS:
            if d >= args.T - 3:
                accs[d] = float("nan"); continue
            correct = 0; total = 0
            for _ in range(0, n_samples, args.batch):
                bs = min(args.batch, n_samples - total)
                ids, ap_, ans = make_batch(bs, args.T, delta=d)
                functional.reset_net(model.snn)
                with torch.amp.autocast(DEV, dtype=torch.bfloat16):
                    out = model(input_ids=ids)
                pred = out.logits[torch.arange(bs, device=DEV), ap_].argmax(-1)
                correct += (pred == ans).sum().item(); total += bs
            accs[d] = correct / total
    finally:
        for l, fp in zip(attn_layers, orig):
            l.forward_parallel = fp
    model.train()
    print(f"  [{label}] recall acc by Δ: " + "  ".join(f"Δ{d}={accs[d]:.3f}" for d in DELTA_EVALS), flush=True)
    return accs


def main():
    LOGF = open(args.out, "a") if args.out else None
    def log(s):
        print(s, flush=True)
        if LOGF: LOGF.write(s + "\n"); LOGF.flush()

    cfg = NeuronSparkConfig(vocab_size=args.vocab, D=args.D, N=args.N, K=args.K_max, num_layers=args.layers,
                            D_ff=2 * args.D, memory_layer_interval=args.mem_interval, spike_mode=args.spike_mode, use_ahp=False)
    model = NeuronSparkForCausalLM(cfg).to(DEV)
    for nm, p in model.named_parameters():
        if nm.endswith(('.w', '.v_th', '.b_beta', '.b_alpha', '.b_th', '.ahp')): p.data = p.data.float()
        else: p.data = p.data.to(torch.bfloat16)
    n_attn = sum(1 for l in model.snn.layers if isinstance(l, SNNAttentionDecoderLayer))
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    log(f"recall probe: vocab={args.vocab} T={args.T} | model D={args.D} N={args.N} K={args.K_max} {args.layers}L "
        f"({n_attn} SNNAttn @ interval {args.mem_interval}) {n_params:.1f}M spike={args.spike_mode}")
    if not args.no_lsuv:
        from utils.lsuv_snn_init import lsuv_snn_init
        _ids0, _, _ = make_batch(2, min(args.T, 256))
        lsuv_snn_init(model, _ids0, target_p_fire=0.3, n_passes=3, verbose=False)
        log("  LSUV v2 init applied")

    from utils.muon_adam_lion import SingleDeviceMoonshotMuonAdamLion, build_muon_adam_lion_param_groups
    opt = SingleDeviceMoonshotMuonAdamLion(build_muon_adam_lion_param_groups(
        model, muon_lr=args.muon_lr, adam_base_lr=args.lr, lion_lr=args.lion_lr, neuron_lr_mult=1.0))
    model.train()
    t0 = time.time()
    for step in range(args.train_steps):
        ids, ap_, ans = make_batch(args.batch, args.T)  # 每例随机 Δ
        functional.reset_net(model.snn)
        opt.zero_grad()
        with torch.amp.autocast(DEV, dtype=torch.bfloat16):
            out = model(input_ids=ids)
            # CE 只在查询位置（其余是 uniform filler，不算 loss）
            logits_q = out.logits[torch.arange(args.batch, device=DEV), ap_]
            loss = F.cross_entropy(logits_q.float(), ans)
        if not torch.isfinite(loss):
            log(f"  NaN at step {step}!"); break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 200 == 0 or step == args.train_steps - 1:
            with torch.no_grad():
                acc = (logits_q.argmax(-1) == ans).float().mean().item()
            log(f"  step {step:5d}: loss={float(loss):.4f}  train_recall_acc={acc:.3f}  ({(time.time()-t0)/(step+1)*1000:.0f} ms/step)")
    log(f"=== EVAL (recall acc by Δ, {args.eval_samples} samples each) ===")
    a_full = eval_recall(model, args.eval_samples, "full     ", no_attn=False)
    a_noatt = eval_recall(model, args.eval_samples, "no-SNNAttn", no_attn=True)
    log("\nRESULT recall-acc vs Δ:")
    log(f"  {'Δ':>6s} | {'full':>8s} | {'no-SNNAttn':>11s} | gap (=SNNAttn 独有贡献)")
    for d in DELTA_EVALS:
        log(f"  {d:6d} | {a_full[d]:8.3f} | {a_noatt[d]:11.3f} | {a_full[d]-a_noatt[d]:+.3f}")
    log("DONE")


if __name__ == "__main__":
    main()
