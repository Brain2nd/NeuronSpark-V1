"""V4.1 记忆归因 — MQAR-style 关联召回探针 v2（Step 1 重做；v1 naive 单-needle 任务太难学不出来）。

任务（标准 associative recall，Mamba/H3 论文那个）：
  分离 vocab：BOS(0), SEP(1), key tokens [2, 2+nk), value tokens [2+nk, 2+nk+nv)。
  序列：BOS · (k_1 v_1) · (k_2 v_2) · ... · (k_N v_N) · SEP · k_q   —— k_q 是某个 k_j（每对的 key 互不相同）。
  目标：SEP·k_q 之后的位置预测 v_j（k_q 对应的 value）。
  → 模型必须 in-context 召回（每例 key/value 随机映射，不能背）。查询第 j 个 pair → 答案 v_j 在 pos 2j，查询 k_q 在 pos 2N+2，距离 ≈ 2N+2 - 2j。
  训练时随机 j；评测时固定 j ∈ {N, 3N/4, N/2, N/4, 1} → 距离 ≈ {2, N/2, N, 3N/2, 2N}。

跑：train 一个小模型学召回（V4.1 quantal / supra / 或 --baseline transformer），eval recall-acc vs 距离 for
{full / no-SNNAttention(cumsum→identity)（仅 V4.1）}。论文图：哪个 component 负责哪段距离 + V4.1 vs transformer 的召回能力。
"""
import sys, os, time, argparse, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuronspark import NeuronSparkConfig, NeuronSparkForCausalLM
from neuronspark.modeling_neuronspark import functional, SNNAttentionDecoderLayer, _apply_rope, _ponder_v3_pick

ap = argparse.ArgumentParser()
ap.add_argument("--n_pairs", type=int, default=128, help="每序列 (key,value) 对数 N → 序列长 ≈ 2N+3")
ap.add_argument("--n_keys", type=int, default=64, help="key vocab 大小（≥ n_pairs）")
ap.add_argument("--n_vals", type=int, default=64, help="value vocab 大小")
ap.add_argument("--model", default="v4", choices=("v4", "transformer"))
ap.add_argument("--spike_mode", default="quantal", choices=("supra", "quantal"))
ap.add_argument("--D", type=int, default=256)
ap.add_argument("--N", type=int, default=8)
ap.add_argument("--K_max", type=int, default=8)
ap.add_argument("--layers", type=int, default=6)
ap.add_argument("--mem_interval", type=int, default=3)
ap.add_argument("--n_heads", type=int, default=4, help="transformer baseline 的 head 数")
ap.add_argument("--train_steps", type=int, default=8000)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--muon_lr", type=float, default=0.005)
ap.add_argument("--lion_lr", type=float, default=1e-4)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--eval_samples", type=int, default=512)
ap.add_argument("--no_lsuv", action="store_true")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=None)
args = ap.parse_args()

DEV = "cuda"
torch.manual_seed(args.seed)
rng = np.random.RandomState(args.seed)
assert args.n_keys >= args.n_pairs, "n_keys 必须 ≥ n_pairs (每对 key 互不相同)"
BOS, SEP = 0, 1
KEY_LO = 2
VAL_LO = 2 + args.n_keys
VOCAB = 2 + args.n_keys + args.n_vals
T = 2 * args.n_pairs + 3  # BOS + 2N + SEP + k_q  (答案在 k_q 之后那个位置 → 总长 = 1 + 2N + 1 + 1 = 2N+3)


def make_batch(B, query_j=None, gen=None):
    """返回 (ids (B,T), ans_pos (B,) = T-1, ans_tok (B,)). query_j: 1-indexed 查第几对; None → 随机."""
    gen = gen or rng
    ids = np.zeros((B, T), dtype=np.int64)
    ans = np.empty(B, dtype=np.int64)
    for b in range(B):
        keys = gen.choice(args.n_keys, size=args.n_pairs, replace=False) + KEY_LO
        vals = gen.randint(0, args.n_vals, size=args.n_pairs) + VAL_LO
        ids[b, 0] = BOS
        for j in range(args.n_pairs):
            ids[b, 1 + 2 * j] = keys[j]
            ids[b, 2 + 2 * j] = vals[j]
        ids[b, 1 + 2 * args.n_pairs] = SEP   # pos 2N+1
        qj = (query_j - 1) if query_j is not None else int(gen.randint(0, args.n_pairs))
        ids[b, 2 + 2 * args.n_pairs] = keys[qj]  # pos 2N+2 = T-1; 答案预测在它之后 → 但序列就到这里 → 我们用 logits[T-1] 预测「下一个」
        ans[b] = vals[qj]
    # 答案是「k_q 之后应出现的 token」= v_j —— 用 logits at pos T-1 (= k_q 的位置) 预测下一 token
    return torch.from_numpy(ids).to(DEV), torch.full((B,), T - 1, device=DEV), torch.from_numpy(ans).to(DEV)


# ---- minimal causal transformer baseline ----
class TinyGPT(nn.Module):
    def __init__(self, vocab, D, n_heads, n_layers, max_T):
        super().__init__()
        self.emb = nn.Embedding(vocab, D)
        self.pos = nn.Embedding(max_T, D)
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(nn.ModuleDict({
                "ln1": nn.LayerNorm(D), "attn": nn.MultiheadAttention(D, n_heads, batch_first=True),
                "ln2": nn.LayerNorm(D), "mlp": nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D)),
            }))
        self.lnf = nn.LayerNorm(D)
        self.head = nn.Linear(D, vocab, bias=False)
        self.max_T = max_T

    def forward(self, input_ids, **kw):
        B, T = input_ids.shape
        x = self.emb(input_ids) + self.pos(torch.arange(T, device=input_ids.device))[None]
        cmask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        for blk in self.blocks:
            h = blk["ln1"](x)
            a, _ = blk["attn"](h, h, h, attn_mask=cmask, need_weights=False)
            x = x + a
            x = x + blk["mlp"](blk["ln2"](x))
        logits = self.head(self.lnf(x))
        class O: pass
        o = O(); o.logits = logits; return o


def build_model():
    if args.model == "transformer":
        m = TinyGPT(VOCAB, args.D, args.n_heads, args.layers, T + 4).to(DEV)
        return m, 0
    cfg = NeuronSparkConfig(vocab_size=VOCAB, D=args.D, N=args.N, K=args.K_max, num_layers=args.layers,
                            D_ff=2 * args.D, memory_layer_interval=args.mem_interval, spike_mode=args.spike_mode, use_ahp=False)
    m = NeuronSparkForCausalLM(cfg).to(DEV)
    for nm, p in m.named_parameters():
        if nm.endswith(('.w', '.v_th', '.b_beta', '.b_alpha', '.b_th', '.ahp')): p.data = p.data.float()
        else: p.data = p.data.to(torch.bfloat16)
    n_attn = sum(1 for l in m.snn.layers if isinstance(l, SNNAttentionDecoderLayer))
    return m, n_attn


# ---- no-SNNAttention forward (cumsum→identity) for V4.1 ----
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
    rope_cos = self.rope_cos[pos:pos + seq_len].unsqueeze(1).to(q.dtype); rope_sin = self.rope_sin[pos:pos + seq_len].unsqueeze(1).to(q.dtype)
    q = _apply_rope(q, rope_cos, rope_sin); k = _apply_rope(k, rope_cos, rope_sin)
    self.pos_offset = pos + seq_len
    gate = self._gate_neuron_parallel(h_normed)
    k = F.normalize(k, dim=-1)
    M_all = gate.unsqueeze(-1) * (k.unsqueeze(-1) * v.unsqueeze(-2))   # no cumsum → per-position
    attn_out = torch.einsum('sbk,sbkv->sbv', q, M_all)
    attn_out = self.attn_out_norm(attn_out)
    res_attn = self.attn_out_proj(attn_out.reshape(seq_len * batch, self.D_value)).reshape(seq_len, batch, D)
    res_attn = res_attn - res_attn.mean(dim=-1, keepdim=True)
    h = h + res_attn
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


def _reset(model):
    if hasattr(model, "snn"):
        functional.reset_net(model.snn)


def _fwd(model, ids):
    _reset(model)
    if hasattr(model, "snn"):
        with torch.amp.autocast(DEV, dtype=torch.bfloat16):
            return model(input_ids=ids)
    return model(input_ids=ids)


@torch.no_grad()
def eval_recall(model, label, no_attn=False):
    model.eval()
    js = sorted(set([args.n_pairs, max(1, 3 * args.n_pairs // 4), args.n_pairs // 2, max(1, args.n_pairs // 4), 1]))
    is_v4 = hasattr(model, "snn")
    import types
    if no_attn and is_v4:
        attn_layers = [l for l in model.snn.layers if isinstance(l, SNNAttentionDecoderLayer)]
        orig = [l.forward_parallel for l in attn_layers]
        for l in attn_layers: l.forward_parallel = types.MethodType(_attn_forward_no_xpos, l)
    try:
        accs = {}
        for qj in js:
            dist = 2 * (args.n_pairs - qj)  # 答案距查询的 token 数
            correct = total = 0
            while total < args.eval_samples:
                bs = min(args.batch, args.eval_samples - total)
                ids, ap_, ans = make_batch(bs, query_j=qj)
                out = _fwd(model, ids)
                pred = out.logits[torch.arange(bs, device=DEV), ap_].argmax(-1)
                correct += (pred == ans).sum().item(); total += bs
            accs[dist] = correct / total
    finally:
        if no_attn and is_v4:
            for l, fp in zip(attn_layers, orig): l.forward_parallel = fp
    model.train()
    print(f"  [{label}] recall acc by dist: " + "  ".join(f"d{d}={accs[d]:.3f}" for d in sorted(accs)), flush=True)
    return accs


def main():
    LOGF = open(args.out, "a") if args.out else None
    def log(s):
        print(s, flush=True)
        if LOGF: LOGF.write(s + "\n"); LOGF.flush()

    model, n_attn = build_model()
    np_ = sum(p.numel() for p in model.parameters()) / 1e6
    log(f"MQAR recall probe v2: N_pairs={args.n_pairs} n_keys={args.n_keys} n_vals={args.n_vals} vocab={VOCAB} T={T} | "
        f"model={args.model}{('('+args.spike_mode+', '+str(args.layers)+'L, '+str(n_attn)+' SNNAttn)') if args.model=='v4' else ('('+str(args.layers)+'L, '+str(args.n_heads)+'h)')} {np_:.1f}M")
    if args.model == "v4" and not args.no_lsuv:
        from utils.lsuv_snn_init import lsuv_snn_init
        _ids0, _, _ = make_batch(2)
        lsuv_snn_init(model, _ids0[:, :min(T, 256)], target_p_fire=0.3, n_passes=3, verbose=False)
        log("  LSUV v2 init applied")
    # optimizer
    if args.model == "v4":
        from utils.muon_adam_lion import SingleDeviceMoonshotMuonAdamLion, build_muon_adam_lion_param_groups
        opt = SingleDeviceMoonshotMuonAdamLion(build_muon_adam_lion_param_groups(model, muon_lr=args.muon_lr, adam_base_lr=args.lr, lion_lr=args.lion_lr, neuron_lr_mult=1.0))
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    model.train()
    t0 = time.time(); best_acc = 0.0
    for step in range(args.train_steps):
        ids, ap_, ans = make_batch(args.batch)
        opt.zero_grad()
        out = _fwd(model, ids)
        logits_q = out.logits[torch.arange(args.batch, device=DEV), ap_].float()
        loss = F.cross_entropy(logits_q, ans)
        if not torch.isfinite(loss):
            log(f"  NaN at step {step}!"); break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 200 == 0 or step == args.train_steps - 1:
            with torch.no_grad():
                acc = (logits_q.argmax(-1) == ans).float().mean().item()
            best_acc = max(best_acc, acc)
            log(f"  step {step:5d}: loss={float(loss):.4f}  train_recall_acc={acc:.3f}  best={best_acc:.3f}  ({(time.time()-t0)/(step+1)*1000:.0f} ms/step)")
    log(f"=== EVAL (recall acc by distance, {args.eval_samples} samples each) ===")
    a_full = eval_recall(model, "full" + ("/" + args.model), no_attn=False)
    if args.model == "v4":
        a_no = eval_recall(model, "no-SNNAttn", no_attn=True)
        log("\nRESULT recall-acc vs distance:")
        log(f"  {'dist':>6s} | {'full':>8s} | {'no-SNNAttn':>11s} | gap (=SNNAttn 独有)")
        for d in sorted(a_full):
            log(f"  {d:6d} | {a_full[d]:8.3f} | {a_no.get(d, float('nan')):11.3f} | {a_full[d]-a_no.get(d, 0):+.3f}")
    else:
        log("\nRESULT recall-acc vs distance (transformer baseline):")
        for d in sorted(a_full):
            log(f"  dist {d:6d}: {a_full[d]:.3f}")
    log("DONE")


if __name__ == "__main__":
    main()
