"""Track B / B1 — Induction Heads (Mamba §4.1.2 招牌合成任务 + 长度外推).

任务 (严格照 HazyResearch/safari `src/dataloaders/synthetics.py::generate_induction_head`):
  序列 = [input_seq_len 个随机 token, COPY_PREFIX] —— 然后在前面随机 num_triggers 个位置插入
  COPY_PREFIX, 紧跟 induction_len 个 "to_copy" token (= 第一个 trigger 后面那几个原始随机 token);
  最后把 to_copy 再拼到序列尾部. 模型当作**标准自回归 LM** 训 —— 在第一个 COPY_PREFIX 之后的每个
  位置上算 next-token CE loss (前面随机段 mask 掉 -100, 学不了也不该学). "答案" = 序列尾部那
  induction_len 个 to_copy token (跟在最后一个 COPY_PREFIX 后面, 模型必须靠 induction head 召回).

长度外推: 训短序列 (train_len ~64-256) → eval 时把 input_seq_len 拉到远超训练长度, 报答案
  token 准确率 vs eval seq_len. Mamba / SSM / 线性注意力外推不掉点; softmax-attention 掉点.

模型: v4 (NeuronSpark, 含 SNNAttention) / v4 --no_xpos (关 SNNAttention 跨位置混合, 消融) /
      transformer (RoPE baseline, 用来 sanity-check 配方 —— 它在 in-dist 上必须学到 ~100%).

参考超参 (safari induction_head config): vocab~16-20, 小模型 (d~32-64, 2-4 层), AdamW lr=5e-4
  wd=0.1 + cosine warmup, batch 32, 训到收敛 (~40k+ 步, 见 Mamba issue #62). v4 用自己的 MAL.

用法: CUDA_VISIBLE_DEVICES=N python scripts/v4_induction_heads.py --model v4 --train_len 256 \
        --steps 40000 [--vocab 16 --induction_len 1 --batch 32 --no_xpos] [--out log.txt]
"""
import sys, os, math, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="v4", choices=("v4", "transformer"))
ap.add_argument("--train_len", type=int, default=64, help="训练时 input_seq_len (随机段长度; 总序列 = +1 COPY_PREFIX +induction_len 尾巴)")
ap.add_argument("--steps", type=int, default=40000)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--vocab", type=int, default=16, help="普通 token 数; COPY_PREFIX = vocab (embedding 大小 = vocab+1)")
ap.add_argument("--induction_len", type=int, default=1, help="每个 trigger 后要复制的 token 数 (=尾部答案长度)")
ap.add_argument("--num_triggers", type=int, default=5, help="序列中插入 COPY_PREFIX 的次数 (越多→可学的 completion 越多, 信号越密)")
ap.add_argument("--loss_on", default="completions", choices=("completions", "all"),
                help="completions=只在紧跟 COPY_PREFIX 的位置算 loss (信号密度与序列长度脱钩, 推荐); all=safari 风格整序列 next-token LM (长序列时可学信号被随机段淹没)")
ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--wd", type=float, default=0.1)
ap.add_argument("--warmup_frac", type=float, default=0.05)
ap.add_argument("--lr_min_frac", type=float, default=0.1, help="cosine 衰减的下限 (× base lr)")
ap.add_argument("--clip", type=float, default=1.0)
ap.add_argument("--log_every", type=int, default=1000)
ap.add_argument("--seed", type=int, default=0)
# v4 config (小模型)
ap.add_argument("--D", type=int, default=64)
ap.add_argument("--N", type=int, default=8)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--num_layers", type=int, default=4)
ap.add_argument("--D_ff", type=int, default=128)
ap.add_argument("--memory_layer_interval", type=int, default=2)
ap.add_argument("--D_key", type=int, default=16)
ap.add_argument("--D_value", type=int, default=16)
ap.add_argument("--no_xpos", action="store_true", help="(v4) 关掉 SNNAttention 跨位置 cumsum 混合 → 消融")
ap.add_argument("--muon_lr", type=float, default=0.005)
ap.add_argument("--adam_lr", type=float, default=2e-4)
ap.add_argument("--lion_lr", type=float, default=1e-4)
# transformer baseline config (小, 同 safari 量级但用 RoPE 以支持外推对照)
ap.add_argument("--t_d", type=int, default=64)
ap.add_argument("--t_layers", type=int, default=2)
ap.add_argument("--t_heads", type=int, default=2)
ap.add_argument("--eval_lens", default="64,128,256,512,1024,2048,4096,8192,16384")
ap.add_argument("--eval_examples", type=int, default=512)
ap.add_argument("--out", default=None)
args = ap.parse_args()

DEV = "cuda" if torch.cuda.is_available() else "cpu"
COPY_PREFIX = args.vocab           # special token id
EMB_SIZE = args.vocab + 1          # embedding / output dim
IGN = -100
LOGF = open(args.out, "a") if args.out else None
def log(s):
    print(s, flush=True)
    if LOGF: LOGF.write(s + "\n"); LOGF.flush()


# ---------------- data: induction-heads (基于 safari generate_induction_head; 多 trigger + completion-only loss) ----------------
_torch_gen = torch.Generator(device=DEV)

def _build_labels(ids_t, first_cp, comp_cols_list, T):
    """labels = ids 的副本, 但:
       loss_on=='all'         → 把首个 COPY_PREFIX 之前(含)的位置设 IGN (safari 风格);
       loss_on=='completions' → 只保留 "紧跟 COPY_PREFIX 的 induction_len 个位置 + 尾部 to_copy" 的 label, 其余全 IGN."""
    labels = ids_t.clone()
    if args.loss_on == "all":
        pos = torch.arange(T, device=DEV)[None, :]
        labels[pos <= first_cp[:, None]] = IGN
        return labels
    mask = torch.zeros_like(labels, dtype=torch.bool)
    for b, cols in enumerate(comp_cols_list):
        mask[b, cols] = True
    labels[~mask] = IGN
    return labels

def make_batch(B, input_seq_len, np_rng):
    """返回 (ids[B,T], labels[B,T], T); T = input_seq_len + 1 + induction_len.
    序列 = [input_seq_len 随机 token, COPY_PREFIX] → 在前面 num_triggers 个(spacing 过滤后)位置插入
    COPY_PREFIX+to_copy → 再把 to_copy 拼到尾部. completion 位置 = 每个插入 trigger 后的 il 个 token
    + 尾部 to_copy 的 il 个 token (= 序列最后 il 列). 答案 = 尾部 to_copy."""
    il = args.induction_len
    T = input_seq_len + 1 + il
    tail_cols = list(range(T - il, T))
    ids = np.empty((B, T), dtype=np.int64)
    first_cp = np.empty(B, dtype=np.int64)
    comp_cols_list = []
    for b in range(B):
        s = np_rng.integers(0, args.vocab, size=input_seq_len).tolist()
        s.append(COPY_PREFIX)                                          # pos input_seq_len = COPY_PREFIX (start_seq)
        raw = np.sort(np_rng.integers(input_seq_len - (1 + il), size=max(args.num_triggers, 1)))
        pf = []
        for i, q in enumerate(raw):
            if i == 0 or q - pf[-1] > il:                              # spacing 过滤, 同 safari
                pf.append(int(q))
        tc = [s[pf[0] + 1 + i] for i in range(il)]                     # to_copy = 第一个 trigger 后的 il 个原始 token
        for q in pf:
            s[q] = COPY_PREFIX
            for i in range(il):
                s[q + 1 + i] = tc[i]
        ids[b] = s + tc
        first_cp[b] = pf[0]
        cc = []
        for q in pf:
            cc += list(range(q + 1, q + 1 + il))
        cc += tail_cols
        comp_cols_list.append(cc)
    ids_t = torch.from_numpy(ids).to(DEV)
    labels = _build_labels(ids_t, torch.from_numpy(first_cp).to(DEV), comp_cols_list, T)
    return ids_t, labels, T


def lm_loss(logits, labels):
    """next-token CE: logits[:, :-1] 预测 labels[:, 1:]. labels 里 IGN 的位置不算."""
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1), ignore_index=IGN)


def answer_acc(logits, ids, T):
    """尾部 induction_len 个 token 的预测准确率 (= induction-head 召回是否对). 位置 [T-il, T-1] 由 logits[T-il-1, T-2] 预测."""
    il = args.induction_len
    pred = logits[:, T - il - 1: T - 1, :].argmax(-1)  # (B, il)
    tgt = ids[:, T - il:]                              # (B, il)
    return (pred == tgt).float().mean().item()


# ---------------- models ----------------
def build_v4():
    from neuronspark import NeuronSparkConfig, NeuronSparkForCausalLM
    cfg = NeuronSparkConfig(vocab_size=EMB_SIZE, D=args.D, N=args.N, K=args.K, num_layers=args.num_layers,
                            D_ff=args.D_ff, memory_layer_interval=args.memory_layer_interval,
                            D_key=args.D_key, D_value=args.D_value, spike_mode="quantal", use_ahp=False)
    m = NeuronSparkForCausalLM(cfg).to(DEV)
    for n, p in m.named_parameters():
        p.data = p.data.to(torch.bfloat16)  # 全 bf16 (含逐通道神经元参数 —— canonical 配置, MAL+SR 训得动)
    if args.no_xpos:
        from neuronspark.modeling_neuronspark import SNNAttentionDecoderLayer
        _orig = SNNAttentionDecoderLayer.forward_parallel
        def _no_xpos_fwd(self, h):
            real_cumsum = torch.cumsum
            torch.cumsum = lambda x, dim=0: x  # noqa  (只影响本次 forward 里 M_all = cumsum(kv_gated))
            try:
                self.M_state = 0.
                out = _orig(self, h)
            finally:
                torch.cumsum = real_cumsum
                self.M_state = 0.  # 永不续传
            return out
        for layer in m.snn.layers:
            if isinstance(layer, SNNAttentionDecoderLayer):
                layer.forward_parallel = _no_xpos_fwd.__get__(layer, SNNAttentionDecoderLayer)
        log("[v4] SNNAttention cross-position mixing DISABLED (no_xpos ablation)")
    return m


class RoPEAttn(nn.Module):
    def __init__(self, d, h):
        super().__init__(); self.h = h; self.dh = d // h
        self.qkv = nn.Linear(d, 3 * d, bias=False); self.o = nn.Linear(d, d, bias=False)
        inv = 1.0 / (10000.0 ** (torch.arange(0, self.dh, 2).float() / self.dh))
        self.register_buffer("inv", inv, persistent=False)
    def rope(self, x):  # x: (B,H,T,dh)
        T = x.shape[-2]
        pos = torch.arange(T, device=x.device, dtype=torch.float32)
        ang = torch.outer(pos, self.inv)
        cos = ang.cos().repeat_interleave(2, -1)[None, None]; sin = ang.sin().repeat_interleave(2, -1)[None, None]
        x1 = x[..., ::2]; x2 = x[..., 1::2]
        rot = torch.stack([-x2, x1], -1).flatten(-2)
        return x * cos + rot * sin
    def forward(self, x):
        B, T, d = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        q = q.view(B, T, self.h, self.dh).transpose(1, 2); k = k.view(B, T, self.h, self.dh).transpose(1, 2); v = v.view(B, T, self.h, self.dh).transpose(1, 2)
        q = self.rope(q); k = self.rope(k)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(o.transpose(1, 2).reshape(B, T, d))


class TinyGPT(nn.Module):
    def __init__(self, vocab, d, layers, heads):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(nn.ModuleDict(dict(
                n1=nn.LayerNorm(d), attn=RoPEAttn(d, heads),
                n2=nn.LayerNorm(d), mlp=nn.Sequential(nn.Linear(d, 4 * d, bias=False), nn.GELU(), nn.Linear(4 * d, d, bias=False)))))
        self.nf = nn.LayerNorm(d); self.head = nn.Linear(d, vocab, bias=False)
    def forward(self, ids):
        x = self.emb(ids)
        for b in self.blocks:
            x = x + b["attn"](b["n1"](x)); x = x + b["mlp"](b["n2"](x))
        return self.head(self.nf(x))


def fwd_logits(model, ids):
    if args.model == "v4":
        from neuronspark.modeling_neuronspark import functional
        functional.reset_net(model.snn)
        with torch.amp.autocast(DEV, dtype=torch.bfloat16):
            out = model(input_ids=ids)
        return out.logits.float()
    return model(ids).float()


# ---------------- train ----------------
torch.manual_seed(args.seed)
model = build_v4() if args.model == "v4" else TinyGPT(EMB_SIZE, args.t_d, args.t_layers, args.t_heads).to(DEV)
n_p = sum(p.numel() for p in model.parameters()) / 1e6
log(f"=== model={args.model} params={n_p:.3f}M | train_len={args.train_len} il={args.induction_len} nt={args.num_triggers} "
    f"| steps={args.steps} batch={args.batch} vocab={args.vocab}(+1) | seed={args.seed} ===")

if args.model == "v4":
    from utils.muon_adam_lion import SingleDeviceMoonshotMuonAdamLion, build_muon_adam_lion_param_groups
    pg = build_muon_adam_lion_param_groups(model, muon_lr=args.muon_lr, adam_base_lr=args.adam_lr, adam_embed_lr=args.adam_lr,
                                           lion_lr=args.lion_lr, neuron_lr_mult=1.0, weight_decay_muon=0.01)
    opt = SingleDeviceMoonshotMuonAdamLion(pg)
    log(f"[v4] optimizer = SingleDeviceMoonshotMuonAdamLion (muon={args.muon_lr} adam={args.adam_lr} lion={args.lion_lr})")
else:
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    log(f"[transformer] optimizer = AdamW (lr={args.lr} wd={args.wd})")

base_lrs = [g["lr"] for g in opt.param_groups]
warmup = max(1, int(args.steps * args.warmup_frac))
def lr_scale(step):  # linear warmup → cosine decay to lr_min_frac×
    if step < warmup:
        return step / warmup
    prog = (step - warmup) / max(1, args.steps - warmup)
    f = args.lr_min_frac
    return f + (1 - f) * 0.5 * (1 + math.cos(math.pi * prog))

np_rng = np.random.default_rng(123); _torch_gen.manual_seed(123)
model.train()
for step in range(args.steps):
    sc = lr_scale(step)
    for g, blr in zip(opt.param_groups, base_lrs):
        g["lr"] = blr * sc
    ids, labels, T = make_batch(args.batch, args.train_len, np_rng)
    logits = fwd_logits(model, ids)
    loss = lm_loss(logits, labels)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip); opt.step()
    if step % args.log_every == 0 or step == args.steps - 1:
        with torch.no_grad():
            aacc = answer_acc(logits, ids, T)
        log(f"  step {step:6d}: lr×{sc:.3f} loss={loss.item():.4f}  answer_acc={aacc:.3f}")

# ---------------- eval: length extrapolation ----------------
model.eval()
np_eval = np.random.default_rng(999); _torch_gen.manual_seed(999)
log(f"=== extrapolation: answer token accuracy vs eval input_seq_len (train_len={args.train_len}) ===")
for L in [int(x) for x in args.eval_lens.split(",")]:
    if L < args.induction_len + 2:
        continue
    B = args.batch if L <= 2048 else max(2, args.batch // 8)
    n_corr = n_tot = 0
    n_iter = max(1, args.eval_examples // B)
    with torch.no_grad():
        for _ in range(n_iter):
            ids, labels, T = make_batch(B, L, np_eval)
            logits = fwd_logits(model, ids)
            il = args.induction_len
            pred = logits[:, T - il - 1: T - 1, :].argmax(-1)
            tgt = ids[:, T - il:]
            n_corr += (pred == tgt).all(dim=1).sum().item(); n_tot += B
    tag = "IN-DIST" if L <= args.train_len else "EXTRAPOL"
    log(f"  input_seq_len={L:7d}: exact-answer acc={n_corr/n_tot:.3f}  ({n_corr}/{n_tot})  [{tag}]")
log("=== DONE ===")
