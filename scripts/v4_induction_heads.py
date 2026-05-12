"""Track B / B1 — Induction Heads (Mamba §4.1 招牌合成任务).

任务: 序列里某个早期位置 p 放一个 SPECIAL token, 紧跟一个随机 X token; 序列末位再放 SPECIAL;
要求模型在末位预测 X (= "上次 SPECIAL 后面跟的那个 token"). 这是关联召回 + **长度外推** 的金标准:
训短序列 (~256) → 测外推到远超训练长度的序列, Mamba/SSM/线性注意力外推不掉点, softmax-attention 掉.

测: V4.1 (含 SNNAttention) vs (可选) V4.1-no-SNNAttention-xpos 消融 vs (可选) 极简 RoPE-Transformer baseline.
报: answer accuracy (argmax(logits[:,-1,:]) == X) vs eval seq_len.

用法: CUDA_VISIBLE_DEVICES=N python scripts/v4_induction_heads.py --model v4 --train_len 256 --steps 5000 \
        [--D 256 --N 8 --K 8 --num_layers 6 --D_ff 512 --vocab 64 --batch 32 --no_xpos] [--out log.txt]
"""
import sys, os, time, argparse, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="v4", choices=("v4", "transformer"))
ap.add_argument("--train_len", type=int, default=256)
ap.add_argument("--steps", type=int, default=5000)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--vocab", type=int, default=64, help="token 词表 (含 1 个 SPECIAL = vocab-1)")
ap.add_argument("--lr", type=float, default=3e-4)
# v4 config
ap.add_argument("--D", type=int, default=256)
ap.add_argument("--N", type=int, default=8)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--num_layers", type=int, default=6)
ap.add_argument("--D_ff", type=int, default=512)
ap.add_argument("--memory_layer_interval", type=int, default=2)
ap.add_argument("--D_key", type=int, default=32)
ap.add_argument("--D_value", type=int, default=32)
ap.add_argument("--no_xpos", action="store_true", help="(v4) 关掉 SNNAttention 的跨位置 cumsum 混合 → 消融")
# transformer baseline config
ap.add_argument("--t_d", type=int, default=256)
ap.add_argument("--t_layers", type=int, default=4)
ap.add_argument("--t_heads", type=int, default=4)
ap.add_argument("--eval_lens", default="64,128,256,512,1024,2048,4096,8192,16384")
ap.add_argument("--eval_batch", type=int, default=16)
ap.add_argument("--out", default=None)
args = ap.parse_args()

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SPECIAL = args.vocab - 1
LOGF = open(args.out, "a") if args.out else None
def log(s):
    print(s, flush=True)
    if LOGF: LOGF.write(s + "\n"); LOGF.flush()


def make_batch(B, L, gen):
    """[t0..., SPECIAL@p, X@p+1, ..., SPECIAL@(L-1)]; target at pos L-1 (next-token) = X.
    其余 token ∈ [0, vocab-2); SPECIAL 只出现在 p 和 L-1; X ∈ [0, vocab-2)."""
    seq = torch.randint(0, args.vocab - 1, (B, L), generator=gen, device=DEV)
    p = torch.randint(1, L - 2, (B,), generator=gen, device=DEV)            # needle 位置 (留 X 的位置 + 不撞末位)
    X = torch.randint(0, args.vocab - 1, (B,), generator=gen, device=DEV)   # 答案 token
    ar = torch.arange(B, device=DEV)
    seq[ar, p] = SPECIAL
    seq[ar, p + 1] = X
    seq[:, L - 1] = SPECIAL
    return seq, X


# ---------------- models ----------------
def build_v4():
    from neuronspark import NeuronSparkConfig, NeuronSparkForCausalLM
    cfg = NeuronSparkConfig(vocab_size=args.vocab, D=args.D, N=args.N, K=args.K, num_layers=args.num_layers,
                            D_ff=args.D_ff, memory_layer_interval=args.memory_layer_interval,
                            D_key=args.D_key, D_value=args.D_value, spike_mode="quantal", use_ahp=False)
    m = NeuronSparkForCausalLM(cfg).to(DEV)
    for n, p in m.named_parameters():
        p.data = p.data.to(torch.bfloat16)  # 全 bf16 (含神经元参数 —— canonical 配置, MAL+SR 训得动)
    if args.no_xpos:
        # 关掉 SNNAttention 跨位置混合: M_all 不做 cumsum (每位置独立), 不携带 M_state.
        from neuronspark.modeling_neuronspark import SNNAttentionDecoderLayer
        _orig = SNNAttentionDecoderLayer.forward_parallel
        def _no_xpos_fwd(self, h):
            # 临时 monkeypatch torch.cumsum 让它在本次调用里变成恒等 (只影响 M_all = cumsum(kv_gated))
            real_cumsum = torch.cumsum
            torch.cumsum = lambda x, dim=0: x  # noqa
            try:
                # 也把 M_state 续传关掉: 临时设成 0.
                saved = self.M_state; self.M_state = 0.
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
        pos = torch.arange(T, device=x.device).float()
        ang = torch.outer(pos, self.inv)  # (T, dh/2)
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
    else:
        return model(ids).float()


# ---------------- train ----------------
torch.manual_seed(0)
model = build_v4() if args.model == "v4" else TinyGPT(args.vocab, args.t_d, args.t_layers, args.t_heads).to(DEV)
n_p = sum(p.numel() for p in model.parameters()) / 1e6
log(f"=== model={args.model} params={n_p:.2f}M train_len={args.train_len} steps={args.steps} vocab={args.vocab} ===")
if args.model == "v4":
    # canonical 配置: MAL (Muon matrices + Adam embed/norm + Lion 逐通道神经元参数 .w/.v_th/.ahp, bf16+SR)
    from utils.muon_adam_lion import SingleDeviceMoonshotMuonAdamLion, build_muon_adam_lion_param_groups
    pg = build_muon_adam_lion_param_groups(model, muon_lr=0.005, adam_base_lr=2e-4, adam_embed_lr=2e-4,
                                           lion_lr=1e-4, neuron_lr_mult=1.0, weight_decay_muon=0.01)
    opt = SingleDeviceMoonshotMuonAdamLion(pg)
    log("[v4] optimizer = SingleDeviceMoonshotMuonAdamLion (muon_lr=0.005 adam=2e-4 lion=1e-4)")
else:
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
gen = torch.Generator(device=DEV).manual_seed(123)
model.train()
for step in range(args.steps):
    ids, X = make_batch(args.batch, args.train_len, gen)
    logits = fwd_logits(model, ids)
    loss = F.cross_entropy(logits[:, -1, :], X)   # 只在答案位置 (末位的 next-token) 上训
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    if step % 200 == 0 or step == args.steps - 1:
        with torch.no_grad():
            acc = (logits[:, -1, :].argmax(-1) == X).float().mean().item()
        log(f"  step {step:5d}: loss={loss.item():.4f}  train_acc={acc:.3f}")

# ---------------- eval extrapolation ----------------
model.eval()
geval = torch.Generator(device=DEV).manual_seed(999)
log("=== extrapolation: answer accuracy vs eval seq_len (train_len={}) ===".format(args.train_len))
for L in [int(x) for x in args.eval_lens.split(",")]:
    B = args.eval_batch if L <= 4096 else max(4, args.eval_batch // 4)
    n_correct = n_total = 0
    n_iter = max(1, 256 // B)
    with torch.no_grad():
        for _ in range(n_iter):
            ids, X = make_batch(B, L, geval)
            logits = fwd_logits(model, ids)
            n_correct += (logits[:, -1, :].argmax(-1) == X).sum().item(); n_total += B
    log(f"  seq_len={L:6d}: acc={n_correct/n_total:.3f}  ({n_correct}/{n_total})  [{'IN-DIST' if L<=args.train_len else 'EXTRAPOL'}]")
log("=== DONE ===")
