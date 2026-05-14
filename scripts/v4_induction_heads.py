"""Track B / B1 — Induction Heads (Mamba §4.1.2 招牌合成任务 + 长度泛化).

术语: 本脚本里的「长度泛化」= 训短序列(train_len)·zero-shot 测远长序列·不改模型, 是一个现象;
与之区分的「外推 / context-length extension」(NoPE/ALiBi/PI/YaRN 那类手段) 本脚本不涉及.

任务 (严格照 HazyResearch/safari `src/dataloaders/synthetics.py::generate_induction_head`):
  序列 = [input_seq_len 个随机 token, COPY_PREFIX] —— 然后在前面随机 num_triggers 个位置插入
  COPY_PREFIX, 紧跟 induction_len 个 "to_copy" token (= 第一个 trigger 后面那几个原始随机 token);
  最后把 to_copy 再拼到序列尾部. 模型当作**标准自回归 LM** 训 —— 在第一个 COPY_PREFIX 之后的每个
  位置上算 next-token CE loss (前面随机段 mask 掉 -100, 学不了也不该学). "答案" = 序列尾部那
  induction_len 个 to_copy token (跟在最后一个 COPY_PREFIX 后面, 模型必须靠 induction head 召回).

长度泛化: 训短序列 (train_len ~64-256) → eval 时把 input_seq_len 拉到远超训练长度, 报答案
  token 准确率 vs eval seq_len. Mamba / SSM / 线性注意力 zero-shot 长度泛化不掉点; softmax-attention 掉点.

模型: v4 (NeuronSpark, 含 SNNAttention) / v4 --no_xpos (关 SNNAttention 跨位置混合 = 纯 SNNBlock SSM, 消融) /
      transformer (RoPE baseline, 原生 RoPE attention 长度泛化会塌) / mamba (selective SSM baseline, 任务原产地模型).
      四个 in-dist 都该到 ~100%; 看点在长度泛化曲线 (transformer 塌 / mamba / v4 / v4-noxpos).

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
ap.add_argument("--model", default="v4", choices=("v4", "transformer", "mamba"))
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
# transformer baseline config (小, 同 safari 量级但用 RoPE; 原生 RoPE attention 长度泛化会塌, 作对照)
ap.add_argument("--t_d", type=int, default=64)
ap.add_argument("--t_layers", type=int, default=2)
ap.add_argument("--t_heads", type=int, default=2)
# mamba baseline config (selective SSM, induction-heads 任务的"原产地"模型 —— 长度泛化应不掉点)
ap.add_argument("--m_d", type=int, default=128)
ap.add_argument("--m_layers", type=int, default=4)
ap.add_argument("--eval_lens", default="64,128,256,512,1024,2048,4096,8192,16384")
ap.add_argument("--eval_examples", type=int, default=512)
ap.add_argument("--rope_eval", default="none", help="逗号分隔, 对训好的 RoPE 模型在 eval 时叠加上下文增程: none/pi/ntk/yarn (动态 scale=max(1, L_eval/L_train)). mamba 无 PE, 这个参数对它无效")
ap.add_argument("--yarn_beta_fast", type=float, default=32.0)
ap.add_argument("--yarn_beta_slow", type=float, default=1.0)
ap.add_argument("--rope_base", type=float, default=10000.0, help="RoPE 基频 (transformer + v4 SNNAttention 都用); 训练就用这个值")
ap.add_argument("--save_ckpt", default=None, help="训完落盘 model + args 到该路径 (后续 --eval_only --load_ckpt 复用, 不用每次重训)")
ap.add_argument("--load_ckpt", default=None, help="从该路径加载训好的 model 权重+架构 args (会覆盖命令行里的架构 args)")
ap.add_argument("--eval_only", action="store_true", help="跳过训练只跑 eval (需要 --load_ckpt). 用来在同一个训好的 ckpt 上扫不同 rope_eval / eval_lens")
ap.add_argument("--eval_multi_gpu", action="store_true", help="把 v4 各层切到 (visible CUDA 数) 张卡上做管线并行 eval, peak 显存/卡 ≈ 单卡/N. 仅 v4 路径生效, eval-only 用; embed/norm/decode 留 cuda:0, layers 平均分发到所有可见 GPU")
ap.add_argument("--eval_chunk_size", type=int, default=0, help="序列分块 eval (v4 路径): 把长 L 切成 chunk_size 一块顺序 forward, 利用 v4 自带的 pos_offset/M_state/v_carry 续传机制. 单块 forward 内存 = O(chunk_size) 而非 O(L), 用来跑 1M+ 长度避免 OOM. 0 = 关闭, 整 L 一次 forward")
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


# ---------------- RoPE 上下文增程 (Context-Length Extension) ----------------
# 动态 scale: 当前 eval 总长 / 训练总长 (>1 才生效, in-dist 时 scale=1 → 透明).
# - "none": 不动 RoPE (训练时怎么样, eval 还是怎么样).
# - "pi"  : Position Interpolation (Chen et al.) — 全频段除以 scale (等价于 θ_i_eff = θ_i/s).
# - "ntk" : NTK-aware (bloc97) — base 升到 base·s^(d/(d-2)) 把低频拉长、高频几乎不动.
# - "yarn": NTK-by-parts + attention temperature (Peng et al. 2023, YaRN) — 按 wavelength 在
#           PI 和无变化之间线性 ramp, 短波长(高频, 训练 ctx 内多旋转 >β_fast)保持外推, 长波长
#           (低频, <β_slow 圈)做 PI; transformer 上额外把 cos/sin 乘 mscale=0.1·ln(s)+1 (锐化 softmax).
_ACTIVE_ROPE = {"method": "none", "train_total_len": 0, "rope_base": float(args.rope_base)}

def _ntk_base_adjust(base, dim, scale):
    return base * (scale ** (dim / max(1, (dim - 2))))

def _yarn_inv_freq(dim, base, scale, train_total_len, beta_fast, beta_slow, device, dtype):
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim))
    inv_interp = inv / scale
    # 训练 ctx 内的旋转圈数 r_i = train_total_len · θ_i / (2π)
    r = train_total_len * inv / (2 * math.pi)
    # gamma: 1 = 保持原 θ (高频, r>β_fast, 短波长); 0 = 全 PI (低频, r<β_slow, 长波长)
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

def _yarn_mscale(scale):
    return 0.1 * math.log(scale) + 1.0 if scale > 1.0 else 1.0


class RoPEAttn(nn.Module):
    def __init__(self, d, h, base=10000.0):
        super().__init__(); self.h = h; self.dh = d // h; self.base = float(base)
        self.qkv = nn.Linear(d, 3 * d, bias=False); self.o = nn.Linear(d, d, bias=False)
    def rope(self, x):  # x: (B,H,T,dh)
        T = x.shape[-2]
        method = _ACTIVE_ROPE["method"]; train_total_len = _ACTIVE_ROPE["train_total_len"]
        scale = max(1.0, T / float(train_total_len)) if train_total_len > 0 else 1.0
        inv = _effective_inv_freq(self.dh, self.base, method, scale, train_total_len, x.device, torch.float32)
        pos = torch.arange(T, device=x.device, dtype=torch.float32)
        ang = torch.outer(pos, inv)
        cos = ang.cos().repeat_interleave(2, -1); sin = ang.sin().repeat_interleave(2, -1)
        if method == "yarn":
            m = _yarn_mscale(scale); cos = cos * m; sin = sin * m
        cos = cos[None, None].to(x.dtype); sin = sin[None, None].to(x.dtype)
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


def _rebuild_v4_rope_for_eval(model, method, eval_total_len, train_total_len):
    """重建 v4 每个 SNNAttention 层的 rope_cos/rope_sin —— 把指定的 context-length extension 方法
    (PI/NTK/YaRN, 动态 scale=max(1,L_eval/L_train)) 烧进 cos/sin 缓存. method='none' 时还原成原始 RoPE."""
    from neuronspark.modeling_neuronspark import SNNAttentionDecoderLayer
    scale = max(1.0, eval_total_len / float(train_total_len)) if train_total_len > 0 else 1.0
    for layer in model.snn.layers:
        if not isinstance(layer, SNNAttentionDecoderLayer):
            continue
        dim = layer.D_key
        dev = layer.rope_cos.device; ddt = layer.rope_cos.dtype
        inv = _effective_inv_freq(dim, _ACTIVE_ROPE["rope_base"], method, scale, train_total_len, dev, torch.float32)
        max_len = max(int(eval_total_len) + 8, 8192)
        t = torch.arange(max_len, device=dev, dtype=torch.float32)
        ang = torch.outer(t, inv)
        cos = ang.cos().to(ddt); sin = ang.sin().to(ddt)
        # YaRN 的 attention temperature 对线性 attention (k 经过 F.normalize) 大部分被吃掉, 此处不应用 mscale; 只用 freq-ramp.
        layer.register_buffer('rope_cos', cos, persistent=False)
        layer.register_buffer('rope_sin', sin, persistent=False)


class TinyGPT(nn.Module):
    def __init__(self, vocab, d, layers, heads):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(nn.ModuleDict(dict(
                n1=nn.LayerNorm(d), attn=RoPEAttn(d, heads, base=args.rope_base),
                n2=nn.LayerNorm(d), mlp=nn.Sequential(nn.Linear(d, 4 * d, bias=False), nn.GELU(), nn.Linear(4 * d, d, bias=False)))))
        self.nf = nn.LayerNorm(d); self.head = nn.Linear(d, vocab, bias=False)
    def forward(self, ids):
        x = self.emb(ids)
        for b in self.blocks:
            x = x + b["attn"](b["n1"](x)); x = x + b["mlp"](b["n2"](x))
        return self.head(self.nf(x))


def build_mamba():
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    from mamba_ssm.models.config_mamba import MambaConfig
    cfg = MambaConfig(d_model=args.m_d, n_layer=args.m_layers, vocab_size=EMB_SIZE,
                      ssm_cfg={"layer": "Mamba1"},   # 原始选择性 SSM (Mamba 论文 §4.1.2 用的那个)
                      rms_norm=True, fused_add_norm=False, residual_in_fp32=True, pad_vocab_size_multiple=1)
    return MambaLMHeadModel(cfg).to(DEV)


def _setup_v4_multi_gpu(model):
    """把 v4 的 snn.layers 平均分发到所有 visible CUDA, monkeypatch forward_parallel 在层间迁移 residual stream.
    embed_tokens / decode_proj / norm 全留 cuda:0 (embed.weight 末尾 F.linear 还要用 + norm/decode 在 cuda:0).
    rope_cos/rope_sin 等 buffer 随 layer.to() 自动迁移. snn_forward 调用 forward_parallel 是直接调 .forward_parallel(...)
    (绕过 __call__), 所以必须 patch forward_parallel 本身, 不能用 forward_pre_hook."""
    import torch as _t
    n_gpus = _t.cuda.device_count()
    snn = model.snn
    layers = snn.layers
    n_layers = len(layers)
    layer_devs = [f"cuda:{i * n_gpus // n_layers}" for i in range(n_layers)]
    snn.embed_tokens.to("cuda:0")
    snn.norm.to("cuda:0")
    snn.decode_proj.to("cuda:0")
    for layer, dev in zip(layers, layer_devs):
        layer.to(dev)
    def make_wrapped(orig_fp, target_dev, move_h_to_zero):
        target_dev_obj = _t.device(target_dev)
        zero = _t.device("cuda:0")
        def wrapped(h, *a, **kw):
            h = h.to(target_dev_obj, non_blocking=True) if h.device != target_dev_obj else h
            with _t.cuda.device(target_dev_obj):   # Triton kernel 用 current-device context 启动, 跨卡时必须显式切
                out = orig_fp(h, *a, **kw)
            # out 通常是 (h, ponder_cost); ponder_cost 在外层会被聚合 sum → 永远搬到 cuda:0
            if isinstance(out, tuple) and len(out) >= 2:
                h_o, pc = out[0], out[1]
                if move_h_to_zero and isinstance(h_o, _t.Tensor) and h_o.device != zero:
                    h_o = h_o.to(zero, non_blocking=True)
                if isinstance(pc, _t.Tensor) and pc.device != zero:
                    pc = pc.to(zero, non_blocking=True)
                return (h_o, pc) + tuple(out[2:])
            return out
        return wrapped
    for i, (layer, dev) in enumerate(zip(layers, layer_devs)):
        orig_fp = layer.forward_parallel
        layer.forward_parallel = make_wrapped(orig_fp, dev, move_h_to_zero=(i == n_layers - 1))
    log(f"[v4 multi-gpu] {n_layers} layers → {n_gpus} GPUs: {layer_devs}; embed/norm/decode on cuda:0; forward_parallel monkeypatched")


def fwd_logits(model, ids):
    if args.model == "v4":
        from neuronspark.modeling_neuronspark import functional
        functional.reset_net(model.snn)
        if args.eval_multi_gpu:
            ids = ids.to("cuda:0")
        with torch.amp.autocast(DEV, dtype=torch.bfloat16):
            out = model(input_ids=ids)
        logits = out.logits.float()
        if args.eval_multi_gpu and logits.device != torch.device("cuda:0"):
            logits = logits.to("cuda:0")
        return logits
    if args.model == "mamba":
        return model(ids).logits.float()
    return model(ids).float()


# ---------------- ckpt: 加载已训模型 (--load_ckpt) → 用其架构 args 覆盖 ----------------
_loaded_state_dict = None
if args.load_ckpt:
    _ck = torch.load(args.load_ckpt, map_location="cpu", weights_only=False)
    _arch_keys = ('model', 'D', 'N', 'K', 'num_layers', 'D_ff', 'memory_layer_interval', 'D_key', 'D_value',
                  't_d', 't_layers', 't_heads', 'm_d', 'm_layers', 'vocab', 'induction_len', 'no_xpos', 'rope_base',
                  'train_len')   # train_len 影响 train_total_len → rope_eval 动态 scale 基准
    saved_args = _ck.get('args', {})
    for k in _arch_keys:
        if k in saved_args:
            setattr(args, k, saved_args[k])
    COPY_PREFIX = args.vocab; EMB_SIZE = args.vocab + 1
    _ACTIVE_ROPE["rope_base"] = float(args.rope_base)
    _loaded_state_dict = _ck['state_dict']
    log(f"[load_ckpt] loaded {args.load_ckpt}: model={args.model} train_len={args.train_len} vocab={args.vocab}")

# ---------------- train ----------------
torch.manual_seed(args.seed)
if args.model == "v4":
    model = build_v4()
elif args.model == "mamba":
    model = build_mamba()
else:
    model = TinyGPT(EMB_SIZE, args.t_d, args.t_layers, args.t_heads).to(DEV)
if _loaded_state_dict is not None:
    model.load_state_dict(_loaded_state_dict, strict=True)
    log(f"[load_ckpt] state_dict loaded ({sum(p.numel() for p in model.parameters())/1e6:.3f}M params)")
if args.eval_multi_gpu and args.model == "v4" and torch.cuda.device_count() > 1:
    _setup_v4_multi_gpu(model)
n_p = sum(p.numel() for p in model.parameters()) / 1e6
log(f"=== model={args.model} params={n_p:.3f}M | train_len={args.train_len} il={args.induction_len} nt={args.num_triggers} "
    f"| steps={args.steps} batch={args.batch} vocab={args.vocab}(+1) | seed={args.seed} | eval_only={args.eval_only} ===")

if not args.eval_only:
    if args.model == "v4":
        from utils.muon_adam_lion import SingleDeviceMoonshotMuonAdamLion, build_muon_adam_lion_param_groups
        pg = build_muon_adam_lion_param_groups(model, muon_lr=args.muon_lr, adam_base_lr=args.adam_lr, adam_embed_lr=args.adam_lr,
                                               lion_lr=args.lion_lr, neuron_lr_mult=1.0, weight_decay_muon=0.01)
        opt = SingleDeviceMoonshotMuonAdamLion(pg)
        log(f"[v4] optimizer = SingleDeviceMoonshotMuonAdamLion (muon={args.muon_lr} adam={args.adam_lr} lion={args.lion_lr})")
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        log(f"[{args.model}] optimizer = AdamW (lr={args.lr} wd={args.wd})")

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

    if args.save_ckpt:
        torch.save({'state_dict': model.state_dict(), 'args': vars(args)}, args.save_ckpt)
        log(f"[save_ckpt] saved model + args to {args.save_ckpt} ({sum(p.numel()*p.element_size() for p in model.parameters())/1e6:.1f} MB)")

# ---------------- eval: length generalization (zero-shot, 训短测长, 不改模型) ----------------
# 若 --rope_eval 指定了多个方法 (none/pi/ntk/yarn), 对训好的 RoPE 模型在 eval 时叠加上下文增程,
# 每个方法跑一遍完整长度泛化曲线. mamba 无 RoPE → 只跑 "none". v4-no_xpos 的 SNNAttention 跨位
# 混合关了, RoPE 仍存在但只影响每位置自身, rope_eval 仍生效但效果有限.
model.eval()
np_eval_seed = 999
train_total_len = args.train_len + 1 + args.induction_len
_ACTIVE_ROPE["train_total_len"] = train_total_len

rope_methods = [m.strip() for m in args.rope_eval.split(",") if m.strip()]
if args.model == "mamba":
    rope_methods = ["none"]  # mamba 没 PE, 增程方法无意义

def _run_lengen_eval(method_label):
    log(f"=== length-generalization [rope_eval={method_label}]: answer token accuracy vs eval input_seq_len (train_len={args.train_len}) ===")
    np_eval = np.random.default_rng(np_eval_seed); _torch_gen.manual_seed(np_eval_seed)
    for L in [int(x) for x in args.eval_lens.split(",")]:
        if L < args.induction_len + 2:
            continue
        L_total = L + 1 + args.induction_len
        # 应用当前 RoPE 增程方法 to this eval length
        _ACTIVE_ROPE["method"] = method_label    # transformer.RoPEAttn.rope 会读这个
        if args.model == "v4":
            _rebuild_v4_rope_for_eval(model, method_label, L_total, train_total_len)
        if L <= 2048: B = args.batch
        elif L <= 4096: B = max(2, args.batch // 8)
        else: B = max(1, args.batch // 32)  # L > 4096 直接 batch=1 (d 较大 transformer 在 8k+ SDPA 显存就吃不下了)
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
        tag = "IN-DIST" if L <= args.train_len else "LEN-GEN"
        log(f"  input_seq_len={L:7d}: exact-answer acc={n_corr/n_tot:.3f}  ({n_corr}/{n_tot})  [{tag}]")

for m in rope_methods:
    _run_lengen_eval(m)
log("=== DONE ===")
