"""验证 v3 Deep Copy 各组件输入输出张量形状。"""

import sys
sys.path.insert(0, '.')

import torch
from model_v3 import (
    NeuronSparkV3ForCausalLM, NeuronSparkV3Config,
    SparkMLP, SparkMOE, SparkTopkRouter, SparkAttention, BioSSMMixer, SparkBlock,
)

# 小规模测试配置
config = NeuronSparkV3Config(
    vocab_size=256,
    hidden_size=128,
    num_hidden_layers=40,
    ssm_N=4,
    ssm_K=8,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    intermediate_size=256,
    n_routed_experts=8,
    num_experts_per_tok=2,
    moe_intermediate_size=128,
    moe_shared_expert_intermediate_size=256,
    n_mtp_heads=1,
)
batch = 2
seq_len = 16

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.bfloat16 if device == 'cuda' else torch.float32

block_types = config.layers_block_type
n_ssm = sum(1 for t in block_types if t == 'ssm')
n_moe = sum(1 for t in block_types if t == 'moe')
n_attn = sum(1 for t in block_types if t == 'attention')

print(f"设备: {device}, 精度: {dtype}")
print(f"配置: D={config.hidden_size}, N={config.ssm_N}, K={config.ssm_K}")
print(f"残差流: (batch, seq_len, D) = ({batch}, {seq_len}, {config.hidden_size})")
print(f"层分布: {n_ssm} SSM + {n_moe} MoE + {n_attn} Attn\n")

# ====== 1. 组件级测试 ======

print("=" * 50)
print("组件级形状测试 (batch, seq_len, D)")
print("=" * 50)

# SparkMLP
mlp = SparkMLP(config, intermediate_size=config.intermediate_size).to(device, dtype)
x = torch.randn(batch, seq_len, config.hidden_size, device=device, dtype=dtype)
y = mlp(x)
print(f"[SparkMLP]       输入: {tuple(x.shape)} → 输出: {tuple(y.shape)}")
assert y.shape == (batch, seq_len, config.hidden_size)

# SparkTopkRouter
router = SparkTopkRouter(config).to(device)
x = torch.randn(batch, seq_len, config.hidden_size, device=device, dtype=dtype)
idx, weights = router(x)
print(f"[SparkTopkRouter] 输入: {tuple(x.shape)} → indices: {tuple(idx.shape)}, "
      f"weights: {tuple(weights.shape)}")
assert idx.shape == (batch * seq_len, config.num_experts_per_tok)
assert weights.shape == (batch * seq_len, config.num_experts_per_tok)

# SparkMOE
moe = SparkMOE(config).to(device, dtype)
x = torch.randn(batch, seq_len, config.hidden_size, device=device, dtype=dtype)
y = moe(x)
print(f"[SparkMOE]       输入: {tuple(x.shape)} → 输出: {tuple(y.shape)}")
assert y.shape == (batch, seq_len, config.hidden_size)

# SparkAttention (HF 接口返回 3 个值: attn_output, attn_weights, past_key_value)
attn = SparkAttention(config, layer_idx=0).to(device, dtype)
x = torch.randn(batch, seq_len, config.hidden_size, device=device, dtype=dtype)
# 使用 NeuronSparkDynamicCache 作为 KV cache
from model_v3 import NeuronSparkDynamicCache
kv_cache = NeuronSparkDynamicCache(config, batch, dtype=dtype, device=device)
y, _, kv_cache_out = attn(x, past_key_value=kv_cache)
print(f"[SparkAttention]  输入: {tuple(x.shape)} → 输出: {tuple(y.shape)}, "
      f"KV cache: K={tuple(kv_cache_out.key_cache[0].shape)}")
assert y.shape == (batch, seq_len, config.hidden_size)

# SparkAttention with KV cache (incremental decode)
x2 = torch.randn(batch, 1, config.hidden_size, device=device, dtype=dtype)
y2, _, kv_cache_out2 = attn(x2, past_key_value=kv_cache_out)
print(f"[SparkAttention+KV] 输入: {tuple(x2.shape)} → 输出: {tuple(y2.shape)}, "
      f"KV cache: K={tuple(kv_cache_out2.key_cache[0].shape)}")
assert y2.shape == (batch, 1, config.hidden_size)
assert kv_cache_out2.key_cache[0].shape[2] == seq_len + 1

# BioSSMMixer
ssm_mixer = BioSSMMixer(config, layer_idx=0).to(device, dtype)
x = torch.randn(batch, seq_len, config.hidden_size, device=device, dtype=dtype)
ssm_mixer.bio_ssm.input_neuron.v = 0.
ssm_mixer.bio_ssm.snn_block.hidden_neuron.v = 0.
y = ssm_mixer(x)
print(f"[BioSSMMixer]    输入: {tuple(x.shape)} → 输出: {tuple(y.shape)}, "
      f"pc: {ssm_mixer.ponder_cost.item():.3f}")
assert y.shape == (batch, seq_len, config.hidden_size)

# MTP Head
from atomic_ops.mtp_head import MTPHead
mtp = MTPHead(D=config.hidden_size, vocab_size=config.vocab_size, n_heads=1).to(device, dtype)
h_bsd = torch.randn(batch, seq_len, config.hidden_size, device=device, dtype=dtype)
target_ids = torch.randint(1, config.vocab_size, (batch, seq_len), device=device)
embed_w = torch.randn(config.vocab_size, config.hidden_size, device=device, dtype=dtype)
lm_w = torch.randn(config.vocab_size, config.hidden_size, device=device, dtype=dtype)
mtp_loss = mtp(h_bsd, target_ids, lm_head_weight=lm_w, embed_weight=embed_w)
print(f"[MTPHead]        输入: h={tuple(h_bsd.shape)}, targets={tuple(target_ids.shape)} "
      f"→ mtp_loss: {mtp_loss.item():.3f}")

print("\n组件级形状测试: 全部通过\n")

# ====== 2. 完整模型测试 ======

print("=" * 50)
print("完整模型形状测试")
print("=" * 50)

model = NeuronSparkV3ForCausalLM(config).to(device, dtype)

total_params = sum(p.numel() for p in model.parameters())
print(f"模型参数量: {total_params / 1e6:.3f}M")

# Forward (训练模式)
token_ids = torch.randint(1, config.vocab_size, (batch, seq_len), device=device)
target_ids = torch.randint(1, config.vocab_size, (batch, seq_len), device=device)

with torch.amp.autocast(device, dtype=dtype):
    out = model(token_ids, target_ids)

print(f"\n[Forward 训练]")
print(f"  last_loss: {tuple(out.last_loss.shape)}, mean={out.last_loss.mean().item():.3f}")
print(f"  ponder_cost: {out.ponder_cost.item():.3f}")
print(f"  ek_floor_cost: {out.ek_floor_cost.item():.3f}")
print(f"  mtp_loss: {out.mtp_loss.item():.3f}")

# Forward (推理模式)
with torch.amp.autocast(device, dtype=dtype):
    out_infer = model(token_ids)

print(f"\n[Forward 推理]")
print(f"  logits: {tuple(out_infer.logits.shape)}")
assert out_infer.logits.shape == (batch, seq_len, config.vocab_size), \
    f"Logits 形状错误: {out_infer.logits.shape}"

# Generate
with torch.amp.autocast(device, dtype=dtype):
    prompt = token_ids[:, :4]
    generated = model.generate(prompt, max_new_tokens=8, temperature=0.8, top_k=10)

print(f"\n[Generate]")
print(f"  输入: {tuple(prompt.shape)} → 输出: {tuple(generated.shape)}")
assert generated.shape[1] <= 4 + 8, f"Generated 长度错误: {generated.shape}"

print("\n完整模型形状测试: 全部通过")
print("\n" + "=" * 50)
print("所有 v3 形状验证通过!")
print("=" * 50)
