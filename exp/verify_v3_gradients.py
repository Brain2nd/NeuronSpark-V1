"""验证 v3 Deep Copy 40 层梯度范数均匀性、专家梯度健康、MPD-AGL alpha。"""

import sys
sys.path.insert(0, '.')

import torch
from model_v3 import NeuronSparkV3ForCausalLM, NeuronSparkV3Config

# 小规模配置
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

print(f"设备: {device}, 配置: D={config.hidden_size}, N={config.ssm_N}, K={config.ssm_K}")

model = NeuronSparkV3ForCausalLM(config).to(device, dtype)

# Forward + backward
token_ids = torch.randint(1, config.vocab_size, (batch, seq_len), device=device)
target_ids = torch.randint(1, config.vocab_size, (batch, seq_len), device=device)

with torch.amp.autocast(device, dtype=dtype):
    out = model(token_ids, target_ids)
    loss = out.last_loss.mean()
    if out.ponder_cost is not None:
        loss = loss + 0.01 * out.ponder_cost
    if out.mtp_loss is not None:
        loss = loss + 0.3 * out.mtp_loss

loss.backward()

# ====== 1. 层级梯度范数 ======
print("\n" + "=" * 60)
print("层级梯度范数 (应大致均匀，不超过 10× 差异)")
print("=" * 60)

layer_grad_norms = []
for block in model.backbone.layers:
    grad_norm = 0.0
    n_params = 0
    for p in block.parameters():
        if p.grad is not None:
            grad_norm += p.grad.norm().item() ** 2
            n_params += 1
    grad_norm = grad_norm ** 0.5
    layer_grad_norms.append(grad_norm)
    print(f"  Layer {block.layer_idx:2d} [{block.block_type:9s}]: "
          f"grad_norm = {grad_norm:.6f} ({n_params} params)")

# 统计
norms = torch.tensor(layer_grad_norms)
valid = norms[norms > 0]
if len(valid) > 0:
    ratio = valid.max() / (valid.min() + 1e-10)
    print(f"\n  范围: [{valid.min():.6f}, {valid.max():.6f}], 比值: {ratio:.1f}x")
    if ratio < 10:
        print("  ✓ 梯度范数差异 < 10×，均匀性良好")
    else:
        print("  ✗ 梯度范数差异 >= 10×，可能存在梯度不均匀")
else:
    print("  ✗ 无有效梯度!")

# ====== 2. BioSSM MPD-AGL alpha ======
print("\n" + "=" * 60)
print("BioSSM MPD-AGL 自适应 surrogate alpha")
print("=" * 60)

for block in model.backbone.layers:
    if block.block_type == 'ssm':
        ssm = block.mixer.bio_ssm
        alpha_in = ssm.input_neuron.surrogate_function.alpha
        alpha_hidden = ssm.snn_block.hidden_neuron.surrogate_function.alpha
        print(f"  Layer {block.layer_idx:2d}: input_alpha={alpha_in:.2f}, "
              f"hidden_alpha={alpha_hidden:.2f}")

# ====== 3. MoE 专家梯度 ======
print("\n" + "=" * 60)
print("MoE 专家梯度健康检查")
print("=" * 60)

for block in model.backbone.layers:
    if block.block_type == 'moe':
        expert_grads = []
        for j, expert in enumerate(block.mixer.experts):
            grad_norm = 0.0
            has_grad = False
            for p in expert.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.norm().item() ** 2
                    has_grad = True
            expert_grads.append(grad_norm ** 0.5 if has_grad else 0.0)

        active = sum(1 for g in expert_grads if g > 0)
        if expert_grads:
            eg = torch.tensor(expert_grads)
            print(f"  Layer {block.layer_idx:2d}: {active}/{len(expert_grads)} 专家有梯度, "
                  f"范围 [{eg.min():.6f}, {eg.max():.6f}]")

# ====== 4. MTP Head 梯度 ======
print("\n" + "=" * 60)
print("MTP Head 梯度")
print("=" * 60)

mtp_grad_norm = 0.0
for p in model.mtp_head.parameters():
    if p.grad is not None:
        mtp_grad_norm += p.grad.norm().item() ** 2
mtp_grad_norm = mtp_grad_norm ** 0.5
print(f"  MTP Head grad_norm: {mtp_grad_norm:.6f}")
print(f"  MTP loss: {out.mtp_loss.item():.3f}")

# ====== 5. Embedding 梯度 ======
print("\n" + "=" * 60)
print("Embedding + Output 梯度")
print("=" * 60)
if model.backbone.embeddings.weight.grad is not None:
    print(f"  embeddings grad_norm: "
          f"{model.backbone.embeddings.weight.grad.norm().item():.6f}")
if model.lm_head.weight.grad is not None:
    print(f"  lm_head grad_norm: {model.lm_head.weight.grad.norm().item():.6f}")

# ====== 6. Router 权重检查 ======
print("\n" + "=" * 60)
print("SparkTopkRouter 权重 (float32)")
print("=" * 60)

for block in model.backbone.layers:
    if block.block_type == 'moe':
        w = block.mixer.gate.weight
        print(f"  Layer {block.layer_idx:2d}: dtype={w.dtype}, "
              f"norm={w.norm().item():.4f}, "
              f"grad_norm={w.grad.norm().item():.6f}" if w.grad is not None
              else f"  Layer {block.layer_idx:2d}: dtype={w.dtype}, no grad")
        break  # 只看第一个 MoE 层

print("\n" + "=" * 60)
print("梯度验证完成!")
print("=" * 60)
