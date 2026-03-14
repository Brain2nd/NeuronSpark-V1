"""参数量和显存估算: NeuronSpark v3 Deep Copy 混合架构。"""

import sys
sys.path.insert(0, '.')

import torch
from model_v3 import NeuronSparkV3ForCausalLM, NeuronSparkV3Config


def count_params(module, name=""):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def format_params(n):
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)


# ====== 目标配置 ======
configs = {
    '研究级 (N=8, 32 experts)': NeuronSparkV3Config(
        hidden_size=1024, ssm_N=8, ssm_K=16, num_hidden_layers=40,
        n_routed_experts=32, num_experts_per_tok=4,
        moe_intermediate_size=1024, moe_shared_expert_intermediate_size=2048,
        intermediate_size=2048,
        num_attention_heads=8, num_key_value_heads=2, head_dim=128,
        n_mtp_heads=1,
    ),
    '紧凑级 (N=4, 16 experts)': NeuronSparkV3Config(
        hidden_size=1024, ssm_N=4, ssm_K=16, num_hidden_layers=40,
        n_routed_experts=16, num_experts_per_tok=4,
        moe_intermediate_size=512, moe_shared_expert_intermediate_size=1024,
        intermediate_size=2048,
        num_attention_heads=8, num_key_value_heads=2, head_dim=128,
        n_mtp_heads=1,
    ),
}

for name, cfg in configs.items():
    print(f"\n{'='*60}")
    print(f"配置: {name}")
    print(f"{'='*60}")

    model = NeuronSparkV3ForCausalLM(cfg)

    # 总参数
    total, trainable = count_params(model)
    print(f"  总参数量: {format_params(total)} (trainable: {format_params(trainable)})")

    # 分层统计
    ssm_total = 0
    moe_total = 0
    moe_active = 0
    attn_total = 0

    block_types = cfg.layers_block_type

    for block in model.backbone.layers:
        t, _ = count_params(block)
        if block.block_type == 'ssm':
            ssm_total += t
        elif block.block_type == 'moe':
            moe_total += t
            top_k = cfg.num_experts_per_tok
            n_experts = cfg.n_routed_experts
            expert_params = sum(p.numel() for p in block.mixer.experts[0].parameters())
            active_layer = t - (n_experts - top_k) * expert_params
            moe_active += active_layer
        elif block.block_type == 'attention':
            attn_total += t

    embed_total, _ = count_params(model.backbone.embeddings)
    output_total = sum(p.numel() for p in [
        model.lm_head.weight,
        model.backbone.norm_f.weight,
    ])
    mtp_total, _ = count_params(model.mtp_head)

    n_ssm = sum(1 for t in block_types if t == 'ssm')
    n_moe = sum(1 for t in block_types if t == 'moe')
    n_attn = sum(1 for t in block_types if t == 'attention')

    print(f"\n  层分布: {n_ssm} SSM + {n_moe} MoE + {n_attn} Attn")
    print(f"  Embedding:   {format_params(embed_total)}")
    print(f"  SSM 层合计:  {format_params(ssm_total)} ({n_ssm} 层, "
          f"平均 {format_params(ssm_total // max(n_ssm, 1))}/层)")
    print(f"  MoE 层合计:  {format_params(moe_total)} total / "
          f"{format_params(moe_active)} active ({n_moe} 层)")
    print(f"  Attn 层合计: {format_params(attn_total)} ({n_attn} 层)")
    print(f"  Output:      {format_params(output_total)}")
    print(f"  MTP Head:    {format_params(mtp_total)}")

    # Active 参数估算
    total_active = total - (moe_total - moe_active)
    print(f"\n  Total: {format_params(total)}, Active: {format_params(total_active)}")

    # 显存估算 (bf16)
    mem_params = total * 2 / 1e9
    mem_grads = total * 2 / 1e9
    mem_optim = total * 8 / 1e9
    mem_total_train = mem_params + mem_grads + mem_optim
    print(f"\n  显存估算 (bf16):")
    print(f"    参数: {mem_params:.2f} GB")
    print(f"    梯度: {mem_grads:.2f} GB")
    print(f"    优化器: {mem_optim:.2f} GB")
    print(f"    合计 (不含激活): {mem_total_train:.2f} GB")

print(f"\n{'='*60}")
print("参数估算完成!")
print(f"{'='*60}")
