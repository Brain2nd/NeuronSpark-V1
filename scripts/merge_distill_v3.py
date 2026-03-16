"""
合并蒸馏模型: NVIDIA 冻结权重 + BioSSM 蒸馏权重 → NeuronSparkForCausalLM

将 NVIDIA 30B 的冻结层权重 (MoE, Attention, Embedding, norm, lm_head) 和
BioSSM 蒸馏权重合并到 NeuronSparkForCausalLM, 保存为独立 HF 模型。

输出目录是完整的 HuggingFace 模型, 可直接:
    model = NeuronSparkForCausalLM.from_pretrained("merged_model_v3")

不依赖 teacher 路径, 不包含 Mamba mixer 权重。

用法:
    python scripts/merge_distill_v3.py \
        --teacher_path /path/to/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
        --bio_ssm_ckpt checkpoints_distill_v3/distill_v3_step8015.pth \
        --output_dir merged_model_v3
"""

import os
import sys
import json
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

from model_v3 import NeuronSparkForCausalLM, NeuronSparkConfig
from distill_wrapper_v3 import DistillHybridModel, BioSSMConfig


def build_neuronspark_config(nvidia_config, bio_cfg):
    """从 NVIDIA config + BioSSM config 构建 NeuronSparkConfig。"""
    return NeuronSparkConfig(
        vocab_size=nvidia_config.vocab_size,
        hidden_size=nvidia_config.hidden_size,
        intermediate_size=nvidia_config.intermediate_size,
        num_hidden_layers=nvidia_config.num_hidden_layers,
        hybrid_override_pattern=nvidia_config.hybrid_override_pattern,
        num_attention_heads=nvidia_config.num_attention_heads,
        head_dim=nvidia_config.head_dim,
        num_key_value_heads=nvidia_config.num_key_value_heads,
        mlp_hidden_act=getattr(nvidia_config, 'mlp_hidden_act', 'relu2'),
        attention_bias=getattr(nvidia_config, 'attention_bias', False),
        mlp_bias=getattr(nvidia_config, 'mlp_bias', False),
        layer_norm_epsilon=nvidia_config.layer_norm_epsilon,
        max_position_embeddings=getattr(nvidia_config, 'max_position_embeddings', 4096),
        # MoE
        n_routed_experts=nvidia_config.n_routed_experts,
        moe_intermediate_size=nvidia_config.moe_intermediate_size,
        moe_shared_expert_intermediate_size=nvidia_config.moe_shared_expert_intermediate_size,
        num_experts_per_tok=nvidia_config.num_experts_per_tok,
        routed_scaling_factor=nvidia_config.routed_scaling_factor,
        n_group=getattr(nvidia_config, 'n_group', 1),
        topk_group=getattr(nvidia_config, 'topk_group', 1),
        norm_topk_prob=getattr(nvidia_config, 'norm_topk_prob', True),
        # BioSSM
        ssm_N=bio_cfg.get('ssm_N', 4),
        ssm_K=bio_cfg.get('ssm_K', 16),
        ssm_v_th_min=bio_cfg.get('ssm_v_th_min', 0.1),
        ssm_ek_floor=bio_cfg.get('ssm_ek_floor', 4.0),
        # MTP (蒸馏阶段不用, 但模型结构需要)
        n_mtp_heads=0,
        max_seq_len=4096,
        # Tokenizer
        pad_token_id=getattr(nvidia_config, 'pad_token_id', 0),
        bos_token_id=getattr(nvidia_config, 'bos_token_id', 1),
        eos_token_id=getattr(nvidia_config, 'eos_token_id', 2),
    )


def map_nvidia_to_neuronspark(nvidia_sd, mamba_indices):
    """映射 NVIDIA state dict → NeuronSpark state dict (跳过 Mamba mixer)。

    Key 差异:
      NVIDIA: backbone.embedding.weight → NeuronSpark: backbone.embeddings.weight
      Mamba mixer 位置: 跳过 (用 BioSSM 替代)
      其他: key 完全相同
    """
    mapped = {}
    mamba_mixer_prefixes = [f'backbone.layers.{idx}.mixer.' for idx in mamba_indices]
    skipped = 0

    for k, v in nvidia_sd.items():
        # 跳过 Mamba mixer 权重
        if any(k.startswith(p) for p in mamba_mixer_prefixes):
            skipped += v.numel()
            continue

        # embedding → embeddings
        new_k = k.replace('backbone.embedding.', 'backbone.embeddings.')
        mapped[new_k] = v.cpu()

    return mapped, skipped


def map_bio_ssm_to_neuronspark(bio_sd, mamba_indices):
    """映射 BioSSM state dict → NeuronSpark state dict。

    蒸馏 checkpoint 的 key 格式: {layer_idx}.bio_ssm.xxx
    NeuronSpark 的 key 格式: backbone.layers.{layer_idx}.mixer.bio_ssm.xxx
    """
    mapped = {}
    for k, v in bio_sd.items():
        # bio_sd key: "{layer_idx}.bio_ssm.xxx" (来自 bio_ssm_modules 的 state_dict)
        # 目标: "backbone.layers.{layer_idx}.mixer.bio_ssm.xxx"
        mapped[f'backbone.layers.{k}'] = v

    return mapped


def merge_and_save(teacher_path, bio_ssm_ckpt_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # ===== 加载 BioSSM checkpoint =====
    print(f"加载 BioSSM: {bio_ssm_ckpt_path}")
    ckpt = torch.load(bio_ssm_ckpt_path, map_location='cpu', weights_only=False)
    bio_cfg = ckpt.get('config', {})
    step = ckpt.get('step', 'unknown')
    bio_sd = ckpt['bio_ssm']
    print(f"  Step: {step}, Keys: {len(bio_sd)}")

    # ===== 加载 NVIDIA 模型 =====
    print(f"加载 Teacher: {teacher_path}")
    sys.path.insert(0, teacher_path)
    nvidia_config = AutoConfig.from_pretrained(teacher_path, trust_remote_code=True)
    nvidia_model = AutoModelForCausalLM.from_pretrained(
        teacher_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    teacher_params = sum(p.numel() for p in nvidia_model.parameters())
    print(f"  Teacher: {teacher_params/1e9:.1f}B params")

    # ===== 获取 Mamba 位置 =====
    mamba_indices = [i for i, block in enumerate(nvidia_model.backbone.layers)
                     if block.block_type == "mamba"]
    print(f"  Mamba 层: {len(mamba_indices)} 个, 位置: {mamba_indices[:5]}...")

    # ===== 构建 NeuronSparkConfig =====
    ns_config = build_neuronspark_config(nvidia_config, bio_cfg)
    # 验证 hybrid_override_pattern 中 S 位置 == mamba_indices
    ssm_indices = [i for i, t in enumerate(ns_config.layers_block_type) if t == 'ssm']
    assert ssm_indices == mamba_indices, \
        f"Pattern S 位置 {ssm_indices} ≠ Mamba 位置 {mamba_indices}"
    print(f"  NeuronSparkConfig: D={ns_config.hidden_size}, L={ns_config.num_hidden_layers}")

    # ===== 映射权重 =====
    print("\n映射权重...")
    nvidia_sd = nvidia_model.state_dict()

    # 1. NVIDIA 冻结层 (跳过 Mamba mixer)
    frozen_sd, skipped_params = map_nvidia_to_neuronspark(nvidia_sd, mamba_indices)
    print(f"  冻结层: {len(frozen_sd)} keys, "
          f"跳过 Mamba mixer: {skipped_params/1e6:.0f}M params ({skipped_params*2/1e9:.1f}GB)")

    # 2. BioSSM 权重
    bio_mapped = map_bio_ssm_to_neuronspark(bio_sd, mamba_indices)
    print(f"  BioSSM: {len(bio_mapped)} keys")

    # 合并
    merged_sd = {**frozen_sd, **bio_mapped}
    print(f"  合并: {len(merged_sd)} keys")

    # 释放 NVIDIA 模型显存
    del nvidia_model, nvidia_sd
    torch.cuda.empty_cache()

    # ===== 创建 NeuronSparkForCausalLM 并加载权重 =====
    print("\n构建 NeuronSparkForCausalLM...")
    ns_model = NeuronSparkForCausalLM(ns_config)

    # 加载合并权重
    missing, unexpected = ns_model.load_state_dict(merged_sd, strict=False)
    if missing:
        # MTP head 权重可能缺失 (蒸馏阶段不用), 忽略
        mtp_missing = [k for k in missing if 'mtp' in k]
        real_missing = [k for k in missing if 'mtp' not in k]
        if real_missing:
            print(f"  [警告] 缺失 {len(real_missing)} keys: {real_missing[:5]}...")
        if mtp_missing:
            print(f"  MTP keys 缺失 (正常): {len(mtp_missing)}")
    if unexpected:
        print(f"  [警告] 多余 {len(unexpected)} keys: {unexpected[:5]}...")

    ns_params = sum(p.numel() for p in ns_model.parameters())
    print(f"  NeuronSpark 模型: {ns_params/1e9:.1f}B params")

    # ===== 保存为 HuggingFace 模型 =====
    print(f"\n保存到 {output_dir} ...")
    ns_model.save_pretrained(output_dir, max_shard_size="5GB")
    print(f"  模型权重已保存")

    # 保存 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(teacher_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)
    print(f"  Tokenizer 已保存")

    # 元信息
    merge_info = {
        'distill_step': step,
        'total_params_B': round(ns_params / 1e9, 1),
        'mamba_mixer_removed_M': round(skipped_params / 1e6, 0),
        'bio_ssm_params_M': round(sum(p.numel() for p in bio_sd.values()) / 1e6, 1),
        'num_ssm_layers': len(mamba_indices),
        'ssm_indices': mamba_indices,
        'num_layers': ns_config.num_hidden_layers,
    }
    with open(os.path.join(output_dir, 'merge_info.json'), 'w') as f:
        json.dump(merge_info, f, indent=2, ensure_ascii=False)

    saved_size = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in os.listdir(output_dir)
        if f.endswith(('.safetensors', '.bin', '.pth'))
    ) / 1e9

    print(f"\n{'='*60}")
    print(f"合并完成: {output_dir}")
    print(f"  模型:    NeuronSparkForCausalLM ({ns_params/1e9:.1f}B params)")
    print(f"  磁盘:    {saved_size:.1f}GB (已去除 Mamba mixer)")
    print(f"  Step:    {step}")
    print(f"{'='*60}")
    print(f"\n推理:")
    print(f"  python generate_distill_v3.py --model_dir {output_dir} --interactive")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher_path', type=str, required=True)
    parser.add_argument('--bio_ssm_ckpt', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='merged_model_v3')
    args = parser.parse_args()
    merge_and_save(args.teacher_path, args.bio_ssm_ckpt, args.output_dir)
