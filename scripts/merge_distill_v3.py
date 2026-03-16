"""
合并蒸馏模型: NVIDIA NemotronH + BioSSM → 完整模型

将 NVIDIA 30B 冻结权重 + BioSSM 蒸馏权重合并为一个完整的 DistillHybridModel,
保存为单个 checkpoint, 之后推理直接加载无需重新合并。

用法:
    python scripts/merge_distill_v3.py \
        --teacher_path /path/to/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
        --bio_ssm_ckpt checkpoints_distill_v3/distill_v3_step8015.pth \
        --output_dir merged_model_v3

输出:
    merged_model_v3/
    ├── config.json          # BioSSM + teacher 配置
    ├── bio_ssm.pth          # BioSSM 权重 (已验证可加载)
    ├── merge_info.json      # 合并元信息
    └── (teacher 模型通过 symlink 或路径引用, 不重复存储 30B 权重)
"""

import os
import sys
import json
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

from distill_wrapper_v3 import DistillHybridModel, BioSSMConfig


def merge_and_save(teacher_path, bio_ssm_ckpt_path, output_dir):
    """合并 NVIDIA + BioSSM 并保存。"""
    os.makedirs(output_dir, exist_ok=True)

    # ===== 加载 BioSSM checkpoint =====
    print(f"加载 BioSSM checkpoint: {bio_ssm_ckpt_path}")
    ckpt = torch.load(bio_ssm_ckpt_path, map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', {})
    step = ckpt.get('step', 'unknown')
    bio_sd = ckpt['bio_ssm']
    print(f"  Step: {step}, Keys: {len(bio_sd)}")

    # ===== 加载 Teacher 配置 (不加载完整权重, 只验证兼容性) =====
    print(f"加载 Teacher 配置: {teacher_path}")
    nvidia_config = AutoConfig.from_pretrained(teacher_path, trust_remote_code=True)
    hidden_size = cfg.get('hidden_size', nvidia_config.hidden_size)
    num_layers = cfg.get('num_hidden_layers', nvidia_config.num_hidden_layers)
    print(f"  Teacher: D={hidden_size}, L={num_layers}")

    # ===== 验证: 实际加载模型并 load_bio_ssm_state 确保权重匹配 =====
    print(f"验证: 加载 Teacher 模型并挂载 BioSSM...")
    sys.path.insert(0, teacher_path)
    nvidia_model = AutoModelForCausalLM.from_pretrained(
        teacher_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        low_cpu_mem_usage=True, device_map='auto',
    )
    teacher_params = sum(p.numel() for p in nvidia_model.parameters())

    bio_config = BioSSMConfig(
        hidden_size=hidden_size,
        ssm_N=cfg.get('ssm_N', 4),
        ssm_K=cfg.get('ssm_K', 16),
        ssm_v_th_min=cfg.get('ssm_v_th_min', 0.1),
        ssm_ek_floor=cfg.get('ssm_ek_floor', 4.0),
        num_hidden_layers=num_layers,
    )

    model = DistillHybridModel(nvidia_model, bio_config)
    model.load_bio_ssm_state(bio_sd)
    bio_params = sum(p.numel() for p in model.bio_ssm_modules.parameters())

    print(f"  Teacher: {teacher_params/1e9:.1f}B params")
    print(f"  BioSSM:  {bio_params/1e6:.1f}M params ({model._num_mamba} 层)")
    print(f"  验证通过: 权重加载成功")

    # ===== 保存 =====
    # 1. BioSSM 权重 (重新从已验证的模型提取)
    bio_save_path = os.path.join(output_dir, 'bio_ssm.pth')
    verified_bio_sd = model.save_bio_ssm_state()
    torch.save(verified_bio_sd, bio_save_path)
    print(f"  BioSSM 权重: {bio_save_path} ({len(verified_bio_sd)} keys)")

    # 2. 配置信息
    config_info = {
        'bio_ssm': {
            'hidden_size': bio_config.hidden_size,
            'ssm_N': bio_config.ssm_N,
            'ssm_K': bio_config.ssm_K,
            'ssm_v_th_min': bio_config.ssm_v_th_min,
            'ssm_ek_floor': bio_config.ssm_ek_floor,
            'num_hidden_layers': bio_config.num_hidden_layers,
        },
        'teacher_path': os.path.abspath(teacher_path),
    }
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config_info, f, indent=2, ensure_ascii=False)

    # 3. 合并元信息
    merge_info = {
        'teacher_path': os.path.abspath(teacher_path),
        'bio_ssm_ckpt': os.path.abspath(bio_ssm_ckpt_path),
        'distill_step': step,
        'teacher_params_B': round(teacher_params / 1e9, 1),
        'bio_ssm_params_M': round(bio_params / 1e6, 1),
        'num_mamba_layers': model._num_mamba,
        'mamba_indices': model.mamba_indices,
        'total_layers': num_layers,
    }
    merge_path = os.path.join(output_dir, 'merge_info.json')
    with open(merge_path, 'w') as f:
        json.dump(merge_info, f, indent=2, ensure_ascii=False)

    # 4. 复制 tokenizer
    print(f"复制 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(teacher_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    print(f"\n{'='*50}")
    print(f"合并完成: {output_dir}")
    print(f"  Teacher:  {teacher_params/1e9:.1f}B (引用路径, 不重复存储)")
    print(f"  BioSSM:   {bio_params/1e6:.1f}M ({model._num_mamba} 层)")
    print(f"  Step:     {step}")
    print(f"{'='*50}")
    print(f"\n推理用法:")
    print(f"  python generate_distill_v3.py \\")
    print(f"      --teacher_path {teacher_path} \\")
    print(f"      --bio_ssm_ckpt {bio_save_path} \\")
    print(f"      --interactive")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="合并 NVIDIA + BioSSM 蒸馏模型")
    parser.add_argument('--teacher_path', type=str, required=True,
                        help='NVIDIA Nemotron-3 模型目录')
    parser.add_argument('--bio_ssm_ckpt', type=str, required=True,
                        help='BioSSM 蒸馏 checkpoint 路径')
    parser.add_argument('--output_dir', type=str, default='merged_model_v3',
                        help='输出目录 (默认: merged_model_v3)')
    args = parser.parse_args()

    merge_and_save(args.teacher_path, args.bio_ssm_ckpt, args.output_dir)
