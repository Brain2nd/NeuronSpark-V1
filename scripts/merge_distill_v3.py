"""
真正合并蒸馏模型: NVIDIA NemotronH + BioSSM → 自包含模型目录

将 NVIDIA 30B 完整权重 + BioSSM 蒸馏权重保存到同一目录,
推理时只需传目录路径, 不再依赖 teacher 路径。

输出目录包含:
  - NVIDIA 模型完整权重 (save_pretrained, 含 config + modeling 代码)
  - bio_ssm.pth: BioSSM 蒸馏权重
  - bio_ssm_config.json: BioSSM 配置
  - tokenizer 文件
  - merge_info.json: 合并元信息

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

from distill_wrapper_v3 import DistillHybridModel, BioSSMConfig


def merge_and_save(teacher_path, bio_ssm_ckpt_path, output_dir):
    """真正合并 NVIDIA + BioSSM 并保存为自包含模型目录。"""
    os.makedirs(output_dir, exist_ok=True)

    # ===== 加载 BioSSM checkpoint =====
    print(f"加载 BioSSM checkpoint: {bio_ssm_ckpt_path}")
    ckpt = torch.load(bio_ssm_ckpt_path, map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', {})
    step = ckpt.get('step', 'unknown')
    bio_sd = ckpt['bio_ssm']
    print(f"  Step: {step}, Keys: {len(bio_sd)}")

    # ===== 加载 Teacher =====
    print(f"加载 Teacher: {teacher_path}")
    sys.path.insert(0, teacher_path)
    nvidia_model = AutoModelForCausalLM.from_pretrained(
        teacher_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    nvidia_config = AutoConfig.from_pretrained(teacher_path, trust_remote_code=True)
    hidden_size = cfg.get('hidden_size', nvidia_config.hidden_size)
    num_layers = cfg.get('num_hidden_layers', nvidia_config.num_hidden_layers)
    teacher_params = sum(p.numel() for p in nvidia_model.parameters())
    print(f"  Teacher: {teacher_params/1e9:.1f}B params, D={hidden_size}, L={num_layers}")

    # ===== 验证 BioSSM 权重可正确加载 =====
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
    print(f"  BioSSM: {model._num_mamba} 层, {bio_params/1e6:.1f}M params")
    print(f"  验证通过")

    # ===== 1. 保存 NVIDIA 完整模型 (权重 + config + modeling 代码) =====
    print(f"\n保存 NVIDIA 模型到 {output_dir} ...")
    nvidia_model.save_pretrained(output_dir)
    print(f"  NVIDIA 模型权重已保存")

    # ===== 2. 保存 BioSSM 权重 =====
    bio_save_path = os.path.join(output_dir, 'bio_ssm.pth')
    verified_bio_sd = model.save_bio_ssm_state()
    torch.save(verified_bio_sd, bio_save_path)
    print(f"  BioSSM 权重: {bio_save_path} ({len(verified_bio_sd)} keys)")

    # ===== 3. 保存 BioSSM 配置 =====
    bio_config_dict = {
        'hidden_size': bio_config.hidden_size,
        'ssm_N': bio_config.ssm_N,
        'ssm_K': bio_config.ssm_K,
        'ssm_v_th_min': bio_config.ssm_v_th_min,
        'ssm_ek_floor': bio_config.ssm_ek_floor,
        'num_hidden_layers': bio_config.num_hidden_layers,
    }
    bio_config_path = os.path.join(output_dir, 'bio_ssm_config.json')
    with open(bio_config_path, 'w') as f:
        json.dump(bio_config_dict, f, indent=2)
    print(f"  BioSSM 配置: {bio_config_path}")

    # ===== 4. 保存 Tokenizer =====
    tokenizer = AutoTokenizer.from_pretrained(teacher_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)
    print(f"  Tokenizer 已保存")

    # ===== 5. 合并元信息 =====
    merge_info = {
        'distill_step': step,
        'teacher_params_B': round(teacher_params / 1e9, 1),
        'bio_ssm_params_M': round(bio_params / 1e6, 1),
        'num_mamba_layers': model._num_mamba,
        'mamba_indices': model.mamba_indices,
        'total_layers': num_layers,
        'source_teacher_path': os.path.abspath(teacher_path),
        'source_bio_ssm_ckpt': os.path.abspath(bio_ssm_ckpt_path),
    }
    merge_info_path = os.path.join(output_dir, 'merge_info.json')
    with open(merge_info_path, 'w') as f:
        json.dump(merge_info, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"合并完成: {output_dir}")
    print(f"  NVIDIA:  {teacher_params/1e9:.1f}B params (完整权重已保存)")
    print(f"  BioSSM:  {bio_params/1e6:.1f}M params ({model._num_mamba} 层)")
    print(f"  Step:    {step}")
    print(f"{'='*60}")
    print(f"\n推理:")
    print(f"  python generate_distill_v3.py --model_dir {output_dir} --interactive")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="合并 NVIDIA + BioSSM → 自包含模型目录")
    parser.add_argument('--teacher_path', type=str, required=True)
    parser.add_argument('--bio_ssm_ckpt', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='merged_model_v3')
    args = parser.parse_args()

    merge_and_save(args.teacher_path, args.bio_ssm_ckpt, args.output_dir)
