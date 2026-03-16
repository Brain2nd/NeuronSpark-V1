"""
导出 HuggingFace 标准格式模型目录。

将 .pth checkpoint 转换为:
  output_dir/
  ├── config.json
  ├── model.safetensors
  ├── configuration_neuronspark.py
  ├── modeling_neuronspark.py
  ├── model.py
  ├── atomic_ops/
  ├── tokenizer.json
  ├── tokenizer_config.json
  └── special_tokens_map.json

用法:
    python export_hf.py --ckpt checkpoints/ckpt_step85000.pth --output_dir NeuronSpark-Pretrain
    python export_hf.py --ckpt checkpoints_sft/ckpt_step6500.pth --output_dir NeuronSpark-SFT
"""

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import save_file

from model import SNNLanguageModel


def export(ckpt_path, output_dir, tokenizer_path="./tokenizer_snn/"):
    os.makedirs(output_dir, exist_ok=True)

    # ====== 1. 加载 checkpoint ======
    print(f"Loading {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model_config = ckpt['model_config']
    state_dict = ckpt['model_state_dict']

    # ====== 2. 创建 config.json ======
    config = {
        "model_type": "neuronspark",
        "architectures": ["NeuronSparkForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_neuronspark.NeuronSparkConfig",
            "AutoModelForCausalLM": "modeling_neuronspark.NeuronSparkForCausalLM"
        },
        "vocab_size": model_config['vocab_size'],
        "D": model_config['D'],
        "N": model_config['N'],
        "K": model_config['K'],
        "num_layers": model_config['num_layers'],
        "D_ff": model_config['D_ff'],
        "v_th_min": 0.1,
        "torch_dtype": "float32",
        "transformers_version": "4.52.0",
        # 训练信息
        "_training_step": ckpt.get('step', 0),
        "_tokens_seen": ckpt.get('tokens_seen', 0),
    }

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"  config.json saved")

    # ====== 3. 转换 state_dict → safetensors ======
    # HuggingFace 模型用 self.model = SNNLanguageModel(...)
    # 所以 key 前缀加 "model."
    hf_state_dict = {}
    for k, v in state_dict.items():
        hf_state_dict[f"model.{k}"] = v.contiguous()

    safetensors_path = os.path.join(output_dir, "model.safetensors")
    save_file(hf_state_dict, safetensors_path)
    size_gb = os.path.getsize(safetensors_path) / 1e9
    print(f"  model.safetensors saved ({size_gb:.2f} GB, {len(hf_state_dict)} tensors)")

    # ====== 4. 复制模型代码 ======
    project_root = os.path.dirname(os.path.abspath(__file__))

    # 核心文件
    for fname in ['configuration_neuronspark.py', 'modeling_neuronspark.py', 'model.py']:
        src = os.path.join(project_root, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, fname))

    # atomic_ops 目录
    atomic_src = os.path.join(project_root, 'atomic_ops')
    atomic_dst = os.path.join(output_dir, 'atomic_ops')
    if os.path.exists(atomic_dst):
        shutil.rmtree(atomic_dst)
    shutil.copytree(atomic_src, atomic_dst)
    # 清理 __pycache__
    for root, dirs, files in os.walk(atomic_dst):
        for d in dirs:
            if d == '__pycache__':
                shutil.rmtree(os.path.join(root, d))

    print(f"  model code copied")

    # ====== 5. 复制 tokenizer ======
    for fname in ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']:
        src = os.path.join(tokenizer_path, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, fname))
    print(f"  tokenizer copied")

    # ====== 6. 验证 ======
    print(f"\n  Output: {output_dir}/")
    for item in sorted(os.listdir(output_dir)):
        full = os.path.join(output_dir, item)
        if os.path.isdir(full):
            print(f"    {item}/")
        else:
            size = os.path.getsize(full)
            if size > 1e6:
                print(f"    {item}  ({size/1e9:.2f} GB)")
            else:
                print(f"    {item}  ({size/1e3:.1f} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--tokenizer_path', type=str, default='./tokenizer_snn/')
    args = parser.parse_args()

    export(args.ckpt, args.output_dir, args.tokenizer_path)
    print("\nDone!")
