"""
蒸馏模型推理: 从合并目录加载 NVIDIA NemotronH + BioSSM 文本生成

用法:
    # 先合并
    python scripts/merge_distill_v3.py \
        --teacher_path /path/to/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
        --bio_ssm_ckpt checkpoints_distill_v3/distill_v3_step8015.pth \
        --output_dir merged_model_v3

    # 推理 (只需传合并目录)
    python generate_distill_v3.py --model_dir merged_model_v3 --interactive
    python generate_distill_v3.py --model_dir merged_model_v3 --prompt "人工智能的发展"
"""

import os
import sys
import json
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from distill_wrapper_v3 import DistillHybridModel, BioSSMConfig


def load_merged_model(model_dir):
    """从合并目录加载完整蒸馏模型 (device_map='auto' 多卡分布)。"""
    # 读取配置
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path) as f:
        config = json.load(f)

    teacher_path = config['teacher_path']
    bio_cfg = config['bio_ssm']

    # 加载 Teacher (多卡分布)
    print(f"加载 Teacher: {teacher_path}")
    sys.path.insert(0, teacher_path)
    nvidia_model = AutoModelForCausalLM.from_pretrained(
        teacher_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        low_cpu_mem_usage=True, device_map='auto',
    )
    print(f"  Teacher: {sum(p.numel() for p in nvidia_model.parameters())/1e9:.1f}B params")

    # 构建蒸馏模型
    bio_config = BioSSMConfig(
        hidden_size=bio_cfg['hidden_size'],
        ssm_N=bio_cfg['ssm_N'],
        ssm_K=bio_cfg['ssm_K'],
        ssm_v_th_min=bio_cfg['ssm_v_th_min'],
        ssm_ek_floor=bio_cfg['ssm_ek_floor'],
        num_hidden_layers=bio_cfg['num_hidden_layers'],
    )
    model = DistillHybridModel(nvidia_model, bio_config)

    # 加载 BioSSM 权重
    bio_ssm_path = os.path.join(model_dir, 'bio_ssm.pth')
    print(f"加载 BioSSM: {bio_ssm_path}")
    bio_sd = torch.load(bio_ssm_path, map_location='cpu', weights_only=False)
    model.load_bio_ssm_state(bio_sd)
    bio_params = sum(p.numel() for p in model.bio_ssm_modules.parameters())
    print(f"  BioSSM: {model._num_mamba} 层, {bio_params/1e6:.1f}M params")

    # BioSSM 放到第一张卡
    first_device = next(nvidia_model.parameters()).device
    model.bio_ssm_modules.to(first_device)
    model.eval()

    # 读取合并信息
    merge_info_path = os.path.join(model_dir, 'merge_info.json')
    if os.path.exists(merge_info_path):
        with open(merge_info_path) as f:
            merge_info = json.load(f)
        print(f"  Distill step: {merge_info.get('distill_step', '?')}")

    return model, first_device


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=256,
             temperature=0.8, top_k=50, device='cuda'):
    """自回归文本生成。"""
    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = model(generated, accumulation_steps=1)

        if out.logits is None:
            print("  [警告] logits 为 None, 无法生成")
            break

        next_logits = out.logits[:, -1, :].float()

        if temperature > 0:
            next_logits = next_logits / temperature
            if top_k > 0:
                topk_vals, _ = torch.topk(next_logits, top_k)
                threshold = topk_vals[:, -1].unsqueeze(-1)
                next_logits[next_logits < threshold] = float('-inf')
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = next_logits.argmax(dim=-1, keepdim=True)

        generated = torch.cat([generated, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated[0], skip_special_tokens=True)


TEST_PROMPTS = [
    "人工智能的发展",
    "北京是中国的",
    "在自然语言处理领域",
    "脉冲神经网络是一种",
    "The development of artificial intelligence",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuronSpark v3 蒸馏模型推理")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='合并后的模型目录 (merge_distill_v3.py 输出)')
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--interactive', action='store_true')

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model, device = load_merged_model(args.model_dir)

    if args.interactive:
        print(f"\n{'='*50}")
        print(f"NeuronSpark v3 蒸馏模型 Interactive")
        print(f"Temperature: {args.temperature} | Top-k: {args.top_k}")
        print(f"{'='*50}\n")
        while True:
            try:
                prompt = input("[distill-v3] > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
            if not prompt:
                continue
            if prompt.lower() == 'quit':
                break
            response = generate(model, tokenizer, prompt,
                                max_new_tokens=args.max_new_tokens,
                                temperature=args.temperature,
                                top_k=args.top_k, device=device)
            print(f"\n{response}\n")
    elif args.prompt:
        response = generate(model, tokenizer, args.prompt,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            top_k=args.top_k, device=device)
        print(f"\n{response}")
    else:
        print(f"\n{'='*50}")
        print(f"蒸馏模型测试 | temp={args.temperature} top_k={args.top_k}")
        print(f"{'='*50}")
        for i, prompt in enumerate(TEST_PROMPTS, 1):
            response = generate(model, tokenizer, prompt,
                                max_new_tokens=args.max_new_tokens,
                                temperature=args.temperature,
                                top_k=args.top_k, device=device)
            print(f"\n[{i}/{len(TEST_PROMPTS)}] Prompt: {prompt}")
            print(f"Output: {response}")
            print("-" * 40)
