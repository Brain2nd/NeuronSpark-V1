"""
蒸馏模型推理: 从合并目录加载 NemotronH + BioSSM

合并目录由 merge_distill_v3.py 生成, 包含 NVIDIA 完整权重 + BioSSM 权重,
不依赖外部 teacher 路径。

用法:
    python generate_distill_v3.py --model_dir merged_model_v3 --interactive
    python generate_distill_v3.py --model_dir merged_model_v3 --prompt "人工智能的发展"
    python generate_distill_v3.py --model_dir merged_model_v3  # 跑默认 prompts
"""

import os
import sys
import json
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from distill_wrapper_v3 import DistillHybridModel, BioSSMConfig


def load_merged_model(model_dir):
    """从合并目录加载完整蒸馏模型。

    NVIDIA 模型通过 device_map='auto' 分布多卡,
    每个 BioSSM 放到对应 Mamba block 所在的 GPU 上, 避免跨卡传输。
    """
    # 加载 NVIDIA 模型 (多卡分布)
    print(f"加载 NVIDIA 模型: {model_dir}")
    sys.path.insert(0, model_dir)
    nvidia_model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True,
        low_cpu_mem_usage=True, device_map='auto',
    )
    print(f"  NVIDIA: {sum(p.numel() for p in nvidia_model.parameters())/1e9:.1f}B params")

    # 加载 BioSSM 配置
    bio_config_path = os.path.join(model_dir, 'bio_ssm_config.json')
    with open(bio_config_path) as f:
        bio_cfg = json.load(f)

    bio_config = BioSSMConfig(
        hidden_size=bio_cfg['hidden_size'],
        ssm_N=bio_cfg['ssm_N'],
        ssm_K=bio_cfg['ssm_K'],
        ssm_v_th_min=bio_cfg['ssm_v_th_min'],
        ssm_ek_floor=bio_cfg['ssm_ek_floor'],
        num_hidden_layers=bio_cfg['num_hidden_layers'],
    )

    # 构建蒸馏模型 + 加载 BioSSM 权重
    model = DistillHybridModel(nvidia_model, bio_config)
    bio_ssm_path = os.path.join(model_dir, 'bio_ssm.pth')
    print(f"加载 BioSSM: {bio_ssm_path}")
    bio_sd = torch.load(bio_ssm_path, map_location='cpu', weights_only=False)
    model.load_bio_ssm_state(bio_sd)
    del bio_sd

    bio_params = sum(p.numel() for p in model.bio_ssm_modules.parameters())
    print(f"  BioSSM: {model._num_mamba} 层, {bio_params/1e6:.1f}M params")

    # 每个 BioSSM 放到对应 Mamba block 的 GPU 上 (避免跨卡 RuntimeError)
    for idx in model.mamba_indices:
        block_device = next(nvidia_model.backbone.layers[idx].parameters()).device
        model.bio_ssm_modules[str(idx)].to(block_device)
    print(f"  BioSSM 设备对齐完成")

    model.eval()

    # 确定输入设备 (embedding 所在的卡)
    input_device = next(nvidia_model.backbone.embeddings.parameters()).device
    print(f"  输入设备: {input_device}")

    # 元信息
    merge_info_path = os.path.join(model_dir, 'merge_info.json')
    if os.path.exists(merge_info_path):
        with open(merge_info_path) as f:
            info = json.load(f)
        print(f"  Distill step: {info.get('distill_step', '?')}")

    return model, input_device


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

        # 取最后一个 token 的 logits (可能在不同卡上, 移到输入设备)
        next_logits = out.logits[:, -1, :].float().to(device)

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
