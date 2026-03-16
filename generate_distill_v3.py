"""
蒸馏模型推理: NVIDIA NemotronH + BioSSM 文本生成

加载 NVIDIA 30B 冻结模型 + 蒸馏得到的 BioSSM 权重, 合并后推理。

用法:
    python generate_distill_v3.py \
        --teacher_path /path/to/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
        --bio_ssm_ckpt checkpoints_distill_v3/distill_v3_step8015.pth \
        --prompt "人工智能的发展"

    # 交互模式
    python generate_distill_v3.py \
        --teacher_path /path/to/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
        --bio_ssm_ckpt checkpoints_distill_v3/distill_v3_step8015.pth \
        --interactive
"""

import sys
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from distill_wrapper_v3 import DistillHybridModel, BioSSMConfig


def load_distill_model(teacher_path, bio_ssm_ckpt_path):
    """加载 NVIDIA 模型 (device_map='auto' 多卡分布) + BioSSM 蒸馏权重。"""
    print(f"加载 Teacher: {teacher_path}")
    sys.path.insert(0, teacher_path)

    nvidia_config = AutoConfig.from_pretrained(teacher_path, trust_remote_code=True)
    nvidia_model = AutoModelForCausalLM.from_pretrained(
        teacher_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        low_cpu_mem_usage=True, device_map='auto',
    )
    print(f"  Teacher: {sum(p.numel() for p in nvidia_model.parameters())/1e9:.1f}B params")

    # 加载 BioSSM checkpoint
    print(f"加载 BioSSM: {bio_ssm_ckpt_path}")
    ckpt = torch.load(bio_ssm_ckpt_path, map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', {})

    bio_config = BioSSMConfig(
        hidden_size=cfg.get('hidden_size', nvidia_config.hidden_size),
        ssm_N=cfg.get('ssm_N', 4),
        ssm_K=cfg.get('ssm_K', 16),
        ssm_v_th_min=cfg.get('ssm_v_th_min', 0.1),
        ssm_ek_floor=cfg.get('ssm_ek_floor', 4.0),
        num_hidden_layers=cfg.get('num_hidden_layers', nvidia_config.num_hidden_layers),
    )

    # 构建蒸馏模型 + 加载 BioSSM 权重
    model = DistillHybridModel(nvidia_model, bio_config)
    model.load_bio_ssm_state(ckpt['bio_ssm'])
    bio_params = sum(p.numel() for p in model.bio_ssm_modules.parameters())
    print(f"  BioSSM: {model._num_mamba} 层, "
          f"{bio_params/1e6:.1f}M params, step={ckpt.get('step', '?')}")

    # BioSSM 模块放到第一张卡 (teacher 已通过 device_map 分布多卡)
    first_device = next(nvidia_model.parameters()).device
    model.bio_ssm_modules.to(first_device)
    model.eval()
    del ckpt
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

        # 取最后一个 token 的 logits
        next_logits = out.logits[:, -1, :].float()

        # Temperature + top-k 采样
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
    parser.add_argument('--teacher_path', type=str, required=True,
                        help='NVIDIA Nemotron-3 模型目录')
    parser.add_argument('--bio_ssm_ckpt', type=str, required=True,
                        help='BioSSM 蒸馏 checkpoint 路径')
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--interactive', action='store_true')

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_path, trust_remote_code=True)
    model, device = load_distill_model(args.teacher_path, args.bio_ssm_ckpt)

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
