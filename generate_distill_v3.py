"""
蒸馏模型推理: 加载独立的 NeuronSparkForCausalLM

从 merge_distill_v3.py 生成的目录加载, 不依赖任何外部模型。

用法:
    python generate_distill_v3.py --model_dir merged_model_v3 --interactive
    python generate_distill_v3.py --model_dir merged_model_v3 --prompt "人工智能的发展"
    python generate_distill_v3.py --model_dir merged_model_v3  # 跑默认 prompts
"""

import argparse

import torch
from transformers import AutoTokenizer

from model_v3 import NeuronSparkForCausalLM, NeuronSparkConfig


def load_model(model_dir):
    """加载 NeuronSparkForCausalLM (device_map='auto' 多卡分布)。"""
    print(f"加载模型: {model_dir}")
    config = NeuronSparkConfig.from_pretrained(model_dir)
    model = NeuronSparkForCausalLM.from_pretrained(
        model_dir, config=config,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        low_cpu_mem_usage=True,
    )
    params = sum(p.numel() for p in model.parameters())
    print(f"  {params/1e9:.1f}B params, D={config.hidden_size}, L={config.num_hidden_layers}")

    input_device = next(model.backbone.embeddings.parameters()).device
    print(f"  输入设备: {input_device}")
    model.eval()
    return model, input_device


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=256,
             temperature=0.8, top_k=50, device='cuda'):
    """自回归文本生成 (使用 HF GenerationMixin)。"""
    input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=temperature > 0,
            eos_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


TEST_PROMPTS = [
    "人工智能的发展",
    "北京是中国的",
    "在自然语言处理领域",
    "脉冲神经网络是一种",
    "The development of artificial intelligence",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuronSpark 蒸馏模型推理")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='合并后的模型目录')
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--interactive', action='store_true')

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model, device = load_model(args.model_dir)

    if args.interactive:
        print(f"\n{'='*50}")
        print(f"NeuronSpark Interactive")
        print(f"Temperature: {args.temperature} | Top-k: {args.top_k}")
        print(f"{'='*50}\n")
        while True:
            try:
                prompt = input("[neuronspark] > ").strip()
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
        print(f"NeuronSpark 测试 | temp={args.temperature} top_k={args.top_k}")
        print(f"{'='*50}")
        for i, prompt in enumerate(TEST_PROMPTS, 1):
            response = generate(model, tokenizer, prompt,
                                max_new_tokens=args.max_new_tokens,
                                temperature=args.temperature,
                                top_k=args.top_k, device=device)
            print(f"\n[{i}/{len(TEST_PROMPTS)}] Prompt: {prompt}")
            print(f"Output: {response}")
            print("-" * 40)
