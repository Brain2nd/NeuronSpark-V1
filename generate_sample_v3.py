"""
推理/采样脚本: NeuronSpark v3 Deep Copy 文本生成

v3 异构状态管理:
  - BioSSM: 神经元 V 状态跨 token 传递 (不 reset)
  - SparkMOE: 无状态，每 token 独立计算
  - SparkAttention: KV cache 累积

用法:
    python generate_sample_v3.py \
        --checkpoint checkpoints_v3/ckpt_step10000.pth \
        --mode pretrain --prompt "人工智能"
"""

import argparse

import torch
from transformers import AutoTokenizer

from model_v3 import NeuronSparkV3ForCausalLM, NeuronSparkV3Config


def load_model(checkpoint_path, device):
    """从 checkpoint 加载 v3 Deep Copy 模型。"""
    print(f"Loading v3 model from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    mc = ckpt.get('model_config', {})
    config = NeuronSparkV3Config(
        vocab_size=mc.get('vocab_size', 6144),
        hidden_size=mc.get('hidden_size', 1024),
        num_hidden_layers=mc.get('num_hidden_layers', 40),
        ssm_N=mc.get('ssm_N', 8),
        ssm_K=mc.get('ssm_K', 16),
        ssm_v_th_min=mc.get('ssm_v_th_min', 0.1),
        ssm_ek_floor=mc.get('ssm_ek_floor', 4.0),
        num_attention_heads=mc.get('num_attention_heads', 8),
        num_key_value_heads=mc.get('num_key_value_heads', 2),
        head_dim=mc.get('head_dim', 128),
        intermediate_size=mc.get('intermediate_size', 2048),
        n_routed_experts=mc.get('n_routed_experts', 32),
        num_experts_per_tok=mc.get('num_experts_per_tok', 4),
        moe_intermediate_size=mc.get('moe_intermediate_size', 1024),
        moe_shared_expert_intermediate_size=mc.get('moe_shared_expert_intermediate_size', 2048),
        routed_scaling_factor=mc.get('routed_scaling_factor', 2.5),
        hybrid_override_pattern=mc.get('hybrid_override_pattern',
                                       "SESESESE*ESESESESE*ESESESESE*ESESESESESE"),
        n_mtp_heads=mc.get('n_mtp_heads', 1),
        max_seq_len=mc.get('max_seq_len', 512),
    )

    model = NeuronSparkV3ForCausalLM(config)

    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)

    model = model.to(device).eval()
    step = ckpt.get('step', '?')
    print(f"  Model loaded (step={step}, D={config.hidden_size}, "
          f"Layers={config.num_hidden_layers})")
    return model


def pretrain_sample(model, tokenizer, prompt, max_new_tokens=256,
                    temperature=0.8, top_k=50, device='cuda'):
    """预训练模型文本续写。"""
    text = f"{tokenizer.bos_token}{prompt}"
    input_ids = tokenizer(text, return_tensors='pt')['input_ids'].to(device)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def sft_sample(model, tokenizer, prompt, max_new_tokens=256,
               temperature=0.8, top_k=50, device='cuda'):
    """SFT 模型对话生成。"""
    messages = [
        {"role": "system", "content": "你是一个AI助手"},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    input_ids = tokenizer(text, return_tensors='pt')['input_ids'].to(device)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    marker = "assistant\n"
    idx = full_text.rfind(marker)
    if idx >= 0:
        response = full_text[idx + len(marker):]
        for end_token in ["<|im_end|>", tokenizer.eos_token]:
            if end_token and response.endswith(end_token):
                response = response[:-len(end_token)]
        return response.strip()
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def interactive_mode(model, tokenizer, args):
    """交互式生成模式。"""
    print(f"\n{'='*50}")
    print(f"NeuronSpark v3 Deep Copy Interactive Generation")
    print(f"Mode: {args.mode} | Temperature: {args.temperature} | Top-k: {args.top_k}")
    print(f"{'='*50}\n")

    mode = args.mode
    sample_fn = sft_sample if mode == 'sft' else pretrain_sample

    while True:
        try:
            prompt = input(f"[v3-{mode}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not prompt:
            continue
        if prompt.lower() == 'quit':
            break
        if prompt.lower().startswith('mode '):
            new_mode = prompt.split()[1].lower()
            if new_mode in ('pretrain', 'sft'):
                mode = new_mode
                sample_fn = sft_sample if mode == 'sft' else pretrain_sample
                print(f"  Switched to {mode} mode")
            continue

        response = sample_fn(
            model, tokenizer, prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=args.device,
        )
        print(f"\n{response}\n")


PRETRAIN_TEST_PROMPTS = [
    "北京大学是",
    "人工智能的发展",
    "中国的四大发明包括",
    "在自然语言处理领域",
]

SFT_TEST_PROMPTS = [
    "你好呀",
    "中国的首都是哪里？",
    "1+12等于多少？",
    "什么是脉冲神经网络？",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuronSpark v3 Text Generation")

    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--tokenizer_path', type=str, default='./tokenizer_snn/')
    parser.add_argument('--device', type=str,
                        default='cuda:0' if torch.cuda.is_available() else 'cpu')

    parser.add_argument('--mode', type=str, default='pretrain', choices=['pretrain', 'sft'])
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--interactive', action='store_true')

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = load_model(args.checkpoint, args.device)

    if args.interactive:
        interactive_mode(model, tokenizer, args)
    elif args.prompt:
        sample_fn = sft_sample if args.mode == 'sft' else pretrain_sample
        response = sample_fn(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=args.device,
        )
        print(f"\n{response}")
    else:
        prompts = SFT_TEST_PROMPTS if args.mode == 'sft' else PRETRAIN_TEST_PROMPTS
        sample_fn = sft_sample if args.mode == 'sft' else pretrain_sample
        print(f"\n{'='*50}")
        print(f"v3 {args.mode} 测试 | temp={args.temperature} top_k={args.top_k}")
        print(f"{'='*50}")
        for i, prompt in enumerate(prompts, 1):
            response = sample_fn(
                model, tokenizer, prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                device=args.device,
            )
            print(f"\n[{i}/{len(prompts)}] Prompt: {prompt}")
            print(f"Output: {response}")
            print("-" * 40)
