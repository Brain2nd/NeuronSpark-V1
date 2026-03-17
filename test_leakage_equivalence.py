"""
转换前后推理对比测试

对比原始模型 (V_post 激活) 和转换模型 (泄漏量激活) 的:
1. logits 数值差异 (L∞, 相对误差)
2. 生成文本一致性 (greedy decode)
3. SFT 对话输出

用法:
    python test_leakage_equivalence.py \
        --original checkpoints_sft/ckpt_step6500_orig.pth \
        --converted checkpoints_sft/ckpt_step6500.pth \
        --mode sft
"""

import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from spikingjelly.activation_based import functional

from model import SNNLanguageModel


def load_model(ckpt_path, device='cpu'):
    """加载模型。"""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get('model_config', {})

    model = SNNLanguageModel(
        vocab_size=config.get('vocab_size', 6144),
        D=config.get('D', 1024),
        N=config.get('N', 8),
        K=config.get('K', 32),
        num_layers=config.get('num_layers', 20),
        D_ff=config.get('D_ff', 3072),
        activation_mode=config.get('activation_mode', 'v2'),
    )
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model = model.to(device).eval()

    step = ckpt.get('step', '?')
    act_type = ckpt.get('activation_type', 'V_post')
    print(f"  Loaded: step={step}, act={act_type}, D={config.get('D')}, L={config.get('num_layers')}")
    return model


def compare_logits(model_orig, model_conv, tokenizer, prompts, device='cpu'):
    """对比两个模型的 logits 数值差异。"""
    print("\n" + "=" * 60)
    print("1. Logits 数值对比")
    print("=" * 60)

    for prompt in prompts:
        text = f"{tokenizer.bos_token}{prompt}"
        input_ids = tokenizer(text, return_tensors='pt')['input_ids'].to(device)

        # 原始模型
        for layer in model_orig.layers:
            functional.reset_net(layer)
        functional.reset_net(model_orig.output_neuron)

        with torch.no_grad():
            out_orig = model_orig(input_ids)
            logits_orig = out_orig.logits

        # 转换模型
        for layer in model_conv.layers:
            functional.reset_net(layer)
        functional.reset_net(model_conv.output_neuron)

        with torch.no_grad():
            out_conv = model_conv(input_ids)
            logits_conv = out_conv.logits

        # 数值差异
        diff = (logits_orig - logits_conv).abs()
        l_inf = diff.max().item()
        l_mean = diff.mean().item()

        scale = logits_orig.abs().clamp(min=1e-8)
        rel_err = (diff / scale).mean().item()

        top1_orig = logits_orig[:, -1, :].argmax(dim=-1)
        top1_conv = logits_conv[:, -1, :].argmax(dim=-1)
        top1_match = (top1_orig == top1_conv).all().item()

        log_prob_orig = F.log_softmax(logits_orig[:, -1, :], dim=-1)
        prob_conv = F.softmax(logits_conv[:, -1, :], dim=-1)
        kl_div = F.kl_div(log_prob_orig, prob_conv, reduction='batchmean').item()

        print(f"\n  Prompt: '{prompt}'")
        print(f"    L∞ 误差:      {l_inf:.6e}")
        print(f"    L_mean 误差:   {l_mean:.6e}")
        print(f"    相对误差:      {rel_err:.6e}")
        print(f"    KL 散度:       {kl_div:.6e}")
        print(f"    Top-1 一致:    {'✓' if top1_match else '✗'}")
        print(f"    原始 top-1:    {tokenizer.decode([top1_orig[0].item()])}")
        print(f"    转换 top-1:    {tokenizer.decode([top1_conv[0].item()])}")


def compare_generation(model_orig, model_conv, tokenizer, prompts, mode='pretrain',
                       max_new_tokens=64, device='cpu'):
    """对比两个模型的 greedy 生成。"""
    print("\n" + "=" * 60)
    print(f"2. Greedy 生成对比 (mode={mode})")
    print("=" * 60)

    for prompt in prompts:
        if mode == 'sft':
            messages = [
                {"role": "system", "content": "你是一个AI助手"},
                {"role": "user", "content": prompt},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        else:
            text = f"{tokenizer.bos_token}{prompt}"

        input_ids = tokenizer(text, return_tensors='pt')['input_ids'].to(device)

        output_orig = model_orig.generate(
            input_ids, max_new_tokens=max_new_tokens,
            temperature=0, top_k=1, eos_token_id=tokenizer.eos_token_id,
        )

        output_conv = model_conv.generate(
            input_ids, max_new_tokens=max_new_tokens,
            temperature=0, top_k=1, eos_token_id=tokenizer.eos_token_id,
        )

        text_orig = tokenizer.decode(output_orig[0], skip_special_tokens=True)
        text_conv = tokenizer.decode(output_conv[0], skip_special_tokens=True)

        tokens_match = torch.equal(output_orig, output_conv)

        print(f"\n  Q: {prompt}")
        print(f"  原始: {text_orig[:200]}")
        print(f"  转换: {text_conv[:200]}")
        print(f"  Token 完全一致: {'✓' if tokens_match else '✗'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="转换前后推理对比测试")
    parser.add_argument('--original', type=str, required=True, help='原始 checkpoint')
    parser.add_argument('--converted', type=str, required=True, help='转换后 checkpoint')
    parser.add_argument('--tokenizer_path', type=str, default='./tokenizer_snn/')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--mode', type=str, default='sft', choices=['pretrain', 'sft'])
    parser.add_argument('--max_new_tokens', type=int, default=64)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    print(f"加载原始模型...")
    model_orig = load_model(args.original, args.device)

    print(f"加载转换模型...")
    model_conv = load_model(args.converted, args.device)

    pretrain_prompts = ["北京大学是", "人工智能的发展", "中国的四大发明包括"]
    sft_prompts = ["你好呀", "中国的首都是哪里？", "1+12等于多少？", "什么是脉冲神经网络？"]

    prompts = sft_prompts if args.mode == 'sft' else pretrain_prompts

    compare_logits(model_orig, model_conv, tokenizer, prompts[:3], args.device)
    compare_generation(model_orig, model_conv, tokenizer, prompts,
                       mode=args.mode, max_new_tokens=args.max_new_tokens, device=args.device)

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
