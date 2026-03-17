import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

"""
推理能力测试 + Surprisal/E[K] 联合分析
"""
import sys
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer
from spikingjelly.activation_based import functional
from model import SNNLanguageModel

print("Loading model...", flush=True)
ckpt = torch.load('checkpoints_sft/ckpt_step6500.pth', map_location='cpu', weights_only=False)
config = ckpt.get('model_config', {})
model = SNNLanguageModel(**{k: config[k] for k in ['vocab_size','D','N','K','num_layers','D_ff']})
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model.eval()
tokenizer = AutoTokenizer.from_pretrained('./tokenizer_snn/')
print(f"Model loaded: D={config['D']}, K={config['K']}, L={config['num_layers']}", flush=True)


def generate_response(prompt, max_new=64, temperature=0.1, top_k=10):
    messages = [{"role":"system","content":"你是一个AI助手"},{"role":"user","content":prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, return_tensors='pt')['input_ids']
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, temperature=temperature,
                             top_k=top_k, eos_token_id=tokenizer.eos_token_id)
    full = tokenizer.decode(out[0], skip_special_tokens=False)
    marker = 'assistant\n'
    idx = full.rfind(marker)
    if idx >= 0:
        resp = full[idx+len(marker):]
        for end in ['<|im_end|>', tokenizer.eos_token]:
            if end and resp.endswith(end): resp = resp[:-len(end)]
        return resp.strip()
    return tokenizer.decode(out[0], skip_special_tokens=True)


def analyze_ek_surprisal(prompt):
    """Forward-only (no generate), collect E[K] + surprisal for prompt tokens."""
    messages = [{"role":"system","content":"你是一个AI助手"},{"role":"user","content":prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, return_tensors='pt')['input_ids']

    # Collect E[K] by temporarily storing in layer attributes
    for layer in model.layers: functional.reset_net(layer)
    functional.reset_net(model.output_neuron)

    with torch.no_grad():
        out = model(ids)

    # Read _ek_min/_ek_max from layers (already stored by forward_parallel)
    layer_ek_means = []
    for layer in model.layers:
        if hasattr(layer, '_ek_min') and hasattr(layer, '_ek_max'):
            layer_ek_means.append((layer._ek_min + layer._ek_max) / 2)

    mean_ek = np.mean(layer_ek_means) if layer_ek_means else 0

    # Surprisal
    log_probs = F.log_softmax(out.logits[0], dim=-1)
    surps = []
    for i in range(ids.shape[1] - 1):
        surps.append(-log_probs[i, ids[0,i+1].item()].item())
    mean_surp = np.mean(surps) if surps else 0

    return mean_ek, mean_surp


# ============================================================
# Test Cases
# ============================================================

test_cases = {
    "Arithmetic": [
        ("1+1等于多少？", "2"),
        ("3+5等于几？", "8"),
        ("10-3等于多少？", "7"),
        ("2乘以4等于多少？", "8"),
        ("100除以5等于多少？", "20"),
        ("12+8等于多少？", "20"),
        ("9乘以9等于多少？", "81"),
        ("15-7等于几？", "8"),
    ],
    "Logic": [
        ("如果小明比小红高，小红比小刚高，那么小明和小刚谁更高？", "小明"),
        ("所有的猫都是动物，小花是猫，小花是动物吗？", "是"),
        ("如果下雨就带伞，今天下雨了，应该怎么做？", "带伞"),
        ("苹果是水果，水果是食物，苹果是食物吗？", "是"),
        ("1、2、3、4、5，下一个数字是什么？", "6"),
        ("北京是中国的首都，中国的首都是哪里？", "北京"),
    ],
    "Commonsense": [
        ("太阳从哪个方向升起？", "东"),
        ("水的化学式是什么？", "H2O"),
        ("一年有多少个月？", "12"),
        ("地球上最大的海洋是什么？", "太平洋"),
        ("人有几只眼睛？", "两/2"),
        ("冰是什么状态的水？", "固"),
        ("彩虹有几种颜色？", "7/七"),
        ("中国最长的河流是什么？", "长江"),
    ],
    "Coherence": [
        ("你好，请问你是谁？", None),
        ("请用一句话介绍人工智能。", None),
        ("今天心情不好，能安慰我一下吗？", None),
        ("帮我写一句生日祝福语。", None),
        ("请解释什么是机器学习。", None),
        ("你能帮我做什么？", None),
    ],
}


# ============================================================
# Run
# ============================================================

print("\n" + "=" * 70, flush=True)
print("REASONING TEST + E[K]/SURPRISAL", flush=True)
print("=" * 70, flush=True)

category_results = {}

for category, cases in test_cases.items():
    print(f"\n--- {category} ({len(cases)} tests) ---", flush=True)
    correct = 0
    total_graded = 0
    eks = []
    surps = []

    for i, (prompt, expected) in enumerate(cases):
        print(f"  [{i+1}/{len(cases)}] {prompt[:30]}...", end=" ", flush=True)

        response = generate_response(prompt, max_new=64)
        ek, surp = analyze_ek_surprisal(prompt)
        eks.append(ek)
        surps.append(surp)

        if expected is not None:
            opts = expected.split('/')
            passed = any(e in response for e in opts)
            if passed: correct += 1
            total_graded += 1
            tag = "PASS" if passed else "FAIL"
        else:
            is_coherent = len(response) > 2 and len(set(response)) > 3
            tag = "OK" if is_coherent else "BAD"

        print(f"[{tag}] E[K]={ek:.1f} Surp={surp:.1f}", flush=True)
        print(f"    A: {response[:100]}", flush=True)

    acc = correct / total_graded * 100 if total_graded > 0 else -1
    category_results[category] = {
        'correct': correct, 'total': total_graded, 'acc': acc,
        'mean_ek': np.mean(eks), 'mean_surp': np.mean(surps),
    }

# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 70, flush=True)
print("SUMMARY", flush=True)
print("=" * 70, flush=True)
print(f"\n{'Category':<15} {'Accuracy':>12} {'Mean E[K]':>10} {'Mean Surp':>10}", flush=True)
print("-" * 50, flush=True)
for cat, r in category_results.items():
    acc_str = f"{r['correct']}/{r['total']} ({r['acc']:.0f}%)" if r['acc'] >= 0 else "qualitative"
    print(f"{cat:<15} {acc_str:>12} {r['mean_ek']:>10.2f} {r['mean_surp']:>10.2f}", flush=True)

print("\nDone.", flush=True)
