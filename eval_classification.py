"""
Zero-shot 文本分类评估：用模型 log-likelihood 排序候选答案。

支持任务：
  - sst2: 情感分类 (positive/negative)
  - mnli: 自然语言推理 (entailment/neutral/contradiction)
  - c3:   中文阅读理解 (多选)

原理：
  对每个候选 label，构造 prompt + label_text，计算 label_text 部分的
  平均 log-likelihood，选最高的作为预测。

用法：
    python eval_classification.py \
        --checkpoint checkpoints/ckpt_step257000 \
        --task sst2 --max_samples 200
"""

import argparse
import math
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from datasets import load_dataset
from model import SNNLanguageModel
from checkpoint_utils import load_config, load_model_weights


def load_model(checkpoint_path, device):
    config = load_config(checkpoint_path)
    model = SNNLanguageModel(
        vocab_size=config.get('vocab_size', 64000),
        D=config.get('D', 1024),
        N=config.get('N', 8),
        K=config.get('K', 12),
        num_layers=config.get('num_layers', 24),
        D_ff=config.get('D_ff', 3072),
    )
    load_model_weights(checkpoint_path, model, device)
    model = model.to(device).eval()
    return model, config


def compute_choice_loglikelihood(model, tokenizer, prompt_text, choice_text, device):
    """计算 choice_text 在 prompt_text 之后的平均 log-likelihood。"""
    full_text = prompt_text + choice_text
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)['input_ids']
    full_ids = tokenizer(full_text, add_special_tokens=False)['input_ids']

    # 加 BOS
    full_ids = [tokenizer.bos_token_id] + full_ids
    prompt_len = len(prompt_ids) + 1  # +1 for BOS

    if len(full_ids) < 2:
        return float('-inf')

    input_ids = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
    target_ids = torch.tensor([full_ids[1:]], dtype=torch.long, device=device)

    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        out = model(input_ids, target_ids)
        # out.last_loss shape: (seq_len,), per-token CE loss
        per_token_loss = out.last_loss

    # choice 部分的 token 位置: prompt_len-1 到末尾
    # (因为 X shift 了 1 位, target 从位置 prompt_len-1 开始是 choice 的第一个 token)
    choice_start = prompt_len - 1
    if choice_start >= len(per_token_loss):
        return float('-inf')

    choice_losses = per_token_loss[choice_start:]
    avg_nll = choice_losses.mean().item()
    return -avg_nll  # 返回 log-likelihood (负的 loss)


def eval_sst2(model, tokenizer, device, max_samples=None):
    ds = load_dataset('glue', 'sst2', split='validation')
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    labels = {0: "negative", 1: "positive"}
    correct = 0
    total = 0

    for i, sample in enumerate(ds):
        sentence = sample['sentence']
        gold = sample['label']

        prompt = f"Review: {sentence}\nSentiment:"

        scores = {}
        for label_id, label_text in labels.items():
            ll = compute_choice_loglikelihood(model, tokenizer, prompt, f" {label_text}", device)
            scores[label_id] = ll

        pred = max(scores, key=scores.get)
        if pred == gold:
            correct += 1
        total += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(ds)}] acc={correct/total:.4f}")

    acc = correct / total
    print(f"\nSST-2 Accuracy: {acc:.4f} ({correct}/{total})")
    return acc


def eval_mnli(model, tokenizer, device, max_samples=None):
    ds = load_dataset('nyu-mll/multi_nli', split='validation_matched')
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    labels = {0: "Yes", 1: "Maybe", 2: "No"}
    correct = 0
    total = 0

    for i, sample in enumerate(ds):
        premise = sample['premise']
        hypothesis = sample['hypothesis']
        gold = sample['label']

        prompt = f"{premise}\nQuestion: {hypothesis}\nAnswer:"

        scores = {}
        for label_id, label_text in labels.items():
            ll = compute_choice_loglikelihood(model, tokenizer, prompt, f" {label_text}", device)
            scores[label_id] = ll

        pred = max(scores, key=scores.get)
        if pred == gold:
            correct += 1
        total += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(ds)}] acc={correct/total:.4f}")

    acc = correct / total
    print(f"\nMNLI Accuracy: {acc:.4f} ({correct}/{total})")
    return acc


def eval_c3(model, tokenizer, device, max_samples=None):
    ds = load_dataset('clue', 'c3', split='validation')
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    correct = 0
    total = 0

    for i, sample in enumerate(ds):
        context = ''.join(sample['context'])
        question = sample['question']
        choices = sample['choice']
        gold_answer = sample['answer']
        gold = choices.index(gold_answer) if gold_answer in choices else 0

        prompt = f"{context}\n问题：{question}\n答案："

        scores = []
        for choice_text in choices:
            ll = compute_choice_loglikelihood(model, tokenizer, prompt, choice_text, device)
            scores.append(ll)

        pred = scores.index(max(scores))
        if pred == gold:
            correct += 1
        total += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(ds)}] acc={correct/total:.4f}")

    acc = correct / total
    print(f"\nC3 Accuracy: {acc:.4f} ({correct}/{total})")
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot Classification Evaluation")
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--tokenizer_path', type=str, default='./tokenizer/')
    parser.add_argument('--task', type=str, required=True, choices=['sst2', 'mnli', 'c3', 'all'])
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--device', type=str,
                        default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model, config = load_model(args.checkpoint, args.device)
    print(f"Model loaded: D={config.get('D')}, L={config.get('num_layers')}, vocab={config.get('vocab_size')}")

    tasks = ['sst2', 'mnli', 'c3'] if args.task == 'all' else [args.task]
    results = {}

    for task in tasks:
        print(f"\n{'='*50}")
        print(f"Evaluating: {task.upper()}")
        print(f"{'='*50}")

        if task == 'sst2':
            results[task] = eval_sst2(model, tokenizer, args.device, args.max_samples)
        elif task == 'mnli':
            results[task] = eval_mnli(model, tokenizer, args.device, args.max_samples)
        elif task == 'c3':
            results[task] = eval_c3(model, tokenizer, args.device, args.max_samples)

    print(f"\n{'='*50}")
    print("Results Summary")
    print(f"{'='*50}")
    random_baseline = {'sst2': 0.5, 'mnli': 0.333, 'c3': 0.25}
    for task, acc in results.items():
        print(f"  {task.upper():>6s}: {acc:.4f}  (random baseline: {random_baseline.get(task, '?')})")

    # 保存结果到 exp/
    import json, os, datetime
    os.makedirs('exp', exist_ok=True)
    ckpt_name = os.path.basename(args.checkpoint.rstrip('/'))
    out = {
        'checkpoint': args.checkpoint,
        'checkpoint_name': ckpt_name,
        'timestamp': datetime.datetime.now().isoformat(),
        'max_samples': args.max_samples,
        'results': {t: {'accuracy': a, 'random_baseline': random_baseline.get(t)} for t, a in results.items()},
        'model_config': config,
    }
    task_suffix = args.task if args.task != 'all' else 'all'
    save_path = f'exp/classification_{task_suffix}_{ckpt_name}.json'
    with open(save_path, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {save_path}")
