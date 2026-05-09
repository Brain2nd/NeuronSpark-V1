"""准备 RL (GRPO) 训练数据.

每条 sample:
  - prompt: ChatML-wrapped 用户问题 (含 enable_thinking=True 触发 think 模式)
  - answer: GT 答案 (用于 reward 计算, 抽 \\boxed{...} 比对)

数据源:
  - gsm8k (7.5k 数学应用题, 答案为整数/小数)
  - MATH (12.5k 难题, latex 答案)
  - 可选: OpenR1-Math (大规模)

输出: HF Arrow dir at data/rl_<source>/

Run:
    python scripts/prep_rl_data.py --source gsm8k --out data/rl_gsm8k
    python scripts/prep_rl_data.py --source math --out data/rl_math
"""
from __future__ import annotations
import argparse
import re
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer


def prep_gsm8k(tokenizer):
    ds = load_dataset("openai/gsm8k", "main", split="train")
    out = []
    for s in ds:
        q = s["question"]
        # GT 答案在 #### 后面
        gt_full = s["answer"]
        if "####" not in gt_full:
            continue
        gt = gt_full.split("####")[-1].strip().replace(",", "")
        # ChatML prompt with think mode
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": q}],
            tokenize=False, add_generation_prompt=True,
            enable_thinking=True,  # 显式开 think
        )
        out.append({"prompt": prompt, "answer": gt, "question": q})
    return out


def prep_math(tokenizer):
    # EleutherAI mirror of Hendrycks competition_math (same content, no gated download)
    # Subjects come as 7 separate splits; concat them.
    from datasets import concatenate_datasets
    SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                "number_theory", "prealgebra", "precalculus"]
    parts = [load_dataset("EleutherAI/hendrycks_math", subj, split="train") for subj in SUBJECTS]
    ds = concatenate_datasets(parts)
    out = []
    for s in ds:
        q = s["problem"]
        sol = s.get("solution", "")
        # 抽 \boxed{...} 内容
        m = re.search(r"\\boxed\{(.+?)\}", sol)
        if not m:
            continue
        gt = m.group(1).strip()
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": q}],
            tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )
        out.append({"prompt": prompt, "answer": gt, "question": q,
                    "level": s.get("level", ""), "subject": s.get("type", "")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["gsm8k", "math"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="tokenizer_v3")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if args.source == "gsm8k":
        rows = prep_gsm8k(tok)
    elif args.source == "math":
        rows = prep_math(tok)
    print(f"prepared {len(rows)} samples from {args.source}", flush=True)

    ds = Dataset.from_list(rows)
    ds.save_to_disk(args.out)
    print(f"saved -> {args.out}", flush=True)
    print(f"\nsample[0] preview:")
    print(f"  prompt[-200:]: ...{rows[0]['prompt'][-200:]!r}")
    print(f"  answer: {rows[0]['answer']!r}")


if __name__ == "__main__":
    main()
