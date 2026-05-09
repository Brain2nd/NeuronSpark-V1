"""准备 RL (GRPO) 训练数据.

每条 sample:
  - prompt:  ChatML-wrapped 用户问题 (enable_thinking=True 留空 assistant prompt 让模型自己 think)
  - answer:  GT 答案 (用于 reward 计算, 提 \\boxed{...} 比对)
  - question / level / subject / source: 元信息

数据源:
  数学:
    - gsm8k       (7.5k 应用题, 答案纯数字)
    - math        (Hendrycks competition_math 7.5k, latex \\boxed{...} 答案)
  通用推理 (RLVR-friendly: 推理有杠杆 + 答案唯一可机器验证):
    - bbh         (Big-Bench Hard 23 sub-task, 逻辑/符号/计数/语义推理, 答案多为字母 / 数字 / yes-no)
    - mmlu_pro    (12k 训练 + 12k 测试, 10-choice 难 STEM/经济/法律 MC, 转 \\boxed{字母})
    - gpqa        (~448 graduate sci, 4-choice, 转 \\boxed{字母}; 可能需要 HF 登录)

为了 boxed reward 的统一, MC 任务的 question 末尾会附加 "Please put your final answer
within \\boxed{}." 提示, 答案统一为字母 (A/B/...). 数学任务保持原始 latex 答案.

输出: HF Arrow dir at data/rl_<source>/

Run:
    python scripts/prep_rl_data.py --source gsm8k     --out data/rl_gsm8k
    python scripts/prep_rl_data.py --source math      --out data/rl_math
    python scripts/prep_rl_data.py --source bbh       --out data/rl_bbh
    python scripts/prep_rl_data.py --source mmlu_pro  --out data/rl_mmlu_pro
    python scripts/prep_rl_data.py --source gpqa      --out data/rl_gpqa
"""
from __future__ import annotations
import argparse
import re
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoTokenizer


def _wrap_prompt(tokenizer, user_content: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False, add_generation_prompt=True,
        enable_thinking=True,
    )


# ----------------------------- math -----------------------------

def prep_gsm8k(tokenizer):
    ds = load_dataset("openai/gsm8k", "main", split="train")
    out = []
    for s in ds:
        q = s["question"]
        gt_full = s["answer"]
        if "####" not in gt_full:
            continue
        gt = gt_full.split("####")[-1].strip().replace(",", "")
        out.append({
            "prompt": _wrap_prompt(tokenizer, q),
            "answer": gt, "question": q, "source": "gsm8k",
        })
    return out


def prep_math(tokenizer):
    SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                "number_theory", "prealgebra", "precalculus"]
    parts = [load_dataset("EleutherAI/hendrycks_math", subj, split="train") for subj in SUBJECTS]
    ds = concatenate_datasets(parts)
    out = []
    for s in ds:
        q = s["problem"]
        sol = s.get("solution", "")
        m = re.search(r"\\boxed\{(.+?)\}", sol)
        if not m:
            continue
        gt = m.group(1).strip()
        out.append({
            "prompt": _wrap_prompt(tokenizer, q),
            "answer": gt, "question": q, "source": "math",
            "level": s.get("level", ""), "subject": s.get("type", ""),
        })
    return out


# ------------------------ general reasoning ------------------------

# BBH 23 sub-tasks (子集精选: 推理有杠杆 + 答案唯一)
BBH_SUBTASKS = [
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_three_objects",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "movie_recommendation",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_three_objects",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "web_of_lies",
    "word_sorting",
]


def prep_bbh(tokenizer):
    """lukaemon/bbh: 每个 subtask ~250 例.
    格式: 每行 (input, target). target 多为字母/数字/字符串.
    我们把 target 直接作为 GT, 提示模型用 \\boxed{} 包答案.
    """
    out = []
    suffix = "\n\nPlease reason step by step and put your final answer within \\boxed{}."
    for sub in BBH_SUBTASKS:
        try:
            ds = load_dataset("lukaemon/bbh", sub, split="test")
        except Exception as e:
            print(f"  [skip] {sub}: {e}")
            continue
        for s in ds:
            q = s["input"]
            gt = str(s["target"]).strip()
            # 部分 BBH target 形如 "(A)" 或 "(Yes)", 去掉外括号便于 boxed match
            if gt.startswith("(") and gt.endswith(")"):
                gt = gt[1:-1].strip()
            out.append({
                "prompt": _wrap_prompt(tokenizer, q + suffix),
                "answer": gt, "question": q,
                "source": "bbh", "subject": sub,
            })
    return out


def prep_mmlu_pro(tokenizer):
    """TIGER-Lab/MMLU-Pro: 10-choice 难 STEM/经济/法律 MC.
    train split 用作 RL prompts (~12k). 答案为 0-9 整数 → 转字母 A-J.
    """
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="validation")  # 1130 train-equivalent
    out = []
    suffix = "\n\nPlease reason step by step and put your final answer letter within \\boxed{}."
    for s in ds:
        q = s["question"]
        opts = s["options"]
        ans_idx = s["answer_index"]
        if ans_idx is None or ans_idx < 0 or ans_idx >= len(opts):
            continue
        gt = chr(ord("A") + ans_idx)
        opt_block = "\n".join(f"{chr(ord('A') + i)}. {o}" for i, o in enumerate(opts))
        full_q = f"{q}\n\n{opt_block}"
        out.append({
            "prompt": _wrap_prompt(tokenizer, full_q + suffix),
            "answer": gt, "question": q,
            "source": "mmlu_pro", "subject": s.get("category", ""),
        })
    return out


def prep_gpqa(tokenizer):
    """Idavidrein/gpqa diamond split — 198 graduate sci 难题, 4-choice.
    HF 上是 gated dataset, 需要 HF token + 同意 license.
    答案为 0-3 整数 → 转字母 A-D.
    """
    # 注意: 'gpqa_diamond' subset, train=198 全用. 用 main 名字也行.
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    out = []
    suffix = "\n\nPlease reason step by step and put your final answer letter within \\boxed{}."
    for s in ds:
        q = s["Question"]
        # GPQA 字段: Correct Answer / Incorrect Answer 1-3, 需要打乱后选位置
        choices = [s["Correct Answer"]] + [s[f"Incorrect Answer {i}"] for i in range(1, 4)]
        # 用 question hash 固定打乱顺序 (可复现)
        import hashlib
        seed = int(hashlib.md5(q.encode()).hexdigest(), 16) % (2**31)
        import random
        rng = random.Random(seed)
        order = list(range(4))
        rng.shuffle(order)
        shuffled = [choices[i] for i in order]
        ans_idx = order.index(0)  # 0 是 Correct Answer 的原始位置
        gt = chr(ord("A") + ans_idx)
        opt_block = "\n".join(f"{chr(ord('A') + i)}. {o}" for i, o in enumerate(shuffled))
        full_q = f"{q}\n\n{opt_block}"
        out.append({
            "prompt": _wrap_prompt(tokenizer, full_q + suffix),
            "answer": gt, "question": q, "source": "gpqa",
        })
    return out


# ------------------------ entry ------------------------

PREP_FNS = {
    "gsm8k": prep_gsm8k,
    "math": prep_math,
    "bbh": prep_bbh,
    "mmlu_pro": prep_mmlu_pro,
    "gpqa": prep_gpqa,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=list(PREP_FNS.keys()))
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="tokenizer_v3")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    rows = PREP_FNS[args.source](tok)
    print(f"prepared {len(rows)} samples from {args.source}", flush=True)

    ds = Dataset.from_list(rows)
    ds.save_to_disk(args.out)
    print(f"saved -> {args.out}", flush=True)
    print(f"\nsample[0] preview:")
    print(f"  prompt[-200:]: ...{rows[0]['prompt'][-200:]!r}")
    print(f"  answer: {rows[0]['answer']!r}")


if __name__ == "__main__":
    main()
