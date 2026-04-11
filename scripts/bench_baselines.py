"""
批量测评基线模型：用 lm-eval-harness 统一流程。

测评任务: arc_easy, arc_challenge, hellaswag, winogrande, boolq (zero-shot)
测评模型: Pythia-1B, OPT-1.3B, TinyLlama-1.1B, Qwen2-0.5B, Qwen2-1.5B,
          Qwen2.5-0.5B, Qwen2.5-1.5B, Qwen3-0.6B, Qwen3-1.7B

用法:
    python scripts/bench_baselines.py
    python scripts/bench_baselines.py --models pythia-1b tinyllama  # 只跑指定模型
"""

import argparse
import json
import os
import datetime
import gc
import torch
import lm_eval

MODELS = {
    "pythia-1b": "EleutherAI/pythia-1b",
    "opt-1.3b": "facebook/opt-1.3b",
    "tinyllama-1.1b": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    "qwen2-0.5b": "Qwen/Qwen2-0.5B",
    "qwen2-1.5b": "Qwen/Qwen2-1.5B",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B-Base",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B-Base",
}

TASKS = ["arc_easy", "arc_challenge", "hellaswag", "winogrande", "boolq"]


def run_eval(model_name, hf_id):
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name} ({hf_id})")
    print(f"{'='*60}")

    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={hf_id},dtype=bfloat16,trust_remote_code=True",
        tasks=TASKS,
        batch_size=1,
    )

    # 提取关键指标
    summary = {}
    for task, res in results["results"].items():
        summary[task] = {k: v for k, v in res.items()
                         if not k.startswith("samples")}

    # 打印
    print(f"\n--- {model_name} Results ---")
    for task, metrics in summary.items():
        acc = metrics.get("acc,none", metrics.get("acc", "?"))
        acc_norm = metrics.get("acc_norm,none", "")
        norm_str = f" (norm: {acc_norm:.4f})" if isinstance(acc_norm, float) else ""
        print(f"  {task:>16s}: {acc:.4f}{norm_str}")

    # 保存
    os.makedirs("exp", exist_ok=True)
    save_path = f"exp/lm_eval_baseline_{model_name}.json"
    out = {
        "model": model_name,
        "hf_id": hf_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "tasks": TASKS,
        "results": summary,
    }
    with open(save_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  Saved to {save_path}")

    # 释放显存
    del results
    gc.collect()
    torch.cuda.empty_cache()

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None,
                        help=f"要测的模型 (可选: {', '.join(MODELS.keys())})")
    args = parser.parse_args()

    if args.models:
        selected = {k: MODELS[k] for k in args.models if k in MODELS}
    else:
        selected = MODELS

    all_results = {}
    for name, hf_id in selected.items():
        try:
            all_results[name] = run_eval(name, hf_id)
        except Exception as e:
            print(f"  ERROR: {name} failed: {e}")
            all_results[name] = {"error": str(e)}

    # 最终汇总表
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    header = f"{'Model':>20s}"
    for t in TASKS:
        header += f" {t:>14s}"
    print(header)
    print("-" * len(header))

    for name, res in all_results.items():
        if "error" in res:
            print(f"{name:>20s}  ERROR: {res['error'][:40]}")
            continue
        row = f"{name:>20s}"
        for t in TASKS:
            acc = res.get(t, {}).get("acc,none", "?")
            row += f" {acc:>14.4f}" if isinstance(acc, float) else f" {'?':>14s}"
        print(row)
