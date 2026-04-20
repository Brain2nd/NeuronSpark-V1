"""Phase A baseline eval: wikitext perplexity + blimp grammaticality + sst2/mnli classification + xnli_zh.

用 lm-eval 的 hf backend 跑, 对比 NS 在 Phase A 任务上的表现.
覆盖 proposal 里的 LM / BLiMP / Classification 评测维度.
"""
import argparse
import datetime
import gc
import json
import os
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
    "mamba-790m": "state-spaces/mamba-790m-hf",
    "mamba-1.4b": "state-spaces/mamba-1.4b-hf",
}

# Phase A 任务: wikitext ppl, blimp (67 subtasks), sst2, mnli, xnli_zh
PHASE_A_TASKS = ["wikitext", "blimp", "sst2", "mnli", "xnli_zh"]


def run_eval(model_name, hf_id, tasks, device='cuda:0'):
    print(f"\n{'='*60}\nEvaluating {model_name} ({hf_id}) on {tasks}\n{'='*60}")
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={hf_id},dtype=bfloat16,trust_remote_code=True",
        tasks=tasks,
        batch_size=1,
        device=device,
    )

    summary = {}
    for task, res in results["results"].items():
        summary[task] = {k: v for k, v in res.items() if not k.startswith("samples")}

    for task, metrics in summary.items():
        if task.startswith(('blimp_nl',)):  # 省略 blimp 子任务个别打印
            continue
        acc = metrics.get("acc,none", metrics.get("acc", "?"))
        acc_norm = metrics.get("acc_norm,none", "")
        ppl = metrics.get("word_perplexity,none", metrics.get("perplexity,none", ""))
        extra = []
        if isinstance(acc_norm, float): extra.append(f"norm={acc_norm:.4f}")
        if isinstance(ppl, float): extra.append(f"ppl={ppl:.2f}")
        extra_str = " " + " ".join(extra) if extra else ""
        acc_str = f"{acc:.4f}" if isinstance(acc, float) else str(acc)
        print(f"  {task:>30s}: acc={acc_str}{extra_str}")

    os.makedirs("exp", exist_ok=True)
    save_path = f"exp/lm_eval_baseline_{model_name}_phaseA.json"
    out = {
        "model": model_name,
        "hf_id": hf_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "tasks": tasks,
        "results": summary,
    }
    with open(save_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  Saved to {save_path}")

    del results
    gc.collect()
    torch.cuda.empty_cache()
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tasks", nargs="+", default=PHASE_A_TASKS)
    args = parser.parse_args()

    selected = {k: MODELS[k] for k in (args.models or MODELS.keys()) if k in MODELS}
    for name, hf_id in selected.items():
        try:
            run_eval(name, hf_id, args.tasks, device=args.device)
        except Exception as e:
            print(f"  ERROR: {name} failed: {type(e).__name__}: {e}")
