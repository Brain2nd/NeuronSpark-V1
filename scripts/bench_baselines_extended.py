"""扩展基线测评: 补齐 mmlu/piqa/openbookqa/lambada_openai/ceval-valid 5 项.

与 bench_baselines.py 的差异:
  - --tasks 可选子集 (默认扩展 5 项, 可和原 5 项叠加)
  - --device 指定 cuda:N (多卡并行运行)
  - --models 过滤要跑的模型
  - 输出 exp/lm_eval_baseline_<model>_ext.json, 后续可 merge

用法 (并行 2 卡):
    CUDA_VISIBLE_DEVICES=2 python scripts/bench_baselines_extended.py \
        --tasks openbookqa piqa lambada_openai --device cuda:0
    CUDA_VISIBLE_DEVICES=3 python scripts/bench_baselines_extended.py \
        --tasks ceval-valid mmlu --device cuda:0
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

DEFAULT_EXT_TASKS = ["openbookqa", "piqa", "lambada_openai", "ceval-valid", "mmlu"]


def run_eval(model_name, hf_id, tasks, device, tag):
    print(f"\n{'=' * 60}")
    print(f"[{device}] Evaluating {model_name} ({hf_id}) on {tasks}")
    print(f"{'=' * 60}")

    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={hf_id},dtype=bfloat16,trust_remote_code=True,device={device}",
        tasks=tasks,
        batch_size=1,
    )

    summary = {
        t: {k: v for k, v in r.items() if not k.startswith("samples")}
        for t, r in results["results"].items()
    }

    print(f"\n--- {model_name} Results on {device} ---")
    for task, metrics in summary.items():
        acc = metrics.get("acc,none", metrics.get("acc", "?"))
        acc_norm = metrics.get("acc_norm,none", "")
        norm_str = f" (norm: {acc_norm:.4f})" if isinstance(acc_norm, float) else ""
        if isinstance(acc, float):
            print(f"  {task:>20s}: {acc:.4f}{norm_str}")

    os.makedirs("exp", exist_ok=True)
    save_path = f"exp/lm_eval_baseline_{model_name}_{tag}.json"
    out = {
        "model": model_name,
        "hf_id": hf_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "tasks": tasks,
        "device": device,
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
    parser.add_argument("--models", nargs="+", default=None,
                        help=f"要测的模型 (可选: {', '.join(MODELS.keys())})")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_EXT_TASKS,
                        help=f"任务列表, 默认 {DEFAULT_EXT_TASKS}")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--tag", type=str, default="ext",
                        help="输出文件后缀, 区分多次运行")
    args = parser.parse_args()

    selected = {k: MODELS[k] for k in (args.models or MODELS.keys()) if k in MODELS}

    for name, hf_id in selected.items():
        try:
            run_eval(name, hf_id, args.tasks, args.device, args.tag)
        except Exception as e:
            print(f"  ERROR: {name} failed: {type(e).__name__}: {e}")
