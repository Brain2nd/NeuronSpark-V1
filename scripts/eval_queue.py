"""按任务请求量从小到大顺序排队跑 lm-eval, 每任务独立写 json.

复用 scripts/eval_full.py 里的 NeuronSparkLM 注册和 run_eval. 任务按预估 request 数排:
  sst2(872) → openbookqa(500) → arc_challenge(1k) → winogrande(1.3k) →
  ceval-valid(1.4k) → piqa(1.8k) → arc_easy(2k) → boolq(3k) → wikitext(slow) →
  xnli_zh(5k) → hellaswag(10k) → mnli(10k) → mmlu(14k)

每任务一个 json: exp/v3_step84000/{task}.json
跑完合并到 exp/v3_step84000/_summary.json
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 触发 NeuronSparkLM 注册
from scripts.eval_full import run_eval  # noqa: F401

TASKS = [
    ("openbookqa", 500),
    ("sst2", 872),
    ("arc_challenge", 1000),
    ("winogrande", 1300),
    ("ceval-valid", 1400),
    ("piqa", 1800),
    ("arc_easy", 2000),
    ("boolq", 3000),
    ("wikitext", 0),         # 量小但 loglikelihood_rolling 慢
    ("xnli_zh", 5000),
    ("hellaswag", 10000),
    ("mnli", 10000),
    ("mmlu", 14000),
]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints_hf_v3_step84000")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out_dir", default="exp/v3_step84000")
    ap.add_argument("--skip_done", action="store_true",
                    help="已存在结果文件就跳过")
    ap.add_argument("--apply_chat_template", action="store_true")
    ap.add_argument("--enable_thinking", action="store_true",
                    help="chat_template 时启用 think 模式")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    summary = {"checkpoint": args.checkpoint, "device": args.device,
               "results": {}, "timings": {}}
    summary_path = os.path.join(args.out_dir, "_summary.json")

    t_total = time.time()
    for task, est_n in TASKS:
        out = os.path.join(args.out_dir, f"{task}.json")
        if args.skip_done and os.path.exists(out):
            print(f"\n[skip] {task} (exists)\n", flush=True)
            with open(out) as f:
                d = json.load(f)
            summary["results"].update(d.get("results", {}))
            continue
        print(f"\n{'='*60}\n[run] {task}  (≈{est_n} reqs)\n{'='*60}", flush=True)
        t0 = time.time()
        try:
            res = run_eval(args.checkpoint, args.device, [task], out,
                           apply_chat_template=args.apply_chat_template,
                           system_instruction=None,
                           enable_thinking=args.enable_thinking)
            dt = time.time() - t0
            summary["timings"][task] = dt
            summary["results"].update(res.get("results", {}))
            print(f"[done] {task} in {dt:.1f}s", flush=True)
        except Exception as e:
            print(f"[FAIL] {task}: {type(e).__name__}: {e}", flush=True)
            summary["timings"][task] = -1
        # 增量保存 summary
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

    print(f"\nTOTAL: {time.time()-t_total:.0f}s", flush=True)
    print(f"Summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
