"""Pretrain 模式 (raw text continuation, 无 chat template) 测事实/数学准确率.

对比 chat+think 模式. 完整保存到 args.out_log.
"""
from __future__ import annotations
import argparse, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    ("en-fact", "The capital of France is", ["Paris"]),
    ("zh-fact", "中华人民共和国的首都是", ["北京"]),
    ("math-easy", "1 + 1 =", ["2"]),
    ("math-mid", "7 * 8 =", ["56"]),
    ("en-qa", "Question: What is the capital of France?\nAnswer:", ["Paris"]),
    ("zh-qa", "问：中华人民共和国的首都是哪？\n答：", ["北京"]),
    ("math-qa", "Question: 7 * 8 = ?\nAnswer:", ["56"]),
]

CONFIGS = [
    ("greedy",         0.0, 1.00,  0, 1.0),
    ("T0.7 p0.9",      0.7, 0.90, 50, 1.1),
    ("T0.6 p0.95 rp1.2", 0.6, 0.95, 50, 1.2),
    ("T0.5 p0.95",     0.5, 0.95, 50, 1.1),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints_hf_v3_step108000")
    ap.add_argument("--max_new_tokens", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_log", default="exp/pretrain_mode_test_step108000.log")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, trust_remote_code=True).cuda().eval()

    import os
    os.makedirs(os.path.dirname(args.out_log) or ".", exist_ok=True)

    lines = []
    def w(s):
        lines.append(s)
        print(s, flush=True)

    w(f"checkpoint: {args.checkpoint}")
    w(f"mode: raw pretrain (NO chat template, NO <|im_start|>)")
    w(f"max_new_tokens: {args.max_new_tokens}")
    w("=" * 78)

    total_hit = 0
    total_runs = 0
    for plabel, prompt, keywords in PROMPTS:
        w(f"\n[{plabel}] prompt: {prompt!r}  (期望: {keywords})")
        ids = tok(prompt, return_tensors="pt").input_ids.cuda()
        for clabel, T, top_p, top_k, rep_pen in CONFIGS:
            torch.manual_seed(args.seed)
            try:
                out = model.generate_cached(
                    input_ids=ids,
                    max_new_tokens=args.max_new_tokens,
                    temperature=T if T > 0 else 0,
                    top_k=top_k if top_k > 0 else 0,
                    top_p=top_p,
                    repetition_penalty=rep_pen,
                )
            except Exception:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = model.generate(
                        input_ids=ids,
                        max_new_tokens=args.max_new_tokens,
                        temperature=T if T > 0 else 1.0,
                        top_k=top_k if top_k > 0 else None,
                        top_p=top_p,
                        repetition_penalty=rep_pen,
                        do_sample=(T > 0),
                    )
            new = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            hit = any(kw.lower() in new.lower() for kw in keywords)
            mark = "✓" if hit else "✗"
            total_hit += int(hit)
            total_runs += 1
            w(f"\n  -- {clabel} -- {mark}")
            w(f"  续写: {new!r}")
    w("\n" + "=" * 78)
    w(f"SUMMARY: {total_hit}/{total_runs} hits ({total_hit/total_runs*100:.1f}%)")

    with open(args.out_log, "w") as f:
        f.write("\n".join(lines))
    print(f"\nlog -> {args.out_log}", flush=True)


if __name__ == "__main__":
    main()
