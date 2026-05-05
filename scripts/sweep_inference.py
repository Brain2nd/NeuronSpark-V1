"""推理超参 sweep: load once, scan temperature × top_p × prompt."""
from __future__ import annotations
import argparse, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    ("EN-fact",   "The capital of France is"),
    ("EN-narr",   "Once upon a time, in a small village,"),
    ("ZH-fact",   "中华人民共和国的首都是"),
    ("ZH-narr",   "今天天气不错，我想"),
    ("CODE",      "def fibonacci(n):"),
    ("MATH",      "Question: 7 * 8 = ?\nAnswer:"),
]

CONFIGS = [
    # (label, temp, top_p, top_k, rep_pen)
    ("greedy",         0.0, 1.00,  0, 1.0),
    ("low T=0.3",      0.3, 0.95, 50, 1.1),
    ("mid T=0.7",      0.7, 0.95, 50, 1.1),
    ("std T=0.8",      0.8, 0.95, 50, 1.1),
    ("high T=1.0",     1.0, 0.95, 50, 1.1),
    ("nucleus only",   0.8, 0.90,  0, 1.1),
    ("no rep_pen",     0.8, 0.95, 50, 1.0),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints_hf_v3_step84000")
    ap.add_argument("--max_new_tokens", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_cache", action="store_true",
                    help="走 generate_cached 路径 (stateful K-frame decode)")
    args = ap.parse_args()

    print(f"loading {args.checkpoint}", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, trust_remote_code=True
    ).cuda().eval()
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    for plabel, prompt in PROMPTS:
        print(f"\n{'='*78}\n[{plabel}] {prompt!r}\n{'='*78}", flush=True)
        ids = tok(prompt, return_tensors="pt").input_ids.cuda()
        for clabel, T, top_p, top_k, rep_pen in CONFIGS:
            torch.manual_seed(args.seed)
            t0 = time.time()
            if args.use_cache:
                out = model.generate_cached(
                    input_ids=ids,
                    max_new_tokens=args.max_new_tokens,
                    temperature=T,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=rep_pen,
                )
            else:
                out = model.generate(
                    input_ids=ids,
                    max_new_tokens=args.max_new_tokens,
                    temperature=T,
                    top_k=top_k if top_k > 0 else None,
                    top_p=top_p,
                    repetition_penalty=rep_pen,
                    do_sample=(T > 0),
                )
            dt = time.time() - t0
            new = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            print(f"\n  -- {clabel} (T={T} top_p={top_p} top_k={top_k} rp={rep_pen}) [{dt:.1f}s] --")
            print(f"  {new}", flush=True)


if __name__ == "__main__":
    main()
