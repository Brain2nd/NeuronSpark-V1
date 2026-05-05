"""Sweep think-mode 超参数, 找能让 pretrain ckpt 给出正确答案的配置."""
from __future__ import annotations
import argparse, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    ("zh-fact-capital", "中华人民共和国的首都是哪？", ["北京", "Beijing"]),
    ("en-fact-capital", "What is the capital of France?", ["Paris"]),
    ("math-easy", "1 + 1 = ?", ["2"]),
    ("math-mid", "7 * 8 = ?", ["56"]),
]

CONFIGS = [
    # (label, T, top_p, top_k, rep_pen)  -- 精选, 上次 sweep 表现较好的
    ("T0.7 p0.9",       0.7, 0.90, 50, 1.1),
    ("T0.6 p0.95 rp1.2", 0.6, 0.95, 50, 1.2),
    ("T0.5 p0.95",      0.5, 0.95, 50, 1.1),
    ("T0.8 p0.95",      0.8, 0.95, 50, 1.1),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints_hf_v3_step108000")
    ap.add_argument("--max_new_tokens", type=int, default=2048,
                    help="give think enough room to complete + answer")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--enable_thinking", default="True", choices=["True", "False"])
    ap.add_argument("--use_cache", action="store_true",
                    help="走 generate_cached (fast)")
    args = ap.parse_args()

    print(f"loading {args.checkpoint}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, trust_remote_code=True
    ).cuda().eval()
    et = (args.enable_thinking == "True")
    print(f"enable_thinking={et}", flush=True)

    for plabel, prompt, keywords in PROMPTS:
        print(f"\n{'='*78}\n[{plabel}] {prompt}  (期望命中: {keywords})\n{'='*78}", flush=True)
        msgs = [{"role":"user","content":prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=et)
        ids = tok(text, return_tensors="pt").input_ids.cuda()
        for clabel, T, top_p, top_k, rep_pen in CONFIGS:
            torch.manual_seed(args.seed)
            if args.use_cache:
                out = model.generate_cached(
                    input_ids=ids,
                    max_new_tokens=args.max_new_tokens,
                    temperature=T if T > 0 else 0,
                    top_k=top_k if top_k > 0 else 0,
                    top_p=top_p,
                    repetition_penalty=rep_pen,
                )
            else:
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
            new = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False)
            # 抽 final answer (after </think>)
            if "</think>" in new:
                think_part = new.split("</think>")[0].replace("<think>", "").strip()
                answer_part = new.split("</think>", 1)[1].strip()
            else:
                think_part = "<no think tag>"
                answer_part = new.strip()
            hit = any(kw.lower() in answer_part.lower() for kw in keywords)
            mark = "✓" if hit else "✗"
            print(f"\n  -- {clabel} -- {mark}")
            print(f"  think({len(think_part)} chars): {think_part[:150]!r}{'...' if len(think_part)>150 else ''}")
            print(f"  answer({len(answer_part)} chars): {answer_part[:300]!r}")


if __name__ == "__main__":
    main()
