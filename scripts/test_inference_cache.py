"""验证 cached generation 与 HF 无 cache 路径在 greedy 下的输出等价。

第一性原则：前向计算正确 > 速度。所以先验证：
  1. greedy 解码逐 token 完全相同（或在 bf16 数值容差内 ≥95% 重合）
  2. logits 数值最大绝对差 < 阈值
  3. 速度对比作为附带信息
"""
from __future__ import annotations
import argparse, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "The capital of France is",
    "Once upon a time, in a small village,",
    "中华人民共和国的首都是",
    "In a series of papers published between 1905 and 1915, Albert Einstein laid the foundations of modern physics with his theories of special and general relativity, fundamentally changing how we understand space, time, and gravity. His work,",
]


def hf_logits_each_step(model, ids, n_steps):
    """每步全序列重算, 返回每步 (last token) logits 列表."""
    cur = ids.clone()
    logits_list = []
    for _ in range(n_steps):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = model(cur)
        last = out.logits[:, -1, :].float().clone()
        logits_list.append(last)
        next_tok = last.argmax(dim=-1, keepdim=True)
        cur = torch.cat([cur, next_tok], dim=1)
    return cur, logits_list


def cached_logits_each_step(model, ids, n_steps):
    """走 cached path 一次拿全部, 但同时记录每步 logits."""
    # 我们手动复刻 SNNLanguageModel.generate, 但记录 logits.
    # functional 是 trust_remote_code 模块里的, 通过 model 的 module 取
    functional = type(model.snn).__module__
    import sys as _sys
    functional = _sys.modules[functional].functional
    snn = model.snn
    for layer_module in snn.layers:
        functional.reset_net(layer_module)
    functional.reset_net(snn.output_neuron)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        # prefill
        h_seq = snn.encode(ids)
        h = h_seq
        for layer_module in snn.layers:
            h, _ = layer_module.forward_parallel(h)
        prompt_len = ids.shape[1]
        logits_full = snn.decode(h, prompt_len)
    last = logits_full[:, -1, :].float().clone()
    logits_list = [last]
    cur = ids.clone()
    next_tok = last.argmax(dim=-1, keepdim=True)
    cur = torch.cat([cur, next_tok], dim=1)

    for _ in range(n_steps - 1):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            frames = snn.encode(next_tok)
            h = frames
            for layer_module in snn.layers:
                h, _ = layer_module.forward_parallel(h)
            logits = snn.decode(h, 1)
        last = logits[:, -1, :].float().clone()
        logits_list.append(last)
        next_tok = last.argmax(dim=-1, keepdim=True)
        cur = torch.cat([cur, next_tok], dim=1)
    return cur, logits_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints_hf_v3_step84000")
    ap.add_argument("--n_steps", type=int, default=20)
    args = ap.parse_args()

    print(f"loading {args.checkpoint}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, trust_remote_code=True
    ).cuda().eval()
    print("loaded", flush=True)

    for prompt in PROMPTS:
        print(f"\n{'='*78}\nPROMPT: {prompt!r}\n{'='*78}", flush=True)
        ids = tok(prompt, return_tensors="pt").input_ids.cuda()

        # No-cache (HF re-run每步)
        t0 = time.time()
        out_nocache, logits_nocache = hf_logits_each_step(model, ids, args.n_steps)
        t_nocache = time.time() - t0

        # Cached
        t0 = time.time()
        out_cached, logits_cached = cached_logits_each_step(model, ids, args.n_steps)
        t_cached = time.time() - t0

        # Token 对比
        new_nc = out_nocache[0, ids.shape[1]:].tolist()
        new_c  = out_cached[0, ids.shape[1]:].tolist()
        match = sum(1 for a, b in zip(new_nc, new_c) if a == b)
        first_div = next((i for i, (a, b) in enumerate(zip(new_nc, new_c)) if a != b), None)

        # Logits 数值对比 (each step's last-token logits)
        max_abs_diffs = []
        rel_argmax_match = []
        for la, lb in zip(logits_nocache, logits_cached):
            d = (la - lb).abs().max().item()
            max_abs_diffs.append(d)
            rel_argmax_match.append((la.argmax(-1) == lb.argmax(-1)).all().item())

        print(f"tokens match: {match}/{args.n_steps} ({100*match/args.n_steps:.0f}%)")
        print(f"first divergence at step: {first_div}")
        print(f"argmax match per step: {sum(rel_argmax_match)}/{len(rel_argmax_match)}")
        print(f"max |Δlogits| per step (first 5): {[f'{d:.4f}' for d in max_abs_diffs[:5]]}")
        print(f"max |Δlogits| overall: {max(max_abs_diffs):.4f}")
        print(f"speed: nocache {t_nocache:.2f}s vs cached {t_cached:.2f}s ({t_nocache/t_cached:.2f}x)")
        print(f"nocache: ...{tok.decode(new_nc[:20])}")
        print(f"cached:  ...{tok.decode(new_c[:20])}")


if __name__ == "__main__":
    main()
