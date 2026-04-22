#!/usr/bin/env python3
"""Verify v3 tokenizer on real corpora: compression ratio, round-trip, dead-vocab, coverage.

Compares v3-128K against Qwen3-151K-full and NS-64K on:
  - 100 MB EN (fineweb-edu)
  - 100 MB ZH (Chinese-DeepSeek-R1)
  - 20 MB CODE (repo Python + transformers src)

Requires /tmp/tok_bench/{en,zh,code}.txt prepared by the earlier benchmark script.
"""
import os
import pickle
import time
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parent.parent.parent
BENCH = Path("/tmp/tok_bench")
if not (BENCH / "en.txt").exists():
    raise SystemExit(f"Corpus missing at {BENCH}/. Run the earlier bench prep first.")

corpora = {
    "EN":   open(BENCH / "en.txt").read(),
    "ZH":   open(BENCH / "zh.txt").read(),
    "CODE": open(BENCH / "code.txt").read(),
}

tokenizers = {
    "Qwen3-151K":  AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base", trust_remote_code=True),
    "v3-128K":     AutoTokenizer.from_pretrained(str(REPO / "tokenizer_v3")),
}

for tname, t in tokenizers.items():
    print(f"{tname:12s}  class={type(t).__name__:25s}  vocab_size={t.vocab_size}  len={len(t)}")

def chunk_encode(tok, text, chunk_bytes=1_000_000):
    out = []
    lines = text.split("\n")
    buf, bsz = [], 0
    for ln in lines:
        buf.append(ln); bsz += len(ln)
        if bsz >= chunk_bytes:
            out.extend(tok.encode("\n".join(buf), add_special_tokens=False))
            buf, bsz = [], 0
    if buf:
        out.extend(tok.encode("\n".join(buf), add_special_tokens=False))
    return out

# === 1. Compression ===
print("\n== Compression (B/tok higher = better; 220MB corpus) ==")
print(f"{'corpus':<6} {'tokenizer':<12} {'bytes':>12} {'tokens':>12} {'B/tok':>7} {'unique':>8} {'util%':>6}")
results = {}
for cname, text in corpora.items():
    nbytes = len(text.encode("utf-8"))
    for tname, tok in tokenizers.items():
        t0 = time.time()
        ids = chunk_encode(tok, text)
        dt = time.time() - t0
        ntok = len(ids)
        uniq = len(set(ids))
        vs = len(tok)
        util = uniq / vs * 100
        results[(cname, tname)] = {"bytes": nbytes, "tokens": ntok, "unique": uniq,
                                    "freq": Counter(ids), "vocab_len": vs}
        print(f"{cname:<6} {tname:<12} {nbytes:>12,} {ntok:>12,} {nbytes/ntok:>7.3f} {uniq:>8,} {util:>5.1f}%  [{dt:.1f}s]")

# === 2. v3 vs Qwen3-full on EN/ZH: should be nearly identical (we only dropped non-EN/ZH langs) ===
print("\n== v3 vs Qwen3-full token-count delta on EN/ZH (should be near-zero) ==")
for cname in ["EN", "ZH", "CODE"]:
    v3t = results[(cname, "v3-128K")]["tokens"]
    q3t = results[(cname, "Qwen3-151K")]["tokens"]
    delta = v3t - q3t
    pct = delta / q3t * 100
    print(f"  {cname}: v3={v3t:>10,}  Qwen3={q3t:>10,}  delta={delta:+d} ({pct:+.3f}%)")

# === 3. Dead vocab across combined corpus ===
print("\n== Dead vocab (never fired on 220MB) ==")
for tname, tok in tokenizers.items():
    combined = Counter()
    for cname in corpora:
        combined.update(results[(cname, tname)]["freq"])
    used = len(combined)
    total = len(tok)
    dead = total - used
    print(f"  {tname:<12}  used={used:>7,} / {total:>7,}  ({used/total*100:.1f}%  live)  dead={dead:,}")

# === 4. Round-trip identity on 1MB sample per corpus ===
print("\n== Round-trip identity (v3 decode(encode(x)) == x, 1MB per corpus) ==")
ok_all = True
for cname, text in corpora.items():
    sample = text[:1_000_000]
    ids = tokenizers["v3-128K"].encode(sample, add_special_tokens=False)
    dec = tokenizers["v3-128K"].decode(ids, skip_special_tokens=False)
    match = dec == sample
    ok_all = ok_all and match
    print(f"  {cname}: {'EXACT' if match else 'MISMATCH'}  (in={len(sample)}B out={len(dec)}B tokens={len(ids)})")

# === 5. Special-token integrity ===
print("\n== Special-token encode/decode ==")
sp = tokenizers["v3-128K"]
specials = [
    "<|endoftext|>", "<|im_start|>", "<|im_end|>",
    "<think>", "</think>",
    "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>",
    "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>",
    "<|repo_name|>", "<|file_sep|>",
]
for s in specials:
    ids = sp.encode(s, add_special_tokens=False)
    single = len(ids) == 1
    tok_id = ids[0] if single else ids
    dec = sp.decode(ids)
    print(f"  {s:<22}  single_id={single}  id={tok_id}  decode={dec!r}")

# === 6. Top-K coverage (Zipf) ===
print("\n== Top-K coverage (% of total tokens covered by top-K most-used) ==")
print(f"{'corpus':<6} {'tokenizer':<12} {'top100':>7} {'top1k':>7} {'top10k':>7} {'top30k':>7}")
for cname in corpora:
    for tname in tokenizers:
        freq = results[(cname, tname)]["freq"]
        total = results[(cname, tname)]["tokens"]
        top = freq.most_common()
        cov = {}
        for K in [100, 1000, 10000, 30000]:
            cov[K] = sum(c for _, c in top[:K]) / total * 100
        print(f"{cname:<6} {tname:<12} {cov[100]:>6.2f}% {cov[1000]:>6.2f}% {cov[10000]:>6.2f}% {cov[30000]:>6.2f}%")

# Save results for downstream
pickle.dump({k: {kk: vv for kk, vv in v.items() if kk != "freq"} | {"freq": dict(v["freq"])}
             for k, v in results.items()},
            open("/tmp/tok_bench/v3_results.pkl", "wb"))

print(f"\n{'All round-trips EXACT.' if ok_all else 'WARN: mismatches above.'}")
print("Results pickled to /tmp/tok_bench/v3_results.pkl")
