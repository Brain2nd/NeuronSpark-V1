"""Pre-tokenize v3_pretrain_mix (parquet dir) → *.bin + *.bos.idx shards.

Eliminates per-step DataLoader tokenize overhead (biggest training bottleneck
when running with `data_path` pointing at raw parquet).

Each input shard `train-NNNNN.parquet` → output `train-NNNNN.bin` + `.bos.idx`:
  - .bin:     uint32 token stream, documents concatenated (no EOS added here;
              tokenizer's BOS/EOS handled by encoder config)
  - .bos.idx: uint64 document start offsets (length == n_docs)

_init_bin in nsdata/pretrain_dataset.py handles multi-shard dirs automatically.

Run (on H200, in parallel 32 workers):
    python scripts/pretokenize_v3.py \
        --input_dir data/v3_pretrain_mix \
        --output_dir data/v3_pretrain_mix_binned \
        --tokenizer_dir tokenizer_v3 \
        --workers 32
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


# worker-global tokenizer (initialized once per worker)
_TOK = None


def _init_worker(tokenizer_dir: str):
    global _TOK
    from transformers import AutoTokenizer
    _TOK = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True, use_fast=True)


def _process_one_shard(args: tuple[str, str, int]) -> dict:
    """Tokenize one parquet → write one .bin + .bos.idx atomically."""
    in_path, out_bin_path, min_chars = args
    out_idx_path = out_bin_path.replace(".bin", ".bos.idx")
    tmp_bin = out_bin_path + ".partial"
    tmp_idx = out_idx_path + ".partial"

    t0 = time.time()
    n_docs = 0
    n_tokens = 0
    n_skip = 0
    offsets: list[int] = []

    pf = pq.ParquetFile(in_path)
    with open(tmp_bin, "wb") as f_bin:
        for batch in pf.iter_batches(batch_size=512, columns=["text"]):
            texts = [t for t in batch.column(0).to_pylist() if t and len(t) >= min_chars]
            if not texts:
                continue
            enc = _TOK(texts, add_special_tokens=False, return_attention_mask=False)
            for ids in enc["input_ids"]:
                if not ids:
                    n_skip += 1
                    continue
                offsets.append(n_tokens)
                arr = np.asarray(ids, dtype=np.uint32)
                arr.tofile(f_bin)
                n_tokens += len(ids)
                n_docs += 1

    np.asarray(offsets, dtype=np.uint64).tofile(tmp_idx)

    os.replace(tmp_bin, out_bin_path)
    os.replace(tmp_idx, out_idx_path)

    dt = time.time() - t0
    return {
        "in_path": os.path.basename(in_path),
        "n_docs": n_docs, "n_tokens": n_tokens, "n_skip": n_skip,
        "elapsed_s": round(dt, 1),
        "bin_size_mb": os.path.getsize(out_bin_path) / 1e6,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="parquet dir (e.g. data/v3_pretrain_mix)")
    ap.add_argument("--output_dir", required=True,
                    help="bin output dir (e.g. data/v3_pretrain_mix_binned)")
    ap.add_argument("--tokenizer_dir", default="tokenizer_v3")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--min_chars", type=int, default=1,
                    help="Skip texts shorter than this (char count)")
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip shards whose .bin already exists")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquets = sorted(in_dir.glob("train-*.parquet"))
    if not parquets:
        parquets = sorted(in_dir.glob("*.parquet"))
    if not parquets:
        print(f"no *.parquet under {in_dir}"); sys.exit(1)

    tasks = []
    for p in parquets:
        out_bin = out_dir / p.name.replace(".parquet", ".bin")
        if args.skip_existing and out_bin.is_file():
            print(f"[skip] {p.name} (already exists)")
            continue
        tasks.append((str(p), str(out_bin), args.min_chars))

    print(f"Pre-tokenizing {len(tasks)} shards × {args.workers} workers → {out_dir}")
    t_start = time.time()

    total_docs = 0
    total_tokens = 0

    with Pool(args.workers, initializer=_init_worker,
              initargs=(args.tokenizer_dir,)) as pool:
        for i, result in enumerate(pool.imap_unordered(_process_one_shard, tasks), 1):
            elapsed = time.time() - t_start
            total_docs += result["n_docs"]
            total_tokens += result["n_tokens"]
            print(f"[{i:3d}/{len(tasks)}] {result['in_path']:<30s}  "
                  f"docs={result['n_docs']:>8,d}  tok={result['n_tokens']/1e6:>8.2f}M  "
                  f"{result['elapsed_s']:>5.1f}s  "
                  f"({total_tokens/1e9:.2f}B tok total, elapsed {elapsed:.0f}s)")

    print(f"\n=== DONE: {total_docs:,} docs, {total_tokens/1e9:.3f}B tokens, "
          f"total {time.time()-t_start:.0f}s ===")


if __name__ == "__main__":
    main()
