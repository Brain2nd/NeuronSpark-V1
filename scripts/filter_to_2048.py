"""Filter existing 8192-binned data to [1000, 2048] range, re-emit at max_length=2048.

不重 tokenize, 直接读 8192 binned 数据:
- nonpad < 1000  → drop (太短)
- nonpad > 2048  → drop (超 cap, 不截断)
- 1000 <= nonpad <= 2048 → 取 ids[:2048] + mask[:2048] 写出
"""
from __future__ import annotations
import argparse
import glob
import os
import time
import numpy as np
from transformers import AutoTokenizer

SRC_ML = 8192
DST_ML = 2048
MIN_NONPAD = 1000
MAX_NONPAD = 2048


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dir', default='data/v3_sft_mix_binned')
    ap.add_argument('--output_dir', default='data/v3_sft_mix_2k_filtered')
    ap.add_argument('--tokenizer_dir', default='tokenizer_v3')
    ap.add_argument('--shard_size', type=int, default=10000)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=True)
    PAD = tok.pad_token_id

    os.makedirs(args.output_dir, exist_ok=True)

    src_bin_files = sorted(f for f in glob.glob(f"{args.input_dir}/train-*.bin")
                           if not f.endswith(".mask.bin"))
    src_mask_files = sorted(glob.glob(f"{args.input_dir}/train-*.mask.bin"))
    print(f'reading {len(src_bin_files)} src shards (max_length={SRC_ML})', flush=True)
    print(f'filter: {MIN_NONPAD} <= nonpad <= {MAX_NONPAD}, output max_length={DST_ML}', flush=True)

    t0 = time.time()
    out_shard_idx = 0
    out_buf_ids: list[np.ndarray] = []
    out_buf_mask: list[np.ndarray] = []
    out_buf_offsets: list[int] = []
    total_seen = 0
    total_kept = 0
    total_short = 0
    total_long = 0

    def flush_shard():
        nonlocal out_shard_idx, out_buf_ids, out_buf_mask, out_buf_offsets
        if not out_buf_ids:
            return
        out_bin = os.path.join(args.output_dir, f'train-{out_shard_idx:05d}.bin')
        out_mask = out_bin.replace('.bin', '.mask.bin')
        out_idx = out_bin.replace('.bin', '.bos.idx')
        np.concatenate(out_buf_ids).astype(np.uint32).tofile(out_bin + '.partial')
        np.concatenate(out_buf_mask).astype(np.uint8).tofile(out_mask + '.partial')
        np.asarray(out_buf_offsets, dtype=np.uint64).tofile(out_idx + '.partial')
        os.replace(out_bin + '.partial', out_bin)
        os.replace(out_mask + '.partial', out_mask)
        os.replace(out_idx + '.partial', out_idx)
        print(f'  shard {out_shard_idx:>3}: {len(out_buf_ids)} docs -> {os.path.basename(out_bin)}', flush=True)
        out_shard_idx += 1
        out_buf_ids = []
        out_buf_mask = []
        out_buf_offsets = []

    for bf, mf in zip(src_bin_files, src_mask_files):
        ids_arr = np.fromfile(bf, dtype=np.uint32).reshape(-1, SRC_ML)
        mask_arr = np.fromfile(mf, dtype=np.uint8).reshape(-1, SRC_ML)
        nonpad_arr = (ids_arr != PAD).sum(axis=1)
        for i in range(ids_arr.shape[0]):
            total_seen += 1
            n = nonpad_arr[i]
            if n < MIN_NONPAD:
                total_short += 1
                continue
            if n > MAX_NONPAD:
                total_long += 1
                continue
            # 取前 DST_ML token (含真实数据 + 少量 pad 填到 2048)
            new_ids = ids_arr[i, :DST_ML].copy()
            new_mask = mask_arr[i, :DST_ML].copy()
            out_buf_ids.append(new_ids)
            out_buf_mask.append(new_mask)
            out_buf_offsets.append(len(out_buf_offsets) * DST_ML)
            total_kept += 1
            if len(out_buf_ids) >= args.shard_size:
                flush_shard()
    flush_shard()

    dt = time.time() - t0
    print(f'\nDONE in {dt:.1f}s', flush=True)
    print(f'  seen:   {total_seen:,}', flush=True)
    print(f'  short:  {total_short:,} ({total_short/total_seen*100:.1f}%)', flush=True)
    print(f'  long:   {total_long:,} ({total_long/total_seen*100:.1f}%)', flush=True)
    print(f'  kept:   {total_kept:,} ({total_kept/total_seen*100:.1f}%)', flush=True)
    print(f'  shards: {out_shard_idx}', flush=True)


if __name__ == '__main__':
    main()
