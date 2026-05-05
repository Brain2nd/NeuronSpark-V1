"""Pre-tokenize SFT dataset (HF Arrow with `messages`) → .bin + .mask.bin + .bos.idx shards.

对齐 pretokenize_v3.py 风格. SFT 与 pretrain 区别:
  - 每样本应用 ChatML chat_template, 截到 max_length, 右填充到 max_length
  - 同时输出 loss_mask (uint8): 1 = assistant token (含 <|im_end|>), 0 = system/user/pad
  - 输出每 shard 三件套:
      train-NNNNN.bin       uint32 token IDs
      train-NNNNN.mask.bin  uint8 loss mask
      train-NNNNN.bos.idx   uint64 doc 起点 (固定 i * max_length, 与 pretrain idx 语义一致)

Run:
    python /tmp/pretokenize_sft_v3.py \
        --input_dir data/v3_sft_v2_mix_raw \
        --output_dir data/v3_sft_v2_mix_binned \
        --tokenizer_dir tokenizer_v3 \
        --max_length 2048 \
        --shard_size 10000 \
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


_TOK = None
_A_OPEN_IDS = None
_IM_END_ID = None
_PAD_ID = 0
_MAX_LEN = 2048


def _init_worker(tokenizer_dir: str, max_length: int):
    global _TOK, _A_OPEN_IDS, _IM_END_ID, _PAD_ID, _MAX_LEN
    from transformers import AutoTokenizer
    _TOK = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True, use_fast=True)
    _A_OPEN_IDS = _TOK("<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
    _IM_END_ID = _TOK.encode("<|im_end|>", add_special_tokens=False)[0]
    _PAD_ID = _TOK.pad_token_id if _TOK.pad_token_id is not None else (_TOK.eos_token_id or 0)
    _MAX_LEN = max_length


def _make_loss_mask(ids: list[int]) -> list[int]:
    mask = [0] * len(ids)
    a = _A_OPEN_IDS
    a_len = len(a)
    n = len(ids)
    i = 0
    while i <= n - a_len:
        if ids[i:i + a_len] == a:
            j = None
            for k in range(i + a_len, n):
                if ids[k] == _IM_END_ID:
                    j = k
                    break
            if j is not None:
                for pos in range(i + a_len, j + 1):
                    if pos < len(mask):
                        mask[pos] = 1
            i += a_len
        else:
            i += 1
    return mask


def _process_one_shard(args: tuple[str, list, int]) -> dict:
    """Tokenize + mask one chunk of samples → write .bin / .mask.bin / .bos.idx."""
    out_bin_path, samples, shard_idx = args
    out_mask_path = out_bin_path.replace(".bin", ".mask.bin")
    out_idx_path = out_bin_path.replace(".bin", ".bos.idx")
    tmp_bin = out_bin_path + ".partial"
    tmp_mask = out_mask_path + ".partial"
    tmp_idx = out_idx_path + ".partial"

    t0 = time.time()
    n_docs = 0
    n_assist = 0
    n_skip = 0
    offsets: list[int] = []

    with open(tmp_bin, "wb") as f_bin, open(tmp_mask, "wb") as f_mask:
        for s in samples:
            msgs = s.get("messages")
            if not msgs:
                n_skip += 1
                continue
            try:
                text = _TOK.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            except Exception:
                n_skip += 1
                continue
            ids = _TOK(text, add_special_tokens=False)["input_ids"][:_MAX_LEN]
            if not ids:
                n_skip += 1
                continue
            text_len = len(ids)
            pad_len = _MAX_LEN - text_len
            if pad_len > 0:
                ids = ids + [_PAD_ID] * pad_len
            mask = _make_loss_mask(ids)
            # 不让 padding 计入 loss (mask 主算法已 0 但保险)
            for p in range(text_len, _MAX_LEN):
                mask[p] = 0
            assist_count = sum(mask)
            if assist_count == 0:
                # 没有 assistant token, 跳过 (无 loss 信号)
                n_skip += 1
                continue
            offsets.append(n_docs * _MAX_LEN)
            np.asarray(ids, dtype=np.uint32).tofile(f_bin)
            np.asarray(mask, dtype=np.uint8).tofile(f_mask)
            n_docs += 1
            n_assist += assist_count

    np.asarray(offsets, dtype=np.uint64).tofile(tmp_idx)
    os.replace(tmp_bin, out_bin_path)
    os.replace(tmp_mask, out_mask_path)
    os.replace(tmp_idx, out_idx_path)

    return {
        "shard_idx": shard_idx,
        "out": os.path.basename(out_bin_path),
        "n_docs": n_docs,
        "n_skip": n_skip,
        "n_assist_tokens": n_assist,
        "elapsed": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="HF Arrow dir with messages column")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--tokenizer_dir", default="tokenizer_v3")
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--shard_size", type=int, default=10000, help="samples per shard")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="dev: limit total samples")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[load] {args.input_dir}", flush=True)
    from datasets import load_from_disk
    ds = load_from_disk(args.input_dir)
    if hasattr(ds, "keys"):
        ds = ds[list(ds.keys())[0]]
    n_total = len(ds) if args.limit is None else min(len(ds), args.limit)
    print(f"  total samples: {n_total}", flush=True)

    # 分 shard
    tasks = []
    shard_idx = 0
    for start in range(0, n_total, args.shard_size):
        end = min(start + args.shard_size, n_total)
        chunk = ds.select(range(start, end)).to_list()
        out_bin = os.path.join(args.output_dir, f"train-{shard_idx:05d}.bin")
        tasks.append((out_bin, chunk, shard_idx))
        shard_idx += 1
    print(f"  shards: {len(tasks)} ({args.shard_size} samples each)", flush=True)

    t0 = time.time()
    total_docs = 0
    total_skip = 0
    total_assist = 0
    with Pool(args.workers, initializer=_init_worker, initargs=(args.tokenizer_dir, args.max_length)) as pool:
        for r in pool.imap_unordered(_process_one_shard, tasks):
            total_docs += r["n_docs"]
            total_skip += r["n_skip"]
            total_assist += r["n_assist_tokens"]
            print(f"  [{r['shard_idx']:>3}] {r['out']} docs={r['n_docs']} skip={r['n_skip']} "
                  f"assist_tok={r['n_assist_tokens']} ({r['elapsed']:.1f}s)", flush=True)

    dt = time.time() - t0
    print(f"\nTOTAL: docs={total_docs} skip={total_skip} assist_tokens={total_assist} "
          f"(loss tokens) in {dt:.1f}s", flush=True)
    print(f"output: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
