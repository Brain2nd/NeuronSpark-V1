"""Pre-tokenize think-focused SFT dataset → .bin + .mask.bin + .bos.idx shards.

与 pretokenize_sft_v3.py 区别:
  - **绕过 chat_template 自动包装**: 该模板对没有 <think> 标记的 assistant 会强制
    注入 '<think>\\n\\n</think>\\n\\n' 包装, 导致 model 学到"空 think 然后答"的捷径.
  - 改用 manual_render_chatml(): 信任 message.content 原样, 不补 think 包装.
  - think_rich 样本 (content 已含 <think>X</think>) 渲染等价于 chat_template.
  - non_think 样本 (content 不含 <think>) 渲染为 raw `<|im_start|>assistant\\nCONTENT<|im_end|>`,
    不包 think 标记 — 模型见到这种样本会学到"这种 prompt 不需要 think".

输出每 shard:
    train-NNNNN.bin       uint32 token IDs
    train-NNNNN.mask.bin  uint8 loss mask (1 = assistant token incl. <|im_end|>)
    train-NNNNN.bos.idx   uint64 doc 起点

Run:
    python scripts/pretokenize_sft_think_v3.py \
        --input_dir data/sft_think_focused_2048 \
        --output_dir data/sft_think_focused_binned_2048 \
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


def manual_render_chatml(messages: list[dict]) -> str:
    """Render messages as ChatML, trusting content as-is.

    不补 think 包装 — content 含 <think>X</think> 则保留 think; 不含则直接 raw.
    """
    parts = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    return "".join(parts)


def _init_worker(tokenizer_dir: str, max_length: int):
    global _TOK, _A_OPEN_IDS, _IM_END_ID, _PAD_ID, _MAX_LEN
    from transformers import AutoTokenizer
    _TOK = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True, use_fast=True)
    _A_OPEN_IDS = _TOK("<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
    _IM_END_ID = _TOK.encode("<|im_end|>", add_special_tokens=False)[0]
    _PAD_ID = _TOK.pad_token_id if _TOK.pad_token_id is not None else (_TOK.eos_token_id or 0)
    _MAX_LEN = max_length


def _make_loss_mask(ids: list[int]) -> list[int]:
    """Mark assistant token positions (after '<|im_start|>assistant\\n' up to '<|im_end|>' incl.)."""
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
            text = manual_render_chatml(msgs)
            ids = _TOK(text, add_special_tokens=False)["input_ids"][:_MAX_LEN]
            if not ids:
                n_skip += 1
                continue
            text_len = len(ids)
            pad_len = _MAX_LEN - text_len
            if pad_len > 0:
                ids = ids + [_PAD_ID] * pad_len
            mask = _make_loss_mask(ids)
            for p in range(text_len, _MAX_LEN):
                mask[p] = 0
            assist_count = sum(mask)
            if assist_count == 0:
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
        "shard": shard_idx, "docs": n_docs, "assist_tokens": n_assist,
        "skipped": n_skip, "elapsed": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--tokenizer_dir", required=True)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--shard_size", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    from datasets import load_from_disk
    ds = load_from_disk(args.input_dir)
    print(f"loaded {args.input_dir}: {len(ds)} samples", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    samples_list = list(ds)
    n = len(samples_list)
    n_shards = (n + args.shard_size - 1) // args.shard_size

    tasks = []
    for i in range(n_shards):
        start = i * args.shard_size
        end = min(start + args.shard_size, n)
        out_bin = os.path.join(args.output_dir, f"train-{i:05d}.bin")
        tasks.append((out_bin, samples_list[start:end], i))

    print(f"sharding into {n_shards} shards of up to {args.shard_size} samples, workers={args.workers}, max_length={args.max_length}", flush=True)
    print(f"  using MANUAL ChatML render (bypass chat_template auto think wrap)", flush=True)
    t0 = time.time()
    with Pool(args.workers, initializer=_init_worker,
              initargs=(args.tokenizer_dir, args.max_length)) as pool:
        total_docs = 0
        total_assist = 0
        total_skip = 0
        for r in pool.imap_unordered(_process_one_shard, tasks):
            total_docs += r["docs"]
            total_assist += r["assist_tokens"]
            total_skip += r["skipped"]
            print(f"  shard {r['shard']}: {r['docs']} docs, {r['assist_tokens']} assist_tok, "
                  f"skip {r['skipped']}, {r['elapsed']:.1f}s", flush=True)

    print(f"\nDONE in {time.time()-t0:.1f}s")
    print(f"  total docs: {total_docs}")
    print(f"  total assistant tokens (loss-counted): {total_assist}")
    print(f"  skipped: {total_skip}")


if __name__ == "__main__":
    main()
