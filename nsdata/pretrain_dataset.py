"""Pretrain dataset with auto-format detection.

Supports three input formats:
  1. Directory containing `.bin` + `.bos.idx` shards (v3 pretrain format) → memory-mapped streaming
  2. HuggingFace Arrow directory (load_from_disk) with `text` column
  3. JSONL file (byte-offset random access) with `{"text": ...}` per line

`.bin` / `.bos.idx` layout (v3):
  shard_NNNNN.bin     — uint32 sequence of token IDs (contiguous docs separated by eos)
  shard_NNNNN.bos.idx — uint64 array of byte offsets where each document begins

Returns `(X, Y, loss_mask)` tensors, same shape as old `dataset.py` so existing
training loops keep working.
"""
from __future__ import annotations

import json
import os
from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset


def _detect_format(path: str) -> str:
    if os.path.isdir(path):
        shards = sorted(glob(os.path.join(path, "*.bin")))
        if shards and os.path.isfile(shards[0] + ".idx") or \
           (shards and os.path.isfile(shards[0].replace(".bin", ".bos.idx"))):
            return "bin"
        # Fall through to HF arrow
        return "hf"
    if path.endswith((".jsonl", ".json")):
        return "jsonl"
    raise ValueError(f"Cannot determine format of {path}")


class PretrainDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048,
                 pad_token_id: int | None = None):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = pad_token_id if pad_token_id is not None else 0

        self._mode = _detect_format(data_path)
        if self._mode == "bin":
            self._init_bin(data_path)
        elif self._mode == "hf":
            self._init_hf(data_path)
        elif self._mode == "jsonl":
            self._init_jsonl(data_path)

    # ---- init variants ----

    def _init_bin(self, path: str):
        shard_paths = sorted(glob(os.path.join(path, "*.bin")))
        self._shards = []
        for sp in shard_paths:
            arr = np.memmap(sp, dtype=np.uint32, mode="r")
            idx_path = sp + ".idx"
            if not os.path.isfile(idx_path):
                idx_path = sp.replace(".bin", ".bos.idx")
            offsets = np.memmap(idx_path, dtype=np.uint64, mode="r")
            self._shards.append((arr, offsets))
        # Flat index: each (shard_i, doc_j) gives one sample
        self._index = []
        for si, (_, offsets) in enumerate(self._shards):
            for di in range(len(offsets)):
                self._index.append((si, di))

    def _init_hf(self, path: str):
        from datasets import load_dataset, load_from_disk, DatasetDict
        # Try save_to_disk format first (dataset_info.json present), else
        # fall back to a dir-of-parquet/jsonl layout (HF Hub raw download).
        if os.path.isfile(os.path.join(path, "dataset_info.json")):
            ds = load_from_disk(path)
            if isinstance(ds, DatasetDict):
                ds = ds[list(ds.keys())[0]]
        else:
            parquets = sorted(glob(os.path.join(path, "*.parquet")))
            if parquets:
                ds = load_dataset("parquet", data_files=parquets, split="train")
            else:
                jsonls = sorted(glob(os.path.join(path, "*.jsonl")))
                if not jsonls:
                    raise RuntimeError(
                        f"{path} has no dataset_info.json / *.parquet / *.jsonl"
                    )
                ds = load_dataset("json", data_files=jsonls, split="train")
        self._hf_dataset = ds

    def _init_jsonl(self, path: str):
        self._data_path = path
        self._offsets = []
        with open(path, "rb") as f:
            self._offsets.append(0)
            while f.readline():
                self._offsets.append(f.tell())
        self._total_lines = len(self._offsets) - 1

    # ---- interface ----

    def __len__(self):
        if self._mode == "bin":
            return len(self._index)
        if self._mode == "hf":
            return len(self._hf_dataset)
        return self._total_lines

    def __getitem__(self, index: int):
        if self._mode == "bin":
            return self._get_bin(index)
        if self._mode == "hf":
            return self._get_text(self._hf_dataset[index]["text"])
        # jsonl
        with open(self._data_path, "rb") as f:
            f.seek(self._offsets[index])
            line = f.readline().decode("utf-8")
        return self._get_text(json.loads(line)["text"])

    def _get_bin(self, index: int):
        shard_i, doc_i = self._index[index]
        arr, offsets = self._shards[shard_i]
        start = int(offsets[doc_i])
        end = int(offsets[doc_i + 1]) if doc_i + 1 < len(offsets) else len(arr)
        ids = np.asarray(arr[start : start + self.max_length], dtype=np.int64)
        text_len = min(len(ids), self.max_length)
        if text_len < self.max_length:
            pad = np.full(self.max_length - text_len, self.pad_token_id, dtype=np.int64)
            ids = np.concatenate([ids, pad])
        loss_mask = np.zeros(self.max_length, dtype=np.int64)
        loss_mask[:text_len] = 1
        X = ids[:-1].astype(np.int64)
        Y = ids[1:].astype(np.int64)
        loss_mask = loss_mask[1:]
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)

    def _get_text(self, text_raw: str):
        # BOS prefix optional: Qwen3-style tokenizers have no bos by default.
        prefix = self.tokenizer.bos_token or ""
        ids = self.tokenizer(prefix + text_raw)["input_ids"][: self.max_length]
        text_len = len(ids)
        pad_len = self.max_length - text_len
        ids = ids + [self.pad_token_id] * pad_len
        loss_mask = [1] * text_len + [0] * pad_len
        arr = np.asarray(ids, dtype=np.int64)
        X = arr[:-1]
        Y = arr[1:]
        loss_mask = np.asarray(loss_mask[1:], dtype=np.int64)
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)
