"""SFT dataset with ChatML loss mask.

Loss is computed only on assistant tokens (between `<|im_start|>assistant\n` and
`<|im_end|>`). Everything else (system/user + padding) is masked to 0.

Supports:
  - HuggingFace Arrow directory (load_from_disk) with `messages` column
  - JSONL file with either {"messages": [...]} or raw chat-template dicts per line
  - Pre-tokenized binned shards (BinnedSFTDataset): scripts/pretokenize_sft_v3.py 产物
    .bin (uint32 token) + .mask.bin (uint8 loss_mask) + .bos.idx (uint64 doc 起点)

Returns `(X, Y, loss_mask)` tuples — same format as `PretrainDataset`.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


def is_binned_sft_dir(path: str) -> bool:
    """检查目录是否包含 pretokenize_sft_v3.py 产生的三件套 shard."""
    if not os.path.isdir(path):
        return False
    bins = [f for f in glob.glob(os.path.join(path, "train-*.bin"))
            if not f.endswith(".mask.bin")]
    if not bins:
        return False
    sample = bins[0]
    return (os.path.exists(sample.replace(".bin", ".mask.bin"))
            and os.path.exists(sample.replace(".bin", ".bos.idx")))


class SFTDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048,
                 pad_token_id: int | None = None):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = pad_token_id if pad_token_id is not None else 0

        # Cache ChatML marker ids once
        self._a_open_ids = tokenizer("<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
        self._im_end_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]

        if os.path.isdir(data_path):
            from datasets import load_from_disk, DatasetDict
            ds = load_from_disk(data_path)
            if isinstance(ds, DatasetDict):
                ds = ds[list(ds.keys())[0]]
            self._hf_dataset = ds
            self._mode = "hf"
        else:
            self._data_path = data_path
            self._offsets = []
            with open(data_path, "rb") as f:
                self._offsets.append(0)
                while f.readline():
                    self._offsets.append(f.tell())
            self._total_lines = len(self._offsets) - 1
            self._mode = "jsonl"

    def __len__(self):
        if self._mode == "hf":
            return len(self._hf_dataset)
        return self._total_lines

    def _make_loss_mask(self, ids: list[int]) -> list[int]:
        """1 at positions inside assistant response, 0 elsewhere.

        Scans for the `<|im_start|>assistant\n` prefix, then marks tokens up to the
        next `<|im_end|>` (inclusive) as loss positions.
        """
        mask = [0] * len(ids)
        a = self._a_open_ids
        a_len = len(a)
        im_end = self._im_end_id
        n = len(ids)
        i = 0
        while i <= n - a_len:
            if ids[i : i + a_len] == a:
                j = None
                for k in range(i + a_len, n):
                    if ids[k] == im_end:
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

    def __getitem__(self, index: int):
        if self._mode == "hf":
            sample = self._hf_dataset[index]
        else:
            with open(self._data_path, "rb") as f:
                f.seek(self._offsets[index])
                sample = json.loads(f.readline().decode("utf-8"))

        if "messages" in sample:
            text = self.tokenizer.apply_chat_template(
                sample["messages"], tokenize=False, add_generation_prompt=False,
            )
        else:
            text = self.tokenizer.apply_chat_template(
                sample, tokenize=False, add_generation_prompt=False,
            )
        ids = self.tokenizer(text)["input_ids"][: self.max_length]
        text_len = len(ids)
        pad_len = self.max_length - text_len
        ids = ids + [self.pad_token_id] * pad_len
        loss_mask = self._make_loss_mask(ids)

        arr = np.asarray(ids, dtype=np.int64)
        X = arr[:-1]
        Y = arr[1:]
        loss_mask = np.asarray(loss_mask[1:], dtype=np.int64)
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)


class BinnedSFTDataset(Dataset):
    """读取 pretokenize_sft_v3.py 产生的二进制分片.

    每 shard 三个文件:
      train-NNNNN.bin       uint32 token IDs, shape (n_docs, max_length) 扁平存储
      train-NNNNN.mask.bin  uint8  loss mask, 同形
      train-NNNNN.bos.idx   uint64 doc 起点偏移 (固定 i*max_length, 与 pretrain idx 语义一致)

    返回 (X, Y, loss_mask) 与 SFTDataset 完全一致:
      X = ids[:-1]
      Y = ids[1:]
      loss_mask = mask[1:]

    使用 numpy memmap 懒加载, DataLoader workers 间共享 OS page cache.
    """

    def __init__(self, data_path: str, max_length: int = 2048):
        super().__init__()
        self.data_path = data_path
        self.max_length = max_length

        bin_files = sorted(
            f for f in glob.glob(os.path.join(data_path, "train-*.bin"))
            if not f.endswith(".mask.bin")
        )
        if not bin_files:
            raise ValueError(f"BinnedSFTDataset: no train-*.bin shards in {data_path}")

        self._shards: list[dict] = []
        cum = 0
        for bf in bin_files:
            mf = bf.replace(".bin", ".mask.bin")
            idx_f = bf.replace(".bin", ".bos.idx")
            if not (os.path.exists(mf) and os.path.exists(idx_f)):
                raise ValueError(f"BinnedSFTDataset: missing companion files for {bf}")
            n_docs = os.path.getsize(idx_f) // 8
            self._shards.append({
                "bin": bf, "mask": mf, "n_docs": n_docs, "global_start": cum,
            })
            cum += n_docs
        self._total = cum

        # 懒加载 memmap (worker fork 后再开, 避免 fork-before-mmap 问题)
        self._bin_mm: dict[int, np.memmap] = {}
        self._mask_mm: dict[int, np.memmap] = {}

    def __len__(self) -> int:
        return self._total

    def _open_shard(self, shard_idx: int) -> tuple[np.memmap, np.memmap]:
        if shard_idx not in self._bin_mm:
            sh = self._shards[shard_idx]
            self._bin_mm[shard_idx] = np.memmap(sh["bin"], dtype=np.uint32, mode="r")
            self._mask_mm[shard_idx] = np.memmap(sh["mask"], dtype=np.uint8, mode="r")
        return self._bin_mm[shard_idx], self._mask_mm[shard_idx]

    def _locate(self, idx: int) -> tuple[int, int]:
        # 二分能更快; n_shards 小所以线性 OK
        for s, sh in enumerate(self._shards):
            if idx < sh["global_start"] + sh["n_docs"]:
                return s, idx - sh["global_start"]
        raise IndexError(idx)

    def __getitem__(self, index: int):
        shard_idx, local = self._locate(index)
        bin_mm, mask_mm = self._open_shard(shard_idx)
        ML = self.max_length
        offset = local * ML
        # copy 出来转 int64 (DataLoader 默认 collate 对 int64 友好)
        ids = np.asarray(bin_mm[offset:offset + ML], dtype=np.int64)
        mask = np.asarray(mask_mm[offset:offset + ML], dtype=np.int64)
        X = ids[:-1]
        Y = ids[1:]
        loss_mask = mask[1:]
        return torch.from_numpy(X.copy()), torch.from_numpy(Y.copy()), torch.from_numpy(loss_mask.copy())
