"""SFT dataset with ChatML loss mask.

Loss is computed only on assistant tokens (between `<|im_start|>assistant\n` and
`<|im_end|>`). Everything else (system/user + padding) is masked to 0.

Supports:
  - HuggingFace Arrow directory (load_from_disk) with `messages` column
  - JSONL file with either {"messages": [...]} or raw chat-template dicts per line

Returns `(X, Y, loss_mask)` tuples — same format as `PretrainDataset`.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


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
