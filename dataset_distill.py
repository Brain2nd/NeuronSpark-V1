"""
蒸馏数据适配器: 统一接口支持多种数据源

支持的数据格式:
  jsonl      — 本地 JSONL 文件, 每行 {"text": "..."} (默认)
  huggingface — 本地或远程 HuggingFace datasets
  parquet    — 本地 Parquet 文件

所有格式统一输出: (X, Y, loss_mask) — 与 PretrainDataset 兼容

用法:
  ds = create_distill_dataset(
      data_path='data/seq-monkey/xxx.jsonl',
      tokenizer=tokenizer,
      max_length=512,
      data_type='jsonl',          # jsonl | huggingface | parquet
      text_column='text',         # 文本列名
  )
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset


# ============================================================
# 基类: 统一 tokenize + shift + padding 逻辑
# ============================================================

class _BaseDistillDataset(Dataset):
    """蒸馏数据集基类。子类只需实现 _get_text(index) → str。"""

    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._pad_id = tokenizer.pad_token_id or 0

    def _get_text(self, index: int) -> str:
        raise NotImplementedError

    def __getitem__(self, index: int):
        text = self._get_text(index)

        # Tokenize (加 bos, 不加 eos 让模型自己学)
        bos = self.tokenizer.bos_token or ''
        input_ids = self.tokenizer(bos + text).data['input_ids'][:self.max_length]
        text_len = len(input_ids)

        # Padding
        pad_len = self.max_length - text_len
        input_ids = input_ids + [self._pad_id] * pad_len
        loss_mask = [1] * text_len + [0] * pad_len

        # Shift: X = input[:-1], Y = input[1:]
        ids = np.array(input_ids, dtype=np.int64)
        X = ids[:-1]
        Y = ids[1:]
        mask = np.array(loss_mask[1:], dtype=np.int64)
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(mask)


# ============================================================
# JSONL 数据集 (本地文件, 懒加载)
# ============================================================

class JsonlDistillDataset(_BaseDistillDataset):
    """本地 JSONL, 每行 {"text": "..."}。

    内存友好: 预计算行偏移量, __getitem__ 时 seek 读取。
    """

    def __init__(self, data_path, tokenizer, max_length=512, text_column='text'):
        super().__init__(tokenizer, max_length)
        self.data_path = data_path
        self.text_column = text_column

        # 预计算字节偏移
        self._offsets = []
        with open(data_path, 'rb') as f:
            self._offsets.append(0)
            while f.readline():
                self._offsets.append(f.tell())
        self._total = len(self._offsets) - 1

    def __len__(self):
        return self._total

    def _get_text(self, index):
        with open(self.data_path, 'rb') as f:
            f.seek(self._offsets[index])
            line = f.readline().decode('utf-8')
        sample = json.loads(line)
        return sample[self.text_column]


# ============================================================
# HuggingFace datasets (本地 load_from_disk 或远程名称)
# ============================================================

class HFDistillDataset(_BaseDistillDataset):
    """HuggingFace datasets 适配。

    data_path 可以是:
      - 本地目录 (load_from_disk): '/data/my_dataset'
      - HF 数据集名称:              'wikitext/wikitext-2-raw-v1'
      - 本地缓存的 HF 数据集

    Args:
        data_path: 数据集路径或名称
        tokenizer: tokenizer
        max_length: 最大序列长度
        text_column: 文本列名 (默认 'text')
        split: 数据集 split (默认 'train')
        subset: 数据集子集名称 (可选)
    """

    def __init__(self, data_path, tokenizer, max_length=512,
                 text_column='text', split='train', subset=None):
        super().__init__(tokenizer, max_length)
        self.text_column = text_column

        from datasets import load_dataset, load_from_disk
        import os

        # 判断: 本地 Arrow 目录 vs HF 名称
        if os.path.isdir(data_path) and os.path.exists(
                os.path.join(data_path, 'dataset_info.json')):
            # load_from_disk (已保存的 Arrow 数据集)
            self._ds = load_from_disk(data_path)
            if isinstance(self._ds, dict):
                self._ds = self._ds[split]
        else:
            # load_dataset (本地文件目录 or HF hub 名称)
            kwargs = {}
            if subset:
                kwargs['name'] = subset
            # 本地 parquet/json 目录也走这条路
            self._ds = load_dataset(data_path, split=split, **kwargs)

        # 过滤空文本
        self._ds = self._ds.filter(
            lambda x: x[text_column] is not None and len(x[text_column].strip()) > 0,
            num_proc=4,
        )

    def __len__(self):
        return len(self._ds)

    def _get_text(self, index):
        return self._ds[index][self.text_column]


# ============================================================
# Parquet 数据集
# ============================================================

class ParquetDistillDataset(_BaseDistillDataset):
    """本地 Parquet 文件。"""

    def __init__(self, data_path, tokenizer, max_length=512, text_column='text'):
        super().__init__(tokenizer, max_length)
        self.text_column = text_column

        import pandas as pd
        import os

        # 支持单文件或目录 (多文件)
        if os.path.isdir(data_path):
            import glob
            files = sorted(glob.glob(os.path.join(data_path, '*.parquet')))
            self._df = pd.concat([pd.read_parquet(f) for f in files],
                                 ignore_index=True)
        else:
            self._df = pd.read_parquet(data_path)

        # 过滤空行
        self._df = self._df[self._df[text_column].notna() &
                             (self._df[text_column].str.len() > 0)].reset_index(drop=True)

    def __len__(self):
        return len(self._df)

    def _get_text(self, index):
        return self._df.iloc[index][self.text_column]


# ============================================================
# 工厂函数
# ============================================================

def create_distill_dataset(data_path, tokenizer, max_length=512,
                           data_type='jsonl', text_column='text',
                           split='train', subset=None):
    """创建蒸馏数据集。

    Args:
        data_path: 数据路径 (文件/目录/HF 名称)
        tokenizer: tokenizer 实例
        max_length: 最大序列长度
        data_type: 'jsonl' | 'huggingface' | 'parquet'
        text_column: 文本列名
        split: HF 数据集 split
        subset: HF 数据集子集

    Returns:
        Dataset 实例, __getitem__ 返回 (X, Y, loss_mask)
    """
    if data_type == 'jsonl':
        return JsonlDistillDataset(data_path, tokenizer, max_length, text_column)
    elif data_type == 'huggingface':
        return HFDistillDataset(data_path, tokenizer, max_length,
                                text_column, split, subset)
    elif data_type == 'parquet':
        return ParquetDistillDataset(data_path, tokenizer, max_length, text_column)
    else:
        raise ValueError(f"不支持的 data_type: {data_type}, "
                         f"可选: jsonl, huggingface, parquet")
