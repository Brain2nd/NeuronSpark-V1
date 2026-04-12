import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    """预训练数据集，支持 JSONL 和 HF Datasets (Arrow/Parquet) 两种格式。

    格式自动判断：
      - 目录路径 → HF Datasets (load_from_disk)
      - .jsonl 文件 → byte-offset 随机访问 JSONL
    """

    def __init__(self, data_path, tokenizer, max_length=2048):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = 0

        if os.path.isdir(data_path):
            # HF Datasets Arrow 格式
            from datasets import load_from_disk, DatasetDict
            ds = load_from_disk(data_path)
            # DatasetDict → 取第一个 split
            if isinstance(ds, DatasetDict):
                ds = ds[list(ds.keys())[0]]
            self._hf_dataset = ds
            self._mode = 'hf'
        else:
            # JSONL 格式（向后兼容）
            self._hf_dataset = None
            self._mode = 'jsonl'
            self.data_path = data_path
            self._offsets = []
            with open(data_path, 'rb') as f:
                self._offsets.append(0)
                while f.readline():
                    self._offsets.append(f.tell())
            self._total_lines = len(self._offsets) - 1

    def __len__(self):
        if self._mode == 'hf':
            return len(self._hf_dataset)
        return self._total_lines

    def __getitem__(self, index: int):
        if self._mode == 'hf':
            text_raw = self._hf_dataset[index]['text']
        else:
            with open(self.data_path, 'rb') as f:
                f.seek(self._offsets[index])
                line = f.readline().decode('utf-8')
            text_raw = json.loads(line)['text']

        text = f"{self.tokenizer.bos_token}{text_raw}"
        input_id = self.tokenizer(text).data['input_ids'][:self.max_length]
        text_len = len(input_id)
        padding_len = self.max_length - text_len
        input_id = input_id + [self.padding] * padding_len
        loss_mask = [1] * text_len + [0] * padding_len

        input_id = np.array(input_id)
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)

class SFTDataset(Dataset):
    """SFT 数据集，支持 JSONL 和 HF Datasets (Arrow) 两种格式。

    格式自动判断：
      - 目录路径 → HF Datasets (load_from_disk)，需含 'messages' 列
      - .jsonl 文件 → byte-offset 随机访问 JSONL
    """

    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = 0

        if os.path.isdir(data_path):
            from datasets import load_from_disk
            self._hf_dataset = load_from_disk(data_path)
            self._mode = 'hf'
        else:
            self._hf_dataset = None
            self._mode = 'jsonl'
            self.data_path = data_path
            self._offsets = []
            with open(data_path, 'rb') as f:
                self._offsets.append(0)
                while f.readline():
                    self._offsets.append(f.tell())
            self._total_lines = len(self._offsets) - 1

    def __len__(self):
        if self._mode == 'hf':
            return len(self._hf_dataset)
        return self._total_lines

    def generate_loss_mask(self, input_ids):
        # 生成 loss mask, 0 表示不计算损失, 1 表示计算损失
        mask = [0] * len(input_ids)
        a_sequence = self.tokenizer("<|im_start|>assistant\n")['input_ids']  # <|im_start|>assistant\n
        im_end_id = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]  # <|im_end|> token id
        a_length = len(a_sequence)
        n = len(input_ids)
        i = 0

        while i <= n - a_length:
            # 检查当前位置是否匹配目标子序列
            match = True
            for k in range(a_length):
                if input_ids[i + k] != a_sequence[k]:
                    match = False
                    break
            if match:
                # 从子序列结束的位置开始查找 <|im_end|>
                j = None
                for idx in range(i + a_length, n):
                    if input_ids[idx] == im_end_id:
                        j = idx
                        break
                if j is not None:
                    start = i + a_length
                    end = j  # 结束位置设为j（包含4）
                    # 标记区间为1（包括start到end）
                    if start <= end:
                        for pos in range(start, end + 1):
                            if pos < len(mask):
                                mask[pos] = 1
                # 跳过当前子序列，避免重叠匹配
                i += a_length
            else:
                i += 1
        return mask

    def __getitem__(self, index: int):
        if self._mode == 'hf':
            sample = self._hf_dataset[index]
        else:
            with open(self.data_path, 'rb') as f:
                f.seek(self._offsets[index])
                line = f.readline().decode('utf-8')
            sample = json.loads(line)

        # messages 格式 → apply_chat_template；旧格式直接用 text
        if 'messages' in sample:
            text = self.tokenizer.apply_chat_template(
                sample['messages'], tokenize=False, add_generation_prompt=False)
        else:
            text = self.tokenizer.apply_chat_template(
                sample, tokenize=False, add_generation_prompt=False)
        input_id = self.tokenizer(text).data['input_ids'][:self.max_length]
        text_len = len(input_id)
        # 没满最大长度的剩余部分
        padding_len = self.max_length - text_len
        input_id = input_id + [self.padding] * padding_len
        # 0表示不计算损失
        loss_mask = self.generate_loss_mask(input_id)

        input_id = np.array(input_id)
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)
