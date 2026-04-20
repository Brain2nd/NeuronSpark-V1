"""过滤 sft_v2_mix: 去掉 apply_chat_template 后 token 数 > 2048 的样本.

避免 seq_len=2048 截断 assistant 段带坏分布.

输出: data/sft_v2_mix_filtered/
"""
import os
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

SRC = '/home/dgxspark/Desktop/NeuronSpark-V1/data/sft_v2_mix'
DST = '/home/dgxspark/Desktop/NeuronSpark-V1/data/sft_v2_mix_filtered'
MAX_LEN = 2048
TOK_PATH = '/home/dgxspark/Desktop/NeuronSpark-V1/tokenizer/'

tok = AutoTokenizer.from_pretrained(TOK_PATH)
ds = load_from_disk(SRC)
print(f'Source: {len(ds)} samples')

# 全量过滤 (30k 测样显示 P99 < 1635, 真实过滤 ~0.4%)
def keep(sample):
    text = tok.apply_chat_template(sample['messages'], tokenize=False, add_generation_prompt=False)
    n = len(tok(text)['input_ids'])
    return n <= MAX_LEN


# 批量过滤
def batch_keep(batch):
    msgs_list = batch['messages']
    texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=False) for m in msgs_list]
    lens = [len(tok(t)['input_ids']) for t in texts]
    return [n <= MAX_LEN for n in lens]


print(f'Filtering with max_len={MAX_LEN} (apply_chat_template 后 token)...')
# 分块并行 tokenize 加速
from concurrent.futures import ProcessPoolExecutor
import numpy as np

# 直接 .filter() 走 batched=True
filtered = ds.filter(
    lambda batch: batch_keep(batch),
    batched=True,
    batch_size=500,
    num_proc=8,
    desc='filtering',
)

print(f'\nBefore: {len(ds)} / After: {len(filtered)} (removed {len(ds)-len(filtered)}, {(1-len(filtered)/len(ds))*100:.2f}%)')

# 按 source 统计过滤前后
from collections import Counter
before = Counter(s['source'] for s in ds)
after = Counter(s['source'] for s in filtered)
print('\nPer-source 过滤影响:')
print(f'{"source":<25} before  after  removed')
for src in sorted(before):
    b = before[src]; a = after[src]; r = b - a
    print(f'{src:<25} {b:<7} {a:<7} {r:<5} ({r/b*100:.1f}%)')

filtered.save_to_disk(DST)
print(f'\n写入: {DST}')
