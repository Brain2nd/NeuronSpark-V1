"""
构建知识性 SFT 数据集：从多个中文知识问答数据源混合。

数据源：
  1. webqa (4.2万) — 百度百科事实问答
  2. COIG-CQIA wiki — 科普知识问答（人工验证）
  3. Chinese-SimpleQA (3K) — 极高质量事实问答
  4. BelleGroup/school_math (24.8万) — 数学解题

输出: data/knowledge-sft/ — HF Dataset, 字段 ['messages']
      messages 格式: [{"role":"system",...}, {"role":"user",...}, {"role":"assistant",...}]

用法:
    python scripts/build_knowledge_sft.py
"""
import os
import json
import random
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict

random.seed(42)
all_samples = []

SYS = "你是一个有帮助的知识助手，请准确回答用户的问题。"

# ============================================================
# 1. webqa — 百度百科问答
# ============================================================
print("=== webqa ===")
try:
    for fname in ["train.json", "dev.json", "test.json"]:
        fpath = os.path.join("data/raw/webqa", fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                q = row.get("input", "").strip()
                a = row.get("output", "").strip()
                if not q or not a or len(a) < 5:
                    continue
                if len(a) > 500:
                    a = a[:500] + "..."
                all_samples.append({
                    "messages": [
                        {"role": "system", "content": SYS},
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": a},
                    ],
                    "source": "webqa",
                })
    print(f"  webqa: {sum(1 for s in all_samples if s['source']=='webqa')} samples")
except Exception as e:
    print(f"  webqa SKIP: {e}")

# ============================================================
# 2. COIG-CQIA wiki — 科普知识
# ============================================================
print("=== COIG-CQIA wiki ===")
before = len(all_samples)
wiki_dir = "data/raw/coig-cqia/wiki"
if os.path.isdir(wiki_dir):
    for fname in os.listdir(wiki_dir):
        if not fname.endswith(".jsonl"):
            continue
        fpath = os.path.join(wiki_dir, fname)
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                q = row.get("instruction", "").strip()
                a = row.get("output", "").strip()
                if not q or not a or len(a) < 10:
                    continue
                if len(a) > 800:
                    a = a[:800] + "..."
                all_samples.append({
                    "messages": [
                        {"role": "system", "content": SYS},
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": a},
                    ],
                    "source": "coig_wiki",
                })
print(f"  coig_wiki: {len(all_samples) - before} samples")

# ============================================================
# 3. Chinese-SimpleQA — 极高质量事实问答
# ============================================================
print("=== Chinese-SimpleQA ===")
before = len(all_samples)
simpleqa_path = "data/raw/chinese-simpleqa/chinese_simpleqa.jsonl"
if os.path.exists(simpleqa_path):
    with open(simpleqa_path, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = row.get("question", "").strip()
            a = row.get("answer", "").strip()
            if not q or not a:
                continue
            all_samples.append({
                "messages": [
                    {"role": "system", "content": SYS},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ],
                "source": "simpleqa",
            })
else:
    print(f"  simpleqa SKIP: {simpleqa_path} not found")
print(f"  simpleqa: {len(all_samples) - before} samples")

# ============================================================
# 4. BelleGroup/school_math — 数学解题
# ============================================================
print("=== school_math ===")
before = len(all_samples)
math_path = "data/pretrain_raw/belle_math/school_math_0.25M.json"
if os.path.exists(math_path):
    count = 0
    with open(math_path) as f:
        for line in f:
            row = json.loads(line)
            q = row.get("instruction", "").strip()
            a = row.get("output", "").strip()
            if not q or not a or len(a) < 10:
                continue
            # 采样 5 万条（不需要全部 24.8 万）
            if count >= 50000:
                break
            all_samples.append({
                "messages": [
                    {"role": "system", "content": SYS},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ],
                "source": "school_math",
            })
            count += 1
    print(f"  school_math: {count} samples (sampled from 248K)")
else:
    print(f"  school_math SKIP: {math_path} not found")


# ============================================================
# 汇总
# ============================================================
random.shuffle(all_samples)

print(f"\n总计: {len(all_samples):,} 条")

from collections import Counter
source_counts = Counter(s["source"] for s in all_samples)
print("\n来源分布:")
for src, cnt in source_counts.most_common():
    print(f"  {src:<18} {cnt:>8,} ({cnt/len(all_samples)*100:.1f}%)")

# 保存
save_dir = "data/knowledge-sft"
# 只保存 messages 字段（和 SFTDataset 兼容）
ds_out = Dataset.from_list([{"messages": s["messages"]} for s in all_samples])
os.makedirs(save_dir, exist_ok=True)
ds_out.save_to_disk(save_dir)
print(f"\n已保存到 {save_dir}")
print(f"样本数: {len(ds_out):,}")
print(f"字段: {ds_out.column_names}")

# 验证
sample = ds_out[0]
print(f"\n验证样例:")
for m in sample["messages"]:
    print(f"  [{m['role']}] {m['content'][:100]}")
