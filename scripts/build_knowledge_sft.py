"""
构建混合 SFT 数据集：BelleGroup 对话 + 知识问答，防止灾难性遗忘。

数据源：
  1. BelleGroup/train_3.5M_CN — 通用对话（保持对话能力）
  2. webqa — 百度百科事实问答
  3. COIG-CQIA wiki — 科普知识问答（人工验证）
  4. Chinese-SimpleQA — 极高质量事实问答
  5. BelleGroup/school_math — 数学解题

数据清洗：
  - 去除 <e> 标签（webqa 分隔符）
  - 去除 HTML 标签
  - 去除过短回答 (<10 字符)
  - 截断过长回答 (>500 字符)
  - 去除重复问题

混合比例：BelleGroup 对话 50% + 知识问答 50%

输出: data/mixed-sft/ — HF Dataset, 字段 ['messages']
"""
import os
import re
import json
import random
from collections import Counter
from datasets import Dataset

random.seed(42)

SYS_CHAT = "你是一个AI助手"
SYS_KNOWLEDGE = "你是一个有帮助的知识助手，请准确回答用户的问题。"


def clean_text(text):
    """清洗文本：去 <e> 标签、HTML、多余空白。"""
    text = re.sub(r'<e>', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def is_quality(q, a, min_a_len=10):
    """质量过滤。"""
    if not q or not a:
        return False
    if len(a.strip()) < min_a_len:
        return False
    # 过滤乱码
    if a.count('�') > 3:
        return False
    return True


# ============================================================
# 知识数据
# ============================================================
knowledge_samples = []

# 1. webqa
print("=== webqa ===")
seen_questions = set()
for fname in ["train.json", "dev.json", "test.json"]:
    fpath = os.path.join("data/raw/webqa", fname)
    if not os.path.exists(fpath):
        continue
    with open(fpath, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = row.get("input", "").strip()
            a = clean_text(row.get("output", ""))
            if not is_quality(q, a):
                continue
            if q in seen_questions:
                continue
            seen_questions.add(q)
            if len(a) > 500:
                # 截断到最后一个完整句子
                cut = a[:500].rfind('。')
                if cut > 100:
                    a = a[:cut+1]
                else:
                    a = a[:500]
            knowledge_samples.append({
                "messages": [
                    {"role": "system", "content": SYS_KNOWLEDGE},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ],
            })
print(f"  webqa: {len(knowledge_samples)} (去重后)")

# 2. COIG-CQIA wiki
print("=== COIG-CQIA wiki ===")
before = len(knowledge_samples)
wiki_dir = "data/raw/coig-cqia/wiki"
if os.path.isdir(wiki_dir):
    for fname in sorted(os.listdir(wiki_dir)):
        if not fname.endswith(".jsonl"):
            continue
        with open(os.path.join(wiki_dir, fname), encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                q = row.get("instruction", "").strip()
                a = clean_text(row.get("output", ""))
                if not is_quality(q, a):
                    continue
                if q in seen_questions:
                    continue
                seen_questions.add(q)
                if len(a) > 800:
                    cut = a[:800].rfind('。')
                    if cut > 200:
                        a = a[:cut+1]
                    else:
                        a = a[:800]
                knowledge_samples.append({
                    "messages": [
                        {"role": "system", "content": SYS_KNOWLEDGE},
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": a},
                    ],
                })
print(f"  coig_wiki: {len(knowledge_samples) - before}")

# 3. Chinese-SimpleQA
print("=== Chinese-SimpleQA ===")
before = len(knowledge_samples)
for simpleqa_path in ["data/raw/chinese-simpleqa/chinese_simpleqa.jsonl",
                       "data/raw/chinese-simpleqa/chinese_simpleqa.csv"]:
    if not os.path.exists(simpleqa_path):
        continue
    if simpleqa_path.endswith(".jsonl"):
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
                knowledge_samples.append({
                    "messages": [
                        {"role": "system", "content": SYS_KNOWLEDGE},
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": a},
                    ],
                })
        break
    elif simpleqa_path.endswith(".csv"):
        import csv
        with open(simpleqa_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                q = row.get("question", "").strip()
                a = row.get("answer", "").strip()
                if not q or not a:
                    continue
                knowledge_samples.append({
                    "messages": [
                        {"role": "system", "content": SYS_KNOWLEDGE},
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": a},
                    ],
                })
        break
print(f"  simpleqa: {len(knowledge_samples) - before}")

# 4. school_math (采样 2 万)
print("=== school_math ===")
before = len(knowledge_samples)
for math_path in ["data/pretrain_raw/belle_math/school_math_0.25M.json",
                   "data/raw/belle_math/school_math_0.25M.json"]:
    if os.path.exists(math_path):
        count = 0
        with open(math_path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                q = row.get("instruction", "").strip()
                a = clean_text(row.get("output", ""))
                if not is_quality(q, a):
                    continue
                if count >= 20000:
                    break
                knowledge_samples.append({
                    "messages": [
                        {"role": "system", "content": SYS_KNOWLEDGE},
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": a},
                    ],
                })
                count += 1
        print(f"  school_math: {count} (采样 20K)")
        break
else:
    print(f"  school_math: SKIP (file not found)")

print(f"\n知识数据小计: {len(knowledge_samples):,}")

# ============================================================
# BelleGroup 对话数据（采样等量混合）
# ============================================================
print("\n=== BelleGroup 对话数据 ===")
belle_samples = []
for belle_path in ["data/sft/sft_data.jsonl",
                    "data/sft/BelleGroup/train_3.5M_CN.json"]:
    if not os.path.exists(belle_path):
        continue

    # 判断格式
    with open(belle_path, encoding="utf-8") as f:
        first_line = f.readline()
        first = json.loads(first_line)

    if isinstance(first, list):
        # deal_dataset.py 处理后的 ChatML list 格式
        with open(belle_path, encoding="utf-8") as f:
            for line in f:
                try:
                    messages = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(messages, list) or len(messages) < 2:
                    continue
                # 检查 assistant 回复质量
                asst = [m['content'] for m in messages if m['role'] == 'assistant']
                if not asst or len(asst[0].strip()) < 10:
                    continue
                belle_samples.append({"messages": messages})
    elif 'conversations' in first:
        # 原始 BelleGroup 格式
        with open(belle_path, encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                convs = item.get('conversations', [])
                messages = [{"role": "system", "content": SYS_CHAT}]
                for c in convs:
                    if c['from'] == 'human':
                        messages.append({"role": "user", "content": c['value']})
                    elif c['from'] == 'assistant':
                        messages.append({"role": "assistant", "content": c['value']})
                if len(messages) < 3:
                    continue
                asst = [m['content'] for m in messages if m['role'] == 'assistant']
                if not asst or len(asst[0].strip()) < 10:
                    continue
                belle_samples.append({"messages": messages})
    break

print(f"  BelleGroup 总量: {len(belle_samples):,}")

# 采样与知识数据等量
n_knowledge = len(knowledge_samples)
if len(belle_samples) > n_knowledge:
    random.shuffle(belle_samples)
    belle_samples = belle_samples[:n_knowledge]
    print(f"  采样到: {len(belle_samples):,} (与知识数据等量)")

# ============================================================
# 混合
# ============================================================
all_samples = knowledge_samples + belle_samples
random.shuffle(all_samples)

print(f"\n{'='*40}")
print(f"总计: {len(all_samples):,} 条")
print(f"  知识数据: {len(knowledge_samples):,} ({100*len(knowledge_samples)/len(all_samples):.0f}%)")
print(f"  对话数据: {len(belle_samples):,} ({100*len(belle_samples)/len(all_samples):.0f}%)")

# 保存
save_dir = "data/mixed-sft"
ds_out = Dataset.from_list([{"messages": s["messages"]} for s in all_samples])
os.makedirs(save_dir, exist_ok=True)
ds_out.save_to_disk(save_dir)
print(f"\n已保存到 {save_dir}")
print(f"样本数: {len(ds_out):,}")

# 验证清洗效果
print("\n=== 清洗验证 ===")
e_count = sum(1 for s in all_samples
              for m in s['messages'] if '<e>' in m['content'])
print(f"  含 <e> 的消息数: {e_count} (应为 0)")

# 打印样例
print("\n=== 样例 ===")
for i in [0, len(all_samples)//4, len(all_samples)//2]:
    s = all_samples[i]
    print(f"\n--- sample {i} ---")
    for m in s['messages']:
        print(f"  [{m['role']}] {m['content'][:120]}")
