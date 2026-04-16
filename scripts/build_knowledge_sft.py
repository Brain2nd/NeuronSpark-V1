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
    """清洗文本：去所有已知异常标签、Markdown、URL、乱码等。

    顺序很重要：先处理会嵌套的（图片/链接）、再 HTML、再单独符号。
    """
    # 1. Markdown 图片（必须先于链接处理）![alt](url) → 删除
    text = re.sub(r'!\[[^\]]*\]\([^)]*\)', '', text)
    # 2. Markdown 链接 [text](url) → text
    text = re.sub(r'\[([^\]]*)\]\(([^)]*)\)', r'\1', text)
    # 3. 特殊标签
    text = re.sub(r'<e>', '', text)
    text = re.sub(r'<\|im_start\|>|<\|im_end\|>', '', text)
    # 4. 处理尖括号：保留数学比较符号 <=, <, >=, > 和中文书名号，删除 HTML 标签
    # <<xxx>> → 《xxx》
    text = re.sub(r'<<([^<>]+)>>', r'《\1》', text)
    # <中文xxx> → 《中文xxx》（中文标签当书名处理）
    text = re.sub(r'<([\u4e00-\u9fff][^<>]{0,30})>', r'《\1》', text)
    # HTML 标签：以字母开头或 / 开头
    text = re.sub(r'<(/?[a-zA-Z][^<>]*)>', '', text)
    # 5. URL (http/https) → 删除，包括残缺的 http://. 这种
    text = re.sub(r'https?://[^\s，。,)\]}]*', '', text)
    # 6. Markdown 代码块 ```xxx``` → 删除
    text = re.sub(r'```[\s\S]*?```', '', text, flags=re.MULTILINE)
    # 如果只有单个 ``` 残留也删掉
    text = re.sub(r'```', '', text)
    # 7. 行内代码 `xxx` → xxx
    text = re.sub(r'`([^`\n]+)`', r'\1', text)
    # 8. Markdown 标题 ## ### 等
    text = re.sub(r'(^|\n)#{1,6}\s*', r'\1', text)
    # 9. Markdown 粗体/斜体（多行配对）
    text = re.sub(r'\*{2,3}([^*]+?)\*{2,3}', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\*([^*\n]+?)\*', r'\1', text)
    # 清理所有剩余的 *
    text = re.sub(r'\*+', '', text)
    # 10. 参考标记 [1] [2] → 删除
    text = re.sub(r'\[\d+\]', '', text)
    # 11. 字面转义字符 \\n \\t → 空格
    text = text.replace('\\n', ' ').replace('\\t', ' ').replace('\\r', ' ')
    # 12. Unicode 转义 \\u0027 等 → 删除（通常是残缺的，正常文本不会有）
    text = re.sub(r'\\u[0-9a-fA-F]{4}', '', text)
    # 13. 方框等特殊符号（Unicode 25a0-25ff）
    text = re.sub(r'[\u25a0-\u25ff]', '', text)
    # 14. 乱码替换符
    text = text.replace('�', '')
    # 15. 连续标点（3+个相同）
    text = re.sub(r'([,，。.])\1{2,}', r'\1', text)
    text = re.sub(r'\?{3,}', '?', text)
    text = re.sub(r'!{3,}', '!', text)
    # 16. 多个换行合并
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 17. 多个空格合并
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r' *\n *', '\n', text)
    text = text.strip()
    return text


def is_quality(q, a, min_a_len=2):
    """质量过滤。

    注意：min_a_len 默认 2，因为 Chinese-SimpleQA 等答案可能只有"北京"这样的短答案。
    过短过滤主要靠调用时传入更高阈值（百科类 >=10）。
    """
    if not q or not a:
        return False
    a_stripped = a.strip()
    if len(a_stripped) < min_a_len:
        return False
    # 乱码过多
    if a.count('�') > 3:
        return False
    # 整段复述问题
    if len(q) > 20 and q in a:
        return False
    # URL 占比过高
    url_chars = sum(len(u) for u in re.findall(r'https?://\S+', a))
    if url_chars > len(a_stripped) * 0.3:
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
            q = clean_text(row.get("input", ""))
            a = clean_text(row.get("output", ""))
            # webqa 百科答案应该有一定长度（至少 5 字符）
            if not is_quality(q, a, min_a_len=5):
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
                q = clean_text(row.get("instruction", ""))
                a = clean_text(row.get("output", ""))
                # COIG wiki 是科普长回答，要求至少 10 字符
                if not is_quality(q, a, min_a_len=10):
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
                q = clean_text(row.get("question", ""))
                a = clean_text(row.get("answer", ""))
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
                q = clean_text(row.get("question", ""))
                a = clean_text(row.get("answer", ""))
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
                q = clean_text(row.get("instruction", ""))
                a = clean_text(row.get("output", ""))
                if not is_quality(q, a, min_a_len=10):
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
                # 清洗每条消息的内容
                cleaned = []
                for m in messages:
                    content = clean_text(m.get('content', ''))
                    if not content:
                        continue
                    cleaned.append({"role": m['role'], "content": content})
                if len(cleaned) < 2:
                    continue
                # 检查 assistant 回复质量
                asst = [m['content'] for m in cleaned if m['role'] == 'assistant']
                if not asst or len(asst[0].strip()) < 10:
                    continue
                belle_samples.append({"messages": cleaned})
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
                    val = clean_text(c.get('value', ''))
                    if not val:
                        continue
                    if c['from'] == 'human':
                        messages.append({"role": "user", "content": val})
                    elif c['from'] == 'assistant':
                        messages.append({"role": "assistant", "content": val})
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

# 验证清洗效果 - 完整扫描
print("\n=== 清洗验证 ===")
check_patterns = {
    '<e>': r'<e>',
    'HTML<tag>': r'<(/?[a-zA-Z])[^<>]*>',
    'Markdown##': r'(^|\n)#{1,6}\s',
    'Markdown**': r'\*\*[^*]+\*\*',
    'Markdown[..](..)': r'\[[^\]]+\]\([^)]+\)',
    'Markdown```': r'```',
    'URL http': r'https?://',
    '字面\\n': r'\\n',
    '字面\\t': r'\\t',
    '字面\\u': r'\\u[0-9a-fA-F]{4}',
    '乱码�': r'�',
    '方框◼': r'[\u25a0-\u25ff]',
    '连续句号...': r'\.{4,}',
    '连续逗号,,,': r',{3,}',
    '参考[1]': r'\[\d+\]',
}

total_bad = 0
for name, pat in check_patterns.items():
    cnt = sum(1 for s in all_samples
              for m in s['messages'] if re.search(pat, m['content']))
    if cnt > 0:
        print(f"  ⚠️  {name}: {cnt} 条残留")
        total_bad += cnt
    else:
        print(f"  ✓ {name}: 0")

# 长度分布统计（不当作异常，只是信息）
lengths = [len(m['content']) for s in all_samples
           for m in s['messages'] if m['role'] == 'assistant']
lengths.sort()
import statistics
print(f"\n  回复长度分布: min={lengths[0]} / "
      f"p25={lengths[len(lengths)//4]} / "
      f"median={statistics.median(lengths):.0f} / "
      f"p75={lengths[3*len(lengths)//4]} / "
      f"max={lengths[-1]}")

if total_bad == 0:
    print("\n  ✅ 所有清洗检查通过")
else:
    print(f"\n  ⚠️  {total_bad} 条清洗异常（需要处理）")

# 打印样例
print("\n=== 样例 ===")
for i in [0, len(all_samples)//4, len(all_samples)//2]:
    s = all_samples[i]
    print(f"\n--- sample {i} ---")
    for m in s['messages']:
        print(f"  [{m['role']}] {m['content'][:120]}")
