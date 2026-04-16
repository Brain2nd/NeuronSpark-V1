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
    """只清洗真正的垃圾。Markdown 等合法格式保留。

    清洗目标：
      - 特殊数据集标签（<e> 等 webqa 残留）
      - 乱码字符 (�)
      - 字面转义字符 \\n \\t \\u (应该已被解码)
      - 残缺/异常的 HTML 遗留（如 <br>、<p>）
      - 残缺 URL (http://.)
      - 连续相同标点 (4+个)
      - 方框等输入残留
      - 特殊标签 <|im_start|> <|im_end|>

    保留：
      - Markdown 标题 ## 、粗体 **、斜体 *、代码块 ```、行内代码 `
      - Markdown 链接 [text](url) 和图片 ![alt](url)
      - 合法 URL
      - 数学符号 <, <=, >, >=
      - 中文书名号 <xxx> → 《xxx》（已有大量《》写法，统一）
      - 参考标记 [1]
    """
    # 1. 特殊数据集标签
    text = re.sub(r'<e>', '', text)
    text = re.sub(r'<\|im_start\|>|<\|im_end\|>', '', text)
    # 2. HTML 标签残留（只删常见的真实 HTML 标签，不删内容）
    text = re.sub(r'</?(?:br|p|div|span|li|ul|ol|table|tr|td|th|hr|html|body|head|meta|h[1-6]|b|i|u|a|img|strong|em|code|pre)(?:\s[^<>]*)?>', '', text, flags=re.IGNORECASE)
    # 3. 字面转义字符 \\n \\t \\r
    text = text.replace('\\n', ' ').replace('\\t', ' ').replace('\\r', ' ')
    # 4. Unicode 转义 \\u0027 等 → 删除（应被 json.loads 解码；出现即数据有问题）
    text = re.sub(r'\\u[0-9a-fA-F]{4}', '', text)
    # 5. 方框等特殊符号（Unicode 25a0-25ff）
    text = re.sub(r'[\u25a0-\u25ff]', '', text)
    # 6. 乱码替换符
    text = text.replace('�', '')
    # 7. 残缺 URL（http://. 这种明显残缺的，删掉）
    text = re.sub(r'https?://[.,\s]*$', '', text)
    text = re.sub(r'https?://\.\s', ' ', text)
    # 8. 省略号压缩（仅处理 ....... 这种明显合法但过长的）
    text = re.sub(r'\.{4,}', '...', text)
    # 9. 零宽字符不属于内容，删除（其他控制字符由 is_quality 判定丢弃）
    text = text.replace('\u200b', '').replace('\u200c', '').replace('\u200d', '').replace('\ufeff', '')
    # 9. 多个换行合并（最多保留 2 个）
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 10. 多个空格合并
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r' *\n *', '\n', text)
    text = text.strip()
    return text


def is_junk(text):
    """检测文本是否含垃圾模式（网页残留、乱码等）。"""
    if not text:
        return False
    # 4+ 连续重复非字母数字字符（---- ____ ???? **** 等）
    if re.search(r'([\W_])\1{3,}', text):
        return True
    # 4+ 连续相同字母（aaaaaa）
    if re.search(r'([a-zA-Z])\1{3,}', text):
        return True
    # 控制字符 + Unicode 私有区
    bad = [c for c in re.findall(r'[\u0000-\u001f\u007f-\u009f\ue000-\uf8ff]', text) if c not in '\n\t']
    if bad:
        return True
    return False


def is_quality(q, a, min_a_len=2):
    """质量过滤：检查 q 和 a，任一含垃圾就丢弃。"""
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
    # 过滤 "As an AI" 拒答类回复（Alpaca 常见问题）
    refuse_phrases = [
        "as an ai language model",
        "as an ai, i cannot",
        "as an ai, i don't",
        "i'm just an ai",
        "i am just an ai",
        "i don't have the ability",
        "i cannot provide",
    ]
    a_lower = a.lower()
    for p in refuse_phrases:
        if p in a_lower:
            return False
    # q 或 a 含任何垃圾模式，丢弃
    if is_junk(q) or is_junk(a):
        return False
    # assistant 语义字符占比过低（全是符号/空白）
    letters = sum(1 for c in a if c.isalpha() or '\u4e00' <= c <= '\u9fff' or c.isdigit())
    if letters < len(a_stripped) * 0.3:
        return False
    return True


def clean_trivia_answer(a):
    """TriviaQA 答案有双引号转义，清理。"""
    # "xxx" "yyy" → xxx yyy （去掉多余引号）
    a = re.sub(r'""([^"]+)""', r'"\1"', a)  # """ → "
    # 前后多余引号
    a = a.strip('"')
    return a


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
            a_raw = clean_text(row.get("output", ""))
            # webqa 格式: "答案。扩展解释1<e>扩展解释2<e>..." (clean_text 已去掉 <e>)
            # 策略: 只保留第一句作为核心答案 + 最多一段简短补充
            # 找第一个 "。" 后是否还有内容
            first_period = a_raw.find('。')
            if first_period > 0 and first_period < len(a_raw) - 1:
                core = a_raw[:first_period + 1]  # 核心答案
                rest = a_raw[first_period + 1:].strip()
                # 再取一小段补充 (最多 200 字)
                if len(rest) > 200:
                    cut = rest[:200].rfind('。')
                    if cut > 30:
                        rest = rest[:cut + 1]
                    else:
                        rest = rest[:200]
                a = core + rest if rest else core
            else:
                a = a_raw

            if not is_quality(q, a, min_a_len=2):
                continue
            if q in seen_questions:
                continue
            seen_questions.add(q)
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
    def add_simpleqa(row):
        q = clean_text(row.get("question", ""))
        a = clean_text(row.get("answer", ""))
        if not q or not a:
            return
        if not is_quality(q, a, min_a_len=1):
            return
        # SimpleQA 答案很短（如"北京"），用领域系统提示让模型对齐
        cat = row.get("primary_category", "")
        sub = row.get("secondary_category", "")
        if cat and sub:
            sys_prompt = f"你是一个{cat}/{sub}领域的知识助手，请简短准确地回答。"
        else:
            sys_prompt = "你是一个知识助手，请简短准确地回答事实性问题。"
        knowledge_samples.append({
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ],
        })

    if simpleqa_path.endswith(".jsonl"):
        with open(simpleqa_path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                add_simpleqa(row)
        break
    elif simpleqa_path.endswith(".csv"):
        import csv
        with open(simpleqa_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                add_simpleqa(row)
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
# 英文知识数据（预训练是双语 55% 英文，SFT 也需要英文）
# ============================================================
print("\n=== 英文知识数据 ===")

SYS_EN = "You are a helpful assistant. Please answer the user's question accurately."

# 1. databricks/databricks-dolly-15k - 英文指令问答（按 category 分情况处理）
#
# 8 种 category 的处理:
#   open_qa / general_qa / brainstorming / creative_writing → 只用 instruction
#   closed_qa / information_extraction / summarization → context 在前（先给材料再问）
#   classification → instruction 在前（先说任务再给选项）
#
dolly_path = "data/raw/dolly-15k"
if os.path.isdir(dolly_path):
    print("=== Dolly-15k ===")
    before = len(knowledge_samples)
    cat_counts = {}
    for fname in os.listdir(dolly_path):
        if not fname.endswith(".jsonl"):
            continue
        with open(os.path.join(dolly_path, fname), encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                q = clean_text(row.get("instruction", ""))
                a = clean_text(row.get("response", ""))
                ctx = clean_text(row.get("context", ""))
                cat = row.get("category", "")

                # 按 category 拼接
                if cat in ("closed_qa", "information_extraction", "summarization"):
                    # 需要参考材料，材料在前
                    if ctx:
                        user_content = f"Context:\n{ctx}\n\nQuestion: {q}"
                    else:
                        user_content = q
                elif cat == "classification":
                    # 分类任务，指令在前（说明任务），context 如果是选项列表跟在后面
                    if ctx:
                        user_content = f"{q}\n\n{ctx}"
                    else:
                        user_content = q
                else:
                    # open_qa / general_qa / brainstorming / creative_writing
                    # 一般不需要 context，有也附加
                    if ctx:
                        user_content = f"{q}\n\n{ctx}"
                    else:
                        user_content = q

                if not is_quality(user_content, a, min_a_len=10):
                    continue
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
                knowledge_samples.append({
                    "messages": [
                        {"role": "system", "content": SYS_EN},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": a},
                    ],
                })
    print(f"  dolly: {len(knowledge_samples) - before}")
    for c, n in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"    {c}: {n}")

# 2. TriviaQA 从 SFT 中移除：
#    - answer 是 1-2 词的英文专有名词（"Easter", "Pilton"）
#    - assistant 只有几个 token，每个是罕见子词，loss 飙到 10+
#    - 已证实是 step 0/80 loss 飙升的主因
#    - TriviaQA 更适合做 RL（答案明确可验证，但长度不适合 SFT）

# 3. Alpaca-cleaned - 英文指令数据（备选）
alpaca_path = "data/raw/alpaca-cleaned"
if os.path.isdir(alpaca_path):
    print("=== Alpaca-cleaned ===")
    before = len(knowledge_samples)
    for fname in os.listdir(alpaca_path):
        if not (fname.endswith(".json") or fname.endswith(".jsonl")):
            continue
        fpath = os.path.join(alpaca_path, fname)
        # 判断是 json 数组还是 jsonl
        with open(fpath, encoding="utf-8") as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == '[':
                items = json.load(f)
            else:
                items = [json.loads(l) for l in f if l.strip()]
        # Alpaca 原模板：instruction 描述任务，input 是输入数据
        # 按原论文: Instruction 在前，Input 跟随 (不加 "### Input:" 标记，简化为换行)
        for row in items:
            q = clean_text(row.get("instruction", ""))
            a = clean_text(row.get("output", ""))
            inp = clean_text(row.get("input", ""))
            if inp:
                user_content = f"{q}\n\n{inp}"
            else:
                user_content = q
            if not is_quality(user_content, a, min_a_len=10):
                continue
            knowledge_samples.append({
                "messages": [
                    {"role": "system", "content": SYS_EN},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": a},
                ],
            })
    print(f"  alpaca: {len(knowledge_samples) - before}")

print(f"\n知识数据总计（中+英）: {len(knowledge_samples):,}")

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

    # 目标：最多 200K 条（减少内存占用和处理时间，足够混合）
    BELLE_LIMIT = 200000
    if isinstance(first, list):
        # deal_dataset.py 处理后的 ChatML list 格式
        with open(belle_path, encoding="utf-8") as f:
            for line in f:
                if len(belle_samples) >= BELLE_LIMIT:
                    break
                try:
                    messages = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(messages, list) or len(messages) < 2:
                    continue
                cleaned = []
                has_junk = False
                for m in messages:
                    content = clean_text(m.get('content', ''))
                    if not content:
                        continue
                    if is_junk(content):
                        has_junk = True
                        break
                    cleaned.append({"role": m['role'], "content": content})
                if has_junk or len(cleaned) < 2:
                    continue
                asst = [m['content'] for m in cleaned if m['role'] == 'assistant']
                if not asst or len(asst[0].strip()) < 10:
                    continue
                belle_samples.append({"messages": cleaned})
                if len(belle_samples) % 20000 == 0:
                    print(f"    ... BelleGroup 已读 {len(belle_samples):,}")
    elif 'conversations' in first:
        # 原始 BelleGroup 格式
        with open(belle_path, encoding="utf-8") as f:
            for line in f:
                if len(belle_samples) >= BELLE_LIMIT:
                    break
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                convs = item.get('conversations', [])
                messages = [{"role": "system", "content": SYS_CHAT}]
                has_junk = False
                for c in convs:
                    val = clean_text(c.get('value', ''))
                    if not val:
                        continue
                    if is_junk(val):
                        has_junk = True
                        break
                    if c['from'] == 'human':
                        messages.append({"role": "user", "content": val})
                    elif c['from'] == 'assistant':
                        messages.append({"role": "assistant", "content": val})
                if has_junk or len(messages) < 3:
                    continue
                asst = [m['content'] for m in messages if m['role'] == 'assistant']
                if not asst or len(asst[0].strip()) < 10:
                    continue
                belle_samples.append({"messages": messages})
                if len(belle_samples) % 20000 == 0:
                    print(f"    ... BelleGroup 已读 {len(belle_samples):,}")
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
    '<e>标签': r'<e>',
    'HTML残留': r'</?(?:br|p|div|span|li|ul|ol|table|tr|td|th|hr|html|body|head|meta|h[1-6])(?:\s|>)',
    '<|im_start|>': r'<\|im_start\|>',
    '<|im_end|>': r'<\|im_end\|>',
    '字面\\n': r'\\n',
    '字面\\t': r'\\t',
    '字面\\u': r'\\u[0-9a-fA-F]{4}',
    '乱码�': r'�',
    '方框◼': r'[\u25a0-\u25ff]',
    '残缺URL': r'https?://\.\s|https?://\s',
    '连续句号4+': r'\.{4,}',
    '连续问号4+': r'\?{4,}',
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
