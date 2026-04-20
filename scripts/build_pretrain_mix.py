"""
将所有 benchmark 数据集转换为预训练格式（纯自然文本）并混合。

转换策略（按数据集类型）：
  - 自然文本类：直接使用原文
  - 选择题类：拼接成 "问题 答案选项 正确答案是X" 的自然文本
  - 场景续写类：上下文 + 正确续写拼接成完整段落
  - 填空类：将占位符替换为正确答案
  - 阅读理解类：段落 + 问题 + 答案拼接成连贯文本

输出: data/neuronspark/ — 单一 Dataset，字段 ['text', 'source', 'lang']
"""
import os
import json
import random
from datasets import load_from_disk, Dataset

save_dir = 'data/benchmark'
out_dir = 'data/neuronspark'

random.seed(42)
all_samples = []


def add(text, source, lang='en'):
    text = text.strip()
    if text and len(text) > 10:
        all_samples.append({'text': text, 'source': source, 'lang': lang})


# ============================================================
# LAMBADA — 已永久移除
# 原因: lambada_openai 是 eval 测试集, 即便作为叙事 pretrain 文本也构成 test-set contamination.
# ============================================================

# ============================================================
# HellaSwag — 上下文 + 正确续写拼接
# ============================================================
print('HellaSwag...')
ds = load_from_disk(os.path.join(save_dir, 'hellaswag'))
before = len(all_samples)
for split in ds:
    for row in ds[split]:
        ctx = row['ctx'] if row['ctx'] else ''
        label = row['label']
        if isinstance(label, str):
            label = int(label) if label.isdigit() else 0
        endings = row['endings']
        if 0 <= label < len(endings):
            text = ctx.strip() + ' ' + endings[label].strip()
            add(text, 'hellaswag')
print(f'  +{len(all_samples) - before}')

# ============================================================
# WinoGrande — 填空替换
# ============================================================
print('WinoGrande...')
ds = load_from_disk(os.path.join(save_dir, 'winogrande'))
before = len(all_samples)
for split in ds:
    for row in ds[split]:
        sentence = row['sentence']
        ans = row['answer']
        if isinstance(ans, str):
            ans = int(ans) if ans.isdigit() else 1
        option = row['option1'] if ans == 1 else row['option2']
        text = sentence.replace('_', option)
        add(text, 'winogrande')
print(f'  +{len(all_samples) - before}')

# ============================================================
# BoolQ — 段落 + 问题 + 答案
# ============================================================
print('BoolQ...')
ds = load_from_disk(os.path.join(save_dir, 'boolq'))
before = len(all_samples)
for split in ds:
    for row in ds[split]:
        passage = row['passage']
        question = row['question']
        answer = 'Yes' if row['answer'] else 'No'
        text = f"{passage}\n\nQuestion: {question}\nAnswer: {answer}"
        add(text, 'boolq')
print(f'  +{len(all_samples) - before}')

# ============================================================
# ARC-Easy / ARC-Challenge — 选择题转自然文本
# ============================================================
for arc_name in ['arc_easy', 'arc_challenge']:
    print(f'{arc_name}...')
    ds = load_from_disk(os.path.join(save_dir, arc_name))
    before = len(all_samples)
    for split in ds:
        for row in ds[split]:
            q = row['question']
            choices = row['choices']
            labels = choices['label']
            texts = choices['text']
            answer_key = row['answerKey']
            # 找正确答案文本
            answer_text = ''
            options_str = ''
            for l, t in zip(labels, texts):
                options_str += f"\n{l}. {t}"
                if l == answer_key:
                    answer_text = t
            text = f"{q}{options_str}\n\nThe answer is {answer_key}: {answer_text}"
            add(text, arc_name)
    print(f'  +{len(all_samples) - before}')

# ============================================================
# MMLU — 多学科选择题
# ============================================================
print('MMLU...')
ds = load_from_disk(os.path.join(save_dir, 'mmlu'))
before = len(all_samples)
choice_labels = ['A', 'B', 'C', 'D']
for split in ds:
    for row in ds[split]:
        q = row['question']
        choices = row['choices']
        answer_idx = row['answer']
        if isinstance(answer_idx, str):
            answer_idx = int(answer_idx) if answer_idx.isdigit() else 0
        options_str = ''
        for i, c in enumerate(choices):
            options_str += f"\n{choice_labels[i]}. {c}"
        answer_label = choice_labels[answer_idx] if 0 <= answer_idx < 4 else 'A'
        answer_text = choices[answer_idx] if 0 <= answer_idx < len(choices) else ''
        subj = row.get('subject', '')
        text = f"Subject: {subj}\n{q}{options_str}\n\nThe answer is {answer_label}: {answer_text}"
        add(text, 'mmlu')
print(f'  +{len(all_samples) - before}')

# ============================================================
# PIQA — 目标 + 正确方案
# ============================================================
print('PIQA...')
ds = load_from_disk(os.path.join(save_dir, 'piqa'))
before = len(all_samples)
for split in ds:
    for row in ds[split]:
        goal = row['goal']
        label = row['label']
        if isinstance(label, str):
            label = int(label) if label.isdigit() else 0
        sol = row['sol1'] if label == 0 else row['sol2']
        text = f"{goal} {sol}"
        add(text, 'piqa')
print(f'  +{len(all_samples) - before}')

# ============================================================
# OpenBookQA — 科学选择题
# ============================================================
print('OpenBookQA...')
ds = load_from_disk(os.path.join(save_dir, 'openbookqa'))
before = len(all_samples)
for split in ds:
    for row in ds[split]:
        q = row['question_stem']
        choices = row['choices']
        labels = choices['label']
        texts = choices['text']
        answer_key = row['answerKey']
        answer_text = ''
        options_str = ''
        for l, t in zip(labels, texts):
            options_str += f"\n{l}. {t}"
            if l == answer_key:
                answer_text = t
        text = f"{q}{options_str}\n\nThe answer is {answer_key}: {answer_text}"
        add(text, 'openbookqa')
print(f'  +{len(all_samples) - before}')

# ============================================================
# SIQA — 社会常识
# ============================================================
print('SIQA...')
ds = load_from_disk(os.path.join(save_dir, 'siqa'))
before = len(all_samples)
for split in ds:
    for row in ds[split]:
        ctx = row['context']
        q = row['question']
        label = row['label']
        if isinstance(label, str):
            label = int(label) if label.isdigit() else 1
        answers = [row['answerA'], row['answerB'], row['answerC']]
        correct = answers[label - 1] if 1 <= label <= 3 else answers[0]
        text = f"{ctx} {q} {correct}"
        add(text, 'siqa')
print(f'  +{len(all_samples) - before}')

# ============================================================
# C3 — 中文阅读理解多选（段落 + 问题 + 正确答案）
# ============================================================
print('C3...')
ds = load_from_disk(os.path.join(save_dir, 'c3'))
before = len(all_samples)
for split in ds:
    for row in ds[split]:
        context = row['context']
        if isinstance(context, list):
            context = '\n'.join(context)
        q = row['question']
        answer = row.get('answer', '')
        if answer:
            text = f"{context}\n\n问题：{q}\n答案：{answer}"
        else:
            # test 集没有答案，只用 context
            choices = row.get('choice', [])
            if choices:
                options = '、'.join(choices)
                text = f"{context}\n\n问题：{q}\n选项：{options}"
            else:
                text = f"{context}\n\n问题：{q}"
        add(text, 'c3', 'zh')
print(f'  +{len(all_samples) - before}')

# ============================================================
# C-Eval — 中文多学科选择题
# ============================================================
print('C-Eval...')
ds = load_from_disk(os.path.join(save_dir, 'ceval'))
before = len(all_samples)
for split in ds:
    for row in ds[split]:
        q = row['question']
        a, b, c, d = row['A'], row['B'], row['C'], row['D']
        answer = row['answer']
        explanation = row.get('explanation', '')
        text = f"{q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\n\n答案是{answer}"
        if explanation:
            text += f"。{explanation}"
        add(text, 'ceval', 'zh')
print(f'  +{len(all_samples) - before}')

# ============================================================
# CMMLU — 中文多学科选择题
# ============================================================
print('CMMLU...')
ds = load_from_disk(os.path.join(save_dir, 'cmmlu'))
before = len(all_samples)
for split in ds:
    for row in ds[split]:
        q = row['Question']
        a, b, c, d = row['A'], row['B'], row['C'], row['D']
        answer = row['Answer']
        text = f"{q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\n\n答案是{answer}"
        add(text, 'cmmlu', 'zh')
print(f'  +{len(all_samples) - before}')

# ============================================================
# 充分混合并保存
# ============================================================
print(f'\n总计: {len(all_samples)} 条')
en = sum(1 for s in all_samples if s['lang'] == 'en')
zh = sum(1 for s in all_samples if s['lang'] == 'zh')
print(f'  英文: {en:,}')
print(f'  中文: {zh:,}')

# 充分打乱
random.shuffle(all_samples)

# 统计
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('./tokenizer/')
lengths = []
for s in random.sample(all_samples, min(5000, len(all_samples))):
    toks = tok(s['text'], add_special_tokens=False)['input_ids']
    lengths.append(len(toks))

avg_len = sum(lengths) // len(lengths)
max_len = max(lengths)
over_512 = sum(1 for l in lengths if l > 512)
over_1024 = sum(1 for l in lengths if l > 1024)
over_2048 = sum(1 for l in lengths if l > 2048)
print(f'  平均 token: {avg_len}')
print(f'  最大 token: {max_len}')
print(f'  >512: {over_512}/{len(lengths)} ({over_512/len(lengths)*100:.1f}%)')
print(f'  >1024: {over_1024}/{len(lengths)} ({over_1024/len(lengths)*100:.1f}%)')
print(f'  >2048: {over_2048}/{len(lengths)} ({over_2048/len(lengths)*100:.1f}%)')

# 按 source 统计
source_counts = {}
for s in all_samples:
    source_counts[s['source']] = source_counts.get(s['source'], 0) + 1
print('\n来源分布:')
for src, cnt in sorted(source_counts.items(), key=lambda x: -x[1]):
    print(f'  {src:<18} {cnt:>8,} ({cnt/len(all_samples)*100:.1f}%)')

# 保存
ds_out = Dataset.from_list(all_samples)
os.makedirs(out_dir, exist_ok=True)
ds_out.save_to_disk(out_dir)
print(f'\n已保存到 {out_dir}')
print(f'字段: {ds_out.column_names}')
print(f'样本数: {len(ds_out):,}')
