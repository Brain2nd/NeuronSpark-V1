"""
将所有 benchmark 数据集转换为 SFT 对话格式（ChatML messages）并混合。

转换策略：
  - 选择题：user 提问 + assistant 回答正确选项并简要解释
  - 续写/补全：user 给上下文要求续写 + assistant 给出正确续写
  - 判断题：user 段落+问题 + assistant 回答 Yes/No
  - 填空题：user 给句子 + assistant 给出正确填充
  - 自然文本：user 要求续写 + assistant 续写

输出: data/neuronspark-sft/ — Dataset，字段 ['messages', 'source', 'lang']
       messages 格式: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
"""
import os
import random
from datasets import load_from_disk, Dataset

save_dir = 'data/benchmark'
out_dir = 'data/neuronspark-sft'

random.seed(42)
all_samples = []

SYS_EN = "You are a helpful assistant."
SYS_ZH = "你是一个有帮助的助手。"


def add(messages, source, lang='en'):
    if messages and len(messages) >= 2:
        all_samples.append({'messages': messages, 'source': source, 'lang': lang})


# ============================================================
# LAMBADA — 已永久移除
# 原因: 1) lambada_openai 的 test 集入 SFT 训练是 test-set 污染;
#       2) lambada 任务形式与 chat/SFT 模型不兼容, 不再评测, 也不该入训练.
# ============================================================

# ============================================================
# HellaSwag — user: 上下文 + 选项 / assistant: 正确续写
# ============================================================
print('HellaSwag...')
ds = load_from_disk(os.path.join(save_dir, 'hellaswag'))
before = len(all_samples)
for split in ds:
    for row in ds[split]:
        ctx = row['ctx'].strip() if row['ctx'] else ''
        label = row['label']
        if isinstance(label, str):
            label = int(label) if label.isdigit() else 0
        endings = row['endings']
        if not (0 <= label < len(endings)):
            continue
        options = '\n'.join([f"{chr(65+i)}. {e}" for i, e in enumerate(endings)])
        correct_label = chr(65 + label)
        correct_text = endings[label]
        add([
            {"role": "system", "content": SYS_EN},
            {"role": "user", "content": f"Choose the most logical continuation:\n\n{ctx}\n\n{options}"},
            {"role": "assistant", "content": f"The answer is {correct_label}: {correct_text}"},
        ], 'hellaswag')
print(f'  +{len(all_samples) - before}')

# ============================================================
# WinoGrande — user: 填空句子 / assistant: 正确答案
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
        o1, o2 = row['option1'], row['option2']
        correct = o1 if ans == 1 else o2
        add([
            {"role": "system", "content": SYS_EN},
            {"role": "user", "content": f"Fill in the blank with the correct option.\n\n{sentence}\n\nA. {o1}\nB. {o2}"},
            {"role": "assistant", "content": f"The answer is {'A' if ans == 1 else 'B'}: {correct}"},
        ], 'winogrande')
print(f'  +{len(all_samples) - before}')

# ============================================================
# BoolQ — user: 段落+问题 / assistant: Yes/No
# ============================================================
print('BoolQ...')
ds = load_from_disk(os.path.join(save_dir, 'boolq'))
before = len(all_samples)
for split in ds:
    for row in ds[split]:
        passage = row['passage']
        question = row['question']
        answer = 'Yes' if row['answer'] else 'No'
        add([
            {"role": "system", "content": SYS_EN},
            {"role": "user", "content": f"Based on the passage below, answer the question with Yes or No.\n\nPassage: {passage}\n\nQuestion: {question}"},
            {"role": "assistant", "content": answer},
        ], 'boolq')
print(f'  +{len(all_samples) - before}')

# ============================================================
# ARC-Easy / ARC-Challenge — user: 科学问题 / assistant: 答案
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
            answer_text = ''
            options = '\n'.join([f"{l}. {t}" for l, t in zip(labels, texts)])
            for l, t in zip(labels, texts):
                if l == answer_key:
                    answer_text = t
            add([
                {"role": "system", "content": SYS_EN},
                {"role": "user", "content": f"{q}\n\n{options}"},
                {"role": "assistant", "content": f"The answer is {answer_key}: {answer_text}"},
            ], arc_name)
    print(f'  +{len(all_samples) - before}')

# ============================================================
# MMLU — user: 学科问题 / assistant: 答案
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
        subj = row.get('subject', '')
        options = '\n'.join([f"{choice_labels[i]}. {c}" for i, c in enumerate(choices)])
        answer_label = choice_labels[answer_idx] if 0 <= answer_idx < 4 else 'A'
        answer_text = choices[answer_idx] if 0 <= answer_idx < len(choices) else ''
        add([
            {"role": "system", "content": SYS_EN},
            {"role": "user", "content": f"[{subj}] {q}\n\n{options}"},
            {"role": "assistant", "content": f"The answer is {answer_label}: {answer_text}"},
        ], 'mmlu')
print(f'  +{len(all_samples) - before}')

# ============================================================
# PIQA — user: 目标 / assistant: 正确方案
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
        s1, s2 = row['sol1'], row['sol2']
        correct_label = 'A' if label == 0 else 'B'
        correct = s1 if label == 0 else s2
        add([
            {"role": "system", "content": SYS_EN},
            {"role": "user", "content": f"{goal}\n\nA. {s1}\nB. {s2}"},
            {"role": "assistant", "content": f"The answer is {correct_label}: {correct}"},
        ], 'piqa')
print(f'  +{len(all_samples) - before}')

# ============================================================
# OpenBookQA
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
        options = '\n'.join([f"{l}. {t}" for l, t in zip(labels, texts)])
        for l, t in zip(labels, texts):
            if l == answer_key:
                answer_text = t
        add([
            {"role": "system", "content": SYS_EN},
            {"role": "user", "content": f"{q}\n\n{options}"},
            {"role": "assistant", "content": f"The answer is {answer_key}: {answer_text}"},
        ], 'openbookqa')
print(f'  +{len(all_samples) - before}')

# ============================================================
# SIQA
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
        correct_label = chr(64 + label)  # 1->A, 2->B, 3->C
        correct = answers[label - 1] if 1 <= label <= 3 else answers[0]
        options = f"A. {row['answerA']}\nB. {row['answerB']}\nC. {row['answerC']}"
        add([
            {"role": "system", "content": SYS_EN},
            {"role": "user", "content": f"{ctx}\n\n{q}\n\n{options}"},
            {"role": "assistant", "content": f"The answer is {correct_label}: {correct}"},
        ], 'siqa')
print(f'  +{len(all_samples) - before}')

# ============================================================
# C3 — 中文阅读理解
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
        choices = row.get('choice', [])
        answer = row.get('answer', '')
        if choices:
            options = '\n'.join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
            if answer:
                # 找到答案对应的选项
                ans_label = ''
                for i, c in enumerate(choices):
                    if c == answer:
                        ans_label = chr(65 + i)
                        break
                if not ans_label:
                    ans_label = 'A'
                add([
                    {"role": "system", "content": SYS_ZH},
                    {"role": "user", "content": f"阅读以下文章，回答问题。\n\n{context}\n\n问题：{q}\n\n{options}"},
                    {"role": "assistant", "content": f"答案是{ans_label}：{answer}"},
                ], 'c3', 'zh')
            else:
                add([
                    {"role": "system", "content": SYS_ZH},
                    {"role": "user", "content": f"阅读以下文章，回答问题。\n\n{context}\n\n问题：{q}\n\n{options}"},
                    {"role": "assistant", "content": f"根据文章内容，{choices[0]}"},
                ], 'c3', 'zh')
print(f'  +{len(all_samples) - before}')

# ============================================================
# C-Eval — 中文多学科
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
        resp = f"答案是{answer}"
        if explanation:
            resp += f"。{explanation}"
        add([
            {"role": "system", "content": SYS_ZH},
            {"role": "user", "content": f"{q}\n\nA. {a}\nB. {b}\nC. {c}\nD. {d}"},
            {"role": "assistant", "content": resp},
        ], 'ceval', 'zh')
print(f'  +{len(all_samples) - before}')

# ============================================================
# CMMLU — 中文多学科
# ============================================================
print('CMMLU...')
ds = load_from_disk(os.path.join(save_dir, 'cmmlu'))
before = len(all_samples)
for split in ds:
    for row in ds[split]:
        q = row['Question']
        a, b, c, d = row['A'], row['B'], row['C'], row['D']
        answer = row['Answer']
        # 找答案文本
        ans_map = {'A': a, 'B': b, 'C': c, 'D': d}
        ans_text = ans_map.get(answer, '')
        add([
            {"role": "system", "content": SYS_ZH},
            {"role": "user", "content": f"{q}\n\nA. {a}\nB. {b}\nC. {c}\nD. {d}"},
            {"role": "assistant", "content": f"答案是{answer}：{ans_text}"},
        ], 'cmmlu', 'zh')
print(f'  +{len(all_samples) - before}')

# ============================================================
# 混合并保存
# ============================================================
print(f'\n总计: {len(all_samples)} 条')
en = sum(1 for s in all_samples if s['lang'] == 'en')
zh = sum(1 for s in all_samples if s['lang'] == 'zh')
print(f'  英文: {en:,}')
print(f'  中文: {zh:,}')

random.shuffle(all_samples)

# 来源分布
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

# 打印几条样例
print('\n=== 样例 ===')
for i in [0, 1, 2]:
    s = all_samples[i]
    print(f'\n--- {s["source"]} ({s["lang"]}) ---')
    for m in s['messages']:
        print(f'  [{m["role"]}]: {m["content"][:120]}...' if len(m["content"]) > 120 else f'  [{m["role"]}]: {m["content"]}')
