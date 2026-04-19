"""用 9 个 benchmark 的 train 集构造 SFT mix.

原则:
  - Prompt 模板严格对齐 lm-eval harness 用的模板 (让 SFT 直接教模型匹配测评形式)
  - Response = 正确选项的文本 (multi-choice) 或 'yes/no/True/False' (判断) 或 'A/B/C/D' (mmlu/ceval)
  - 每任务均匀采样/上采样到 ~10k, 合计 ~90k

输出:
  data/benchmark_sft_mix/{messages, source, lang}
"""
import os
import random
import re
from datasets import load_from_disk, Dataset

BASE = '/home/dgxspark/Desktop/NeuronSpark-V1/data/benchmark'
OUT_DIR = '/home/dgxspark/Desktop/NeuronSpark-V1/data/benchmark_sft_mix'
TARGET_PER_TASK = 10000
random.seed(0)


# ============ lm-eval 兼容的 preprocess 函数 ============

def _preprocess_hellaswag_text(text: str) -> str:
    """lm-eval hellaswag preprocessor: 去 [title] 标签, 去多余空格."""
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


# ============ 单任务格式化: 每题 → (prompt, response, source, lang) ============

def fmt_hellaswag(doc):
    ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
    prompt = _preprocess_hellaswag_text(doc["activity_label"] + ": " + ctx)
    response = _preprocess_hellaswag_text(doc["endings"][int(doc["label"])])
    return prompt, response


def fmt_piqa(doc):
    prompt = f"Question: {doc['goal']}\nAnswer:"
    sol = doc['sol1'] if int(doc['label']) == 0 else doc['sol2']
    return prompt, sol


def fmt_arc(doc):
    prompt = f"Question: {doc['question']}\nAnswer:"
    texts = doc['choices']['text']
    labels = doc['choices']['label']
    idx = labels.index(doc['answerKey']) if doc['answerKey'] in labels else 0
    return prompt, texts[idx]


def fmt_boolq(doc):
    prompt = f"{doc['passage']}\nQuestion: {doc['question']}?\nAnswer:"
    response = "yes" if doc['answer'] else "no"
    return prompt, response


def fmt_openbookqa(doc):
    prompt = doc['question_stem']
    texts = doc['choices']['text']
    labels = doc['choices']['label']
    idx = labels.index(doc['answerKey']) if doc['answerKey'] in labels else 0
    return prompt, texts[idx]


def fmt_winogrande(doc):
    """Winogrande: 用 '_' 前的上下文作 prompt, 正确选项作 response.
    lm-eval 的 partial_context = sentence.split(_)[0] + option, target = sentence.split(_)[1]
    但 SFT 简化为: prompt = 完整句子中把 '_' 替换为 mask, response = 正确选项."""
    sent = doc['sentence']
    idx = int(doc['answer']) - 1  # 1-indexed → 0-indexed
    opt = doc[f'option{idx + 1}']
    # 用 lm-eval 风格: 把 '_' 之前当 prompt 结尾, 正确 option 当续写
    parts = sent.split('_', 1)
    prompt = parts[0].rstrip() + " " + opt
    response = parts[1].lstrip() if len(parts) > 1 else ""
    return prompt, response


def fmt_mmlu(doc):
    choices = doc['choices']
    opts = '\n'.join(f"{c}. {t}" for c, t in zip('ABCD', choices))
    prompt = f"{doc['question']}\n{opts}\nAnswer:"
    response = 'ABCD'[int(doc['answer'])]
    return prompt, response


def fmt_ceval(doc):
    prompt = f"{doc['question']}\nA. {doc['A']}\nB. {doc['B']}\nC. {doc['C']}\nD. {doc['D']}\n答案:"
    return prompt, doc['answer']


# ============ 配置 ============

TASKS = [
    # (name, split, formatter, lang)
    ('hellaswag', 'train', fmt_hellaswag, 'en'),
    ('piqa', 'train', fmt_piqa, 'en'),
    ('arc_easy', 'train', fmt_arc, 'en'),
    ('arc_challenge', 'train', fmt_arc, 'en'),
    ('boolq', 'train', fmt_boolq, 'en'),
    ('openbookqa', 'train', fmt_openbookqa, 'en'),
    ('winogrande', 'train', fmt_winogrande, 'en'),
    ('mmlu', 'auxiliary_train', fmt_mmlu, 'en'),
    ('ceval', 'dev', fmt_ceval, 'zh'),
]


# ============ 采样到均匀 ============

def balance_to_target(items, target):
    """> target: 下采样; < target: 轮询重复 + 最后补齐."""
    if len(items) >= target:
        return random.sample(items, target)
    # upsample: 完整重复 N 次, 最后一次部分采样
    n_full = target // len(items)
    rem = target - n_full * len(items)
    result = items * n_full + random.sample(items, rem)
    random.shuffle(result)
    return result


# ============ 主流程 ============

def main():
    all_messages = []
    all_source = []
    all_lang = []

    for name, split, formatter, lang in TASKS:
        path = f'{BASE}/{name}'
        ds = load_from_disk(path)[split]
        print(f'--- {name}.{split} loaded {len(ds)} examples')
        pairs = []
        fails = 0
        for doc in ds:
            try:
                p, r = formatter(doc)
                if p and r and isinstance(p, str) and isinstance(r, str):
                    pairs.append((p.strip(), r.strip()))
            except Exception:
                fails += 1
        print(f'  formatted {len(pairs)} ({fails} failed)')

        balanced = balance_to_target(pairs, TARGET_PER_TASK)
        print(f'  balanced to {len(balanced)} (target={TARGET_PER_TASK})')

        for prompt, response in balanced:
            msgs = [
                {'role': 'user', 'content': prompt},
                {'role': 'assistant', 'content': response},
            ]
            all_messages.append(msgs)
            all_source.append(name)
            all_lang.append(lang)

    # shuffle 全局, 避免 source 聚簇
    idx = list(range(len(all_messages)))
    random.shuffle(idx)
    all_messages = [all_messages[i] for i in idx]
    all_source = [all_source[i] for i in idx]
    all_lang = [all_lang[i] for i in idx]

    print(f'\n=== 合计 {len(all_messages)} samples, 写入 {OUT_DIR} ===')
    ds_out = Dataset.from_dict({
        'messages': all_messages,
        'source': all_source,
        'lang': all_lang,
    })
    os.makedirs(OUT_DIR, exist_ok=True)
    ds_out.save_to_disk(OUT_DIR)

    # 分布检查
    from collections import Counter
    c = Counter(all_source)
    print(f'  source 分布: {dict(c)}')
    c = Counter(all_lang)
    print(f'  lang 分布:   {dict(c)}')


if __name__ == '__main__':
    main()
