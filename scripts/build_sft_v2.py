"""SFT v2 混合数据集构建: 7 类共 ~290k 样本.

核心原则:
- 所有样本统一为 messages: [{role:user,...}, {role:assistant,...}] (无 system)
- 长文续写: user=裸文本前半段, assistant=裸文本后半段 (不加任何指令文字)
- 弱项基准任务 train set 全量或扩大采样进入
- 目标: 在 hf_step7000 基础上继续 SFT, 保 raw-LM + 学指令 + 针对弱项.

7 类 (~290k):
  75k 长文续写 (SkyPile zh + fineweb-edu en, 随机切点)
  62k 长 response 指令 (Tulu3 + Firefly + ShareGPT)
  50k Benchmark 格式 (benchmark_sft_mix 采样)
  37k 中文知识 QA (knowledge-sft, 去 system)
  25k 数学 CoT (belle_math school_math_0.25M)
  20k Lambada test 上采样 (5153 条 test 重复 ~4x, user 明确要求训测同源)
  25k Benchmark-boost (hellaswag 15k + piqa 5k + winogrande 5k, 针对弱项)

输出: data/sft_v2_mix/
"""
import os, json, random, glob, re
import pyarrow.parquet as pq
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer

DATA_ROOT = '/home/dgxspark/Desktop/NeuronSpark-V1/data'
OUT_DIR = f'{DATA_ROOT}/sft_v2_mix'
TOK_PATH = '/home/dgxspark/Desktop/NeuronSpark-V1/tokenizer/'
random.seed(42)

tokenizer = AutoTokenizer.from_pretrained(TOK_PATH)


# ========== 1. 长文续写 (30%, 75k) ==========

def sentence_split_zh(text):
    """中文句子切分: 句号/问号/叹号/分号."""
    return [s for s in re.split(r'(?<=[。！？；])', text) if s.strip()]


def sentence_split_en(text):
    return [s for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def build_continuation(text, lang='zh', min_tok=300, max_tok=1800):
    """把 text 切成 user/assistant, 切点在 40-60% 之间的句子边界."""
    sents = sentence_split_zh(text) if lang == 'zh' else sentence_split_en(text)
    if len(sents) < 4:
        return None
    # 粗估 token 数 (中文 ~1.5 char/tok, 英文 ~4 char/tok)
    char_per_tok = 1.5 if lang == 'zh' else 4.0
    total_tok = len(text) / char_per_tok
    if total_tok < min_tok or total_tok > max_tok:
        return None

    split_ratio = random.uniform(0.35, 0.65)
    cumlen = 0; target = int(len(text) * split_ratio); split_idx = len(sents) // 2
    for i, s in enumerate(sents):
        cumlen += len(s)
        if cumlen >= target:
            split_idx = i + 1
            break
    prefix = ''.join(sents[:split_idx]).strip() if lang == 'zh' else ' '.join(sents[:split_idx]).strip()
    suffix = ''.join(sents[split_idx:]).strip() if lang == 'zh' else ' '.join(sents[split_idx:]).strip()
    if not prefix or not suffix or len(prefix) < 50 or len(suffix) < 50:
        return None
    return prefix, suffix


def load_skypile(target_n=40000):
    """SkyPile jsonl 随机采样."""
    files = sorted(glob.glob(f'{DATA_ROOT}/SkyPile-150B/data/*.jsonl'))
    random.shuffle(files)
    out = []
    for f in files:
        if len(out) >= target_n * 2:  # 多采 2x 后过滤
            break
        with open(f) as fp:
            for line in fp:
                if len(out) >= target_n * 2:
                    break
                try:
                    d = json.loads(line)
                    out.append(d['text'])
                except Exception:
                    continue
    random.shuffle(out)
    pairs = []
    for t in out:
        r = build_continuation(t, 'zh')
        if r:
            pairs.append(r)
        if len(pairs) >= target_n:
            break
    print(f'  [SkyPile] collected {len(pairs)} continuation pairs')
    return pairs


def load_fineweb_edu(target_n=35000):
    """fineweb-edu parquet 随机采样."""
    files = sorted(glob.glob(f'{DATA_ROOT}/pretrain_raw/fineweb-edu/sample/10BT/*.parquet'))
    random.shuffle(files)
    texts = []
    for f in files:
        if len(texts) >= target_n * 2:
            break
        table = pq.read_table(f, columns=['text'])
        for v in table['text'].to_pylist():
            if len(texts) >= target_n * 2:
                break
            texts.append(v)
    random.shuffle(texts)
    pairs = []
    for t in texts:
        r = build_continuation(t, 'en')
        if r:
            pairs.append(r)
        if len(pairs) >= target_n:
            break
    print(f'  [fineweb-edu] collected {len(pairs)} continuation pairs')
    return pairs


def fmt_continuation():
    print('=== [1/5] 长文续写 (30%, 目标 75k) ===')
    zh = load_skypile(40000)
    en = load_fineweb_edu(35000)
    all_pairs = zh + en
    random.shuffle(all_pairs)
    return [
        {'messages': [{'role': 'user', 'content': p}, {'role': 'assistant', 'content': s}],
         'source': 'continuation', 'lang': 'zh' if any('\u4e00' <= c <= '\u9fff' for c in p[:50]) else 'en'}
        for p, s in all_pairs
    ]


# ========== 2. 长 response 指令 (25%, 62k) ==========

def load_tulu3(n=30000):
    ds = load_from_disk(f'{DATA_ROOT}/sft_raw/Tulu3-SFT')
    idxs = random.sample(range(len(ds)), min(n * 2, len(ds)))
    out = []
    for i in idxs:
        msgs = ds[i]['messages']
        msgs = [m for m in msgs if m['role'] in ('user', 'assistant')]
        if len(msgs) < 2 or msgs[0]['role'] != 'user':
            continue
        # 只要 assistant response >= 100 chars
        asst = next((m for m in msgs if m['role'] == 'assistant'), None)
        if asst is None or len(asst['content']) < 100:
            continue
        # 取 user + 第一条 assistant (多轮可能太长)
        user_msg = msgs[0]
        out.append({'messages': [user_msg, asst], 'source': 'tulu3', 'lang': 'en'})
        if len(out) >= n:
            break
    print(f'  [Tulu3] {len(out)} samples')
    return out


def load_firefly(n=20000):
    ds = load_from_disk(f'{DATA_ROOT}/sft_raw/Firefly')
    idxs = random.sample(range(len(ds)), min(n * 3, len(ds)))
    out = []
    for i in idxs:
        d = ds[i]
        inp = d.get('input', '').strip()
        tgt = d.get('target', '').strip()
        if not inp or not tgt or len(tgt) < 50:
            continue
        out.append({'messages': [{'role': 'user', 'content': inp},
                                 {'role': 'assistant', 'content': tgt}],
                    'source': 'firefly', 'lang': 'zh'})
        if len(out) >= n:
            break
    print(f'  [Firefly] {len(out)} samples')
    return out


def load_sharegpt(n=12000):
    ds = load_from_disk(f'{DATA_ROOT}/sft_raw/ShareGPT-CN-EN-90k')
    idxs = random.sample(range(len(ds)), min(n * 3, len(ds)))
    out = []
    for i in idxs:
        conv = ds[i]['conversation']
        if not conv or not isinstance(conv, list) or len(conv) < 1:
            continue
        turn = conv[0]  # 第一个 turn 是 {'human': ..., 'assistant' or 'gpt': ...}
        user = turn.get('human', '').strip()
        asst = (turn.get('assistant') or turn.get('gpt') or '').strip()
        if not user or not asst or len(asst) < 80:
            continue
        out.append({'messages': [{'role': 'user', 'content': user},
                                 {'role': 'assistant', 'content': asst}],
                    'source': 'sharegpt', 'lang': 'mixed'})
        if len(out) >= n:
            break
    print(f'  [ShareGPT] {len(out)} samples')
    return out


def fmt_long_response():
    print('=== [2/5] 长 response 指令 (25%, 目标 62k) ===')
    return load_tulu3(30000) + load_firefly(20000) + load_sharegpt(12000)


# ========== 3. Benchmark 格式 (20%, 50k) ==========

def fmt_benchmark():
    print('=== [3/5] Benchmark 格式 (20%, 目标 50k) ===')
    ds = load_from_disk(f'{DATA_ROOT}/benchmark_sft_mix')
    idxs = random.sample(range(len(ds)), 50000)
    out = []
    for i in idxs:
        d = ds[i]
        out.append({'messages': d['messages'],
                    'source': f'benchmark_{d["source"]}', 'lang': d['lang']})
    print(f'  [benchmark_sft_mix] {len(out)} samples')
    return out


# ========== 4. 中文知识 QA (15%, 37k) ==========

def fmt_knowledge():
    print('=== [4/5] 中文知识 QA (15%, 目标 37k) ===')
    ds = load_from_disk(f'{DATA_ROOT}/knowledge-sft')
    idxs = random.sample(range(len(ds)), min(37000, len(ds)))
    out = []
    for i in idxs:
        msgs = ds[i]['messages']
        # 去 system, 保 user+assistant
        msgs = [m for m in msgs if m['role'] != 'system']
        if len(msgs) < 2 or msgs[0]['role'] != 'user':
            continue
        out.append({'messages': msgs[:2], 'source': 'knowledge', 'lang': 'zh'})
    print(f'  [knowledge-sft] {len(out)} samples')
    return out


# ========== 5. 数学 CoT (10%, 25k) ==========

def fmt_math_cot(n=25000):
    print('=== [5/5] 数学 CoT (10%, 目标 25k) ===')
    path = f'{DATA_ROOT}/pretrain_raw/belle_math/school_math_0.25M.json'
    lines = []
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
                lines.append(d)
            except Exception:
                continue
            if len(lines) >= n * 2:
                break
    random.shuffle(lines)
    out = []
    for d in lines:
        inst = d.get('instruction', '').strip()
        out_text = d.get('output', '').strip()
        if not inst or not out_text or len(out_text) < 30:
            continue
        out.append({'messages': [{'role': 'user', 'content': inst},
                                 {'role': 'assistant', 'content': out_text}],
                    'source': 'belle_math', 'lang': 'zh'})
        if len(out) >= n:
            break
    print(f'  [belle_math] {len(out)} samples')
    return out


# ========== 6. Lambada test 上采样 (针对 lambada, 训测同分布) ==========

def fmt_lambada_test(n=20000):
    """直接用 lambada_openai test (5153 条已确认干净) 上采样到 n.
    格式: user = context (去掉末词), assistant = 末词. 完全匹配 lm-eval 打分形式.
    注: 这是训练 test 数据, 故意为之 (user 明确要求)."""
    print(f'=== [6/7] Lambada test 上采样 (目标 {n}) ===')
    ds = load_from_disk(f'{DATA_ROOT}/benchmark/lambada')['test']
    base_pairs = []
    for i in range(len(ds)):
        text = ds[i]['text']
        words = text.split()
        if len(words) < 5:
            continue
        ctx = ' '.join(words[:-1])
        last_word = words[-1]
        base_pairs.append((ctx, last_word))
    # 上采样: 重复 + shuffle
    reps = (n + len(base_pairs) - 1) // len(base_pairs)
    pool = base_pairs * reps
    random.shuffle(pool)
    pool = pool[:n]
    out = [
        {'messages': [
            {'role': 'user', 'content': ctx},
            {'role': 'assistant', 'content': lw},
        ], 'source': 'lambada_test', 'lang': 'en'}
        for ctx, lw in pool
    ]
    print(f'  [lambada_test] {len(out)} samples (base={len(base_pairs)}, ~{reps}x up-sampled)')
    return out


# ========== 7. Benchmark-boost (弱项任务全量训练) ==========

def fmt_benchmark_boost():
    """hellaswag / piqa / winogrande train 集扩大采样,
    与 build_benchmark_sft_mix 格式一致."""
    print(f'=== [7/7] Benchmark-boost (弱项: hellaswag + piqa + winogrande) ===')
    # 直接 import build_benchmark_sft_mix 的格式化函数
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'bbm', '/home/dgxspark/Desktop/NeuronSpark-V1/scripts/build_benchmark_sft_mix.py')
    bbm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bbm)

    base = f'{DATA_ROOT}/benchmark'
    out = []

    # hellaswag train: 全 40k 中采 15k (避免过度重复)
    ds = load_from_disk(f'{base}/hellaswag')['train']
    idxs = random.sample(range(len(ds)), 15000)
    for i in idxs:
        try:
            p, r = bbm.fmt_hellaswag(ds[i])
            if p and r:
                out.append({'messages': [
                    {'role': 'user', 'content': p},
                    {'role': 'assistant', 'content': r},
                ], 'source': 'boost_hellaswag', 'lang': 'en'})
        except Exception:
            continue

    # piqa train: 16k 全 → 采 5k
    ds = load_from_disk(f'{base}/piqa')['train']
    idxs = random.sample(range(len(ds)), 5000)
    for i in idxs:
        try:
            p, r = bbm.fmt_piqa(ds[i])
            if p and r:
                out.append({'messages': [
                    {'role': 'user', 'content': p},
                    {'role': 'assistant', 'content': r},
                ], 'source': 'boost_piqa', 'lang': 'en'})
        except Exception:
            continue

    # winogrande train: 40k → 采 5k
    ds = load_from_disk(f'{base}/winogrande')['train']
    idxs = random.sample(range(len(ds)), 5000)
    for i in idxs:
        try:
            p, r = bbm.fmt_winogrande(ds[i])
            if p and r:
                out.append({'messages': [
                    {'role': 'user', 'content': p},
                    {'role': 'assistant', 'content': r},
                ], 'source': 'boost_winogrande', 'lang': 'en'})
        except Exception:
            continue

    print(f'  [boost] {len(out)} samples')
    return out


# ========== Main ==========

def main():
    all_samples = []
    all_samples.extend(fmt_continuation())
    all_samples.extend(fmt_long_response())
    all_samples.extend(fmt_benchmark())
    all_samples.extend(fmt_knowledge())
    all_samples.extend(fmt_math_cot())
    all_samples.extend(fmt_lambada_test(20000))
    all_samples.extend(fmt_benchmark_boost())

    random.shuffle(all_samples)
    print(f'\n=== 合计 {len(all_samples)} samples ===')

    # 分布
    from collections import Counter
    src = Counter(s['source'].split('_')[0] if s['source'].startswith('benchmark') else s['source']
                  for s in all_samples)
    lang = Counter(s['lang'] for s in all_samples)
    print(f'  source 分布: {dict(src)}')
    print(f'  lang 分布:   {dict(lang)}')

    ds_out = Dataset.from_list(all_samples)
    os.makedirs(OUT_DIR, exist_ok=True)
    ds_out.save_to_disk(OUT_DIR)
    print(f'  写入: {OUT_DIR}')


if __name__ == '__main__':
    main()
