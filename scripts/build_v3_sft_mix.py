"""Build v3 SFT mix v3 (final): English-dominant, modern model only.

组成（目标 ~620k）：
  - 现有 SFT-v2-Mix 减 lambada               ~272.5k
  - smoltalk2 everyday_conversations_no_think  30k    (Qwen3-32B EN 多轮)
  - smoltalk2 systemchats_30k_no_think 多轮部分  20k   (Qwen3-32B EN 多轮系统)
  - smoltalk2 LongAlign_no_think ZH 部分        10k    (Qwen3-32B ZH 长 ctx 单轮)
  - WildChat-1M GPT-4 + EN + ≥3 轮             26k    (真人 + GPT-4)
  - WildChat-1M GPT-4 + ZH + ≥3 轮             ~5.7k  (真人 + GPT-4)
  - OpenThoughts-114k                          73k    (R1 EN thinking 单轮)
  - QwQ-LongCoT-130K                           73k    (QwQ EN thinking 单轮)
  - Congliu/Chinese-DeepSeek-R1-Distill-110k-SFT ~110k (R1 ZH thinking 单轮)

输出: data/v3_sft_mix_raw, schema (messages/source/lang)
"""
from __future__ import annotations
import argparse, re, random
from datasets import load_dataset, load_from_disk, Dataset, concatenate_datasets

_THOUGHT_RE = re.compile(
    r'<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>.*?<\|begin_of_solution\|>(.*?)<\|end_of_solution\|>',
    re.DOTALL,
)


def parse_open_thoughts(s):
    convs = s.get('conversations') or []
    if len(convs) != 2 or convs[0].get('from') != 'user': return None
    user = (convs[0].get('value') or '').strip()
    asst = convs[1].get('value') or ''
    m = _THOUGHT_RE.search(asst)
    if not m or not user: return None
    th, sol = m.group(1).strip(), m.group(2).strip()
    if not th or not sol: return None
    return {'messages':[{'role':'user','content':user},
                        {'role':'assistant','content':f'<think>\n{th}\n</think>\n\n{sol}'}],
            'source':'open-thoughts','lang':'en'}


def parse_qwq(s):
    p = (s.get('problem') or '').strip()
    q = (s.get('qwq') or '').strip()
    if not p or not q or '**Final Answer**' not in q: return None
    parts = q.split('**Final Answer**', 1)
    r, f = parts[0].strip(), parts[1].strip()
    if not r or not f: return None
    return {'messages':[{'role':'user','content':p},
                        {'role':'assistant','content':f'<think>\n{r}\n</think>\n\n**Final Answer**\n\n{f}'}],
            'source':'qwq-longcot','lang':'en'}


def parse_congliu(s):
    inst = (s.get('instruction') or '').strip()
    inp = (s.get('input') or '').strip()
    out = (s.get('output') or '').strip()
    if not inst or not out: return None
    user = f'{inst}\n{inp}' if inp else inst
    return {'messages':[{'role':'user','content':user},
                        {'role':'assistant','content':out}],
            'source':'congliu-r1-zh','lang':'zh'}


def parse_smoltalk_passthrough(source_label, lang_label='en', require_multi=False, require_lang_zh=False):
    def fn(s):
        msgs = s.get('messages') or []
        if not msgs or len(msgs) < 2 or len(msgs) % 2 != 0: return None
        for m in msgs:
            if m.get('role') not in ('user','assistant','system'): return None
            if not (m.get('content') or '').strip(): return None
        if msgs[0]['role'] not in ('user','system'): return None
        if msgs[-1]['role'] != 'assistant': return None
        if require_multi and len(msgs) <= 2: return None
        if require_lang_zh:
            text = ''.join(m['content'][:300] for m in msgs[:2])
            zh = sum(1 for c in text if '一'<=c<='鿿')
            if zh / max(len(text),1) < 0.3: return None
        return {'messages':msgs,'source':source_label,'lang':lang_label}
    return fn


def parse_wildchat(lang_filter):
    def fn(s):
        model = (s.get('model') or '').lower()
        if not model.startswith('gpt-4'): return None
        if (s.get('language') or '').lower() != lang_filter: return None
        if s.get('toxic'): return None
        if (s.get('turn') or 0) < 3: return None
        convs = s.get('conversation') or []
        if not convs or len(convs) % 2 != 0: return None
        msgs = []
        for c in convs:
            role = c.get('role')
            if role not in ('user','assistant'): return None
            content = (c.get('content') or '').strip()
            if not content: return None
            msgs.append({'role':role,'content':content})
        if msgs[0]['role'] != 'user': return None
        return {'messages':msgs,'source':f'wildchat-gpt4-{lang_filter[:2]}',
                'lang':'en' if lang_filter=='english' else 'zh'}
    return fn


def collect(ds, parser, target, label):
    out, seen = [], 0
    for s in ds:
        seen += 1
        p = parser(s)
        if p is not None:
            out.append(p)
            if len(out) % 5000 == 0:
                print(f'  [{label}] {len(out)}/{target} (seen {seen})', flush=True)
        if len(out) >= target: break
    print(f'  [{label}] final: {len(out)} (seen {seen})', flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--existing', default='data/v3_sft_v2_mix_raw')
    ap.add_argument('--out', default='data/v3_sft_mix_raw')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--ot_n',         type=int, default=73000)
    ap.add_argument('--qwq_n',        type=int, default=73000)
    ap.add_argument('--congliu_n',    type=int, default=110000)
    ap.add_argument('--smolt_everyday_n', type=int, default=30000)
    ap.add_argument('--smolt_systemchat_n', type=int, default=20000)
    ap.add_argument('--smolt_longalign_zh_n', type=int, default=10000)
    ap.add_argument('--wildchat_en_n', type=int, default=26000)
    ap.add_argument('--wildchat_zh_n', type=int, default=10000)  # 实际拿到 ~5.7k
    args = ap.parse_args()

    random.seed(args.seed)
    parts = []

    # 1) existing minus lambada
    print('[1] existing -lambada', flush=True)
    ex = load_from_disk(args.existing)
    keep = ex.filter(lambda x: x['source'] != 'lambada_test', num_proc=8)
    print(f'  {len(ex)} -> {len(keep)}', flush=True)

    # 2) OpenThoughts
    print('\n[2] OpenThoughts-114k', flush=True)
    ds = load_dataset('open-thoughts/OpenThoughts-114k', split='train').shuffle(seed=args.seed)
    parts.append(collect(ds, parse_open_thoughts, args.ot_n, 'OpenThoughts'))

    # 3) QwQ
    print('\n[3] QwQ-LongCoT-130K', flush=True)
    ds = load_dataset('amphora/QwQ-LongCoT-130K', split='train').shuffle(seed=args.seed)
    parts.append(collect(ds, parse_qwq, args.qwq_n, 'QwQ'))

    # 4) Congliu R1 ZH
    print('\n[4] Congliu Chinese-R1-Distill-110k-SFT', flush=True)
    ds = load_dataset('Congliu/Chinese-DeepSeek-R1-Distill-data-110k-SFT', split='train').shuffle(seed=args.seed)
    parts.append(collect(ds, parse_congliu, args.congliu_n, 'Congliu-zh'))

    # 5) smoltalk2 everyday convs (multi-turn, EN, no_think)
    print('\n[5] smoltalk2 everyday_conversations_no_think', flush=True)
    ds = load_dataset('HuggingFaceTB/smoltalk2', 'SFT',
                      split='smoltalk_smollm3_everyday_conversations_no_think', streaming=True)
    parts.append(collect(ds, parse_smoltalk_passthrough('smolt-everyday','en',require_multi=True),
                         args.smolt_everyday_n, 'smolt-everyday'))

    # 6) smoltalk2 systemchats (multi-turn, EN, no_think)
    print('\n[6] smoltalk2 systemchats_30k_no_think (multi-turn only)', flush=True)
    ds = load_dataset('HuggingFaceTB/smoltalk2', 'SFT',
                      split='smoltalk_smollm3_systemchats_30k_no_think', streaming=True)
    parts.append(collect(ds, parse_smoltalk_passthrough('smolt-systemchat','en',require_multi=True),
                         args.smolt_systemchat_n, 'smolt-systemchat'))

    # 7) smoltalk2 LongAlign ZH 部分 (long context, ZH, no_think)
    print('\n[7] smoltalk2 LongAlign ZH 子集', flush=True)
    ds = load_dataset('HuggingFaceTB/smoltalk2', 'SFT',
                      split='LongAlign_64k_context_lang_annotated_lang_6_no_think', streaming=True)
    parts.append(collect(ds, parse_smoltalk_passthrough('smolt-longalign-zh','zh',require_lang_zh=True),
                         args.smolt_longalign_zh_n, 'LongAlign-zh'))

    # 8) WildChat EN
    print('\n[8] WildChat-1M EN gpt-4 ≥3 轮', flush=True)
    ds = load_dataset('allenai/WildChat-1M', split='train', streaming=True)
    parts.append(collect(ds, parse_wildchat('english'), args.wildchat_en_n, 'WildChat-en'))

    # 9) WildChat ZH (上面流到中后段了，重新流一遍)
    print('\n[9] WildChat-1M ZH gpt-4 ≥3 轮', flush=True)
    ds = load_dataset('allenai/WildChat-1M', split='train', streaming=True)
    parts.append(collect(ds, parse_wildchat('chinese'), args.wildchat_zh_n, 'WildChat-zh'))

    # concat all
    print('\n[concat]', flush=True)
    all_new = sum(parts, [])
    new_ds = Dataset.from_list(all_new)
    final = concatenate_datasets([keep, new_ds]).shuffle(seed=args.seed)

    # stats
    from collections import Counter
    print(f'\n=== final stats ===', flush=True)
    src_c = Counter(final['source']); lang_c = Counter(final['lang'])
    print(f'  total: {len(final)}')
    print(f'  by source:')
    for s, c in src_c.most_common(): print(f'    {c:>8}  {s}')
    print(f'  by lang:')
    for l, c in lang_c.most_common():
        print(f'    {c:>8}  {l}  ({c/len(final)*100:.1f}%)')

    final.save_to_disk(args.out)
    print(f'\nDONE -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
