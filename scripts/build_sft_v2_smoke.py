"""Smoke test: 每类跑极小量确认数据源都能读."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_sft_v2 import (
    load_skypile, load_fineweb_edu, load_tulu3, load_firefly, load_sharegpt,
    fmt_benchmark, fmt_knowledge, fmt_math_cot,
)
import random
random.seed(42)

# 用原函数但小量: 每类 100 条
print('--- skypile ---')
s = load_skypile(100)
print(f'  got {len(s)}; first prefix[:80]: {s[0][0][:80] if s else "EMPTY"}')
print(f'  first suffix[:80]: {s[0][1][:80] if s else "EMPTY"}')

print('--- fineweb-edu ---')
f = load_fineweb_edu(100)
print(f'  got {len(f)}; first prefix[:80]: {f[0][0][:80] if f else "EMPTY"}')

print('--- tulu3 ---')
t = load_tulu3(100)
print(f'  got {len(t)}; first user[:80]: {t[0]["messages"][0]["content"][:80] if t else "EMPTY"}')

print('--- firefly ---')
ff = load_firefly(100)
print(f'  got {len(ff)}; first user[:80]: {ff[0]["messages"][0]["content"][:80] if ff else "EMPTY"}')

print('--- sharegpt ---')
sg = load_sharegpt(100)
print(f'  got {len(sg)}; first user[:80]: {sg[0]["messages"][0]["content"][:80] if sg else "EMPTY"}')
