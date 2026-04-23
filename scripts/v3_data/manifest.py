"""v3 pretrain data manifest — source declarations + target weights.

All source paths are on H100 pod (relative paths resolved from repo root).
Per-source `category` + `weight` define the interleaved mix.

Spec: docs/v3_pretrain_design.md §3
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Source:
    name: str
    category: str              # "en_web" / "zh_web" / "synthetic" / "code" / "r1_distill" / "math" / "zh_pro"
    path: str                  # relative to repo root; can be HF Arrow dir, parquet glob, or JSONL file
    format: str                # "hf_arrow" / "parquet" / "jsonl"
    text_column: str = "text"  # column containing plain text (for hf_arrow / parquet)
    apply_chat_template: bool = False  # True for R1-distill sources
    max_samples: int | None = None     # None = use all; set for cap per source
    repeat_factor: float = 1.0 # >1 = oversample (e.g., 1.8 for R1-distill)
    # Parquet-only: keep rows whose `filter_column` value is in this set. Used by
    # codeparrot/github-code-clean to select {Python, JavaScript, TypeScript} out
    # of 32 languages.
    filter_column: str | None = None
    filter_values: tuple[str, ...] = ()


# ---- Target mix (sums to 1.0) ----
# v3.2: added `narrative` (novels/fiction) — counters the encyclopedia/wiki/edu
# bias and teaches long-form coherence, dialogue, and story flow.
TARGET_WEIGHTS = {
    "en_web":     0.22,
    "zh_web":     0.20,
    "synthetic":  0.13,
    "code":       0.15,
    "r1_distill": 0.10,
    "math":       0.07,
    "narrative":  0.10,   # ZH-heavy mix via source-size ratio (~6% zh + ~4% en)
    "zh_pro":     0.03,
}

# ---- Target total tokens ----
DEFAULT_TARGET_TOKENS = 25_000_000_000   # 25B, stop-anywhere-safe


SOURCES: list[Source] = [
    # ========== EN web (28%) ==========
    Source(
        name="fineweb-edu-10BT",
        category="en_web",
        path="data/pretrain_raw/fineweb-edu/sample/10BT",
        format="parquet",
        text_column="text",
    ),
    # ========== ZH web (22%) ==========
    Source(
        name="skypile-150B",
        category="zh_web",
        path="data/SkyPile-150B/data",       # 620 GB jsonl
        format="jsonl",
        text_column="text",
    ),
    Source(
        name="seq-monkey",
        category="zh_web",
        path="data/seq-monkey/mobvoi_seq_monkey_general_open_corpus.jsonl",
        format="jsonl",
        text_column="text",
    ),
    # ========== 合成教科书 (13%) ==========
    Source(
        name="cosmopedia",
        category="synthetic",
        path="data/pretrain_raw/cosmopedia/data",     # parquet under sub-dirs
        format="parquet",
        text_column="text",
    ),
    # Benchmark train+val+test merged into pretrain plain text (~0.1B tokens).
    # User directive: 没有区分训练集测试集的直接混。Skipped lambada per project policy.
    # Covers MMLU/ARC/BoolQ/HellaSwag/PIQA/SIQA/Winogrande/OpenBookQA (EN)
    # + C3/CEval/ChID/CMMLU/CMRC2018 (ZH). Half EN half ZH by volume.
    Source(
        name="benchmarks-pretrain",
        category="synthetic",
        path="data/benchmark-pretrain",
        format="jsonl",
        text_column="text",
    ),
    # ========== 代码 (15%) ==========
    # codeparrot/github-code-clean = 313 GB open-license dump of GitHub code, 32 langs mixed.
    # We download a random subset of its 880 shards and keep only {Python, JavaScript, TypeScript}.
    Source(
        name="github-code-py-js-ts",
        category="code",
        path="data/github-code-clean",
        format="parquet",
        text_column="code",
        filter_column="language",
        filter_values=("Python", "JavaScript", "TypeScript"),
    ),
    # ========== R1-distill CoT (10%) ==========
    Source(
        name="open-thoughts-114k",
        category="r1_distill",
        path="data/r1_bootstrap/open-thoughts__OpenThoughts-114k/data",
        format="parquet",
        apply_chat_template=True,
        repeat_factor=1.8,
    ),
    Source(
        name="qwq-longcot-130k",
        category="r1_distill",
        path="data/r1_bootstrap/amphora__QwQ-LongCoT-130K",
        format="parquet",
        apply_chat_template=True,
        repeat_factor=1.8,
    ),
    Source(
        name="chinese-r1-110k",
        category="r1_distill",
        path="data/r1_bootstrap/Congliu__Chinese-DeepSeek-R1-Distill-data-110k/distill_r1_110k.jsonl",
        format="jsonl",
        apply_chat_template=True,
        repeat_factor=1.8,
    ),
    Source(
        name="bespoke-stratos-17k",
        category="r1_distill",
        path="data/r1_bootstrap/bespokelabs__Bespoke-Stratos-17k/data",
        format="parquet",
        apply_chat_template=True,
        repeat_factor=1.8,
    ),
    Source(
        name="s1K-1.1",
        category="r1_distill",
        path="data/r1_bootstrap/simplescaling__s1K-1.1/data",
        format="parquet",
        apply_chat_template=True,
        repeat_factor=1.8,
    ),
    Source(
        name="GAIR-LIMO",
        category="r1_distill",
        path="data/r1_bootstrap/GAIR__LIMO/limo.jsonl",
        format="jsonl",
        apply_chat_template=True,
        repeat_factor=1.8,
    ),
    # Chinese CoT reasoning — added v3.3 to balance r1_distill (was ~50% ZH).
    Source(
        name="mxode-reasoning-distil",
        category="r1_distill",
        path="data/mxode-reasoning-distil",
        format="jsonl",
        apply_chat_template=True,
    ),
    Source(
        name="zake-openscience-zh",
        category="r1_distill",
        path="data/zake-openscience-zh",
        format="parquet",
        apply_chat_template=True,
    ),
    # ========== 数学 (7%) ==========
    Source(
        name="openwebmath",
        category="math",
        path="data/pretrain_raw/openwebmath/data",
        format="parquet",
        text_column="text",
    ),
    Source(
        name="numinamath-cot",
        category="math",
        path="data/NuminaMath-CoT",      # need download, 1.2 GB
        format="parquet",
        apply_chat_template=False,        # treat as text
    ),
    # Chinese math CoT — added v3.3 to balance math category (was 100% EN).
    Source(
        name="mxode-cmid-math",
        category="math",
        path="data/mxode-cmid-math",
        format="jsonl",
        apply_chat_template=True,         # custom formatter in build script
    ),
    Source(
        name="mxode-school-math",
        category="math",
        path="data/mxode-school-math",
        format="jsonl",
        apply_chat_template=True,
    ),
    Source(
        name="almonster-mathinstruct-zh",
        category="math",
        path="data/almonster-mathinstruct-zh",
        format="parquet",
        apply_chat_template=True,
    ),
    # ========== 中文专业 (3%) ==========
    Source(
        name="belle-math",
        category="zh_pro",
        path="data/pretrain_raw/belle_math/school_math_0.25M.json",
        format="jsonl",
    ),
    Source(
        name="coig-cqia",
        category="zh_pro",
        path="data/raw/coig-cqia",       # local raw
        format="jsonl",
    ),
    Source(
        name="chinese-simpleqa",
        category="zh_pro",
        path="data/raw/chinese-simpleqa",
        format="jsonl",
    ),
    # wangrui6/Zhihu-KOL: 1.5 GB parquet, 1M+ high-quality Chinese Zhihu Q&A
    # Schema: INSTRUCTION, RESPONSE, SOURCE, METADATA
    Source(
        name="zhihu-kol",
        category="zh_pro",
        path="data/zhihu-kol",
        format="parquet",
    ),
    # shibing624/medical pretrain subset: 0.63 GB, Chinese medical encyclopedia + textbooks
    # Schema: {"text": "..."}  (jsonl lines, despite .json extension)
    Source(
        name="medical-zh",
        category="zh_pro",
        path="data/medical-zh",
        format="jsonl",
    ),
    # ========== 叙事 narrative (5%) ==========
    # Added in v3.2 to counter the encyclopedia/wiki/edu bias of the base mix.
    # Teaches long-form coherence, dialogue, and story flow.
    # wdndev/webnovel-chinese: 39 GB Chinese web novels (we take 3 of 10 shards = ~12 GB).
    # Schema: {"title", "chapter", "text"} — one chapter per line.
    Source(
        name="webnovel-zh",
        category="narrative",
        path="data/webnovel-zh",
        format="jsonl",
        text_column="text",
    ),
    # sedthh/gutenberg_english: 10.75 GB public-domain English literature (Dickens, Austen,
    # Twain, etc.) — classical long-form narrative, MIT-licensed.
    Source(
        name="gutenberg-en",
        category="narrative",
        path="data/gutenberg-en",
        format="parquet",
        text_column="TEXT",   # Gutenberg parquet uses uppercase column name
    ),
    # roneneldan/TinyStories: 7.62 GB GPT-3.5/4-generated short children's stories.
    # Shown in the original TinyStories paper to teach narrative to sub-billion-param LMs.
    Source(
        name="tinystories",
        category="narrative",
        path="data/tinystories",
        format="parquet",
        text_column="text",
    ),
]


def list_by_category() -> dict[str, list[Source]]:
    out: dict[str, list[Source]] = {c: [] for c in TARGET_WEIGHTS}
    for s in SOURCES:
        if s.category not in out:
            raise ValueError(f"Source {s.name} has unknown category {s.category}")
        out[s.category].append(s)
    return out


def resolve_path(repo_root: Path, source: Source) -> Path:
    return repo_root / source.path


if __name__ == "__main__":
    # Sanity print
    import json
    cats = list_by_category()
    total_w = sum(TARGET_WEIGHTS.values())
    print(f"Target weights sum: {total_w:.3f} (should be ≈1.0)")
    print(f"Target total tokens: {DEFAULT_TARGET_TOKENS / 1e9:.1f} B")
    print()
    for cat, sources in cats.items():
        w = TARGET_WEIGHTS[cat]
        print(f"== {cat} ({w*100:.0f}%, {w * DEFAULT_TARGET_TOKENS / 1e9:.2f}B tokens) ==")
        for s in sources:
            repeat_marker = f" [×{s.repeat_factor}]" if s.repeat_factor != 1.0 else ""
            chat_marker = " [ChatML]" if s.apply_chat_template else ""
            print(f"    {s.name:<30s}  {s.format:<8s}  {s.path}{repeat_marker}{chat_marker}")
