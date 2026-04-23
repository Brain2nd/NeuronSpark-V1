"""Build v3 pretrain mix → data/v3_pretrain_mix/train-NNNNN.parquet

Design goals (from user direction 2026-04-23):
  1. OUTPUT FORMAT = same as Brain2nd/NeuronSpark-V1 on HF:
     Parquet, two columns: text (string), source (string). NOT pre-tokenized.
     H100 tokenizes on-the-fly at training time.
  2. Stop-anywhere-safe: any prefix of shards has category ratio ≈ TARGET_WEIGHTS.
     Achieved by weighted round-robin interleave at WRITE time (not read time).
  3. Memory-bounded: never load a full source into RAM. Stream everything.
     Peak RAM target < 1 GB for Pass 2 even with 14 sources open simultaneously.
  4. Resumable in two stages:
     Pass 1 : per-source stream → staging/{source}.parquet (downsampled, formatted)
     Pass 2 : weighted interleave staging/* → out/train-NNNNN.parquet

Run:
  python scripts/v3_data/build_pretrain_mix.py --pass all \\
      --target_tokens 20000000000 \\
      --out_dir data/v3_pretrain_mix --staging_dir data/v3_staging

  # Or stepwise:
  python scripts/v3_data/build_pretrain_mix.py --pass 1   # downsample per source
  python scripts/v3_data/build_pretrain_mix.py --pass 2   # interleave
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.v3_data.manifest import SOURCES, Source, TARGET_WEIGHTS, list_by_category, resolve_path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# ============================================================
# Tokens-per-byte heuristic (for budgeting without real tokenize)
# ============================================================
# Rough avg; good enough for proportional downsampling
CHAR_PER_TOKEN = {
    "en_web":     4.0,
    "zh_web":     1.7,   # CJK is 2-3 chars/token in qwen3 tokenizer, but JSON/punct mixed lower
    "synthetic":  4.0,
    "code":       3.3,
    "r1_distill": 2.2,   # CoT has lots of math/special tokens
    "math":       3.0,
    "zh_pro":     2.0,
    "narrative":  2.8,   # mixed ZH novels (~1.7) + EN literature (~4); weighted average
}

MIN_CHARS = 200     # matches V2.5 filter
MIN_TOKENS_EST = 16


# ============================================================
# Per-source row → formatted text generator
# ============================================================

def _stream_raw_text(source: Source, path: Path) -> Iterator[str]:
    """Yield plain text strings per doc (streaming, O(1) memory per doc)."""
    col = source.text_column
    filt_col = source.filter_column
    filt_vals = set(source.filter_values or ())
    if source.format == "parquet":
        for pq_file in _parquet_files(path):
            pf = pq.ParquetFile(pq_file)
            schema_names = set(pf.schema_arrow.names)
            have_text_col = col in schema_names
            have_filt_col = filt_col in schema_names if filt_col else False
            if have_text_col and (filt_col is None or have_filt_col):
                read_cols = [col] + ([filt_col] if filt_col else [])
                for batch in pf.iter_batches(batch_size=512, columns=read_cols):
                    text_arr = batch.column(0).to_pylist()
                    if filt_col:
                        filt_arr = batch.column(1).to_pylist()
                        for val, fv in zip(text_arr, filt_arr):
                            if val and fv in filt_vals:
                                yield val
                    else:
                        for val in text_arr:
                            if val:
                                yield val
            else:
                # text column missing — fall back to heuristic row-level mapping
                for batch in pf.iter_batches(batch_size=512):
                    for row in batch.to_pylist():
                        t = _fallback_text(row)
                        if t:
                            yield t
    elif source.format == "jsonl":
        for jf in _jsonl_files(path):
            with open(jf, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    val = d.get(col) if col else None
                    if not val:
                        val = _fallback_text(d)
                    if val:
                        yield val
    else:
        raise ValueError(f"unknown format {source.format}")


def _fallback_text(d: dict) -> str | None:
    """Map instruction-tuning / QA rows to plain text. Case-insensitive keys
    (some datasets use UPPERCASE, e.g. Zhihu-KOL has INSTRUCTION/RESPONSE)."""
    lower = {k.lower(): v for k, v in d.items() if isinstance(k, str)}
    if "instruction" in lower and "output" in lower:
        inst = lower.get("instruction", "")
        inp = lower.get("input", "") or ""
        out = lower.get("output", "")
        if inp:
            return f"{inst}\n{inp}\n{out}"
        return f"{inst}\n{out}"
    if "instruction" in lower and "response" in lower:
        return f"{lower['instruction']}\n{lower['response']}"
    if "question" in lower and "answer" in lower:
        return f"{lower['question']}\n{lower['answer']}"
    if "prompt" in lower and "response" in lower:
        return f"{lower['prompt']}\n{lower['response']}"
    if "problem" in lower and "solution" in lower:
        return f"{lower['problem']}\n{lower['solution']}"
    return None


# ---- R1-distill formatters: convert structured rows → ChatML plain text ----
# We do NOT call tokenizer.apply_chat_template here (avoids tokenizer dep in Pass 1
# and keeps output as plain text like the HF reference dataset).

_CHATML_TEMPLATE = "<|im_start|>{role}\n{content}<|im_end|>"


def _chatml(msgs: list[dict]) -> str:
    parts = [_CHATML_TEMPLATE.format(role=m["role"], content=m["content"]) for m in msgs]
    return "\n".join(parts)


def _r1_open_thoughts(path: Path) -> Iterator[str]:
    """open-thoughts / bespoke-stratos: cols = [system, conversations(list{from,value})]"""
    for pq_file in _parquet_files(path):
        pf = pq.ParquetFile(pq_file)
        for batch in pf.iter_batches(batch_size=256):
            for row in batch.to_pylist():
                msgs = []
                if row.get("system"):
                    msgs.append({"role": "system", "content": row["system"]})
                for turn in row.get("conversations") or []:
                    role = {"human": "user", "gpt": "assistant", "system": "system"}.get(
                        turn.get("from"), "user"
                    )
                    msgs.append({"role": role, "content": turn.get("value", "")})
                if len(msgs) >= 2:
                    yield _chatml(msgs)


def _r1_qwq(path: Path) -> Iterator[str]:
    """amphora__QwQ-LongCoT-130K: cols = [problem, qwq]"""
    for pq_file in _parquet_files(path):
        pf = pq.ParquetFile(pq_file)
        for batch in pf.iter_batches(batch_size=256, columns=["problem", "qwq"]):
            for prob, qwq in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
                if prob and qwq:
                    yield _chatml([
                        {"role": "user", "content": prob},
                        {"role": "assistant", "content": qwq},
                    ])


def _r1_s1k(path: Path) -> Iterator[str]:
    """simplescaling__s1K-1.1: deepseek_thinking_trajectory + deepseek_attempt"""
    for pq_file in _parquet_files(path):
        pf = pq.ParquetFile(pq_file)
        for batch in pf.iter_batches(batch_size=256):
            for row in batch.to_pylist():
                q = row.get("question")
                if not q:
                    continue
                thinking = row.get("deepseek_thinking_trajectory") or row.get("gemini_thinking_trajectory")
                answer = row.get("deepseek_attempt") or row.get("gemini_attempt") or row.get("solution")
                if not answer:
                    continue
                assistant = (f"<think>\n{thinking}\n</think>\n\n{answer}"
                             if thinking else str(answer))
                yield _chatml([
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": assistant},
                ])


def _r1_chinese(path: Path) -> Iterator[str]:
    """Congliu__Chinese-DeepSeek-R1-Distill: input + reasoning_content + content"""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            inp = d.get("input")
            reasoning = d.get("reasoning_content", "")
            content = d.get("content", "")
            if not inp or not content:
                continue
            assistant = (f"<think>\n{reasoning}\n</think>\n\n{content}"
                         if reasoning else content)
            yield _chatml([
                {"role": "user", "content": inp},
                {"role": "assistant", "content": assistant},
            ])


def _r1_limo(path: Path) -> Iterator[str]:
    """GAIR__LIMO: question + solution + answer"""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = d.get("question")
            sol = d.get("solution")
            ans = d.get("answer")
            if not q or not sol:
                continue
            assistant = sol if not ans else f"{sol}\n\n**Final answer:** {ans}"
            yield _chatml([
                {"role": "user", "content": q},
                {"role": "assistant", "content": assistant},
            ])


def _r1_mxode_prompt_reasoning_response(path: Path) -> Iterator[str]:
    """Mxode/* jsonl with {prompt, reasoning, response} fields (Chinese CoT).
    Used by mxode-reasoning-distil, mxode-school-math, mxode-cmid-math."""
    for jf in _jsonl_files(path):
        with open(jf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                p = d.get("prompt") or d.get("query")
                r = d.get("reasoning") or ""
                a = d.get("response") or d.get("answer")
                if not p or not a:
                    continue
                assistant = (f"<think>\n{r}\n</think>\n\n{a}" if r else str(a))
                yield _chatml([
                    {"role": "user", "content": str(p)},
                    {"role": "assistant", "content": assistant},
                ])


def _chat_messages_parquet(path: Path) -> Iterator[str]:
    """Generic ChatML parquet: each row has `messages: list[{role, content}]`.
    Used by zake7749/OpenScience-Chinese-Reasoning + ALmonster/MathInstruct-Chinese."""
    for pq_file in _parquet_files(path):
        pf = pq.ParquetFile(pq_file)
        if "messages" not in pf.schema_arrow.names:
            continue
        for batch in pf.iter_batches(batch_size=256, columns=["messages"]):
            for msgs in batch.column(0).to_pylist():
                if not msgs:
                    continue
                out = []
                for m in msgs:
                    role = m.get("role", "user")
                    content = m.get("content", "")
                    if content:
                        out.append({"role": role, "content": content})
                if len(out) >= 2:
                    yield _chatml(out)


_CHAT_STREAMS = {
    "open-thoughts-114k":  _r1_open_thoughts,
    "bespoke-stratos-17k": _r1_open_thoughts,
    "qwq-longcot-130k":    _r1_qwq,
    "s1K-1.1":             _r1_s1k,
    "chinese-r1-110k":     _r1_chinese,
    "GAIR-LIMO":           _r1_limo,
    # v3.3 Chinese CoT / math additions
    "mxode-reasoning-distil":   _r1_mxode_prompt_reasoning_response,
    "mxode-school-math":        _r1_mxode_prompt_reasoning_response,
    "mxode-cmid-math":          _r1_mxode_prompt_reasoning_response,
    "zake-openscience-zh":      _chat_messages_parquet,
    "almonster-mathinstruct-zh": _chat_messages_parquet,
}


def stream_text(source: Source) -> Iterator[str]:
    path = resolve_path(REPO_ROOT, source)
    if source.apply_chat_template:
        if source.name not in _CHAT_STREAMS:
            raise ValueError(f"{source.name} needs chat template but has no formatter")
        yield from _CHAT_STREAMS[source.name](path)
    else:
        yield from _stream_raw_text(source, path)


# ============================================================
# File helpers
# ============================================================

def _parquet_files(path: Path) -> list[Path]:
    if path.is_file() and path.suffix == ".parquet":
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"parquet path missing: {path}")
    files = sorted(path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no *.parquet under {path}")
    return files


def _jsonl_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"jsonl path missing: {path}")
    files = sorted(list(path.rglob("*.jsonl")) + list(path.rglob("*.json"))
                   + list(path.rglob("*.csv")))  # some local raw dirs have csv too; ignored by reader
    files = [f for f in files if f.suffix in (".jsonl", ".json")]
    if not files:
        raise FileNotFoundError(f"no *.jsonl/*.json under {path}")
    return files


def _estimate_source_chars(source: Source, path: Path) -> int:
    """Rough byte/char estimate by summing file sizes (matches UTF-8 byte≈char enough)."""
    if source.format == "parquet":
        files = _parquet_files(path)
    elif source.format == "jsonl":
        files = _jsonl_files(path)
    else:
        return 0
    return sum(f.stat().st_size for f in files)


# ============================================================
# Category → per-source token budget
# ============================================================

def allocate_budgets(target_tokens: int,
                     present_only: bool = True) -> dict[str, int]:
    """Return {source_name: target_tokens_for_that_source}.

    Strategy:
      - Category budget = TARGET_WEIGHTS[cat] * target_tokens (renormalized if some cats skipped)
      - Within category: split by relative file size (proxy for raw token count)
    """
    cats = list_by_category()
    # Which categories have at least one usable source on disk?
    usable_cats: dict[str, list[Source]] = {}
    for cat, sources in cats.items():
        if not sources:
            continue
        usable = []
        for s in sources:
            p = resolve_path(REPO_ROOT, s)
            if p.exists():
                usable.append(s)
            else:
                print(f"[WARN] skipping {s.name}: path missing {p}")
        if usable:
            usable_cats[cat] = usable

    # Renormalize target weights over usable categories
    w_sum = sum(TARGET_WEIGHTS[c] for c in usable_cats)
    cat_budget = {c: int(target_tokens * TARGET_WEIGHTS[c] / w_sum) for c in usable_cats}

    # Within-category allocation by file-size proxy
    out: dict[str, int] = {}
    for cat, sources in usable_cats.items():
        sizes = []
        for s in sources:
            p = resolve_path(REPO_ROOT, s)
            sizes.append(max(1, _estimate_source_chars(s, p)))
        total_sz = sum(sizes)
        for s, sz in zip(sources, sizes):
            share = sz / total_sz
            # Cap by what we estimate the source can actually provide (× repeat_factor)
            est_tokens_avail = int(sz / CHAR_PER_TOKEN[cat] * s.repeat_factor)
            allocated = int(cat_budget[cat] * share)
            out[s.name] = min(allocated, est_tokens_avail)

    return out


# ============================================================
# Pass 1: per-source stream → staging parquet
# ============================================================

PARQUET_WRITE_BATCH = 5000


def pass_1(sources: list[Source], budgets: dict[str, int], staging_dir: Path,
           seed: int) -> dict:
    """Each source → staging/{name}.parquet (text, source).

    Uses Bernoulli downsampling with keep_prob = target_tokens / estimated_total_tokens
    so we never load the full source — just skip most rows.

    Returns stats dict.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    for s in sources:
        target_t = budgets.get(s.name, 0)
        if target_t <= 0:
            continue
        p = resolve_path(REPO_ROOT, s)
        est_bytes = _estimate_source_chars(s, p)
        est_tokens = int(est_bytes / CHAR_PER_TOKEN[s.category] * s.repeat_factor)
        if est_tokens <= 0:
            print(f"[skip] {s.name}: empty source")
            continue
        # keep_prob chosen with slight over-sampling (×1.15) for safety margin.
        # If the source already has a content filter (e.g. language=Python|JS|TS for
        # github-code-clean), skip Bernoulli — the filter itself handles sparsity.
        if s.filter_column:
            keep_prob = 1.0
        else:
            keep_prob = min(1.0, 1.15 * target_t / est_tokens)
        stats[s.name] = {"target_tokens": target_t, "est_tokens_avail": est_tokens,
                         "keep_prob": keep_prob}

        out_path = staging_dir / f"{s.name}.parquet"
        tmp_path = staging_dir / f"{s.name}.parquet.partial"
        if out_path.exists():
            print(f"[keep] {s.name}: staging already present at {out_path}")
            continue

        rng = random.Random(f"{seed}-{s.name}")
        schema = pa.schema([pa.field("text", pa.string()),
                            pa.field("source", pa.string())])

        repeats = max(1, int(np.ceil(s.repeat_factor)))
        print(f"\n[pass1] {s.name}  cat={s.category}  target={target_t/1e9:.3f}B tok  "
              f"keep={keep_prob:.4f}  repeats={repeats}")

        docs_written = 0
        tokens_written = 0
        chars_scanned = 0
        t0 = time.time()

        with pq.ParquetWriter(tmp_path, schema, compression="zstd") as writer:
            buf_text, buf_src = [], []
            done = False
            for r in range(repeats):
                if done:
                    break
                src_iter = stream_text(s)
                for txt in src_iter:
                    if not txt:
                        continue
                    if len(txt) < MIN_CHARS:
                        continue
                    chars_scanned += len(txt)
                    if rng.random() > keep_prob:
                        continue
                    buf_text.append(txt)
                    buf_src.append(s.name)
                    docs_written += 1
                    tokens_written += int(len(txt) / CHAR_PER_TOKEN[s.category])

                    if len(buf_text) >= PARQUET_WRITE_BATCH:
                        writer.write_table(pa.table({"text": buf_text, "source": buf_src},
                                                    schema=schema))
                        buf_text.clear(); buf_src.clear()

                    if tokens_written >= target_t:
                        done = True
                        break

                    if docs_written % 50000 == 0:
                        dt = time.time() - t0
                        print(f"    [{s.name}] {docs_written:,} docs, "
                              f"{tokens_written/1e9:.3f}B est-tok, {dt:.0f}s")

            if buf_text:
                writer.write_table(pa.table({"text": buf_text, "source": buf_src},
                                            schema=schema))

        os.replace(tmp_path, out_path)
        dt = time.time() - t0
        stats[s.name].update({
            "docs": docs_written, "tokens_est": tokens_written,
            "chars_scanned": chars_scanned, "elapsed_sec": round(dt, 1),
            "path": str(out_path), "size_bytes": out_path.stat().st_size,
        })
        print(f"[done] {s.name}: {docs_written:,} docs, {tokens_written/1e9:.3f}B est-tok, "
              f"{out_path.stat().st_size/1e9:.2f} GB, {dt:.0f}s")

    return stats


# ============================================================
# Pass 2: weighted interleave staging → final shards
# ============================================================

class StagingIterator:
    """Streams docs from one staging parquet. Wraps on exhaustion (for weight balance)."""
    def __init__(self, name: str, category: str, path: Path, rng: random.Random):
        self.name = name
        self.category = category
        self.path = path
        self.rng = rng
        self.n_rows = pq.ParquetFile(path).metadata.num_rows
        self._open()
        self.wraps = 0

    def _open(self):
        pf = pq.ParquetFile(self.path)
        # Random row-group order for shuffle-ish streaming
        rg_indices = list(range(pf.num_row_groups))
        self.rng.shuffle(rg_indices)
        self._pf = pf
        self._rg_order = rg_indices
        self._rg_pos = 0
        self._batch_iter = None

    def _next_batch_iter(self):
        if self._rg_pos >= len(self._rg_order):
            # wrap
            self.wraps += 1
            self._open()
        rg = self._rg_order[self._rg_pos]
        self._rg_pos += 1
        tbl = self._pf.read_row_group(rg)
        rows = list(zip(tbl.column("text").to_pylist(),
                        tbl.column("source").to_pylist()))
        self.rng.shuffle(rows)
        self._batch_iter = iter(rows)

    def next_doc(self) -> tuple[str, str]:
        while True:
            if self._batch_iter is None:
                self._next_batch_iter()
            try:
                return next(self._batch_iter)
            except StopIteration:
                self._batch_iter = None


def pass_2(staging_dir: Path, out_dir: Path, target_tokens: int,
           shard_docs: int, seed: int) -> dict:
    """Weighted round-robin interleave staging/*.parquet → out/train-NNNNN.parquet.

    Memory profile:
      - one ParquetFile handle per source (metadata only, negligible)
      - at most one row_group worth of docs per source cached at a time (~200MB worst case)
      - one output ParquetWriter with ~30MB batch buffer
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    # Build iterators for every source with a non-empty staging file
    iters: list[StagingIterator] = []
    for s in SOURCES:
        p = staging_dir / f"{s.name}.parquet"
        if not p.is_file():
            continue
        pf_meta = pq.ParquetFile(p)
        if pf_meta.metadata.num_rows == 0 or pf_meta.num_row_groups == 0:
            print(f"[skip] {s.name}: staging parquet is empty")
            continue
        iters.append(StagingIterator(s.name, s.category, p, random.Random(f"{seed}-{s.name}-pass2")))
    if not iters:
        raise RuntimeError(f"no staging parquet found under {staging_dir}")

    # --- Per-source avg_tokens_per_doc (prefer pass1_manifest, fallback: sample parquet) ---
    avg_tok: dict[str, float] = {}
    m_path = staging_dir / "pass1_manifest.json"
    if m_path.is_file():
        try:
            m = json.load(open(m_path))
            for name, info in m.get("pass1", {}).items():
                docs = info.get("docs", 0)
                toks = info.get("tokens_est", 0)
                if docs > 0:
                    avg_tok[name] = toks / docs
        except Exception as e:
            print(f"[WARN] couldn't parse pass1_manifest: {e}")

    for it in iters:
        if it.name in avg_tok and avg_tok[it.name] > 0:
            continue
        # Fallback: read first row_group, average char len / char_per_token[cat]
        pf = pq.ParquetFile(it.path)
        tbl = pf.read_row_group(0, columns=["text"])
        texts = tbl.column("text").to_pylist()
        if texts:
            mean_chars = sum(len(t) for t in texts) / len(texts)
            avg_tok[it.name] = max(1.0, mean_chars / CHAR_PER_TOKEN.get(it.category, 3.5))
        else:
            avg_tok[it.name] = 500.0  # guardrail

    # --- Sampling weights: each source's TOKEN share within its category ---
    # We want each source to contribute proportionally to its available TOKEN pool,
    # not its row count. A source of 85K short rows (belle-math, 12.6M tok) must
    # NOT be asked to match 85K long rows (zhihu, 193M tok); otherwise the small
    # source wraps 6-8× = duplicated data.
    #
    # weight_i = target_w[cat] × (tokens_i / cat_tokens) / avg_tok_i
    #          = target_w[cat] × (rows_i × avg_tok_i) / cat_tokens / avg_tok_i
    #          = target_w[cat] × rows_i / cat_tokens
    # where cat_tokens = Σ_{j in cat} rows_j × avg_tok_j
    cat_tokens: dict[str, float] = {}
    for it in iters:
        cat_tokens[it.category] = cat_tokens.get(it.category, 0.0) + it.n_rows * avg_tok[it.name]

    weights = []
    for it in iters:
        cat_w = TARGET_WEIGHTS.get(it.category, 0.0)
        w = cat_w * it.n_rows / max(1.0, cat_tokens[it.category])
        weights.append(w)
    wsum = sum(weights)
    weights = [w / wsum for w in weights]

    print(f"\n[pass2] interleaving {len(iters)} sources, target {target_tokens/1e9:.1f}B est-tok")
    for it, w in zip(iters, weights):
        print(f"  {it.name:<25s}  cat={it.category:<11s}  rows={it.n_rows:>10,}  "
              f"avg_tok={avg_tok[it.name]:>7.0f}  w={w:.4f}")

    schema = pa.schema([pa.field("text", pa.string()), pa.field("source", pa.string())])

    shard_idx = 0
    total_docs = 0
    total_tokens = 0
    per_cat_tokens: dict[str, int] = {c: 0 for c in TARGET_WEIGHTS}
    per_src_tokens: dict[str, int] = {it.name: 0 for it in iters}

    def _open_shard(i):
        return pq.ParquetWriter(out_dir / f"train-{i:05d}.parquet", schema, compression="zstd")

    t0 = time.time()
    shard_writer = _open_shard(shard_idx)
    buf_text, buf_src = [], []
    shard_rows = 0

    try:
        while total_tokens < target_tokens:
            idx = rng.choices(range(len(iters)), weights=weights, k=1)[0]
            it = iters[idx]
            try:
                text, src_name = it.next_doc()
            except Exception as e:
                print(f"[WARN] iter {it.name} failed: {e}")
                continue

            buf_text.append(text); buf_src.append(src_name)
            shard_rows += 1
            total_docs += 1
            est_tok = int(len(text) / CHAR_PER_TOKEN[it.category])
            total_tokens += est_tok
            per_cat_tokens[it.category] = per_cat_tokens.get(it.category, 0) + est_tok
            per_src_tokens[it.name] += est_tok

            if len(buf_text) >= PARQUET_WRITE_BATCH:
                shard_writer.write_table(pa.table({"text": buf_text, "source": buf_src},
                                                  schema=schema))
                buf_text.clear(); buf_src.clear()

            if shard_rows >= shard_docs:
                if buf_text:
                    shard_writer.write_table(pa.table({"text": buf_text, "source": buf_src},
                                                      schema=schema))
                    buf_text.clear(); buf_src.clear()
                shard_writer.close()
                dt = time.time() - t0
                print(f"  [shard {shard_idx}] {shard_rows:,} docs  total "
                      f"{total_tokens/1e9:.3f}B/{target_tokens/1e9:.1f}B tok  {dt:.0f}s")
                shard_idx += 1
                shard_rows = 0
                shard_writer = _open_shard(shard_idx)

            if total_docs % 100000 == 0:
                pct = total_tokens / target_tokens * 100
                dt = time.time() - t0
                print(f"  [{pct:>5.1f}%] {total_tokens/1e9:.3f}B est-tok  "
                      f"{total_docs:,} docs  {dt:.0f}s")
    finally:
        if buf_text:
            shard_writer.write_table(pa.table({"text": buf_text, "source": buf_src},
                                              schema=schema))
        shard_writer.close()

    stats = {
        "total_docs": total_docs,
        "total_tokens_est": total_tokens,
        "target_tokens": target_tokens,
        "n_shards": shard_idx + 1,
        "shard_docs": shard_docs,
        "per_category_tokens_est": per_cat_tokens,
        "per_source_tokens_est": per_src_tokens,
        "target_weights": TARGET_WEIGHTS,
        "actual_weights": {c: per_cat_tokens.get(c, 0) / max(1, total_tokens)
                           for c in TARGET_WEIGHTS},
        "wraps": {it.name: it.wraps for it in iters},
        "seed": seed,
    }
    print(f"\n=== pass2 done: {total_docs:,} docs in {stats['n_shards']} shards, "
          f"{total_tokens/1e9:.3f}B est-tok ===")
    return stats


# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="stage", choices=["1", "2", "all"], default="all")
    ap.add_argument("--target_tokens", type=int, default=20_000_000_000)
    ap.add_argument("--staging_dir", type=str, default="data/v3_staging")
    ap.add_argument("--out_dir", type=str, default="data/v3_pretrain_mix")
    ap.add_argument("--shard_docs", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only_source", type=str, default=None,
                    help="(pass1 debug) only process this source, skip others")
    args = ap.parse_args()

    staging_dir = REPO_ROOT / args.staging_dir
    out_dir = REPO_ROOT / args.out_dir
    staging_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    budgets = allocate_budgets(args.target_tokens)
    print("Per-source token budget (est):")
    for name, t in sorted(budgets.items(), key=lambda x: -x[1]):
        print(f"  {name:<25s}  {t/1e9:.3f}B tokens")
    print(f"  TOTAL: {sum(budgets.values())/1e9:.2f}B tokens")

    all_stats = {"budgets": budgets, "target_tokens": args.target_tokens}

    if args.stage in ("1", "all"):
        sources = SOURCES
        if args.only_source:
            sources = [s for s in SOURCES if s.name == args.only_source]
        pass1_stats = pass_1(sources, budgets, staging_dir, args.seed)
        all_stats["pass1"] = pass1_stats
        with open(staging_dir / "pass1_manifest.json", "w") as f:
            json.dump(all_stats, f, indent=2, ensure_ascii=False)

    if args.stage in ("2", "all"):
        pass2_stats = pass_2(staging_dir, out_dir, args.target_tokens,
                             args.shard_docs, args.seed)
        all_stats["pass2"] = pass2_stats
        with open(out_dir / "manifest.json", "w") as f:
            json.dump(all_stats, f, indent=2, ensure_ascii=False)
        print("\n=== Category ratios (target vs actual) ===")
        for c in TARGET_WEIGHTS:
            t = TARGET_WEIGHTS[c]
            a = pass2_stats["actual_weights"].get(c, 0)
            print(f"  {c:<12s}  target {t:.3f}  actual {a:.3f}  delta {a-t:+.3f}")


if __name__ == "__main__":
    main()
