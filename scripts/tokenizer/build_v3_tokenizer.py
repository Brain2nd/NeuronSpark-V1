#!/usr/bin/env python3
"""Build v3 tokenizer: Qwen3 (151K) -> drop non-EN/ZH language tokens -> remap to contiguous IDs.

Output: tokenizer_v3/ in repo root with tokenizer.json + tokenizer_config.json.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUT = REPO / "tokenizer_v3"
DEFAULT_SRC = "Qwen/Qwen3-1.7B-Base"


def bytes_to_unicode():
    """GPT-2 / ByteLevel BPE byte<->unicode mapping (from HF tokenizers)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip([chr(c) for c in cs], bs))


BYTE_DECODER = bytes_to_unicode()


def decode_token(tok):
    try:
        raw = bytes(BYTE_DECODER[c] for c in tok)
        return raw.decode("utf-8", errors="replace"), raw
    except KeyError:
        return None, None


def classify_char(ch):
    if ch == "\ufffd":
        return "REPL"
    cp = ord(ch)
    if cp < 128:
        return "EN"
    if cp <= 0x024F:
        return "EN"
    if 0x2000 <= cp <= 0x206F:
        return "SYM"
    if (0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF) \
       or (0xF900 <= cp <= 0xFAFF) or (0x20000 <= cp <= 0x2A6DF):
        return "ZH"
    if (0x3000 <= cp <= 0x303F) or (0xFF00 <= cp <= 0xFFEF):
        return "ZH"
    if 0x2100 <= cp <= 0x27FF:
        return "SYM"
    if 0x2070 <= cp <= 0x209F:
        return "SYM"
    if 0xAC00 <= cp <= 0xD7AF:
        return "OTHER"
    if 0x3040 <= cp <= 0x30FF or 0x31F0 <= cp <= 0x31FF:
        return "OTHER"
    if 0x0370 <= cp <= 0x1CFF:
        return "OTHER"
    if 0x1F000 <= cp <= 0x1FAFF or 0x2600 <= cp <= 0x27BF:
        return "SYM"
    return "OTHER"


def should_keep(tok_str, added_content_set):
    if tok_str in added_content_set:
        return True
    dec, raw = decode_token(tok_str)
    if dec is None:
        return True
    if raw is not None and len(raw) == 1 and raw[0] >= 0x80:
        return True
    cats = [classify_char(c) for c in dec if not c.isspace()]
    cats = [c for c in cats if c != "REPL"]
    if not cats:
        return True
    if "OTHER" in cats:
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC, help="HF repo id for source tokenizer")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output directory")
    args = ap.parse_args()

    out = Path(args.out)

    print(f"== Fetching {args.src} tokenizer files ==")
    src_dir = Path(snapshot_download(
        args.src,
        allow_patterns=["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"],
    ))
    print(f"  src: {src_dir}")

    with open(src_dir / "tokenizer.json", encoding="utf-8") as f:
        tj = json.load(f)

    vocab = tj["model"]["vocab"]
    merges = tj["model"]["merges"]
    added = tj.get("added_tokens", [])

    # Merge specials from tokenizer.json and tokenizer_config.json added_tokens_decoder
    # (Qwen3 keeps 4 extra specials — <tool_response>, </tool_response>, <think>, </think> —
    # only in tokenizer_config, not in tokenizer.json's added_tokens list.)
    with open(src_dir / "tokenizer_config.json", encoding="utf-8") as f:
        src_cfg = json.load(f)
    atd_src = src_cfg.get("added_tokens_decoder", {})
    by_content = {a["content"]: a for a in added}
    for old_id_str, info in atd_src.items():
        c = info["content"]
        if c not in by_content:
            a2 = {
                "id": int(old_id_str),
                "content": c,
                "single_word": info.get("single_word", False),
                "lstrip": info.get("lstrip", False),
                "rstrip": info.get("rstrip", False),
                "normalized": info.get("normalized", False),
                "special": info.get("special", True),
            }
            by_content[c] = a2
            added.append(a2)
    added_content_set = set(by_content.keys())

    print(f"\n== Source stats ==")
    print(f"  regular vocab : {len(vocab):,}")
    print(f"  merges        : {len(merges):,}")
    print(f"  specials total: {len(added):,}  ({len(atd_src)} from config, {len(tj.get('added_tokens', []))} from tokenizer.json)")

    # Classify
    print("\n== Classifying tokens ==")
    keep_strs = set()
    cat_stats = Counter()
    for tok_str in vocab:
        if should_keep(tok_str, added_content_set):
            keep_strs.add(tok_str)
            cat_stats["keep"] += 1
        else:
            cat_stats["drop"] += 1
    print(f"  keep    : {cat_stats['keep']:,}")
    print(f"  drop    : {cat_stats['drop']:,}  ({cat_stats['drop']/len(vocab)*100:.1f}%)")

    # Remap: preserve old-id ordering
    sorted_keep = sorted(
        [(tok, old_id) for tok, old_id in vocab.items() if tok in keep_strs],
        key=lambda x: x[1],
    )
    new_vocab = {tok: new_id for new_id, (tok, _) in enumerate(sorted_keep)}
    old_to_new = {old_id: new_id for new_id, (_, old_id) in enumerate(sorted_keep)}

    print(f"\n== Remapped vocab: {len(new_vocab):,} tokens (IDs 0..{len(new_vocab)-1}) ==")

    # Filter merges
    print("\n== Filtering merges ==")
    new_merges = []
    for m in merges:
        # tokenizer.json merges may be stored as ["A", "B"] OR "A B"
        if isinstance(m, list):
            if len(m) != 2:
                continue
            a, b = m
        else:
            parts = m.split(" ", 1)
            if len(parts) != 2:
                continue
            a, b = parts
        merged = a + b
        if a in new_vocab and b in new_vocab and merged in new_vocab:
            new_merges.append(m)
    print(f"  merges: {len(merges):,} -> {len(new_merges):,} ({len(new_merges)/len(merges)*100:.1f}%)")

    # Append specials at the tail of the new vocab. We do NOT try to remap their original
    # IDs (151643..151668) through old_to_new — those IDs never live in model.vocab in Qwen3.
    # Instead we allocate fresh new IDs right after the filtered regular vocab.
    sorted_added = sorted(added, key=lambda a: a["id"])
    next_id = len(new_vocab)
    new_added = []
    old_to_new_specials = {}
    for a in sorted_added:
        a2 = dict(a)
        a2["id"] = next_id
        old_to_new_specials[a["id"]] = next_id
        new_added.append(a2)
        next_id += 1
    print(f"  specials appended: {len(new_added)} at IDs {len(new_vocab)}..{next_id-1}")

    # Assemble new tokenizer.json
    new_tj = {k: v for k, v in tj.items()}
    new_tj["model"] = {k: v for k, v in tj["model"].items()}
    new_tj["model"]["vocab"] = new_vocab
    new_tj["model"]["merges"] = new_merges
    new_tj["added_tokens"] = new_added

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "tokenizer.json", "w", encoding="utf-8") as f:
        json.dump(new_tj, f, ensure_ascii=False)
    print(f"\n== Written {out}/tokenizer.json ==")

    # Update tokenizer_config.json: remap added_tokens_decoder keys via specials map
    cfg = src_cfg  # already loaded above
    atd_old = cfg.get("added_tokens_decoder", {})
    atd_new = {}
    for old_id_str, info in atd_old.items():
        old_id = int(old_id_str)
        if old_id in old_to_new_specials:
            atd_new[str(old_to_new_specials[old_id])] = info
    cfg["added_tokens_decoder"] = atd_new

    # bos/eos/pad are referenced by string content (not id) in HF config — no remap needed.
    # model_max_length / chat_template preserved as-is.
    # Add provenance note
    cfg["_origin"] = f"filtered from {args.src}: dropped non-EN/ZH language tokens ({cat_stats['drop']} of {len(vocab)})"

    with open(out / "tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print(f"== Written {out}/tokenizer_config.json ==")

    # Sanity check via HF loader
    print("\n== HF load sanity ==")
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained(str(out), trust_remote_code=True)
    print(f"  class          : {type(t).__name__}")
    print(f"  vocab_size     : {t.vocab_size}")
    print(f"  len(tokenizer) : {len(t)}")
    print(f"  pad_token      : {t.pad_token} (id={t.pad_token_id})")
    print(f"  eos_token      : {t.eos_token} (id={t.eos_token_id})")
    print(f"  bos_token      : {t.bos_token}")

    samples = [
        ("EN",    "Hello world, how are you today?"),
        ("ZH",    "你好，世界。今天天气怎么样？"),
        ("CODE",  "def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)"),
        ("THINK", "<think>Let me reason step by step.</think>The answer is 42."),
        ("CHAT",  "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"),
        ("TOOL",  "<tool_call>{\"name\":\"x\"}</tool_call>"),
        ("MIXED", "请用 Python 写一个 quicksort，重点讲 pivot 选择"),
    ]
    print("\n  Encoding samples (round-trip check):")
    all_ok = True
    for tag, text in samples:
        ids = t.encode(text, add_special_tokens=False)
        dec = t.decode(ids, skip_special_tokens=False)
        match = dec == text
        status = "OK" if match else "MISMATCH"
        if not match:
            all_ok = False
        preview = (dec[:60] + "...") if len(dec) > 60 else dec
        print(f"    [{status:8}] {tag:6} ids={len(ids):3d}  -> {preview!r}")

    print(f"\n{'All samples round-trip exact.' if all_ok else 'WARNING: some mismatches above — investigate.'}")
    print(f"\nDone. New tokenizer at {out}/")


if __name__ == "__main__":
    main()
