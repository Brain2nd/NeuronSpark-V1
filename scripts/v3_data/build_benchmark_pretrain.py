"""Convert local eval-benchmark datasets (data/benchmark/*) into plain pretrain
text and dump to data/benchmark-pretrain/train.jsonl.

User directive (2026-04-23): "测评指标的训练集也混进来，没有区分训练集测试集的直接混"
→ use ALL splits (train + val + test), since these benchmarks are published
and the decision is explicit to treat them as plain knowledge corpus.

Skips `lambada` permanently (memory rule: lambada_openai → never use).

Per-benchmark handler writes each row as a single natural-language block mixing
the question, options, and ground-truth answer. Output schema = {text, source}.

Run:
  python scripts/v3_data/build_benchmark_pretrain.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from datasets import load_from_disk

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BENCH_DIR = REPO_ROOT / "data/benchmark"
OUT_DIR = REPO_ROOT / "data/benchmark-pretrain"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_CHARS = 20
SKIP = {"lambada"}   # permanently removed per project policy


# ============================================================
# Per-benchmark row → text converters
# ============================================================

def _hellaswag(row) -> str:
    ctx = (row.get("ctx") or "").strip()
    label = row.get("label")
    if isinstance(label, str) and label.isdigit():
        label = int(label)
    endings = row.get("endings") or []
    if isinstance(label, int) and 0 <= label < len(endings):
        return f"{ctx} {endings[label].strip()}".strip()
    return ""


def _winogrande(row) -> str:
    sent = row.get("sentence") or ""
    ans = row.get("answer")
    if isinstance(ans, str) and ans.isdigit():
        ans = int(ans)
    o1 = row.get("option1") or ""
    o2 = row.get("option2") or ""
    opt = o1 if ans == 1 else o2
    return sent.replace("_", opt)


def _boolq(row) -> str:
    passage = row.get("passage") or ""
    question = row.get("question") or ""
    ans = "Yes" if row.get("answer") else "No"
    return f"{passage}\n\nQuestion: {question}\nAnswer: {ans}"


def _arc(row) -> str:
    q = row.get("question") or ""
    choices = row.get("choices") or {}
    labels = choices.get("label", [])
    texts = choices.get("text", [])
    key = row.get("answerKey", "")
    opts = ""
    ans_text = ""
    for l, t in zip(labels, texts):
        opts += f"\n{l}. {t}"
        if l == key:
            ans_text = t
    if not ans_text:
        return ""
    return f"{q}{opts}\n\nThe answer is {key}: {ans_text}"


def _mmlu(row) -> str:
    q = row.get("question") or ""
    choices = row.get("choices") or []
    ans = row.get("answer")
    if isinstance(ans, str) and ans.isdigit():
        ans = int(ans)
    labels = ["A", "B", "C", "D"]
    opts = ""
    for i, c in enumerate(choices):
        if i < len(labels):
            opts += f"\n{labels[i]}. {c}"
    if not isinstance(ans, int) or not (0 <= ans < len(choices)):
        return ""
    subj = row.get("subject", "") or ""
    header = f"Subject: {subj}\n" if subj else ""
    return f"{header}{q}{opts}\n\nThe answer is {labels[ans]}: {choices[ans]}"


def _piqa(row) -> str:
    goal = row.get("goal") or ""
    label = row.get("label")
    if isinstance(label, str) and label.isdigit():
        label = int(label)
    s1 = row.get("sol1") or ""
    s2 = row.get("sol2") or ""
    sol = s1 if label == 0 else s2
    return f"{goal} {sol}"


def _siqa(row) -> str:
    ctx = row.get("context") or ""
    q = row.get("question") or ""
    label = row.get("label")
    if isinstance(label, str) and label.isdigit():
        label = int(label)
    answers = [row.get(f"answerA"), row.get(f"answerB"), row.get(f"answerC")]
    idx = max(0, (label or 1) - 1)   # siqa labels are 1/2/3
    if 0 <= idx < len(answers) and answers[idx]:
        return f"{ctx} {q} {answers[idx]}"
    return ""


def _openbookqa(row) -> str:
    # Schema: question_stem + choices.{text,label} + answerKey
    q = row.get("question_stem") or ""
    choices = row.get("choices") or {}
    labels = choices.get("label", [])
    texts = choices.get("text", [])
    key = row.get("answerKey", "")
    opts = ""
    ans_text = ""
    for l, t in zip(labels, texts):
        opts += f"\n{l}. {t}"
        if l == key:
            ans_text = t
    if not ans_text:
        return ""
    return f"{q}{opts}\n\nThe answer is {key}: {ans_text}"


# ---- Chinese benchmarks ----

def _c3(row) -> str:
    # 中文阅读理解：one question per row. Fields: context (list), question, choice (list), answer.
    # test split has empty answer — we skip those.
    ctx_list = row.get("context") or []
    passage = "\n".join(ctx_list) if isinstance(ctx_list, list) else str(ctx_list)
    q = row.get("question") or ""
    choices = row.get("choice") or []
    ans = row.get("answer") or ""
    if not ans:
        return ""
    opts = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
    return f"{passage}\n\n问题：{q}\n{opts}\n答：{ans}"


def _ceval(row) -> str:
    # 52 科目考试题: {question, A, B, C, D, answer, explanation?}
    q = row.get("question") or ""
    subj = row.get("subject_name") or row.get("subject", "")
    opts = ""
    ans_text = ""
    for letter in ["A", "B", "C", "D"]:
        t = row.get(letter, "")
        opts += f"\n{letter}. {t}"
        if row.get("answer") == letter:
            ans_text = t
    expl = row.get("explanation", "") or ""
    header = f"科目：{subj}\n" if subj else ""
    body = f"{header}{q}{opts}\n\n答案：{row.get('answer','')}：{ans_text}"
    if expl:
        body += f"\n解析：{expl}"
    return body


def _chid(row) -> str:
    # 成语完形: content is list of passages with #idiomNNNN# placeholders,
    # candidates is list of 10 idiom options, answers.text gives correct idioms in order.
    contents = row.get("content") or []
    ans = row.get("answers") or {}
    ans_texts = ans.get("text", []) if isinstance(ans, dict) else []
    if not ans_texts:
        return ""
    # Join all passages
    body = "\n".join(contents) if isinstance(contents, list) else str(contents)
    # Replace each #idiomNNNN# placeholder (in order of occurrence) with the answer idiom.
    # Pattern: #idiom<digits>#
    import re
    placeholders = re.findall(r"#idiom\d+#", body)
    out = body
    for ph, correct in zip(placeholders, ans_texts):
        out = out.replace(ph, correct, 1)
    return out


def _cmmlu(row) -> str:
    q = row.get("Question") or row.get("question") or ""
    A = row.get("A", ""); B = row.get("B", "")
    C = row.get("C", ""); D = row.get("D", "")
    ans_letter = row.get("Answer") or row.get("answer", "")
    map_ = {"A": A, "B": B, "C": C, "D": D}
    ans_text = map_.get(ans_letter, "")
    return (f"{q}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n"
            f"答案：{ans_letter}：{ans_text}")


def _cmrc2018(row) -> str:
    # {context, question, answers: {text:[...], answer_start:[...]}}
    ctx = row.get("context") or ""
    q = row.get("question") or ""
    ans = row.get("answers") or {}
    texts = ans.get("text", []) if isinstance(ans, dict) else []
    ans_text = texts[0] if texts else ""
    if not ans_text:
        return ""
    return f"{ctx}\n\n问题：{q}\n答：{ans_text}"


CONVERTERS = {
    "hellaswag":     _hellaswag,
    "winogrande":    _winogrande,
    "boolq":         _boolq,
    "arc_easy":      _arc,
    "arc_challenge": _arc,
    "mmlu":          _mmlu,
    "piqa":          _piqa,
    "siqa":          _siqa,
    "openbookqa":    _openbookqa,
    "c3":            _c3,
    "ceval":         _ceval,
    "chid":          _chid,
    "cmmlu":         _cmmlu,
    "cmrc2018":      _cmrc2018,
}


# ============================================================
# Main
# ============================================================

def main():
    out_path = OUT_DIR / "train.jsonl"
    tmp_path = OUT_DIR / "train.jsonl.partial"

    counts = {}
    total_chars = 0
    with open(tmp_path, "w", encoding="utf-8") as out:
        for bench_name in sorted(os.listdir(BENCH_DIR)):
            if bench_name.startswith(".") or bench_name in SKIP:
                continue
            bench_path = BENCH_DIR / bench_name
            if not (bench_path / "dataset_dict.json").is_file():
                print(f"[skip] {bench_name}: not a HF dataset_dict")
                continue
            if bench_name not in CONVERTERS:
                print(f"[skip] {bench_name}: no converter registered")
                continue
            conv = CONVERTERS[bench_name]

            try:
                dd = load_from_disk(str(bench_path))
            except Exception as e:
                print(f"[ERR ] {bench_name}: load_from_disk failed: {e}")
                continue

            n_written = 0
            for split_name, split in dd.items():
                for row in split:
                    try:
                        text = conv(row).strip()
                    except Exception:
                        continue
                    if not text or len(text) < MIN_CHARS:
                        continue
                    out.write(json.dumps({"text": text, "source": f"{bench_name}/{split_name}"},
                                         ensure_ascii=False) + "\n")
                    n_written += 1
                    total_chars += len(text)
            print(f"[done] {bench_name:<15s} +{n_written:,} rows (all splits merged)")
            counts[bench_name] = n_written

    os.replace(tmp_path, out_path)
    print()
    print(f"=== {out_path} ===")
    print(f"Total rows: {sum(counts.values()):,}")
    print(f"Total chars: {total_chars:,} ({total_chars/1e6:.1f}M, ~{total_chars/3e9:.2f}B est tokens)")


if __name__ == "__main__":
    main()
