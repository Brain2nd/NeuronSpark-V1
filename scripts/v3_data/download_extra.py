"""Download the two remaining v3 pretrain-mix sources:
  - codeparrot/github-code-clean  (random 200 shards × 0.36 GB ≈ 72 GB)
    → data/github-code-clean/  (parquet, will be filtered to Python/JS/TS in pass1)
  - AI-MO/NuminaMath-CoT  (1.23 GB)  → data/NuminaMath-CoT/

Why sample-only for github-code-clean:
  The full repo is 313 GB. We only need ~10 GB of Python/JS/TS text after filtering,
  so 200 random shards is plenty (yields ~11 GB of filtered code ≈ 3.3B tokens).

Run:
  python scripts/v3_data/download_extra.py
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


def download_github_code(out_dir: Path, n_shards: int, seed: int = 42):
    total_shards = 880
    rng = random.Random(seed)
    indices = rng.sample(range(total_shards), n_shards)
    indices.sort()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[github-code-clean] downloading {n_shards}/{total_shards} random shards (seed={seed})")
    print(f"  → {out_dir}")
    for i, shard in enumerate(indices, 1):
        fname = f"data/train-{shard:05d}-of-00880.parquet"
        local = out_dir / fname.split("/")[-1]
        if local.is_file():
            print(f"  [{i:>3}/{n_shards}] skip (exists): {local.name}")
            continue
        path = hf_hub_download(
            "codeparrot/github-code-clean",
            fname,
            repo_type="dataset",
            local_dir=str(out_dir),
        )
        # Flatten layout: HF puts it under out_dir/data/..., move up
        src = Path(path)
        if src != local and src.is_file():
            src.rename(local)
        sz = local.stat().st_size / 1e9 if local.is_file() else 0
        print(f"  [{i:>3}/{n_shards}] {local.name}  {sz:.2f} GB")
    # Cleanup leftover HF cache dir inside out_dir
    nested = out_dir / "data"
    if nested.is_dir() and not any(nested.iterdir()):
        nested.rmdir()


def download_numinamath(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[NuminaMath-CoT] downloading → {out_dir}")
    snapshot_download(
        "AI-MO/NuminaMath-CoT",
        repo_type="dataset",
        local_dir=str(out_dir),
        allow_patterns=["data/*", "*.json", "*.md"],
    )
    print("  done")


def download_zhihu_kol(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Zhihu-KOL] downloading → {out_dir}")
    snapshot_download(
        "wangrui6/Zhihu-KOL",
        repo_type="dataset",
        local_dir=str(out_dir),
        allow_patterns=["data/*.parquet", "*.md"],
    )
    print("  done")


def download_medical_zh(out_dir: Path):
    """Only the pretrain/ subset (~0.63 GB) — avoid finetune/ which is instruction style."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[shibing624/medical pretrain] downloading → {out_dir}")
    snapshot_download(
        "shibing624/medical",
        repo_type="dataset",
        local_dir=str(out_dir),
        allow_patterns=["pretrain/*.json", "*.md"],
    )
    print("  done")


def download_webnovel_zh(out_dir: Path, n_shards: int = 6, seed: int = 42):
    """wdndev/webnovel-chinese = 10 jsonl shards × ~4 GB each (Chinese web novels).
    We only take `n_shards` random shards to cap disk usage."""
    total_shards = 10
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    indices = rng.sample(range(total_shards), min(n_shards, total_shards))
    indices.sort()
    print(f"\n[webnovel-chinese] downloading {len(indices)}/{total_shards} shards (seed={seed})")
    for i, shard in enumerate(indices, 1):
        fname = f"data/webnovel_{shard}.jsonl"
        local = out_dir / f"webnovel_{shard}.jsonl"
        if local.is_file():
            print(f"  [{i:>2}/{len(indices)}] skip (exists): {local.name}")
            continue
        path = hf_hub_download("wdndev/webnovel-chinese", fname,
                               repo_type="dataset", local_dir=str(out_dir))
        src = Path(path)
        if src != local and src.is_file():
            src.rename(local)
        sz = local.stat().st_size / 1e9 if local.is_file() else 0
        print(f"  [{i:>2}/{len(indices)}] {local.name}  {sz:.2f} GB")
    nested = out_dir / "data"
    if nested.is_dir() and not any(nested.iterdir()):
        nested.rmdir()


def download_gutenberg(out_dir: Path):
    """sedthh/gutenberg_english = 37 parquet shards × ~0.3 GB each (public-domain Eng lit).
    Download all of it — only ~10.75 GB total."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[gutenberg_english] downloading → {out_dir}")
    snapshot_download("sedthh/gutenberg_english", repo_type="dataset",
                      local_dir=str(out_dir),
                      allow_patterns=["data/*.parquet", "*.md"])
    print("  done")


def download_tinystories(out_dir: Path):
    """roneneldan/TinyStories parquet subset (data/*.parquet ≈ 1 GB)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[TinyStories] downloading → {out_dir}")
    snapshot_download("roneneldan/TinyStories", repo_type="dataset",
                      local_dir=str(out_dir),
                      allow_patterns=["data/*.parquet", "*.md"])
    print("  done")


def download_mxode_sources(repo_root: Path):
    """Mxode trio: Chinese-Reasoning-Distil-Data + CMID-Math-Instruct + School-Math-R1-Distil.
    All three are plain jsonl with {prompt, reasoning, response} or {query, response}."""
    targets = [
        ("Mxode/Chinese-Reasoning-Distil-Data",          "mxode-reasoning-distil"),
        ("Mxode/CMID-Chinese_Math_Instruct_Dataset",     "mxode-cmid-math"),
        ("Mxode/School-Math-R1-Distil-Chinese-220K",     "mxode-school-math"),
    ]
    for repo, local in targets:
        out_dir = repo_root / f"data/{local}"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{repo}] → {out_dir}")
        snapshot_download(repo, repo_type="dataset", local_dir=str(out_dir),
                          allow_patterns=["*.jsonl", "*.json", "*.md"])


def download_zake_openscience(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[zake7749/OpenScience-Chinese-Reasoning-SFT] → {out_dir}")
    snapshot_download("zake7749/OpenScience-Chinese-Reasoning-SFT",
                      repo_type="dataset", local_dir=str(out_dir),
                      allow_patterns=["data/*.parquet", "*.md"])


def download_almonster_mathinstruct(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[ALmonster/MathInstruct-Chinese] → {out_dir}")
    snapshot_download("ALmonster/MathInstruct-Chinese",
                      repo_type="dataset", local_dir=str(out_dir),
                      allow_patterns=["*.parquet", "*.md"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", default=str(Path(__file__).resolve().parent.parent.parent))
    ap.add_argument("--n_code_shards", type=int, default=200,
                    help="Random github-code-clean shards to pull (~0.36 GB each)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_code", action="store_true")
    ap.add_argument("--skip_math", action="store_true")
    ap.add_argument("--skip_zhihu", action="store_true")
    ap.add_argument("--skip_medical", action="store_true")
    ap.add_argument("--skip_webnovel", action="store_true")
    ap.add_argument("--skip_gutenberg", action="store_true")
    ap.add_argument("--skip_tinystories", action="store_true")
    ap.add_argument("--skip_mxode", action="store_true")
    ap.add_argument("--skip_zake", action="store_true")
    ap.add_argument("--skip_almonster", action="store_true")
    ap.add_argument("--n_webnovel_shards", type=int, default=6,
                    help="Random webnovel-chinese shards (of 10 total, ~4 GB each)")
    args = ap.parse_args()

    repo_root = Path(args.repo_root)
    if not args.skip_code:
        download_github_code(repo_root / "data/github-code-clean", args.n_code_shards, args.seed)
    if not args.skip_math:
        download_numinamath(repo_root / "data/NuminaMath-CoT")
    if not args.skip_zhihu:
        download_zhihu_kol(repo_root / "data/zhihu-kol")
    if not args.skip_medical:
        download_medical_zh(repo_root / "data/medical-zh")
    if not args.skip_webnovel:
        download_webnovel_zh(repo_root / "data/webnovel-zh", args.n_webnovel_shards, args.seed)
    if not args.skip_gutenberg:
        download_gutenberg(repo_root / "data/gutenberg-en")
    if not args.skip_tinystories:
        download_tinystories(repo_root / "data/tinystories")
    if not args.skip_mxode:
        download_mxode_sources(repo_root)
    if not args.skip_zake:
        download_zake_openscience(repo_root / "data/zake-openscience-zh")
    if not args.skip_almonster:
        download_almonster_mathinstruct(repo_root / "data/almonster-mathinstruct-zh")


if __name__ == "__main__":
    main()
