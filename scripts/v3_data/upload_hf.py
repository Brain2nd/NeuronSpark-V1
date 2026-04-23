"""Upload data/v3_pretrain_mix/ to HuggingFace as Brain2nd/NeuronSpark-Pretrain-v3."""
from __future__ import annotations

import argparse
from pathlib import Path
from huggingface_hub import HfApi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", default="Brain2nd/NeuronSpark-Pretrain-v3")
    ap.add_argument("--folder", default="data/v3_pretrain_mix")
    ap.add_argument("--private", action="store_true", default=False)
    args = ap.parse_args()

    folder = Path(args.folder).resolve()
    assert folder.is_dir(), f"not a dir: {folder}"

    api = HfApi()
    print(f"Creating/ensuring repo {args.repo_id} (private={args.private}) …")
    api.create_repo(args.repo_id, repo_type="dataset",
                    private=args.private, exist_ok=True)

    print(f"Uploading {folder} → {args.repo_id} …")
    api.upload_large_folder(
        folder_path=str(folder),
        repo_id=args.repo_id,
        repo_type="dataset",
    )
    print(f"\nDONE: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
