"""验证 NoThink SFT loss 真假.

加载 binned data + ckpt, 对若干 sample:
  - 列出 mask=1 的 token 数
  - 解码这些 mask token (应该是 assistant 答案)
  - 算每个 mask token 的真实 CE loss (不被 ignore_index 或 denom clamp 影响)
  - 看 loss 分布: 模板前缀几乎 0 vs 真实答案 token 多少
"""
from __future__ import annotations
import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from nsdata.sft_dataset import BinnedSFTDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--tokenizer", default="tokenizer_v3")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--n_samples", type=int, default=5)
    args = ap.parse_args()

    print(f"loading {args.ckpt}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt, trust_remote_code=True, dtype=torch.bfloat16,
    ).cuda().eval()

    print(f"loading dataset {args.data}", flush=True)
    ds = BinnedSFTDataset(args.data, max_length=args.max_length)
    print(f"  total docs: {len(ds)}")

    print(f"\n=== analyzing {args.n_samples} random samples ===\n")
    import random
    random.seed(42)
    indices = random.sample(range(len(ds)), args.n_samples)

    for i, idx in enumerate(indices):
        X, Y, mask = ds[idx]
        X = X.unsqueeze(0).cuda()
        Y = Y.unsqueeze(0).cuda()
        mask = mask.cuda()

        with torch.no_grad():
            out = model.snn(X, Y)

        per_token = out.last_loss          # (seq_len,) per-token CE
        mask_flat = mask.view(-1).float()
        valid = mask_flat.bool()           # which positions count

        # 真实 mean: 只在 mask=1 处
        mask_sum = mask_flat.sum().item()
        # 全 vocab CE (不被 ignore_index 影响)
        # 重新算一遍, 用 ignore_index=-100 以便所有 mask 位置都参与
        # 实际 model.snn 用 ignore_index=0, 跳过 target=0 的位置
        # 看 mask=1 中有多少 target=0 (本应被算 loss 但被 ignore 了)
        ignored_in_mask = (valid & (Y.view(-1) == 0)).sum().item()

        masked_losses = per_token[valid]
        if masked_losses.numel() == 0:
            print(f"\n[{i}] idx={idx}: ALL MASK ZERO (no assistant tokens visible??)")
            continue

        # 分布
        ml = masked_losses.cpu().float()
        print(f"\n[{i}] idx={idx}")
        print(f"  mask coverage : {int(mask_sum)} / {len(mask_flat)} ({mask_sum/len(mask_flat):.2%})")
        print(f"  ignored (target=0 inside mask): {ignored_in_mask}")
        print(f"  raw masked loss stats: mean={ml.mean():.4f} median={ml.median():.4f} "
              f"min={ml.min():.4f} max={ml.max():.4f} std={ml.std():.4f}")

        # 解码 mask 区段
        mask_positions = valid.nonzero(as_tuple=True)[0]
        if len(mask_positions) > 0:
            start, end = int(mask_positions[0]), int(mask_positions[-1]) + 1
            # 解码原始 sample 中从 start 到 end 的位置 (Y[start:end] 是预测目标)
            target_ids = Y.view(-1)[start:end].tolist()
            text = tok.decode(target_ids, skip_special_tokens=False)
            print(f"  assistant target text: {text[:300]!r}")

            # 把 token-level loss 与 token 配对显示前 10 个
            print(f"  per-token (token | loss):")
            for j, (tid, lv) in enumerate(zip(target_ids[:15], ml[:15].tolist())):
                tok_str = tok.decode([tid], skip_special_tokens=False)
                print(f"    {j:2d}  {repr(tok_str):20s}  loss={lv:.4f}")
            if len(target_ids) > 15:
                print(f"    ... ({len(target_ids)-15} more)")


if __name__ == "__main__":
    main()
