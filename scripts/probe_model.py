"""Load a NeuronSparkConfig + build the model, print per-module param counts,
and optionally run one forward+backward at a given batch/seq to measure peak
memory (single-GPU — for quick OOM check).

Usage:
    python scripts/probe_model.py --config configs/smoke_1p16b.json
    python scripts/probe_model.py --config configs/smoke_1p16b.json --batch 3 --seq 2048
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# suppress torch.compile failures on older torch (e.g. 2.4)
import torch._dynamo
torch._dynamo.config.suppress_errors = True

from neuronspark import NeuronSparkConfig, NeuronSparkForCausalLM


def _fmt(n):
    if n >= 1e9: return f"{n/1e9:.3f} B"
    if n >= 1e6: return f"{n/1e6:.2f} M"
    if n >= 1e3: return f"{n/1e3:.1f} K"
    return f"{n}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--batch", type=int, default=0,
                    help="If >0, run one forward+backward to measure peak memory.")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    cfg_kwargs = json.load(open(args.config))
    cfg = NeuronSparkConfig(**cfg_kwargs)
    print(f"=== Config ===")
    for k in ["D", "N", "K", "num_layers", "D_ff", "vocab_size", "memory_layer_interval"]:
        v = getattr(cfg, k, "?")
        print(f"  {k}: {v}")

    model = NeuronSparkForCausalLM(cfg)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n=== Total params: {_fmt(total)}  (trainable: {_fmt(trainable)}) ===\n")

    # Group by top-level module prefix
    groups = OrderedDict()
    for name, p in model.named_parameters():
        # Bucket: take first 2-3 path segments
        parts = name.split(".")
        if "layers." in name:
            # e.g. snn.layers.3.attn.q_proj.weight → snn.layers.*.attn.q_proj
            key = ".".join(p for p in parts if not p.isdigit())
        else:
            key = ".".join(parts[:3])
        groups[key] = groups.get(key, 0) + p.numel()

    # Print top-level blobs (aggregated across 24 layers for layer.*)
    print(f"=== Per-module breakdown ===")
    layer_buckets = OrderedDict()
    other_buckets = OrderedDict()
    for k, v in groups.items():
        if "layers." in k:
            layer_buckets[k] = v
        else:
            other_buckets[k] = v

    print(f"  --- Non-layer ---")
    for k, v in sorted(other_buckets.items(), key=lambda x: -x[1]):
        pct = v / total * 100
        print(f"    {k:<50s}  {_fmt(v):>10s}  ({pct:5.2f}%)")

    print(f"  --- Per-layer × {cfg.num_layers} (summed) ---")
    layer_total = sum(layer_buckets.values())
    for k, v in sorted(layer_buckets.items(), key=lambda x: -x[1]):
        pct = v / total * 100
        per_layer = v / cfg.num_layers
        print(f"    {k:<50s}  {_fmt(v):>10s}  ({pct:5.2f}%)  [{_fmt(per_layer)}/layer]")
    print(f"    {'(all layers)':<50s}  {_fmt(layer_total):>10s}  ({layer_total/total*100:.2f}%)")

    # Optional memory probe
    if args.batch > 0:
        if not torch.cuda.is_available():
            print("CUDA not available, skipping memory probe."); return
        print(f"\n=== Memory probe: batch={args.batch}, seq={args.seq}, dtype={args.dtype} ===")
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
        device = torch.device("cuda")
        model = model.to(device).to(dtype)
        torch.cuda.reset_peak_memory_stats(device)

        x = torch.randint(0, cfg.vocab_size, (args.batch, args.seq), device=device)
        y = torch.randint(0, cfg.vocab_size, (args.batch, args.seq), device=device)

        torch.cuda.synchronize()
        with torch.amp.autocast("cuda", dtype=dtype):
            out = model.snn(x, y)
            loss = out.last_loss.mean()
        print(f"  forward loss: {loss.item():.4f}")
        peak_fwd = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"  peak (forward only): {peak_fwd:.2f} GB")

        loss.backward()
        torch.cuda.synchronize()
        peak_bwd = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"  peak (forward+backward): {peak_bwd:.2f} GB")


if __name__ == "__main__":
    main()
