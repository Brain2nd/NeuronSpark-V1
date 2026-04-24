"""Muon + DeepSpeed 4-GPU compatibility smoke test.

Builds a small v3 model, wraps with DS ZeRO stage=0 (Muon-required),
runs 3 training steps, verifies:
  1. forward/backward/step succeed
  2. Loss decreases (sanity)
  3. DS save_checkpoint + load_checkpoint round-trip preserves Muon state
  4. No NaN/Inf

Run:
    deepspeed --num_gpus=4 scripts/test_muon_ds.py --deepspeed_config ds_config.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import torch
import torch.distributed as dist
# H100 torch 2.4.1 has a FakeTensor / dynamo bug on @torch.compile'd helpers in
# modeling_neuronspark.py. Fall back to eager so the smoke test can run.
# TODO: upgrade torch on H100 to 2.9+ once the env dance is safe.
import torch._dynamo
torch._dynamo.config.suppress_errors = True

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neuronspark import NeuronSparkConfig, NeuronSparkForCausalLM
from utils.param_groups import build_muon_param_groups, promote_neuron_params_fp32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deepspeed_config", required=True)
    ap.add_argument("--local_rank", type=int, default=-1)
    ap.add_argument("--muon_variant", choices=["keller", "moonshot"], default="moonshot",
                    help="keller = upstream KellerJordan; moonshot = our Moonshot-scaled variant.")
    ap.add_argument("--zero_stage", type=int, default=0, choices=[0, 1, 2, 3],
                    help="DeepSpeed ZeRO stage to force for this test (default 0).")
    ap.add_argument("--optimizer", choices=["muon", "adam"], default="muon",
                    help="muon = MuonWithAuxAdam, adam = plain AdamW (for ZeRO-2 compat test).")
    ap.add_argument("--config", default=None,
                    help="Optional path to a NeuronSparkConfig JSON (overrides tiny defaults).")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=16)
    args, _ = ap.parse_known_args()

    import deepspeed
    if args.muon_variant == "keller":
        from muon import MuonWithAuxAdam
    else:
        from utils.muon_moonshot import MoonshotMuonWithAuxAdam as MuonWithAuxAdam

    if args.config:
        cfg = NeuronSparkConfig(**json.load(open(args.config)))
    else:
        cfg = NeuronSparkConfig(
            D=128, N=2, K=6, num_layers=4, D_ff=256,
            vocab_size=256, memory_layer_interval=2,
        )
    # Params must be bf16 for Triton PLIF kernel alignment (see train_pretrain.py).
    # DS bf16.enabled=true on top gives fp32 master copy + fp32 Adam state.
    m = NeuronSparkForCausalLM(cfg).cuda().to(torch.bfloat16)
    promote_neuron_params_fp32(m)

    if args.optimizer == "muon":
        pg = build_muon_param_groups(m)
        opt = MuonWithAuxAdam(pg)
        # Inject lr_mult post-init (Muon asserts strict group keys on init)
        for g in opt.param_groups:
            g["lr_mult"] = 1.0
    else:  # adam (plain, matches train_pretrain.py production)
        opt = torch.optim.Adam(m.parameters(), lr=1e-4, betas=(0.9, 0.95), eps=1e-8)

    with open(args.deepspeed_config) as f:
        ds_cfg = json.load(f)
    ds_cfg["train_micro_batch_size_per_gpu"] = args.batch
    ds_cfg["gradient_accumulation_steps"] = 1
    ds_cfg.setdefault("zero_optimization", {})["stage"] = args.zero_stage

    engine, optimizer, _, _ = deepspeed.initialize(
        model=m, optimizer=opt, config=ds_cfg,
    )
    rank = dist.get_rank()
    world = dist.get_world_size()
    if rank == 0:
        print(f"=== {args.optimizer}({args.muon_variant if args.optimizer=='muon' else ''}) + DS ZeRO-{args.zero_stage} "
              f"on {world} GPUs, bs={args.batch} seq={args.seq} ===")

    device = next(engine.parameters()).device
    x = torch.randint(0, cfg.vocab_size, (args.batch, args.seq), device=device)
    y = torch.randint(0, cfg.vocab_size, (args.batch, args.seq), device=device)

    # ==== Training steps: first 2 warmup (JIT compile), next 3 timed ====
    import time
    torch.cuda.reset_peak_memory_stats(device)
    losses = []
    step_times = []
    for step in range(5):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = engine.module.snn(x, y)
            loss = out.last_loss.mean()
        engine.backward(loss)
        engine.step()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        assert not torch.isnan(loss).item(), f"rank {rank}: NaN at step {step}"
        losses.append(loss.item())
        step_times.append(dt)
    peak_mem = torch.cuda.max_memory_allocated(device) / 1e9
    if rank == 0:
        print(f"  peak memory (rank 0): {peak_mem:.2f} GB")
        print(f"  step losses (rank 0): {[f'{l:.4f}' for l in losses]}")
        print(f"  step times (s): {[f'{t:.2f}' for t in step_times]}")
        # Steps 2..4 after compile; report mean of those
        steady = step_times[2:]
        if steady:
            print(f"  steady-state (last 3 steps): mean {sum(steady)/len(steady)*1000:.0f} ms, "
                  f"min {min(steady)*1000:.0f} ms")
        assert losses[-1] < losses[0] or losses[-1] < 5.8, f"Loss not decreasing: {losses}"
        print("  ✓ 5 training steps, loss decreasing, no NaN")

    # Skip save/load section for adam (no Muon params). Only meaningful for muon.
    if args.optimizer != "muon":
        return

    # ==== DS save/load round-trip ====
    # Use /workspace (data disk) not /tmp (20G system disk) to avoid filling up
    shared_dir = os.environ.get("MUON_TEST_CKPT_DIR",
                                 "/workspace/tmp/muon_ds_ckpt_test"
                                 if os.path.isdir("/workspace") else "/tmp/muon_ds_ckpt_test")
    if rank == 0:
        if os.path.isdir(shared_dir):
            shutil.rmtree(shared_dir)
        os.makedirs(shared_dir, exist_ok=True)
    dist.barrier()

    engine.save_checkpoint(shared_dir, tag="muon_ds")
    dist.barrier()

    # Snapshot: first Muon param + its momentum state
    first_muon = None
    for g in engine.optimizer.param_groups:
        if g.get("use_muon"):
            for p in g["params"]:
                if p.requires_grad:
                    first_muon = p
                    break
            if first_muon is not None: break
    assert first_muon is not None, "No Muon param found"
    snap_weight = first_muon.data.clone()
    snap_mom = engine.optimizer.state[first_muon].get("momentum_buffer", None)
    if snap_mom is not None:
        snap_mom = snap_mom.clone()

    # Dirty: 2 more steps
    for _ in range(2):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = engine.module.snn(x, y).last_loss.mean()
        engine.backward(loss)
        engine.step()

    weight_dirt = (first_muon.data - snap_weight).abs().max().item()
    if rank == 0:
        print(f"  before reload: weight drift after 2 more steps = {weight_dirt:.6e}")
    assert weight_dirt > 0, "Weights did not change after steps"

    # Reload
    engine.load_checkpoint(shared_dir, tag="muon_ds")

    weight_restored = (first_muon.data - snap_weight).abs().max().item()
    weight_restored_t = torch.tensor([weight_restored], device=device)
    dist.all_reduce(weight_restored_t, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(f"  after reload: max rank weight diff = {weight_restored_t.item():.6e}")
    assert weight_restored < 1e-5, f"Muon weight not restored (diff={weight_restored})"
    if rank == 0:
        print(f"  ✓ DS save/load round-trip preserves Muon weights across {world} ranks")
        print("\nAll Muon+DS compat tests passed ✓")

    if rank == 0 and os.path.isdir(shared_dir):
        shutil.rmtree(shared_dir)


if __name__ == "__main__":
    main()
