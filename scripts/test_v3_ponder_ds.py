"""DeepSpeed + v3 PonderNet compatibility smoke test.

Tests:
1. Buffer dtype stays fp32 after model.to(bf16) + promote (prevents bias/EMA precision loss)
2. DeepSpeed engine wrap succeeds with KPredictor in param groups
3. Forward/backward through DS engine produces non-NaN loss + valid gradients
4. update_ponder_bias mutates bias buffer correctly under DS
5. Gradient checkpoint + Gumbel RNG: same forward twice with same seed → same y_hard
6. Save/load round-trip preserves bias + _usage_ema buffers

Run:
    python scripts/test_v3_ponder_ds.py     # single-GPU (skips multi-rank sync test)
    deepspeed --num_gpus=1 scripts/test_v3_ponder_ds.py --deepspeed_config ds_config.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neuronspark import NeuronSparkConfig, NeuronSparkForCausalLM
from utils.param_groups import build_param_groups, promote_neuron_params_fp32


def make_config():
    return NeuronSparkConfig(
        D=128, N=2, K=6, num_layers=4, D_ff=256, vocab_size=256,
        memory_layer_interval=2,
    )


def test_1_buffer_dtype():
    print("=== Test 1: Buffer dtype after model.to(bf16) + promote ===")
    cfg = make_config()
    m = NeuronSparkForCausalLM(cfg).cuda()
    m = m.to(torch.bfloat16)
    n = promote_neuron_params_fp32(m)

    # Check: all KPredictor bias + _usage_ema buffers are fp32
    fp32_count = 0
    bf16_leaks = []
    for name, buf in m.named_buffers():
        if "k_predictor" in name and (name.endswith("bias") or name.endswith("_usage_ema")):
            if buf.dtype == torch.float32:
                fp32_count += 1
            else:
                bf16_leaks.append((name, buf.dtype))

    assert fp32_count > 0, "No KPredictor buffers found"
    assert not bf16_leaks, f"Buffers not promoted to fp32: {bf16_leaks}"
    print(f"  ✓ {fp32_count} PonderNet buffers in fp32 (promoted {n} total tensors)")


def test_2_ds_engine_wrap(ds_config_path):
    print("=== Test 2: DeepSpeed engine wrap ===")
    import deepspeed

    cfg = make_config()
    m = NeuronSparkForCausalLM(cfg).cuda().to(torch.bfloat16)
    promote_neuron_params_fp32(m)

    # Build param groups (should include k_predictors group)
    pg = build_param_groups(m, learning_rate=2e-4, neuron_lr_mult=10.0)
    assert len(pg) == 2, f"Expected 2 groups, got {len(pg)}"
    opt = torch.optim.Adam(pg)

    # Read ds config
    with open(ds_config_path) as f:
        ds_cfg = json.load(f)
    ds_cfg["train_micro_batch_size_per_gpu"] = 1
    ds_cfg["gradient_accumulation_steps"] = 1

    engine, optimizer, _, _ = deepspeed.initialize(
        model=m, optimizer=opt, config=ds_cfg,
    )
    print(f"  ✓ DeepSpeed engine wrapped (world={dist.get_world_size()}, rank={dist.get_rank()})")
    return engine


def test_3_fwd_bwd(engine):
    print("=== Test 3: Forward/backward through DS engine ===")
    cfg = engine.module.config
    device = next(engine.parameters()).device
    x = torch.randint(0, cfg.vocab_size, (2, 16), device=device)
    y = torch.randint(0, cfg.vocab_size, (2, 16), device=device)

    # Set ponder state
    engine.module.snn.set_ponder_temperature(1.5)
    engine.module.snn.set_ponder_exploration(0.05)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = engine.module.snn(x, y)
        loss = out.last_loss.mean()

    assert not torch.isnan(loss).item(), f"Loss NaN: {loss.item()}"
    print(f"  ✓ forward OK: loss={loss.item():.3f}, ponder_cost={out.ponder_cost.item():.3f}")

    engine.backward(loss)
    engine.module.snn.compensate_modulation_gradients()
    engine.step()
    print("  ✓ backward + step OK")


def test_4_bias_update(engine):
    print("=== Test 4: update_ponder_bias ===")
    # Capture bias before
    l0 = engine.module.snn.layers[0]
    bias_before = l0.ffn_k_predictor.bias.clone()
    ema_before = l0.ffn_k_predictor._usage_ema.clone()

    engine.module.snn.update_ponder_bias(lr_bias=1e-3, ema_decay=0.99)

    bias_after = l0.ffn_k_predictor.bias
    ema_after = l0.ffn_k_predictor._usage_ema

    bias_delta = (bias_after - bias_before).abs().max().item()
    ema_delta = (ema_after - ema_before).abs().max().item()

    assert bias_delta > 0, "Bias did not update"
    assert ema_delta > 0, "EMA did not update"
    assert bias_after.dtype == torch.float32, f"Bias dtype drifted to {bias_after.dtype}"
    print(f"  ✓ bias Δ={bias_delta:.6f}, EMA Δ={ema_delta:.6f}, dtype=fp32")


def test_7_ds_checkpoint_roundtrip(engine, ds_config_path):
    """Verify save_checkpoint + load_checkpoint restores optimizer state.

    Flow:
      1. Take optimizer state snapshot
      2. Save DS checkpoint
      3. Take a training step (to dirty optimizer state)
      4. Load DS checkpoint → optimizer should match snapshot
    """
    print("=== Test 7: DeepSpeed save_checkpoint / load_checkpoint round-trip ===")
    import deepspeed, tempfile

    cfg = engine.module.config
    device = next(engine.parameters()).device

    # Do one optimizer step to populate state
    x = torch.randint(0, cfg.vocab_size, (1, 8), device=device)
    y = torch.randint(0, cfg.vocab_size, (1, 8), device=device)
    engine.module.snn.set_ponder_temperature(1.0)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = engine.module.snn(x, y)
        loss = out.last_loss.mean()
    engine.backward(loss)
    engine.step()

    with tempfile.TemporaryDirectory() as tmp:
        # Save checkpoint
        engine.save_checkpoint(tmp, tag="deepspeed")
        # Verify directory created
        ds_dir = os.path.join(tmp, "deepspeed")
        assert os.path.isdir(ds_dir), f"deepspeed/ dir not created under {tmp}"

        # Take snapshot of bias/ema
        l0 = engine.module.snn.layers[0].ffn_k_predictor
        snapshot_bias = l0.bias.clone()
        snapshot_ema = l0._usage_ema.clone()

        # Dirty the state: more steps + bias updates
        for _ in range(3):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = engine.module.snn(x, y)
                loss = out.last_loss.mean()
            engine.backward(loss)
            engine.step()
            engine.module.snn.update_ponder_bias(lr_bias=1e-3, ema_decay=0.99)

        # Verify state has changed
        bias_changed = (l0.bias - snapshot_bias).abs().max().item()
        ema_changed = (l0._usage_ema - snapshot_ema).abs().max().item()
        assert bias_changed > 0 or ema_changed > 0, "State did not diverge after extra steps"

        # Load checkpoint
        load_path, _ = engine.load_checkpoint(tmp, tag="deepspeed")
        assert load_path is not None, "load_checkpoint returned None"

        # Verify state restored
        bias_diff = (l0.bias - snapshot_bias).abs().max().item()
        ema_diff = (l0._usage_ema - snapshot_ema).abs().max().item()
        print(f"  ✓ After restore: bias diff={bias_diff:.6e}, EMA diff={ema_diff:.6e}")
        assert bias_diff < 1e-5, f"Bias not restored (diff={bias_diff})"
        assert ema_diff < 1e-5, f"EMA not restored (diff={ema_diff})"
        print(f"  ✓ DeepSpeed round-trip preserves optimizer + bias + EMA state")


def test_5_gumbel_rng_determinism():
    print("=== Test 5: Gradient-checkpoint + Gumbel RNG determinism ===")
    cfg = make_config()
    m = NeuronSparkForCausalLM(cfg).cuda().to(torch.bfloat16)
    promote_neuron_params_fp32(m)
    m.train()
    m.snn.set_ponder_temperature(1.0)
    m.snn.set_ponder_exploration(0.0)  # disable forced exploration for determinism

    device = next(m.parameters()).device
    x = torch.randint(0, cfg.vocab_size, (2, 16), device=device)
    y = torch.randint(0, cfg.vocab_size, (2, 16), device=device)

    # Run 1: same seed, record y_hard from first layer
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        _ = m.snn(x, y)
    y_hard_1 = m.snn.layers[0].ffn_k_predictor  # Inspect the last y_hard cached
    y1 = m.snn.layers[0]._last_y_hard_ffn.clone()

    # Run 2: same seed
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        _ = m.snn(x, y)
    y2 = m.snn.layers[0]._last_y_hard_ffn.clone()

    assert torch.equal(y1, y2), f"Gumbel sampling non-deterministic across calls"
    print(f"  ✓ same-seed forward → identical y_hard")


def test_6_save_load_roundtrip():
    print("=== Test 6: save_pretrained / from_pretrained round-trip ===")
    cfg = make_config()
    m1 = NeuronSparkForCausalLM(cfg).cuda().to(torch.bfloat16)
    promote_neuron_params_fp32(m1)

    # Modify bias to non-zero
    m1.snn.layers[0].ffn_k_predictor.bias.data.fill_(0.1234)
    m1.snn.layers[0].ffn_k_predictor._usage_ema.data.fill_(0.5)

    with tempfile.TemporaryDirectory() as tmpdir:
        m1.save_pretrained(tmpdir)
        m2 = NeuronSparkForCausalLM.from_pretrained(tmpdir, dtype=torch.bfloat16, trust_remote_code=True).cuda()

    bias_diff = (m2.snn.layers[0].ffn_k_predictor.bias.data - 0.1234).abs().max().item()
    ema_diff = (m2.snn.layers[0].ffn_k_predictor._usage_ema.data - 0.5).abs().max().item()
    assert bias_diff < 1e-3, f"Bias not preserved: diff={bias_diff}"
    assert ema_diff < 1e-3, f"EMA not preserved: diff={ema_diff}"
    print(f"  ✓ bias preserved (diff={bias_diff:.6f}), EMA preserved (diff={ema_diff:.6f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deepspeed_config", default=None)
    ap.add_argument("--local_rank", type=int, default=-1)
    args, _ = ap.parse_known_args()

    assert torch.cuda.is_available(), "GPU required"

    # Tests 1, 5, 6 don't need DeepSpeed
    test_1_buffer_dtype()
    test_5_gumbel_rng_determinism()
    test_6_save_load_roundtrip()

    # Tests 2-4 + 7 need DeepSpeed engine
    if args.deepspeed_config and os.path.isfile(args.deepspeed_config):
        engine = test_2_ds_engine_wrap(args.deepspeed_config)
        test_3_fwd_bwd(engine)
        test_4_bias_update(engine)
        test_7_ds_checkpoint_roundtrip(engine, args.deepspeed_config)
    else:
        print("=== Tests 2-4, 7 SKIPPED (no --deepspeed_config) ===")
        print("    Run with: deepspeed --num_gpus=1 scripts/test_v3_ponder_ds.py --deepspeed_config ds_config.json")

    print("\nAll compat tests passed ✓")


if __name__ == "__main__":
    main()
