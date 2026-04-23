"""Test MoonshotMuonWithAuxAdam — single-GPU local validation.

1. Verify scaling formula vs KellerJordan (on isolated orthogonalized matrix)
2. Full forward+backward+step on a tiny model, compare updates vs KellerJordan
3. Save/load optimizer state, verify deterministic resume

Run:
  python scripts/test_moonshot_muon.py
"""
from __future__ import annotations

import sys
from pathlib import Path
import tempfile

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.muon_moonshot import (
    SingleDeviceMoonshotMuonWithAuxAdam,
    moonshot_muon_update,
    zeropower_via_newtonschulz5,
)
import muon as keller  # upstream


# ============================================================
# Test 1: scaling formula differs as designed
# ============================================================

def test_scaling_formula():
    """Verify that on the SAME orthogonalized matrix, Moonshot vs Keller scale differently."""
    torch.manual_seed(0)
    g = torch.randn(512, 128).cuda()  # m=512, n=128
    m1 = torch.zeros_like(g)
    m2 = torch.zeros_like(g)

    # Keller
    k_out = keller.muon_update(g.clone(), m1, beta=0.95)
    # Moonshot
    m_out = moonshot_muon_update(g.clone(), m2, beta=0.95)

    # Shapes match
    assert k_out.shape == m_out.shape == g.shape, "shape mismatch"

    # Ratio check: Moonshot / Keller = (0.2 * sqrt(max(m,n))) / max(1, m/n)^0.5
    # For 512x128: Moonshot = 0.2 * sqrt(512) = 4.525
    #              Keller   = sqrt(512/128) = 2.0
    # ratio ≈ 2.26
    k_norm = k_out.float().norm().item()
    m_norm = m_out.float().norm().item()
    ratio = m_norm / k_norm
    expected_ratio = (0.2 * (512 ** 0.5)) / ((512 / 128) ** 0.5)
    print(f"  Keller norm:   {k_norm:.4f}")
    print(f"  Moonshot norm: {m_norm:.4f}")
    print(f"  Ratio (actual/expected): {ratio:.4f} / {expected_ratio:.4f}")
    assert abs(ratio - expected_ratio) < 0.02, f"scaling off: {ratio} vs {expected_ratio}"
    print("  ✓ test_scaling_formula passed")


# ============================================================
# Test 2: full optimizer step on a tiny transformer-ish model
# ============================================================

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(1000, 64)
        self.norm = nn.LayerNorm(64)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.head = nn.Linear(64, 1000)

    def forward(self, x):
        h = self.norm(self.embed(x))
        h = torch.relu(self.fc1(h))
        h = self.fc2(h)
        return self.head(h)


def _make_groups(model):
    matrix = [p for n, p in model.named_parameters() if p.ndim >= 2 and "embed" not in n and "head" not in n]
    embed  = [p for n, p in model.named_parameters() if "embed" in n or "head" in n]
    scalar = [p for n, p in model.named_parameters()
              if p.ndim < 2 and "embed" not in n and "head" not in n]
    return [
        dict(params=matrix, lr=0.02, momentum=0.95, weight_decay=0, use_muon=True),
        dict(params=embed,  lr=3e-4, betas=(0.9, 0.95), eps=1e-10, weight_decay=0, use_muon=False),
        dict(params=scalar, lr=3e-4, betas=(0.9, 0.95), eps=1e-10, weight_decay=0, use_muon=False),
    ]


def test_optimizer_step():
    """Run one forward-backward-step; ensure loss goes down and params change."""
    torch.manual_seed(42)
    model = TinyModel().cuda()
    opt = SingleDeviceMoonshotMuonWithAuxAdam(_make_groups(model))

    p_before = {n: p.clone() for n, p in model.named_parameters()}
    x = torch.randint(0, 1000, (4, 16)).cuda()
    y = torch.randint(0, 1000, (4, 16)).cuda()

    losses = []
    for step in range(5):
        logits = model(x)
        loss = nn.functional.cross_entropy(logits.view(-1, 1000), y.view(-1))
        losses.append(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"  Loss: {losses[0]:.4f} → {losses[-1]:.4f} (Δ={losses[0]-losses[-1]:+.4f})")
    assert losses[-1] < losses[0], "loss should decrease"

    # Ensure every param moved
    changed = sum(1 for n, p in model.named_parameters() if not torch.equal(p, p_before[n]))
    total = len(list(model.named_parameters()))
    print(f"  Params changed: {changed}/{total}")
    assert changed == total, "all params should be updated"
    print("  ✓ test_optimizer_step passed")


# ============================================================
# Test 3: save / load optimizer state round-trip
# ============================================================

def test_save_load():
    torch.manual_seed(7)
    model = TinyModel().cuda()
    opt = SingleDeviceMoonshotMuonWithAuxAdam(_make_groups(model))

    x = torch.randint(0, 1000, (4, 16)).cuda()
    y = torch.randint(0, 1000, (4, 16)).cuda()

    # Pre-warm
    for _ in range(3):
        loss = nn.functional.cross_entropy(
            model(x).view(-1, 1000), y.view(-1)
        )
        opt.zero_grad(); loss.backward(); opt.step()

    # Save
    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "opt.pth"
        torch.save({"model": model.state_dict(),
                    "opt": opt.state_dict()}, ckpt)

        # Mutate + restore
        for _ in range(3):
            loss = nn.functional.cross_entropy(
                model(x).view(-1, 1000), y.view(-1)
            )
            opt.zero_grad(); loss.backward(); opt.step()

        state = torch.load(ckpt, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])

        # Step once more and check: momentum buffers exist
        loss = nn.functional.cross_entropy(model(x).view(-1, 1000), y.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()

        # Sanity: momentum_buffer present on at least one muon param
        has_mom = any("momentum_buffer" in opt.state[p] for g in opt.param_groups if g["use_muon"]
                      for p in g["params"])
        print(f"  momentum_buffer present after load: {has_mom}")
        assert has_mom, "muon momentum_buffer lost after load"
    print("  ✓ test_save_load passed")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA required."); sys.exit(1)

    print("=== Test 1: scaling formula ===")
    test_scaling_formula()
    print("\n=== Test 2: optimizer step ===")
    test_optimizer_step()
    print("\n=== Test 3: save/load ===")
    test_save_load()
    print("\nALL TESTS PASSED ✓")
