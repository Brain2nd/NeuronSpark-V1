"""Optimizer param-group helpers for NeuronSpark SNN training.

Builds 2-group optimizer layout: neuron params get `neuron_lr_mult × base_lr`,
matrix/linear params use base_lr. Uses `NeuronSparkForCausalLM.snn.get_param_groups()`
for the functional classification.
"""
from __future__ import annotations

import torch
import torch.nn as nn


NEURON_GROUP_KEYS = (
    "input_neurons",
    "b_beta",
    "b_alpha",
    "b_th",
    "block_output_neuron",
    "ffn_neurons",
    "output_neuron",
)


def _get_inner_snn(model: nn.Module):
    """Unwrap DeepSpeed/DDP/FSDP + HF wrapper → raw SNNLanguageModel."""
    raw = model.module if hasattr(model, "module") else model
    # HF wrapper stores inner model at .snn
    if hasattr(raw, "snn"):
        return raw.snn
    return raw


def promote_neuron_params_fp32(model: nn.Module) -> int:
    """Cast neuron-specific parameters + PonderNet bias buffers to fp32.

    Neuron parameters (LIF β/α/threshold, gain `w`, `v_th`) need fp32 precision
    because sigmoid/softplus saturate in bf16 and tiny updates get quantized
    away. PonderNet `bias` / `_usage_ema` buffers need fp32 because
    EMA-accumulation of small fractions in bf16 loses signal entirely.

    Matrix weights stay bf16 via autocast in model forward.

    Returns number of tensors promoted.
    """
    count = 0
    for name, p in model.named_parameters():
        if name.endswith((".w", ".v_th", ".b_beta", ".b_alpha", ".b_th")):
            if p.dtype != torch.float32:
                p.data = p.data.float()
                count += 1
    # PonderNet KPredictor buffers: bias + _usage_ema
    for name, buf in model.named_buffers():
        if name.endswith((".bias", "._usage_ema")) and "k_predictor" in name:
            if buf.dtype != torch.float32:
                buf.data = buf.data.float()
                count += 1
    return count


def _partition_params_for_muon(model: nn.Module):
    """Classify NeuronSpark model parameters into Muon vs Adam groups.

    Muon eligibility: 2D+ matrix parameters in transformer blocks (Linear weights),
    EXCLUDING embedding (large vocab makes Newton-Schulz pointless) and norms.
    All other parameters (biases, norms, embeddings, 1D neuron params) go to Adam.
    Neuron params go to a SEPARATE Adam group with LR × multiplier.

    Returns:
        muon_params:       list of parameters (matrix Linear weights) for Muon
        adam_embed_params: list (embedding weight, typically shared with LM head)
        adam_norm_params:  list (RMSNorm gains, all 1D scalars)
        adam_neuron_params:list (PLIF .w, .v_th, .b_beta, .b_alpha, .b_th — fp32 master)
    """
    snn = _get_inner_snn(model)
    pg = snn.get_param_groups()

    muon_params = []
    adam_embed = []
    adam_norm = []
    adam_neuron = []

    # Neurons (already identified by name pattern)
    for k in NEURON_GROUP_KEYS:
        adam_neuron.extend(pg.get(k, []))

    # Embedding (vocab × D) — keep in Adam
    adam_embed.extend(pg.get("embedding", []))

    # Norms (1D) and halt (k_predictor handled separately)
    adam_norm.extend(pg.get("rms_norms", []))
    adam_norm.extend(pg.get("norm", []))        # top-level snn.norm (RMSNorm gain)

    # decode_proj (D × D) — candidate for Muon
    muon_params.extend(pg.get("decode", []))

    # All Linear weight matrices → Muon
    for key in ("residual_projs",
                "W_in", "W_beta", "W_alpha", "W_th", "W_gate", "W_skip", "W_out",
                "ffn_gate_proj", "ffn_up_proj", "ffn_down_proj", "ffn_skip_proj",
                "k_predictors"):
        for p in pg.get(key, []):
            if p.ndim >= 2:
                muon_params.append(p)
            else:
                # Fallback: 1D slipped in, goes to adam norm group
                adam_norm.append(p)

    # Conv1d weight is 3D (Mamba-style depthwise), NOT matrix-orthogonalizable
    # → route to adam_norm (generic 2+ moment optimizer is fine for this)
    for p in pg.get("conv1d", []):
        adam_norm.append(p)

    # Sanity: dedup and ensure all trainable params covered
    seen = set()
    def dedup(lst):
        out = []
        for p in lst:
            if id(p) not in seen:
                seen.add(id(p))
                out.append(p)
        return out

    muon_params = dedup(muon_params)
    adam_embed = dedup(adam_embed)
    adam_norm = dedup(adam_norm)
    adam_neuron = dedup(adam_neuron)

    total_in = sum(p.numel() for p in muon_params) + \
               sum(p.numel() for p in adam_embed) + \
               sum(p.numel() for p in adam_norm) + \
               sum(p.numel() for p in adam_neuron)
    total_model = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if total_in != total_model:
        grouped_ids = seen
        missing = [p for p in model.parameters()
                   if p.requires_grad and id(p) not in grouped_ids]
        n_missing = sum(p.numel() for p in missing)
        print(f"[muon] WARN: {n_missing:,} params ({n_missing/total_model*100:.3f}%) "
              f"unclassified; falling back to adam_norm group.")
        adam_norm.extend(missing)

    return muon_params, adam_embed, adam_norm, adam_neuron


def build_muon_param_groups(
    model: nn.Module,
    muon_lr: float = 0.02,
    muon_momentum: float = 0.95,
    adam_base_lr: float = 2e-4,
    adam_embed_lr: float = 2e-4,
    neuron_lr_mult: float = 10.0,
    weight_decay_muon: float = 0.0,
    weight_decay_adam: float = 0.0,
) -> list[dict]:
    """Build param_groups compatible with `muon.MuonWithAuxAdam`.

    MuonWithAuxAdam expects each group to have `use_muon: bool` flag.
      - Muon groups: {params, lr, momentum, weight_decay, use_muon=True}
      - Adam groups: {params, lr, betas, eps, weight_decay, use_muon=False}

    Our 4-group layout:
      0. Muon matrix params (transformer Linear weights, decode_proj, k_predictor nets)
      1. Aux Adam: embedding (vocab × D)
      2. Aux Adam: norms / 1D scalars
      3. Aux Adam: neuron fp32 params (LR × mult, typically 10×)

    Each group carries `lr_mult` so cosine schedules can rescale.
    """
    muon_params, adam_embed, adam_norm, adam_neuron = _partition_params_for_muon(model)

    groups: list[dict] = []

    # NOTE: MuonWithAuxAdam asserts group keys == exact set, so we can't put
    # `lr_mult` inside the dict pre-init. Instead, append lr_mult AFTER the
    # optimizer is built (done by caller). We emit _pending_lr_mult alongside.
    if muon_params:
        groups.append({
            "params": muon_params,
            "lr": muon_lr,
            "momentum": muon_momentum,
            "weight_decay": weight_decay_muon,
            "use_muon": True,
        })
    if adam_embed:
        groups.append({
            "params": adam_embed,
            "lr": adam_embed_lr,
            "betas": (0.9, 0.95),
            "eps": 1e-10,
            "weight_decay": weight_decay_adam,
            "use_muon": False,
        })
    if adam_norm:
        groups.append({
            "params": adam_norm,
            "lr": adam_base_lr,
            "betas": (0.9, 0.95),
            "eps": 1e-10,
            "weight_decay": weight_decay_adam,
            "use_muon": False,
        })
    if adam_neuron:
        groups.append({
            "params": adam_neuron,
            "lr": adam_base_lr * neuron_lr_mult,
            "betas": (0.9, 0.95),
            "eps": 1e-10,
            "weight_decay": 0.0,                # neurons never weight_decay
            "use_muon": False,
        })

    # Sanity summary
    counts = {
        "muon": sum(p.numel() for p in muon_params),
        "adam_embed": sum(p.numel() for p in adam_embed),
        "adam_norm": sum(p.numel() for p in adam_norm),
        "adam_neuron": sum(p.numel() for p in adam_neuron),
    }
    total = sum(counts.values())
    print(f"[build_muon_param_groups] "
          f"muon={counts['muon']/1e6:.1f}M  embed={counts['adam_embed']/1e6:.1f}M  "
          f"norm={counts['adam_norm']/1e6:.1f}M  neuron={counts['adam_neuron']/1e6:.3f}M  "
          f"total={total/1e6:.1f}M")
    return groups


def build_param_groups(
    model: nn.Module,
    learning_rate: float,
    neuron_lr_mult: float = 10.0,
    weight_decay_other: float = 0.0,
    weight_decay_neuron: float = 0.0,
) -> list[dict]:
    """Build Adam/AdamW param_groups with neuron × `neuron_lr_mult` LR.

    Args:
        model:       raw NeuronSparkForCausalLM (or unwrapped inner SNNLanguageModel).
        learning_rate: base LR for matrix/linear params.
        neuron_lr_mult: LR multiplier for neuron params (default 10×).
        weight_decay_other / weight_decay_neuron: L2 regularization per group.

    Returns:
        list[dict] ready to pass to `torch.optim.Adam(...)` / `AdamW(...)`.

    Each dict carries a `lr_mult` key so cosine schedules can rescale correctly.
    """
    snn = _get_inner_snn(model)
    pg = snn.get_param_groups()

    neuron_params = [p for k in NEURON_GROUP_KEYS for p in pg.get(k, [])]
    other_params = [p for k, ps in pg.items() if k not in NEURON_GROUP_KEYS for p in ps]

    # Sanity: grouped params should cover (almost) all trainable params.
    n_neuron = sum(p.numel() for p in neuron_params)
    n_other = sum(p.numel() for p in other_params)
    n_total = n_neuron + n_other
    model_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n_total != model_total:
        # Collect missing params (not in any group) and append to 'other' group so
        # nothing goes untrained. get_param_groups() coverage may drift as
        # architecture evolves on v3; we trade strict parity for robustness.
        grouped_ids = {id(p) for ps in pg.values() for p in ps}
        missing = [p for p in model.parameters() if p.requires_grad and id(p) not in grouped_ids]
        n_missing = sum(p.numel() for p in missing)
        print(
            f"[build_param_groups] WARN: {n_missing:,} params ({n_missing/model_total*100:.3f}%) "
            f"not classified by get_param_groups(); adding to 'other' group with base LR."
        )
        other_params = other_params + missing

    groups = [
        {
            "params": other_params,
            "lr": learning_rate,
            "lr_mult": 1.0,
            "weight_decay": weight_decay_other,
        },
        {
            "params": neuron_params,
            "lr": learning_rate * neuron_lr_mult,
            "lr_mult": float(neuron_lr_mult),
            "weight_decay": weight_decay_neuron,
        },
    ]
    return groups
