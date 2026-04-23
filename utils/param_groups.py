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
