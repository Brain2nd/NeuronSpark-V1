# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NeuronSpark is a **100% SNN (Spiking Neural Network) language model** (874M params). All neurons are PLIF-based — no standard ANN components (no Transformer attention, no conventional MLP). Training infrastructure is aligned with the [happy-llm](https://github.com/datawhalechina/happy-llm) tutorial; the SNN architecture itself (`model.py` + `atomic_ops/`) is original.

**License**: CC BY-NC-SA 4.0 (non-commercial).

## Environment

```bash
conda activate SNN          # Python 3.10
export TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas  # Required on DGX Spark / Blackwell GPU
```

Dependencies: `torch`, `spikingjelly`, `triton`, `transformers`, `tokenizers`, `pandas`, `numpy`, `tqdm`, `modelscope`, `huggingface_hub`.

## Common Commands

### Data Pipeline
```bash
bash scripts/download_dataset.sh                    # Download pretrain + SFT data
python scripts/deal_dataset.py                       # Preprocess (--pretrain_only / --sft_only)
python scripts/train_tokenizer.py --data_path data/seq-monkey/seq_monkey_datawhale.jsonl --vocab_size 6144
```

### Training
```bash
# Pretrain (single GPU)
TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas \
python -u train.py --D 1024 --D_ff 3072 --num_layers 20 --batch_size 2 --accumulation_steps 32

# Pretrain (multi-GPU DDP)
TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas \
torchrun --nproc_per_node=4 train_ddp.py --D 1024 --D_ff 3072 --batch_size 2 --accumulation_steps 8

# Resume from checkpoint
python train.py --resume checkpoints/ckpt_step5000.pth

# SFT (single GPU)
python sft.py --pretrained_ckpt checkpoints/ckpt_step85000.pth --sft_data_path data/sft/sft_data.jsonl \
    --D 1024 --D_ff 3072 --batch_size 4 --learning_rate 5e-5 --epochs 3

# SFT (multi-GPU DDP)
torchrun --nproc_per_node=4 sft_ddp.py --pretrained_ckpt checkpoints/ckpt_step85000.pth \
    --sft_data_path data/sft/sft_data.jsonl --D 1024 --D_ff 3072
```

### Inference
```bash
# Text continuation (pretrain model)
python generate_sample.py --checkpoint checkpoints/ckpt_step85000.pth --mode pretrain --prompt "人工智能的发展"

# Dialogue (SFT model)
python generate_sample.py --checkpoint checkpoints_sft/ckpt_step6500.pth --mode sft --prompt "什么是脉冲神经网络？"

# Interactive mode
python generate_sample.py --checkpoint checkpoints_sft/ckpt_step6500.pth --interactive
```

### Experiments & Diagnostics
Scripts in `exp/`: `verify_*.py` (correctness), `bench_*.py` (performance), `diagnose_*.py` (training diagnostics).

## Architecture

### Three-Stage Forward Pass (`model.py: SNNLanguageModel`)

```
token_ids → Embedding(D=1024) → repeat K=32 frames
  → 20 × SNNDecoderLayer (with gradient checkpointing):
      RMSNorm → PLIFNode → spike_current → SNNBlock → PonderNet K-aggregation → out_proj → residual
      RMSNorm → PLIFNode → spike_current → SNNFFN  → PonderNet K-aggregation → out_proj → residual
  → RMSNorm → PLIFNode → spike_current → K-frame mean → decode_proj → LateralInhibition → Embedding^T (tied weights) → logits
```

- **`model.encode()`**: Embeds tokens and repeats K times → `(seq_len*K, batch, D)`
- **`model.snn_forward()`**: Runs through 20 decoder layers, returns `(h_out, ponder_cost)`
- **`model.decode()`**: Output neuron spike_current + K-frame aggregation → logits `(batch, seq_len, vocab_size)`

### Core SNN Components (`atomic_ops/`)

**SNNBlock** (`snn_block.py`) — Attention-equivalent layer:
- 7 parallel projections: W_in, W_β, W_α, W_th, W_gate, W_skip, W_out
- SelectivePLIFNode with input-dependent dynamic parameters β(t), α(t), V_th(t)
- Gate (sigmoid on spike) and skip (residual) paths

**SNNFFN** (`snn_ffn.py`) — Feed-forward equivalent:
- Three branches: gate_proj, up_proj, skip_proj
- SwiGLU-style gating: `sc_gate * sc_up` (spike current V_th×spike replaces SiLU)
- down_proj reduction + skip

**PLIFNode** (`plif_node.py`) — Fixed-parameter PLIF neuron (D-dimensional, learnable β and V_th per dim):
- `V_pre = β·V_post_prev + (1-β)·x`, spike = Θ(V_pre - V_th), `V_post = V_pre - V_th·spike`

**SelectivePLIFNode** (`selective_plif.py`) — Dynamic-parameter neuron (no learnable params, modulated by SNNBlock):
- Receives external β(t)=σ(W_β·x), α(t)=softplus(W_α·x), V_th(t)=|W_th·x|+min
- `V = β(t)·V_prev + α(t)·I`

**SNNDecoderLayer** (`snn_decoder_layer.py`):
- Pre-LN RMSNorm + PLIF input neuron + SNNBlock/SNNFFN + PonderNet dynamic-K aggregation + residual

### Parallel Scan (Triton Kernels — `parallel_scan.py`)

- **Fused PLIF kernel**: Single-pass forward (scan + spike + soft-reset), reverse accumulation backward
- **Row-param kernel**: β/V_th as row parameters (batch-wise, memory efficient)
- **Linear recurrence**: General V[k] = A[k]·V[k-1] + B[k]
- Surrogate gradient: sigmoid-based `∂spike/∂V_pre`

### PonderNet Dynamic-K

Each decoder sub-layer learns a halt probability via `halt_proj(D→1)`. Geometric distribution weights `λ_k = p_k · ∏(1-p_j)` aggregate K frames — different tokens use different effective step counts. `ponder_cost` regularizes toward early stopping.

### Key Design Principle: Spike Current + Continuous Residual Flow

Neurons output **spike current** (V_th × spike, sparse), which is projected and accumulated into a continuous residual stream. Layers transmit continuous **h** (residual), not binary spikes or raw membrane potentials. This solves deep-layer gradient vanishing while maintaining SNN spike semantics.

## Model Configuration (Current V8.0Pre)

| Param | Value | Notes |
|-------|-------|-------|
| D | 1024 | Hidden dimension |
| N | 8 | State expansion (D×N hidden neurons) |
| K | 32 | Max SNN timesteps per token (PonderNet dynamic) |
| num_layers | 20 | Decoder layers |
| D_ff | 3072 | FFN intermediate (3×D) |
| vocab_size | 6144 | BPE tokenizer |
| seq_max_length | 512 | Sequence length |
| Parameters | 874M | Total |

## Code Conventions

- **Language**: Code comments and docstrings are in Chinese (中文). Follow this convention.
- **Framework**: SpikingJelly (`spikingjelly.activation_based`) for neuron primitives and surrogate gradients.
- **Precision**: bfloat16 mixed precision throughout (`torch.autocast`).
- **Neuron LR**: Neuron parameters (β, V_th) use 10× learning rate multiplier (`neuron_lr_mult`).
- **Checkpoint management**: Auto-saves every `save_interval` steps as `ckpt_step{N}.pth`, keeps latest 5.
- **Gradient checkpointing**: All 20 layers use `torch.utils.checkpoint.checkpoint` to reduce memory.
- **No test suite**: Project has no automated tests. Correctness is validated via `exp/verify_*.py` scripts.
