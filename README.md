# NeuronSpark — SNN Hidden State Space Language Model

A language model **built entirely on Spiking Neural Networks (SNNs)**. Hidden neurons with dynamic parameters β(t), α(t), V_th(t) serve as input-dependent modulation signals for selective information filtering. **The entire network is pure SNN — no standard ANN components.**

> **Language**: Currently supports **Chinese only**, as the model was trained on Chinese corpora (Seq-Monkey + BelleGroup).

> **Training data note**: Due to limited compute resources (single DGX Spark), both pretraining and SFT used only small subsets of their respective datasets (~1.4B of ~10B tokens for pretraining; ~6.5K steps for SFT). Despite this minimal data budget, the model demonstrates emergent language capabilities, validating the architectural viability of pure SNN language models. We plan to continue scaling with more data and compute.

## Model Downloads

| Model | HuggingFace | ModelScope | Description |
|-------|-------------|------------|-------------|
| NeuronSpark-0.9B | [Brain2nd/NeuronSpark-0.9B](https://huggingface.co/Brain2nd/NeuronSpark-0.9B) | [Brain2nd/NeuronSpark-0.9B](https://www.modelscope.ai/models/Brain2nd/NeuronSpark-0.9B) | Pretrained 85K steps |
| NeuronSpark-0.9B-Chat | [Brain2nd/NeuronSpark-0.9B-Chat](https://huggingface.co/Brain2nd/NeuronSpark-0.9B-Chat) | [Brain2nd/NeuronSpark-0.9B-Chat](https://www.modelscope.ai/models/Brain2nd/NeuronSpark-0.9B-Chat) | SFT chat version |

### Quick Inference (HuggingFace)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Brain2nd/NeuronSpark-0.9B-Chat", trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("Brain2nd/NeuronSpark-0.9B-Chat")

messages = [
    {"role": "system", "content": "你是一个AI助手"},
    {"role": "user", "content": "中国的首都是哪里？"},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
output_ids = model.generate(input_ids, max_new_tokens=256, temperature=0.1, top_k=10,
                            eos_token_id=tokenizer.eos_token_id)
response = tokenizer.decode(output_ids[0], skip_special_tokens=False)
print(response.split("assistant\n")[-1].replace("<|im_end|>", "").strip())
# Output: 中国的首都在北京。
```


## Architecture

### Overview

NeuronSpark processes each token as K=16 SNN temporal frames. The model has 3 stages:

```
                          ┌─────────────────────────────────────────────────────────┐
 token_ids ──► Embedding ─┤  repeat K=16 times along temporal dim                   │
                          │  (batch, seq_len, D) → (seq_len*K, batch, D)            │
                          └──────────────────────────┬──────────────────────────────┘
                                                     │
                          ┌──────────────────────────▼──────────────────────────────┐
                          │  L=20 × SNNDecoderLayer (continuous residual stream h)  │
                          │                                                         │
                          │  Sublayer 1 — SNNBlock (attention analogue):             │
                          │    h → RMSNorm → PLIFNode → SNNBlock → PonderNet        │
                          │      → out_proj → residual add back to h                │
                          │                                                         │
                          │  Sublayer 2 — SNNFFN (MLP analogue):                    │
                          │    h → RMSNorm → PLIFNode → SNNFFN → PonderNet          │
                          │      → out_proj → residual add back to h                │
                          └──────────────────────────┬──────────────────────────────┘
                                                     │
                          ┌──────────────────────────▼──────────────────────────────┐
                          │  Output PLIFNode → K-frame mean → decode_proj            │
                          │  → LateralInhibition → tied Embedding head → logits      │
                          └─────────────────────────────────────────────────────────┘
```

### PLIF Neuron — The Basic Building Block

Every neuron in NeuronSpark is a **Parametric Leaky Integrate-and-Fire (PLIF)** neuron:

```
V_pre[t]  = β · V_post[t-1] + (1-β) · x[t]      # charge: exponential decay + input
spike[t]  = Θ(V_pre[t] - V_th)                     # fire: threshold comparison
V_post[t] = V_pre[t] - V_th · spike[t]             # soft reset: subtract threshold
```

- **β = sigmoid(w)**: per-dimension learnable decay rate (how fast the neuron "forgets")
- **V_th**: per-dimension learnable firing threshold
- Training uses **surrogate gradient** (Sigmoid) to backprop through the non-differentiable Θ

**Leakage Activation**: Inter-layer signals use `(1-β) · V_post` — the amount of membrane potential that will leak away. This naturally emphasizes fast-responding neurons (large 1-β) and attenuates slow-memory neurons (small 1-β).

### SNNBlock — Selective State Space Block (Attention Analogue)

The SNNBlock replaces Transformer's self-attention. It contains D×N=896×8=7168 hidden spiking neurons with **input-dependent dynamic parameters** — making it a selective state space model:

```
                    ┌──► W_in ──────────► I[t] (input current, D*N dim)
                    │
                    ├──► W_β + b_β ─► sigmoid ──► β(t)   (decay rate, controls memory)
                    │
 leak_input ────────├──► W_α + b_α ─► softplus ─► α(t)   (input gain, controls writing)
 (D dim)            │
                    ├──► W_th + b_th ► |·|+V_min ► V_th(t) (threshold, controls firing)
                    │
                    ├──► W_gate ────► sigmoid ──► gate    (output gating, D dim)
                    │
                    └──► W_skip ────────────────► I_skip  (skip connection, D dim)

                              ▼
              SelectivePLIF Hidden Neurons (D*N dim):
              V[t] = β(t)·V[t-1] + α(t)·I[t]   ← selective state space recurrence
              spike[t] = Θ(V[t] - V_th(t))       ← spike with dynamic threshold
              V[t] -= V_th(t) · spike[t]          ← soft reset

                              ▼
              Output: W_out · V_post ⊙ gate + I_skip   (D dim)
```

**SNN–SSM Duality**: The recurrence `V[t] = β(t)·V[t-1] + α(t)·I[t]` is structurally identical to Mamba's selective SSM `h[t] = A̅(t)·h[t-1] + B̅(t)·x[t]`, with β mapping to A̅ and α to B̅. The key difference is the spike-and-reset mechanism that introduces discrete nonlinearity.

### SNNFFN — SNN Feed-Forward Network (MLP Analogue)

Replaces the standard SwiGLU MLP with spiking neurons:

```
                    ┌──► gate_proj → PLIF gate_neuron → leak_gate ──┐
 leak_input ────────┤                                                ├──► × (element-wise)
                    └──► up_proj   → PLIF up_neuron   → leak_up   ──┘         │
                                                                               ▼
                                                        down_proj(gated) + skip_proj(input) → output
```

The element-wise product of two leakage signals replaces SiLU(x)⊙x gating in SwiGLU. The PLIF dynamics provide implicit nonlinearity through integrate-fire-reset.

### PonderNet Adaptive Timesteps

Each token is represented as K=16 SNN frames, but not all tokens need all 16 steps. PonderNet learns per-frame halt probabilities:

```
For each frame k = 1, ..., K:
  p_k = sigmoid(halt_proj(frame_k))          # halt probability
  S_k = ∏(1 - p_j) for j < k                # survival probability
  λ_k = p_k · S_k                            # geometric distribution weight
  λ̂_k = λ_k / Σ λ_k                          # normalize to sum=1

output = Σ λ̂_k · frame_k                     # weighted aggregation
E[K]   = Σ k · λ̂_k                           # expected steps (ponder cost)
```

Simple tokens halt early (E[K] ≈ 2-3), complex tokens use more steps (E[K] ≈ 10-15). The ponder cost E[K] is regularized to encourage efficiency.

### Each Decoder Layer (Pre-LN Pattern)

Matches Qwen3/LLaMA's Pre-LN structure:

```
h ──────────────────────────────────────────────────────── + ──► h  (sublayer 1)
 └─► RMSNorm → PLIFNode(leak) → SNNBlock → PonderNet → out_proj ─┘

h ──────────────────────────────────────────────────────── + ──► h  (sublayer 2)
 └─► RMSNorm → PLIFNode(leak) → SNNFFN   → PonderNet → out_proj ─┘
```

The residual stream `h` carries **continuous values** throughout. Only inside the SNN sublayers do spike/membrane dynamics operate. This solves the vanishing gradient problem in deep SNNs.

### Triton Fused PLIF Kernels

The PLIF sequential recurrence (with spike-and-reset) cannot be naively parallelized. We implement custom Triton kernels:

- **Fused forward**: Single kernel for charge → fire → soft reset across all K steps
- **Fused backward**: Single kernel with inline Sigmoid surrogate gradient
- **Row-parameter variant**: For PLIFNode (fixed β/V_th), loads parameters into registers once — 40% faster
- All computations in fp32 internally, bf16 for storage

## Acknowledgments

Training infrastructure (data processing, tokenizer training, pretrain/SFT pipeline) is **strictly aligned with the [happy-llm](https://github.com/datawhalechina/happy-llm) tutorial project** (Datawhale open-source community). We replaced only the model architecture with SNN while keeping the training pipeline identical.

### Training Infrastructure (from happy-llm)

| Component | happy-llm Source | NeuronSpark File | Alignment |
|-----------|-----------------|------------------|-----------|
| Data download | `download_dataset.sh` | `scripts/download_dataset.sh` | Fully aligned |
| Data preprocessing | `deal_dataset.py` | `scripts/deal_dataset.py` | Fully aligned |
| Tokenizer training | `train_tokenizer.py` | `scripts/train_tokenizer.py` | Fully aligned: BPE, vocab=6144 |
| Dataset loading | `dataset.py` | `dataset.py` | Fully aligned: byte-offset JSONL |
| Pretrain loop | `ddp_pretrain.py` | `train.py` / `train_ddp.py` | Highly aligned: Adam, Cosine LR, bf16 |
| SFT training | `ddp_sft_full.py` | `sft.py` / `sft_ddp.py` | Highly aligned |
| Inference | `model_sample.py` | `generate_sample.py` | Aligned |

### Original SNN Architecture

| Component | Description |
|-----------|-------------|
| `model.py` | SNNLanguageModel: encode/snn_forward/decode + autoregressive generate() |
| `atomic_ops/selective_plif.py` | SelectivePLIFNode: dynamic-parameter PLIF neuron |
| `atomic_ops/plif_node.py` | PLIFNode: D-dim fixed-parameter PLIF neuron |
| `atomic_ops/snn_block.py` | SNNBlock: SNN attention analogue (7 parallel paths + gating) |
| `atomic_ops/snn_ffn.py` | SNNFFN: SNN feed-forward (SwiGLU-style 3-branch spike gating) |
| `atomic_ops/snn_decoder_layer.py` | SNNDecoderLayer: Pre-LN + Block + FFN + PonderNet + residual |
| `atomic_ops/parallel_scan.py` | Triton Fused PLIF Kernel + Row-param Kernel |
| `atomic_ops/lateral_inhibition.py` | Triton lateral inhibition normalization (divisive normalization) |

## Configuration

| Parameter | Value |
|-----------|-------|
| Parameters | 874M |
| Hidden dim (D) | 896 |
| State expansion (N) | 8 |
| Max SNN steps (K) | 16 (PonderNet adaptive) |
| Layers | 20 |
| FFN dim (D_ff) | 2688 |
| Vocab size | 6144 |
| Context length | 512 |
| Training data | Seq-Monkey (small subset, ~1.4B / 10B tokens) |
| SFT data | BelleGroup (~6.5K steps / 3.5M samples) |
| Hardware | 1x NVIDIA DGX Spark (GB10, 128GB unified memory) |

## Getting Started

### Environment

```bash
conda create -n SNN python=3.10
conda activate SNN

pip install torch torchvision torchaudio
pip install spikingjelly transformers tokenizers pandas numpy tqdm safetensors
pip install modelscope huggingface_hub

# DGX Spark / Blackwell GPU: Triton ptxas config
export TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas
```

### Training

```bash
# Pretrain (single GPU)
python train.py --D 896 --D_ff 2688 --K 16 --num_layers 20 \
    --batch_size 2 --accumulation_steps 32 --warmup_iters 1000

# Pretrain (multi-GPU DDP)
torchrun --nproc_per_node=4 train_ddp.py \
    --D 896 --D_ff 2688 --K 16 --batch_size 8 --accumulation_steps 8

# SFT
python sft.py --pretrained_ckpt checkpoints/ckpt_step85000.pth \
    --sft_data_path data/sft/sft_data.jsonl \
    --D 896 --D_ff 2688 --learning_rate 5e-5 --epochs 3

# Resume training
python train.py --resume checkpoints/ckpt_step5000.pth
```

### Export to HuggingFace Format

```bash
python export_hf.py --ckpt checkpoints/ckpt_step85000.pth --output_dir NeuronSpark-Pretrain
python export_hf.py --ckpt checkpoints_sft/ckpt_step6500.pth --output_dir NeuronSpark-SFT
```

## Project Structure

```
NeuronSpark/
├── model.py                          # SNNLanguageModel (encode/snn_forward/decode + generate)
├── train.py / train_ddp.py           # Pretraining scripts (single/multi-GPU)
├── sft.py / sft_ddp.py               # SFT scripts (single/multi-GPU)
├── generate_sample.py                # Inference / text generation
├── dataset.py                        # PretrainDataset + SFTDataset
├── configuration_neuronspark.py      # HuggingFace config class
├── modeling_neuronspark.py           # HuggingFace model wrapper
├── export_hf.py                      # Export to HF format (safetensors)
├── atomic_ops/                       # Core SNN operators
│   ├── selective_plif.py             # SelectivePLIFNode
│   ├── plif_node.py                  # PLIFNode
│   ├── snn_block.py                  # SNNBlock (attention analogue)
│   ├── snn_ffn.py                    # SNNFFN (MLP analogue)
│   ├── snn_decoder_layer.py          # SNNDecoderLayer
│   ├── parallel_scan.py              # Triton Fused PLIF Kernels
│   ├── lateral_inhibition.py         # Triton lateral inhibition
│   └── rms_norm.py                   # RMSNorm
├── scripts/                          # Data processing (aligned with happy-llm)
├── tokenizer_snn/                    # Trained BPE tokenizer (6144 vocab)
├── paper/                            # Research paper (LaTeX)
└── docs/                             # Architecture design documents
```

## Citation

```bibtex
@misc{neuronspark2025,
    title={NeuronSpark: A Spiking Neural Network Language Model with Selective State Space Dynamics},
    author={Zhengzheng Tang},
    year={2025},
    url={https://github.com/Brain2nd/NeuronSpark}
}
```

## Contact

- **Author**: Zhengzheng Tang
- **Email**: zztangbu@bu.edu
- **GitHub**: [Brain2nd/NeuronSpark](https://github.com/Brain2nd/NeuronSpark)

## License

Apache License 2.0
