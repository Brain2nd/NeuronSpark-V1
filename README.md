# NeuronSpark — SNN Hidden State Space Language Model

A language model **built entirely on Spiking Neural Networks (SNNs)**. Hidden neurons with dynamic parameters β(t), α(t), V_th(t) serve as input-dependent modulation signals for selective information filtering. **The entire network is pure SNN — no standard ANN components.**

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
    {"role": "system", "content": "You are an AI assistant"},
    {"role": "user", "content": "What is the capital of China?"},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
output_ids = model.generate(input_ids, max_new_tokens=256, temperature=0.1, top_k=10,
                            eos_token_id=tokenizer.eos_token_id)
response = tokenizer.decode(output_ids[0], skip_special_tokens=False)
print(response.split("assistant\n")[-1].replace("<|im_end|>", "").strip())
# Output: The capital of China is Beijing.
```


## Architecture

```
token -> Embedding(D=896) -> repeat K=16 frames
  -> L=20 x SNNDecoderLayer:
      RMSNorm(h) -> PLIFNode(leak) -> SNNBlock -> PonderNet K-agg -> out_proj -> residual
      RMSNorm(h) -> PLIFNode(leak) -> SNNFFN   -> PonderNet K-agg -> out_proj -> residual
  -> RMSNorm -> PLIFNode(leak) -> K-frame mean -> decode_proj -> LateralInhibition -> tied head -> logits
```

### Key Design Choices

- **Membrane Potential Leakage Activation**: PLIFNode outputs `(1-β)·V_post` (leak current), naturally emphasizing fast-responding neurons over slow-memory neurons
- **PonderNet Adaptive K**: Each sublayer learns per-frame halt probabilities with geometric distribution weighting; different tokens use different effective timesteps ∈ [1, K_max]
- **Selective State Space**: PLIF neurons with input-dependent dynamic β(t), α(t), V_th(t) — structurally identical to Mamba's selective SSM
- **Continuous Residual Stream**: Inter-layer signals are continuous values; only SNN sublayers operate on spike/membrane dynamics
- **Pre-LN RMSNorm**: Branch normalization controlling PLIFNode input scale + residual centering to eliminate DC drift
- **7 Parallel Input Paths**: W_in, W_β, W_α, W_th, W_gate, W_skip, W_out
- **Triton Fused PLIF Kernels**: Single kernel for scan + spike + soft reset + surrogate gradient
- **Natural Gradient Compensation**: Sigmoid/softplus saturation compensation + cross-layer gradient equalization
- **Training**: Surrogate Gradient + standard backpropagation via SpikingJelly

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
