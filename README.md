# NeuronSpark: Selective Spiking State Space Models with Adaptive Computation for Language Modeling

> **Anonymous submission to NeurIPS 2026.** All identifying information has been removed for double-blind review.

This repository contains the source code, tokenizer, and a fine-tuned (SFT) checkpoint
for **NeuronSpark-1.1B**, a 24-layer, 1.16B-parameter self-attention-free language model
built from input-dependent spiking membrane dynamics (SelectivePLIF), gated associative
memory, and adaptive temporal aggregation (PonderNet-style early-exit over short internal
processing steps per token). The reported pretrain run uses 2.1B tokens.

## Repository layout

```
.
├── neuronsparkcheckpoint/        # Pretrained + SFT checkpoint, ready for HuggingFace AutoModel.
│                                 # 2.32 GB safetensors (Git LFS), tokenizer (vocab 64002),
│                                 # ChatML chat_template, modeling code with trust_remote_code.
├── neuronspark/                  # HuggingFace-compatible model classes
│   ├── configuration_neuronspark.py
│   └── modeling_neuronspark.py
├── atomic_ops/                   # Core architectural building blocks
│   ├── selective_plif.py         #   SelectivePLIF neuron (input-dependent β/α/V_th)
│   ├── snn_associative_memory.py #   Gated associative memory (linear-attention style)
│   ├── snn_block.py              #   SNNBlock (selective spiking state-space layer)
│   ├── snn_attention_decoder_layer.py  # Decoder layer with PonderNet adaptive K
│   ├── snn_decoder_layer.py
│   ├── plif_node.py / parallel_scan.py / rms_norm.py / lateral_inhibition.py / snn_base.py / snn_ffn.py
├── tokenizer/                    # Tokenizer (vocab 64002) used by the released checkpoint
├── model.py                      # Native (non-HF) model definition + native checkpoint loader
├── checkpoint_utils.py           # Native checkpoint utilities (used by training scripts)
├── dataset.py / nsdata/          # Pretraining + SFT data loading (parquet / jsonl / arrow)
├── train_ddp.py                  # PyTorch DDP pretraining (4090 / single-node)
├── train_fsdp.py                 # PyTorch FSDP pretraining (multi-node H100)
├── train_ds.py / train_ds_neuron.py  # DeepSpeed ZeRO-{1,2,3} pretraining
├── sft_ddp.py / sft_ds.py / sft_ds_neuron.py  # SFT entry points (DDP / DeepSpeed)
├── rl_train.py                   # RLVR (GRPO) training entry
├── generate_sample.py            # Inference / sampling demo for the released checkpoint
├── eval_classification.py        # Classification probe evaluation
├── scripts/
│   ├── eval_full.py              # lm-evaluation-harness driver (with NeuronSpark LM adapter)
│   ├── bench_baselines.py / bench_baselines_extended.py / bench_baselines_phaseA.py
│   ├── build_pretrain_mix.py     # Reproduce the 2.1B-token pretraining mixture
│   ├── build_sft_mix.py          # Reproduce the benchmark-derived SFT mixture
│   ├── build_sft_v2.py           # Reproduce the SFT v2 mixture used by the released ckpt
│   ├── build_benchmark_sft_mix.py / build_knowledge_sft.py / build_rl_domain_data.py
│   ├── prepare_data.py / filter_sft_v2.py / resize_embedding.py
│   ├── convert_to_hf.py          # Native → HuggingFace format converter
│   ├── train_tokenizer.py        # Reproduce the BPE tokenizer
│   └── ...                       # (smoke / dryrun / equivalence tests)
├── ds_config.json                # DeepSpeed ZeRO-2 config used in the reported run
├── docs/                         # Architecture notes referenced by the paper
└── LICENSE
```

## Quick start: load the released checkpoint

The `neuronsparkcheckpoint/` directory is a self-contained HuggingFace artifact. Loading
requires `trust_remote_code=True` because the architecture is custom.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

ckpt = "./neuronsparkcheckpoint"
tokenizer = AutoTokenizer.from_pretrained(ckpt)
model = AutoModelForCausalLM.from_pretrained(
    ckpt, trust_remote_code=True, dtype=torch.bfloat16,
).cuda().eval()

# Chat-style inference (the released ckpt is SFT-tuned)
messages = [{"role": "user", "content": "What is the capital of France?"}]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
)
input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
with torch.no_grad():
    out = model.generate(
        input_ids,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
    )
print(tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True))
```

A reference inference script with multiple sampling configurations and both raw / chat
modes is provided in `generate_sample.py`.

### Files inside `neuronsparkcheckpoint/`

| File | Purpose |
|------|---------|
| `model.safetensors`            | bf16 weights (2.32 GB; tracked via Git LFS) |
| `config.json`                  | Model hyperparameters (D=1024, 24 layers, K_max=12, vocab=64002, ~1.16B params) |
| `generation_config.json`       | Default generation parameters |
| `configuration_neuronspark.py` | HuggingFace `PretrainedConfig` subclass |
| `modeling_neuronspark.py`      | HuggingFace `PreTrainedModel` + `GenerationMixin` implementation, including the inference cache (per-token `conv_state` ring buffer + RoPE `pos_offset`) used in our system measurements |
| `tokenizer.json` / `tokenizer_config.json` | Tokenizer (vocab=64002) |
| `chat_template.jinja`          | ChatML-compatible chat template |

## Reproducing the pretraining and SFT runs

### Environment

* Python 3.10
* PyTorch 2.4+ with CUDA
* `transformers>=4.40`, `datasets>=2.16`, `deepspeed>=0.14`, `huggingface_hub`,
  `lm-eval==0.4.x`, `safetensors`, `tokenizers`, `numpy`, `tqdm`, `tensorboard`.

### Data

The full pretraining and SFT pipelines (curation, deduplication, packing into
fixed-length shards, length-uniform bucket selection, ChatML masking) are reproducible
end-to-end from public sources via `scripts/build_pretrain_mix.py`,
`scripts/build_sft_mix.py`, and `scripts/build_sft_v2.py`. See each script's docstring
for the upstream HuggingFace dataset IDs and curation parameters.

### Pretraining

```bash
# DeepSpeed ZeRO-2, 8 GPUs (matches the reported run)
deepspeed --num_gpus=8 train_ds.py \
  --deepspeed_config ds_config.json \
  --data data/pretrain_mix \
  --tokenizer_path tokenizer/ \
  --out_dir runs/pretrain
```

`train_ddp.py` and `train_fsdp.py` provide single-node DDP and multi-node FSDP entry
points using the same model code. Architectural ablations referenced in the paper
(removing selective modulation / continuous health control / E[K] floor / etc.) are
controlled by command-line flags documented in those scripts.

### SFT

```bash
deepspeed --num_gpus=8 sft_ds.py \
  --deepspeed_config ds_config.json \
  --data data/sft_v2_mix \
  --tokenizer_path tokenizer/ \
  --pretrained_ckpt runs/pretrain/ckpts/ckpt_stepN \
  --out_dir runs/sft
```

The released `neuronsparkcheckpoint/` corresponds to the SFT v2 run, which fine-tunes
the pretrained 2.1B-token checkpoint on the mixture produced by `build_sft_v2.py`.

### Evaluation

```bash
# lm-evaluation-harness with the NeuronSpark adapter (registered as model='neuronspark')
python scripts/eval_full.py \
  --checkpoint ./neuronsparkcheckpoint \
  --num_gpus 1
```

The eight zero-shot tasks reported in the paper (ARC-E, ARC-C, WinoGrande, BoolQ, PIQA,
OpenBookQA, MMLU, C-Eval) are the default suite in `scripts/eval_full.py`. Per-task
JSONs and an aggregate summary are written to `exp/`.

## License

Released under the license in `LICENSE`. Author and affiliation information has been
withheld for double-blind review and will be added upon acceptance.
