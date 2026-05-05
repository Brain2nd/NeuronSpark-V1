"""NeuronSpark v3 RLVR (GRPO) - DeepSeek-R1-Zero style.

直接从 pretrain ckpt 起步, rule-based reward (\\boxed 答案正误 + format).
TRL GRPOTrainer (HF-native). Group-normalized advantage + KL ref penalty.

Usage:
    deepspeed --num_gpus=8 train_rl_grpo.py \
        --pretrained_ckpt runs/.../ckpts/ckpt_step108000 \
        --data_path data/rl_gsm8k \
        --tokenizer_path tokenizer_v3 \
        --out_dir runs/rl_zero/ckpts \
        --num_generations 8 \
        --max_completion_length 1536 \
        --learning_rate 1e-6 \
        --deepspeed ds_config.json
"""
from __future__ import annotations
import argparse
import os
import re
import warnings

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from neuronspark import NeuronSparkForCausalLM
from utils.param_groups import promote_neuron_params_fp32

warnings.filterwarnings("ignore")

# ===== Reward functions =====

_BOX_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def extract_boxed(text: str) -> str | None:
    m = _BOX_RE.search(text)
    return m.group(1).strip() if m else None


def reward_outcome(prompts, completions, answer=None, **kwargs):
    """\\boxed{X} match GT = 1 else 0. 数值 / 字符串两种比对."""
    rewards = []
    for c, gt in zip(completions, answer):
        ext = extract_boxed(c)
        if ext is None:
            rewards.append(0.0); continue
        gt_clean = (gt or "").strip().replace(",", "")
        ext_clean = ext.strip().replace(",", "").rstrip(".")
        try:
            if float(ext_clean) == float(gt_clean):
                rewards.append(1.0); continue
        except (ValueError, TypeError):
            pass
        rewards.append(1.0 if ext_clean == gt_clean else 0.0)
    return rewards


def reward_format(prompts, completions, **kwargs):
    """think 标签闭合 + boxed 出现 = 0.1."""
    rewards = []
    for c in completions:
        if "<think>" in c and "</think>" in c and "\\boxed" in c:
            rewards.append(0.1)
        else:
            rewards.append(0.0)
    return rewards


# ===== Main =====

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_ckpt", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--tokenizer_path", default="tokenizer_v3")
    ap.add_argument("--out_dir", default="runs/rl_zero/ckpts")
    ap.add_argument("--dashboard_dir", default=None)
    ap.add_argument("--max_prompt_length", type=int, default=512)
    ap.add_argument("--max_completion_length", type=int, default=1536)
    ap.add_argument("--num_generations", type=int, default=8,
                    help="每 prompt 采 N 个 rollout (GRPO group)")
    ap.add_argument("--learning_rate", type=float, default=1e-6)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--num_train_epochs", type=int, default=1)
    ap.add_argument("--beta", type=float, default=0.04, help="KL ref penalty")
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--log_steps", type=int, default=10)
    ap.add_argument("--deepspeed", default=None)
    ap.add_argument("--local_rank", type=int, default=-1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading model from {args.pretrained_ckpt}", flush=True)
    model = NeuronSparkForCausalLM.from_pretrained(
        args.pretrained_ckpt, dtype=torch.bfloat16, trust_remote_code=True,
    )
    promote_neuron_params_fp32(model)

    ds = load_from_disk(args.data_path)
    print(f"Dataset: {len(ds)} samples from {args.data_path}", flush=True)

    config = GRPOConfig(
        output_dir=args.out_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        bf16=True,
        save_steps=args.save_steps,
        logging_steps=args.log_steps,
        beta=args.beta,
        deepspeed=args.deepspeed,
        report_to="tensorboard" if args.dashboard_dir else "none",
        logging_dir=args.dashboard_dir,
        save_safetensors=True,
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_outcome, reward_format],
        args=config,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    print("Starting RLVR (GRPO) training...", flush=True)
    trainer.train()
    print("DONE.", flush=True)


if __name__ == "__main__":
    main()
