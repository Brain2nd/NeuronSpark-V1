"""NeuronSpark v3 RL — REINFORCE with KL-anchor, HF-native.

Core loop:
  1. Load SFT checkpoint as policy + reference (policy trainable, reference frozen)
  2. For each prompt batch:
     a. Roll out with `policy.generate()` (no grad)
     b. Score with reward function (keyword match + quality; extensible)
     c. Compute log_prob(response | prompt) via `policy.snn` forward (with grad)
     d. Compute KL(policy || reference) on same tokens
     e. Loss = -advantage * log_prob + kl_coef * KL

Usage:
    python train_rl.py \
        --sft_ckpt checkpoints_v3_sft/ckpt_stepN/ \
        --data_path data/rl-domain/ \
        --tokenizer_path tokenizer_v3/ \
        --out_dir checkpoints_v3_rl/ \
        --kl_coef 0.05 --learning_rate 1e-5
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import random
import time
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from neuronspark import NeuronSparkForCausalLM
from utils.param_groups import build_param_groups, promote_neuron_params_fp32


# ============================================================
# Reward
# ============================================================

class RewardFunction:
    """R = R_keyword + R_quality.

    R_keyword: +1.0 if any kw hits, -0.5 if none, up to +1.5 for multi-hit.
    R_quality: length / repetition / topical-relevance heuristics, [-1.0, +0.2].
    """

    def compute(self, text: str, keywords: list[str], question: str = ""):
        detail = {"r_keyword": 0.0, "r_quality": 0.0}
        if not text or not text.strip():
            return -1.5, {"r_keyword": -0.5, "r_quality": -1.0}

        t = text.strip()
        lower = t.lower()
        hits = sum(1 for kw in keywords if kw.lower() in lower)
        if hits == 0:
            detail["r_keyword"] = -0.5
        elif hits == 1:
            detail["r_keyword"] = 1.0
        else:
            detail["r_keyword"] = min(1.0 + 0.2 * (hits - 1), 1.5)

        # Quality
        chars = len(t)
        q = 0.0
        if chars < 5:
            q -= 0.3
        elif chars > 500:
            q -= 0.1
        elif 10 <= chars <= 200:
            q += 0.1
        if chars >= 16:
            grams = [t[i:i+4] for i in range(chars - 3)]
            ur = len(set(grams)) / len(grams)
            if ur < 0.3:
                q -= 0.5
            elif ur < 0.5:
                q -= 0.3
            elif ur < 0.7:
                q -= 0.1
        if question:
            q_chars = set(question)
            overlap = sum(1 for c in t if c in q_chars and '\u4e00' <= c <= '\u9fff')
            if overlap > 3:
                q += 0.1
        detail["r_quality"] = max(q, -1.0)
        return sum(detail.values()), detail


# ============================================================
# Dataset
# ============================================================

class RLDomainDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_prompt_length: int = 512):
        from datasets import load_from_disk, DatasetDict
        ds = load_from_disk(data_path)
        if isinstance(ds, DatasetDict):
            ds = ds[list(ds.keys())[0]]
        self.data = []
        skipped = 0
        for row in ds:
            kws = row.get("keywords") or []
            if not kws:
                skipped += 1
                continue
            msgs = row["prompt_messages"]
            prompt = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )
            pid_len = len(tokenizer.encode(prompt))
            if pid_len > max_prompt_length:
                skipped += 1
                continue
            question = next((m["content"] for m in msgs if m["role"] == "user"), "")
            self.data.append({
                "prompt": prompt,
                "keywords": kws,
                "domain": row.get("domain", ""),
                "question": question,
            })
        print(f"  RL data: {len(self.data)} valid / {len(ds)} total (skipped {skipped})")
        for dom, n in Counter(d["domain"] for d in self.data).most_common():
            print(f"    {dom:<15} {n}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate(batch):
    return batch  # list of dicts


# ============================================================
# Log-prob / KL helpers
# ============================================================

def response_log_prob(model, input_ids: torch.Tensor, response_start: int, device):
    """avg log p_model(response | prompt).

    Uses `model.snn(prev, next)` to get per-token CE loss, then takes -CE[response].
    """
    snn = model.snn if hasattr(model, "snn") else model
    seq_len = input_ids.shape[1]
    if seq_len < 2 or response_start >= seq_len:
        return torch.tensor(0.0, device=device, requires_grad=True)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = snn(input_ids[:, :-1], input_ids[:, 1:])
    loss_start = max(response_start - 1, 0)
    resp_ce = out.last_loss[loss_start:]
    if resp_ce.numel() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return -resp_ce.mean()


@torch.no_grad()
def ref_log_prob(ref_model, input_ids: torch.Tensor, response_start: int, device):
    return response_log_prob(ref_model, input_ids, response_start, device).detach()


# ============================================================
# RL step
# ============================================================

def rl_step(policy, reference, tokenizer, batch, device, reward_fn,
            max_gen_tokens: int, temperature: float, kl_coef: float,
            eos_id: int):
    total_r = 0.0
    hits = 0
    n = 0
    rk_sum = rq_sum = 0.0
    pairs = []  # (policy_log_prob, ref_log_prob, reward)

    for item in batch:
        prompt_ids = tokenizer(item["prompt"], return_tensors="pt").input_ids.to(device)
        prompt_len = prompt_ids.shape[1]

        policy.eval()
        with torch.no_grad():
            gen_ids = policy.generate(
                prompt_ids,
                max_new_tokens=max_gen_tokens,
                temperature=temperature,
                top_k=50, top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=eos_id,
            )
        policy.train()

        resp_ids = gen_ids[0, prompt_len:]
        if resp_ids.numel() == 0:
            n += 1
            total_r -= 1.5
            continue
        resp_text = tokenizer.decode(resp_ids, skip_special_tokens=True)

        reward, detail = reward_fn.compute(resp_text, item["keywords"], item.get("question", ""))
        total_r += reward
        n += 1
        rk_sum += detail["r_keyword"]
        rq_sum += detail["r_quality"]
        if detail["r_keyword"] > 0.5:
            hits += 1

        full = gen_ids[:, :prompt_len + resp_ids.numel()]
        lp = response_log_prob(policy, full, prompt_len, device)
        ref_lp = ref_log_prob(reference, full, prompt_len, device)
        pairs.append((lp, ref_lp, reward))

    if not pairs:
        return None, {"reward": 0.0, "hit_rate": 0.0, "kl": 0.0}

    mean_r = sum(p[2] for p in pairs) / len(pairs)
    loss_terms = []
    kl_sum = 0.0
    for lp, ref_lp, r in pairs:
        adv = r - mean_r
        # REINFORCE objective + KL penalty: -adv*lp + kl_coef * (lp - ref_lp)
        kl_term = (lp - ref_lp).detach() * lp if kl_coef > 0 else torch.zeros([], device=device)
        loss_terms.append(-adv * lp + kl_coef * kl_term)
        kl_sum += float(lp.detach() - ref_lp)
    loss = torch.stack([t if t.ndim == 0 else t.squeeze() for t in loss_terms]).mean()

    stats = {
        "reward": total_r / n,
        "hit_rate": hits / n,
        "r_keyword": rk_sum / n,
        "r_quality": rq_sum / n,
        "kl": kl_sum / len(pairs),
        "mean_log_prob": sum(float(lp.detach()) for lp, _, _ in pairs) / len(pairs),
    }
    return loss, stats


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft_ckpt", required=True,
                    help="HF SFT checkpoint to initialize policy + reference")
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--tokenizer_path", default="tokenizer_v3/")
    ap.add_argument("--out_dir", default="checkpoints_v3_rl/")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_gen_tokens", type=int, default=128)
    ap.add_argument("--max_prompt_length", type=int, default=512)
    ap.add_argument("--learning_rate", type=float, default=1e-5)
    ap.add_argument("--neuron_lr_mult", type=float, default=5.0)
    ap.add_argument("--kl_coef", type=float, default=0.05)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--total_steps", type=int, default=2000)
    ap.add_argument("--warmup_steps", type=int, default=50)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--save_interval", type=int, default=200)
    ap.add_argument("--eval_interval", type=int, default=100)
    ap.add_argument("--log_interval", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading policy from {args.sft_ckpt}")
    policy = NeuronSparkForCausalLM.from_pretrained(args.sft_ckpt, dtype=torch.bfloat16, trust_remote_code=True).to(device)
    print("Cloning as reference (frozen)")
    reference = NeuronSparkForCausalLM.from_pretrained(args.sft_ckpt, dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()
    for p in reference.parameters():
        p.requires_grad_(False)
    promote_neuron_params_fp32(policy)

    param_groups = build_param_groups(
        policy, learning_rate=args.learning_rate, neuron_lr_mult=args.neuron_lr_mult,
    )
    optimizer = torch.optim.Adam(param_groups)

    ds = RLDomainDataset(args.data_path, tokenizer, max_prompt_length=args.max_prompt_length)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    reward_fn = RewardFunction()
    im_end = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]

    print(f"\n{'='*60}")
    print(f"NeuronSpark v3 RL (REINFORCE + KL-anchor)")
    print(f"  SFT ckpt       : {args.sft_ckpt}")
    print(f"  Data           : {args.data_path}  ({len(ds)} samples)")
    print(f"  Batch / gen    : {args.batch_size} × {args.max_gen_tokens} new tokens")
    print(f"  LR             : {args.learning_rate} (neuron × {args.neuron_lr_mult})")
    print(f"  KL coef        : {args.kl_coef}")
    print(f"  Total steps    : {args.total_steps}")
    print(f"{'='*60}\n")

    policy.train()
    step = 0
    t0 = time.time()
    for epoch in range(10**9):
        for batch in loader:
            if step >= args.total_steps:
                break
            # LR schedule: warmup → constant (RL rarely benefits from cosine)
            lr_factor = min(step / max(args.warmup_steps, 1), 1.0)
            for g in optimizer.param_groups:
                g["lr"] = args.learning_rate * g.get("lr_mult", 1.0) * lr_factor

            loss, stats = rl_step(
                policy, reference, tokenizer, batch, device, reward_fn,
                max_gen_tokens=args.max_gen_tokens, temperature=args.temperature,
                kl_coef=args.kl_coef, eos_id=im_end,
            )
            if loss is None:
                step += 1
                continue

            optimizer.zero_grad()
            loss.backward()
            # SNN-specific gradient compensation
            policy.snn.compensate_modulation_gradients()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            optimizer.step()

            if step % args.log_interval == 0:
                elapsed = time.time() - t0
                print(
                    f"step {step}/{args.total_steps}  loss {loss.item():+.3f}  "
                    f"reward {stats['reward']:+.3f}  hit {stats['hit_rate']*100:.1f}%  "
                    f"kl {stats['kl']:+.3f}  lr {optimizer.param_groups[0]['lr']:.2e}  "
                    f"elapsed {elapsed:.0f}s"
                )

            if step > 0 and step % args.save_interval == 0:
                save_dir = os.path.join(args.out_dir, f"ckpt_step{step}")
                policy.save_pretrained(save_dir, safe_serialization=True)
                torch.save({"step": step}, os.path.join(save_dir, "training_state.pth"))
                print(f"  → saved {save_dir}")

            step += 1
        if step >= args.total_steps:
            break

    final_dir = os.path.join(args.out_dir, "final")
    policy.save_pretrained(final_dir, safe_serialization=True)
    print(f"RL done. Final saved to {final_dir}")


if __name__ == "__main__":
    main()
