"""NeuronSpark v3 inference: load HF checkpoint + model.generate().

Supports:
  - Interactive chat (ChatML template)
  - Single-prompt batch completion
  - Pretrain-style continuation (no chat template)

Usage:
    # Pretrain-style
    python generate.py --checkpoint checkpoints_v3/ckpt_stepN/ --mode pretrain \
        --prompt "Once upon a time"

    # Chat-style (SFT)
    python generate.py --checkpoint checkpoints_v3_sft/ckpt_stepN/ --mode chat \
        --prompt "Explain attention in one sentence."

    # Interactive
    python generate.py --checkpoint ... --mode interactive
"""
from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load(ckpt_path: str, tokenizer_path: str | None = None, device: str = "cuda"):
    tok_src = tokenizer_path or ckpt_path
    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path, dtype=torch.float32, trust_remote_code=True,
    )
    model = model.to(device).eval()
    return model, tokenizer


def generate_pretrain(model, tokenizer, prompt, max_new_tokens=256,
                       temperature=0.8, top_k=50, top_p=0.95, repetition_penalty=1.1):
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    out = model.generate(
        input_ids=ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k, top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    new_ids = out[0, ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def generate_chat(model, tokenizer, user_msg, system_msg=None, history=None,
                   max_new_tokens=512, temperature=0.8, top_k=50, top_p=0.95,
                   repetition_penalty=1.1):
    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_msg})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    im_end_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    eos_id = im_end_id[0] if im_end_id else tokenizer.eos_token_id
    out = model.generate(
        input_ids=ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k, top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_token_id=eos_id,
    )
    new_ids = out[0, ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def interactive(model, tokenizer, args):
    print("Interactive chat (type 'exit' to quit, 'reset' to clear history).")
    history = []
    while True:
        user = input("user> ").strip()
        if user.lower() in {"exit", "quit"}:
            break
        if user.lower() == "reset":
            history = []
            print("(history cleared)")
            continue
        if not user:
            continue
        reply = generate_chat(
            model, tokenizer, user, system_msg=args.system,
            history=history, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        history.append({"role": "user", "content": user})
        history.append({"role": "assistant", "content": reply})
        print(f"assistant> {reply}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="HF checkpoint directory")
    ap.add_argument("--tokenizer_path", default=None,
                    help="tokenizer directory (default: use --checkpoint)")
    ap.add_argument("--mode", choices=["pretrain", "chat", "interactive"], default="chat")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--system", default=None, help="system prompt for chat mode")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--repetition_penalty", type=float, default=1.1)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model, tokenizer = load(args.checkpoint, args.tokenizer_path, args.device)
    print(f"Loaded model with {sum(p.numel() for p in model.parameters())/1e6:.1f} M params")

    if args.mode == "interactive":
        interactive(model, tokenizer, args)
        return

    if not args.prompt:
        raise SystemExit("--prompt required for pretrain/chat mode")

    if args.mode == "pretrain":
        out = generate_pretrain(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        print(f"[pretrain] {args.prompt!r} → {out!r}")
    else:  # chat
        out = generate_chat(
            model, tokenizer, args.prompt, system_msg=args.system,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        print(f"[chat] user: {args.prompt}\nassistant: {out}")


if __name__ == "__main__":
    main()
