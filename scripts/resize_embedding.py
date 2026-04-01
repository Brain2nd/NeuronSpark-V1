"""Resize model embedding from 64000 to 64002 for thinking tokens."""
import torch
from safetensors.torch import load_file, save_file
import json
import sys

ckpt_dir = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/ckpt_step224000"

state = load_file(ckpt_dir + "/model.safetensors", device="cpu")
old_emb = state["embed_tokens.weight"]
print("Old:", old_emb.shape)

D = old_emb.shape[1]
new_rows = torch.randn(2, D, dtype=old_emb.dtype) * 0.02
state["embed_tokens.weight"] = torch.cat([old_emb, new_rows], dim=0)
print("New:", state["embed_tokens.weight"].shape)
save_file(state, ckpt_dir + "/model.safetensors")

with open(ckpt_dir + "/config.json") as f:
    config = json.load(f)
config["vocab_size"] = 64002
with open(ckpt_dir + "/config.json", "w") as f:
    json.dump(config, f, indent=2)
print("Done: vocab_size=64002")
