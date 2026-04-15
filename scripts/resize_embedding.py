"""Resize model embedding from 64000 to 64002 for thinking tokens.

同时清理旧字段、补全缺失字段。
"""
import torch
from safetensors.torch import load_file, save_file
import json
import sys

ckpt_dir = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/ckpt_step224000"

state = load_file(ckpt_dir + "/model.safetensors", device="cpu")
old_emb = state["embed_tokens.weight"]
print("Old:", old_emb.shape)

if old_emb.shape[0] >= 64002:
    print("Already 64002+, skip embedding resize")
else:
    D = old_emb.shape[1]
    new_rows = torch.randn(2, D, dtype=old_emb.dtype) * 0.02
    state["embed_tokens.weight"] = torch.cat([old_emb, new_rows], dim=0)
    print("New:", state["embed_tokens.weight"].shape)
    save_file(state, ckpt_dir + "/model.safetensors")

with open(ckpt_dir + "/config.json") as f:
    config = json.load(f)
config["vocab_size"] = 64002
# 清理废弃字段
config.pop("activation_mode", None)
# 补全缺失字段
config.setdefault("v_th_min", 0.1)
config.setdefault("memory_layer_interval", 4)
config.setdefault("D_key", 128)
config.setdefault("D_value", 128)
with open(ckpt_dir + "/config.json", "w") as f:
    json.dump(config, f, indent=2)
print("Done: vocab_size=64002")
