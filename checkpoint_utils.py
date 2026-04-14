"""
Checkpoint 工具：HuggingFace safetensors 格式保存/加载

每个 checkpoint 是一个目录，包含：
  model.safetensors  — 模型权重（HF 标准格式）
  config.json        — 模型配置
  training_state.pth — 优化器/scaler/step/epoch 等训练状态（resume 用）

用法：
    save_checkpoint(save_dir, model, optimizer, scaler, step, epoch, loss, tokens)
    step, epoch, loss, tokens = load_checkpoint(path, model, optimizer, scaler, device)
"""

import os
import json
import glob
import shutil

import torch
from safetensors.torch import save_file, load_file


def save_checkpoint(save_dir, model, optimizer, scaler, step, epoch, best_loss, tokens_seen,
                    max_keep=5):
    """保存 checkpoint 为 HF safetensors 格式目录。

    目录结构: save_dir/ckpt_step{step}/
        model.safetensors  — 模型权重
        config.json        — 模型配置
        training_state.pth — 训练状态
    """
    os.makedirs(save_dir, exist_ok=True)
    raw = model.module if hasattr(model, 'module') else model
    ckpt_dir = os.path.join(save_dir, f'ckpt_step{step}')
    os.makedirs(ckpt_dir, exist_ok=True)

    # 1. 模型权重 → safetensors
    state_dict = raw.state_dict()
    # safetensors 不支持非 tensor，过滤并转 contiguous
    tensor_dict = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            tensor_dict[k] = v.contiguous().cpu()
    save_file(tensor_dict, os.path.join(ckpt_dir, 'model.safetensors'))

    # 2. 模型配置 → config.json
    config = {
        'vocab_size': raw.vocab_size,
        'D': raw.D,
        'N': raw.N,
        'K': raw.K,
        'num_layers': raw.num_layers,
        'D_ff': raw.D_ff,
        'v_th_min': raw.v_th_min,
        'memory_layer_interval': raw.memory_layer_interval,
        'D_key': raw.D_key,
        'D_value': raw.D_value,
    }
    with open(os.path.join(ckpt_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # 3. 训练状态 → training_state.pth
    torch.save({
        'optimizer_state': optimizer.state_dict(),
        'scaler_state': scaler.state_dict() if scaler is not None else None,
        'step': step,
        'epoch': epoch,
        'best_loss': best_loss,
        'tokens_seen': tokens_seen,
    }, os.path.join(ckpt_dir, 'training_state.pth'))

    print(f"  → Checkpoint saved: {ckpt_dir}")

    # 清理旧 checkpoint 目录，按 mtime 排序，仅保留最新 max_keep 个
    ckpt_dirs = sorted(glob.glob(os.path.join(save_dir, 'ckpt_step*')),
                       key=os.path.getmtime)
    while len(ckpt_dirs) > max_keep:
        old = ckpt_dirs.pop(0)
        shutil.rmtree(old)
        print(f"  → Removed old checkpoint: {old}")


def load_checkpoint(path, model, optimizer, scaler, device):
    """加载 checkpoint，恢复训练状态。

    Args:
        path: checkpoint 目录路径（含 model.safetensors）或旧格式 .pth 文件路径
    """
    # 兼容旧格式 .pth
    if path.endswith('.pth') and os.path.isfile(path):
        return _load_legacy_checkpoint(path, model, optimizer, scaler, device)

    print(f"Loading checkpoint from {path}...")
    raw = model.module if hasattr(model, 'module') else model

    # 1. 加载模型权重
    safetensors_path = os.path.join(path, 'model.safetensors')
    state_dict = load_file(safetensors_path, device=str(device))
    raw.load_state_dict(state_dict, strict=False)

    # 2. 加载训练状态
    training_state_path = os.path.join(path, 'training_state.pth')
    if os.path.exists(training_state_path):
        ts = torch.load(training_state_path, map_location=device, weights_only=False)

        if ts.get('optimizer_state') is not None and optimizer is not None:
            optimizer.load_state_dict(ts['optimizer_state'])

        if 'scaler_state' in ts and scaler is not None and ts['scaler_state'] is not None:
            scaler.load_state_dict(ts['scaler_state'])

        step = ts.get('step', 0)
        epoch = ts.get('epoch', 0)
        best_loss = ts.get('best_loss', float('inf'))
        tokens_seen = ts.get('tokens_seen', 0)
    else:
        step, epoch, best_loss, tokens_seen = 0, 0, float('inf'), 0

    print(f"  Resumed: step={step}, epoch={epoch}, tokens={tokens_seen:,}")
    return step, epoch, best_loss, tokens_seen


def load_model_weights(path, model, device):
    """仅加载模型权重（推理或 SFT 加载预训练权重用）。

    Args:
        path: checkpoint 目录路径或旧格式 .pth 文件路径
    """
    raw = model.module if hasattr(model, 'module') else model

    if path.endswith('.pth') and os.path.isfile(path):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        if 'model_state_dict' in ckpt:
            raw.load_state_dict(ckpt['model_state_dict'], strict=False)
        elif 'trainable_state_dict' in ckpt:
            raw.load_state_dict(ckpt['trainable_state_dict'], strict=False)
        return

    safetensors_path = os.path.join(path, 'model.safetensors')
    state_dict = load_file(safetensors_path, device=str(device))
    raw.load_state_dict(state_dict, strict=False)


def load_config(path):
    """从 checkpoint 读取模型配置。

    Args:
        path: checkpoint 目录路径或旧格式 .pth 文件路径

    Returns:
        dict: 模型配置
    """
    if path.endswith('.pth') and os.path.isfile(path):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        return ckpt.get('model_config', {})

    config_path = os.path.join(path, 'config.json')
    with open(config_path, 'r') as f:
        return json.load(f)


def _load_legacy_checkpoint(path, model, optimizer, scaler, device):
    """加载旧格式 .pth checkpoint（向后兼容）。"""
    print(f"Loading legacy checkpoint from {path}...")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    raw = model.module if hasattr(model, 'module') else model

    if 'model_state_dict' in ckpt:
        raw.load_state_dict(ckpt['model_state_dict'], strict=False)
    elif 'trainable_state_dict' in ckpt:
        raw.load_state_dict(ckpt['trainable_state_dict'], strict=False)

    if 'optimizer_state' in ckpt and optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer_state'])

    if 'scaler_state' in ckpt and scaler is not None:
        scaler.load_state_dict(ckpt['scaler_state'])

    step = ckpt.get('step', 0)
    epoch = ckpt.get('epoch', 0)
    best_loss = ckpt.get('best_loss', float('inf'))
    tokens_seen = ckpt.get('tokens_seen', 0)
    print(f"  Resumed: step={step}, epoch={epoch}, tokens={tokens_seen:,}")
    return step, epoch, best_loss, tokens_seen
