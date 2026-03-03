"""
吞吐量基准测试：SNN 语言模型前向+反向性能评估

使用合成数据测量不同 batch size 下的吞吐量、延迟和显存峰值。
5 步 warmup + 50 步计时，输出表格。

用法：
    # 默认扫描 batch size 1,2,4,8
    python exp/bench_throughput.py --D 1024 --D_ff 3072

    # 自定义 batch size 列表
    python exp/bench_throughput.py --batch_sizes 1 2 4 8 16

    # 指定特定 batch size
    python exp/bench_throughput.py --batch_sizes 4
"""

import os
import sys
import time
import math
import argparse
import warnings

import torch
import torch.nn.functional as F
from contextlib import nullcontext

warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model import SNNLanguageModel
from spikingjelly.activation_based import functional


def run_benchmark(model, batch_size, seq_len, vocab_size, device, ctx,
                  warmup_steps=5, bench_steps=50):
    """对指定 batch_size 进行基准测试。

    Returns:
        dict: {tokens_per_sec, step_ms, peak_mem_gb, loss, oom} 或 oom=True 表示显存不足
    """
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    try:
        # Warmup
        for _ in range(warmup_steps):
            X = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)
            Y = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)

            with ctx:
                out = model(X, Y)
                loss = out.last_loss.mean()
            loss.backward()

            model.zero_grad(set_to_none=True)
            # 重置神经元状态（每步 forward 内部已做，这里确保干净）
            for layer_module in model.layers:
                functional.reset_net(layer_module)
            functional.reset_net(model.output_neuron)

        # 计时
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        total_loss = 0.0

        for step in range(bench_steps):
            X = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)
            Y = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)

            with ctx:
                out = model(X, Y)
                loss = out.last_loss.mean()
            loss.backward()
            total_loss += loss.item()

            model.zero_grad(set_to_none=True)

        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

        tokens_total = batch_size * seq_len * bench_steps
        tokens_per_sec = tokens_total / elapsed
        step_ms = (elapsed / bench_steps) * 1000
        peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9
        avg_loss = total_loss / bench_steps

        return {
            'tokens_per_sec': tokens_per_sec,
            'step_ms': step_ms,
            'peak_mem_gb': peak_mem_gb,
            'loss': avg_loss,
            'oom': False,
        }

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        model.zero_grad(set_to_none=True)
        return {'oom': True}


def main():
    parser = argparse.ArgumentParser(description="SNN LM Throughput Benchmark")

    # 模型参数
    parser.add_argument('--vocab_size', type=int, default=6144)
    parser.add_argument('--D', type=int, default=1024)
    parser.add_argument('--N', type=int, default=8)
    parser.add_argument('--K', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=20)
    parser.add_argument('--D_ff', type=int, default=3072)
    parser.add_argument('--v_th_min', type=float, default=0.1)
    parser.add_argument('--max_length', type=int, default=512)

    # 基准参数
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[1, 2, 4, 8],
                        help="要测试的 batch size 列表")
    parser.add_argument('--warmup_steps', type=int, default=5)
    parser.add_argument('--bench_steps', type=int, default=50)
    parser.add_argument('--device', type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    if args.device == 'cpu':
        print("警告: CPU 模式，结果仅供参考。")

    device = torch.device(args.device)

    # 性能优化
    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True

    ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()

    # 创建模型
    print(f"正在创建模型 (D={args.D}, N={args.N}, K={args.K}, "
          f"layers={args.num_layers}, D_ff={args.D_ff})...")
    model = SNNLanguageModel(
        vocab_size=args.vocab_size,
        D=args.D,
        N=args.N,
        K=args.K,
        num_layers=args.num_layers,
        D_ff=args.D_ff,
        v_th_min=args.v_th_min,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"参数量: {total_params / 1e6:.1f}M")

    if device.type == 'cuda':
        baseline_mem = torch.cuda.memory_allocated(device) / 1e9
        print(f"基准显存: {baseline_mem:.2f} GB")

    print(f"序列长度: {args.max_length}, Warmup: {args.warmup_steps} 步, 计时: {args.bench_steps} 步")
    print()

    # 表头
    header = f" {'Batch':>5} | {'Tokens/s':>10} | {'Step (ms)':>10} | {'Peak Mem (GB)':>14} | {'Loss':>8}"
    sep = '-' * len(header)
    print(header)
    print(sep)

    # 扫描 batch size
    results = []
    for bs in args.batch_sizes:
        result = run_benchmark(
            model, bs, args.max_length, args.vocab_size, device, ctx,
            warmup_steps=args.warmup_steps, bench_steps=args.bench_steps,
        )

        if result['oom']:
            print(f" {bs:>5} |        OOM |       OOM |           OOM |      OOM")
            results.append({'batch_size': bs, 'oom': True})
            # OOM 后更大的 batch size 也会 OOM
            for bs2 in args.batch_sizes[args.batch_sizes.index(bs)+1:]:
                print(f" {bs2:>5} |        OOM |       OOM |           OOM |      OOM")
                results.append({'batch_size': bs2, 'oom': True})
            break
        else:
            print(f" {bs:>5} | {result['tokens_per_sec']:>10,.0f} | {result['step_ms']:>10.1f} "
                  f"| {result['peak_mem_gb']:>14.1f} | {result['loss']:>8.2f}")
            results.append({'batch_size': bs, **result})

    print(sep)

    # 找最优 batch size
    valid = [r for r in results if not r.get('oom')]
    if valid:
        best = max(valid, key=lambda r: r['tokens_per_sec'])
        print(f"\n最优吞吐量: batch_size={best['batch_size']}, "
              f"{best['tokens_per_sec']:,.0f} tokens/s, "
              f"峰值显存 {best['peak_mem_gb']:.1f} GB")


if __name__ == '__main__':
    main()
