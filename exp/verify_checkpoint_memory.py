"""显存优化验证 — 逐组件剖析 + 正确性 + 全模型峰值。

四项测试：
  1. 正确性：小模型 forward+backward 梯度非零
  2. SNNBlock 逐组件显存剖析（单层，无 checkpoint，定位瓶颈）
  3. 逐层显存累积（gradient checkpoint 模式）
  4. 全模型 forward+backward 峰值显存
"""

import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import functional
from torch.utils.checkpoint import checkpoint


def get_gpu_mb():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024 / 1024


def test_correctness():
    """测试 1: 小模型 forward+backward 正确性。"""
    print("=" * 60)
    print("测试 1: 正确性 — 小模型 forward+backward")
    print("=" * 60)

    from atomic_ops import SNNDecoderLayer

    D, N, K = 128, 4, 8
    batch, seq_len = 2, 16
    TK = seq_len * K
    device = "cuda"
    dtype = torch.bfloat16

    layer = SNNDecoderLayer(
        D=D, N=N, D_ff=D * 3, v_th_min=0.1,
        ffn_v_threshold=0.5, K=K, num_layers=1, layer_idx=0,
    ).to(device=device, dtype=dtype)

    h = torch.randn(TK, batch, D, device=device, dtype=dtype, requires_grad=True)

    # forward
    functional.reset_net(layer)
    out, ponder_cost = layer(h)

    # backward
    loss = out.sum() + ponder_cost * 0.01
    loss.backward()

    grad_norm = h.grad.norm().item()
    print(f"  输出 shape: {out.shape}")
    print(f"  ponder_cost: {ponder_cost.item():.4f}")
    print(f"  输入梯度 L2: {grad_norm:.6f}")
    print(f"  梯度非零: {'PASS' if grad_norm > 1e-8 else 'FAIL'}")

    # 检查参数梯度
    n_params = 0
    n_grad = 0
    for name, p in layer.named_parameters():
        n_params += 1
        if p.grad is not None and p.grad.norm().item() > 0:
            n_grad += 1
    print(f"  参数梯度: {n_grad}/{n_params} 非零")
    ok = grad_norm > 1e-8 and n_grad > n_params * 0.6
    print(f"  结果: {'PASS' if ok else 'FAIL'}")
    print()

    del layer, h, out, loss
    gc.collect()
    torch.cuda.empty_cache()
    return ok


def test_snnblock_profile():
    """测试 2: SNNBlock 逐组件显存剖析。"""
    print("=" * 60)
    print("测试 2: SNNBlock 逐组件显存剖析（单层，无 checkpoint）")
    print("=" * 60)

    from atomic_ops.snn_block import SNNBlock
    from atomic_ops.parallel_scan import plif_parallel_forward
    from atomic_ops import spike_current_activation

    D, N, K = 1024, 8, 32
    batch, seq_len = 2, 512
    TK = seq_len * K  # 16384
    DN = D * N  # 8192
    device = "cuda"
    dtype = torch.bfloat16

    block = SNNBlock(D=D, N=N, v_th_min=0.1).to(device=device, dtype=dtype)
    functional.reset_net(block)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    tensor_gb = TK * batch * DN * 2 / 1024**3  # bf16 DN tensor size
    tensor_d_gb = TK * batch * D * 2 / 1024**3
    print(f"  参考: (TK,batch,DN) bf16 = {tensor_gb:.3f} GB, (TK,batch,D) bf16 = {tensor_d_gb:.3f} GB")

    spike_in = torch.randn(TK, batch, D, device=device, dtype=dtype)
    flat = spike_in.reshape(TK * batch, D)

    base = get_gpu_mb()
    print(f"\n  基线: {base:.1f} MB")

    # D-sized projections
    gate_all = torch.sigmoid(F.linear(flat, block.W_gate.weight).reshape(TK, batch, D))
    I_skip_all = F.linear(flat, block.W_skip.weight).reshape(TK, batch, D)
    m1 = get_gpu_mb()
    print(f"  gate+skip (D-sized): {m1:.1f} MB (+{m1-base:.1f} MB)")

    # beta: project → sigmoid_
    beta_all = F.linear(flat, block.W_beta_x.weight).reshape(TK, batch, DN)
    beta_all.add_(block.b_beta).sigmoid_()
    m2 = get_gpu_mb()
    print(f"  beta (in-place sigmoid): {m2:.1f} MB (+{m2-m1:.1f} MB)")

    # v_th: project → abs_ + min
    v_th_all = F.linear(flat, block.W_th_x.weight).reshape(TK, batch, DN)
    v_th_all.add_(block.b_th).abs_().add_(block.v_th_min)
    m3 = get_gpu_mb()
    print(f"  v_th (in-place abs): {m3:.1f} MB (+{m3-m2:.1f} MB)")

    # alpha → u: project alpha, project I, mul_
    raw_alpha = F.linear(flat, block.W_alpha_x.weight).reshape(TK, batch, DN)
    raw_alpha.add_(block.b_alpha)
    u_hidden = F.linear(flat, block.W_in.weight).reshape(TK, batch, DN)
    del flat
    m4a = get_gpu_mb()
    print(f"  raw_alpha + I_all (peak): {m4a:.1f} MB (+{m4a-m3:.1f} MB)")

    alpha = F.softplus(raw_alpha)
    del raw_alpha
    u_hidden.mul_(alpha)
    del alpha
    m4 = get_gpu_mb()
    print(f"  u_hidden (after mul+del): {m4:.1f} MB (+{m4-m3:.1f} MB)")

    # PLIF parallel scan
    v_init = torch.zeros(batch, DN, device=device, dtype=dtype)
    s_hidden, V_post_hidden, _ = plif_parallel_forward(
        beta_all, u_hidden, v_th_all, v_init, max_iter=3,
        surrogate_function=block.hidden_neuron.surrogate_function,
    )
    m5a = get_gpu_mb()
    print(f"  plif_parallel (peak): {m5a:.1f} MB (+{m5a-m4:.1f} MB)")

    del beta_all, u_hidden
    block.hidden_neuron.v = V_post_hidden[-1].detach()
    del V_post_hidden
    m5 = get_gpu_mb()
    print(f"  plif (after cleanup): {m5:.1f} MB (+{m5-m4:.1f} MB)")

    # Output projection
    sc_hidden = spike_current_activation(s_hidden, v_th_all)
    m6a = get_gpu_mb()
    print(f"  spike_current: {m6a:.1f} MB (+{m6a-m5:.1f} MB)")

    del s_hidden, v_th_all
    sc_flat = sc_hidden.reshape(TK * batch, DN)
    I_out_all = F.linear(sc_flat, block.W_out.weight).reshape(TK, batch, D)
    del sc_hidden, sc_flat
    m6 = get_gpu_mb()
    print(f"  W_out + cleanup: {m6:.1f} MB (+{m6-m5:.1f} MB)")

    I_total = I_out_all * gate_all + I_skip_all
    m7 = get_gpu_mb()
    print(f"  gate+skip combine: {m7:.1f} MB (+{m7-m6:.1f} MB)")

    peak = torch.cuda.max_memory_allocated() / 1024 / 1024
    total_inc = m7 - base
    print(f"\n  总增量: {total_inc:.1f} MB")
    print(f"  峰值显存: {peak:.1f} MB")
    print()

    del spike_in, gate_all, I_skip_all, I_out_all, I_total, v_init
    gc.collect()
    torch.cuda.empty_cache()
    return True


def test_per_layer_memory():
    """测试 3: 逐层显存累积（gradient checkpoint 模式）。"""
    print("=" * 60)
    print("测试 3: 逐层显存累积（gradient checkpoint）")
    print("=" * 60)

    from atomic_ops import SNNDecoderLayer

    D, N, K = 1024, 8, 32
    batch, seq_len = 2, 64
    TK = seq_len * K
    device = "cuda"
    dtype = torch.bfloat16
    num_test_layers = 5

    layers = nn.ModuleList([
        SNNDecoderLayer(
            D=D, N=N, D_ff=D * 3, v_th_min=0.1,
            ffn_v_threshold=0.5, K=K, num_layers=num_test_layers, layer_idx=i,
        )
        for i in range(num_test_layers)
    ]).to(device=device, dtype=dtype)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    h = torch.randn(TK, batch, D, device=device, dtype=dtype, requires_grad=True)
    mem_before = get_gpu_mb()
    print(f"  输入 h 后显存: {mem_before:.1f} MB")

    increments = []
    mem_prev = mem_before

    for i, layer in enumerate(layers):
        functional.reset_net(layer)

        def run_layer(x, layer_module=layer):
            return layer_module(x)

        h, pc = checkpoint(run_layer, h, use_reentrant=False)
        mem_now = get_gpu_mb()
        delta = mem_now - mem_prev
        increments.append(delta)
        print(f"  Layer {i}: {mem_now:.1f} MB (+{delta:.1f} MB)")
        mem_prev = mem_now

    # backward
    loss = h.sum()
    mem_before_bwd = get_gpu_mb()
    loss.backward()
    mem_after_bwd = get_gpu_mb()
    peak = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f"  Backward 前: {mem_before_bwd:.1f} MB")
    print(f"  Backward 后: {mem_after_bwd:.1f} MB")
    print(f"  峰值显存: {peak:.1f} MB")

    avg_inc = sum(increments[1:]) / len(increments[1:]) if len(increments) > 1 else increments[0]
    print(f"\n  平均每层增量（跳过第一层）: {avg_inc:.1f} MB")
    ok = avg_inc < 200
    print(f"  阈值 < 200 MB: {'PASS' if ok else 'FAIL'}")
    print()

    del layers, h, loss
    gc.collect()
    torch.cuda.empty_cache()
    return ok


def test_full_model_memory():
    """测试 4: 全模型 forward+backward 峰值显存。"""
    print("=" * 60)
    print("测试 4: 全模型 forward+backward 峰值显存")
    print("=" * 60)

    from model import SNNLanguageModel
    from spikingjelly.activation_based import functional

    device = "cuda"
    dtype = torch.bfloat16

    model = SNNLanguageModel(
        vocab_size=6144, D=1024, N=8, K=32,
        num_layers=20, D_ff=3072, v_th_min=0.1,
    ).to(device=device, dtype=dtype)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    param_mem = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
    print(f"  模型参数: {n_params:.1f}M, 参数显存: {param_mem:.1f} MB")

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    batch, seq_len = 2, 512
    input_ids = torch.randint(0, 6144, (batch, seq_len), device=device)
    labels = input_ids.clone()

    mem_before = get_gpu_mb()
    print(f"  Forward 前显存: {mem_before:.1f} MB")

    functional.reset_net(model)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(input_ids, labels)

    mem_after_fwd = get_gpu_mb()
    print(f"  Forward 后显存: {mem_after_fwd:.1f} MB")

    loss = output.last_loss.mean() + output.ponder_cost * 0.01
    loss.backward()

    mem_after_bwd = get_gpu_mb()
    peak = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f"  Backward 后显存: {mem_after_bwd:.1f} MB")
    print(f"  峰值显存: {peak:.1f} MB ({peak / 1024:.2f} GB)")

    total_gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
    print(f"  GPU 总显存: {total_gpu_mem:.0f} MB ({total_gpu_mem / 1024:.1f} GB)")
    print(f"  峰值占比: {peak / total_gpu_mem * 100:.1f}%")
    print()

    del model, input_ids, labels, output, loss
    gc.collect()
    torch.cuda.empty_cache()
    return True


if __name__ == "__main__":
    print("显存优化验证 — in-place modulation + spike uint8 + view broadcast")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.get_device_name(0)}")
    print(f"总显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

    results = {}
    results["正确性"] = test_correctness()
    results["SNNBlock 剖析"] = test_snnblock_profile()
    results["逐层显存"] = test_per_layer_memory()
    results["全模型显存"] = test_full_model_memory()

    print("=" * 60)
    print("汇总")
    print("=" * 60)
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    all_pass = all(results.values())
    print(f"\n总结: {'ALL PASS' if all_pass else 'SOME FAILED'}")
