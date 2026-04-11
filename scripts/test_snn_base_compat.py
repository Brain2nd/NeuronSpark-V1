"""
Vendored snn_base.py vs spikingjelly 一致性验证。

在有 spikingjelly 的环境中运行，对比两个版本的行为。
全部通过标准: CPU/fp32 严格 0 diff + Triton 快路径命中。

用法:
    python scripts/test_snn_base_compat.py
"""

import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# === 导入两个版本 ===
# 原版
from spikingjelly.activation_based import base as sj_base
from spikingjelly.activation_based import neuron as sj_neuron
from spikingjelly.activation_based import surrogate as sj_surrogate
from spikingjelly.activation_based import layer as sj_layer
from spikingjelly.activation_based import functional as sj_functional

# Vendored 版
from atomic_ops.snn_base import (
    MemoryModule, BaseNode, Sigmoid, Linear, reset_net,
    surrogate as v_surrogate, functional as v_functional, layer as v_layer,
)

passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  ✓ {name}")
        passed += 1
    else:
        print(f"  ✗ {name} — {detail}")
        failed += 1


# ============================================================
print("\n=== 2.1 Sigmoid vs surrogate.Sigmoid ===")

sj_sig = sj_surrogate.Sigmoid(alpha=4.0)
v_sig = v_surrogate.Sigmoid(alpha=4.0)

# 属性检查
check("类名 == 'Sigmoid'", type(v_sig).__name__ == 'Sigmoid')
check("hasattr alpha", hasattr(v_sig, 'alpha'))
check("alpha 值", v_sig.alpha == sj_sig.alpha)
check("hasattr spiking", hasattr(v_sig, 'spiking'))
check("spiking 值", v_sig.spiking == sj_sig.spiking)
check("isinstance nn.Module", isinstance(v_sig, nn.Module))

# 前向: 随机输入
torch.manual_seed(42)
x = torch.randn(100, requires_grad=True)
x_sj = x.detach().clone().requires_grad_(True)

out_v = v_sig(x)
out_sj = sj_sig(x_sj)
check("前向输出一致", torch.equal(out_v.detach(), out_sj.detach()),
      f"max diff={( out_v.detach() - out_sj.detach()).abs().max().item()}")

# dtype 保留
for dt in [torch.float32, torch.float16, torch.bfloat16]:
    x_dt = torch.randn(10, dtype=dt)
    out_dt = v_sig(x_dt)
    check(f"dtype 保留 ({dt})", out_dt.dtype == dt)

# 反向梯度一致
out_v.sum().backward()
out_sj.sum().backward()
check("反向梯度一致", torch.allclose(x.grad, x_sj.grad, atol=0, rtol=0),
      f"max diff={( x.grad - x_sj.grad).abs().max().item()}" if x.grad is not None else "no grad")

# 边界值
for val, name in [(0.0, "x=0"), (100.0, "x=100"), (-100.0, "x=-100")]:
    xb = torch.tensor([val])
    check(f"边界 {name}", torch.equal(v_sig(xb), sj_sig(xb)))

# spiking=False 模式
v_sig_ns = Sigmoid(alpha=4.0, spiking=False)
sj_sig_ns = sj_surrogate.Sigmoid(alpha=4.0, spiking=False)
x_ns = torch.randn(50)
check("spiking=False 一致", torch.allclose(v_sig_ns(x_ns), sj_sig_ns(x_ns), atol=1e-7))


# ============================================================
print("\n=== 2.2 MemoryModule vs base.MemoryModule ===")

class TestMM_V(MemoryModule):
    def __init__(self):
        super().__init__()
        self.register_memory('state', 0.)
    def single_step_forward(self, x):
        self.state = self.state + x
        return self.state

class TestMM_SJ(sj_base.MemoryModule):
    def __init__(self):
        super().__init__()
        self.register_memory('state', 0.)
    def single_step_forward(self, x):
        self.state = self.state + x
        return self.state

mm_v = TestMM_V()
mm_sj = TestMM_SJ()

# 初始状态
check("初始 state 一致", mm_v.state == mm_sj.state)

# 前向修改状态
mm_v.single_step_forward(torch.tensor(3.0))
mm_sj.single_step_forward(torch.tensor(3.0))
check("前向后 state 一致", torch.equal(
    mm_v.state if isinstance(mm_v.state, torch.Tensor) else torch.tensor(mm_v.state),
    mm_sj.state if isinstance(mm_sj.state, torch.Tensor) else torch.tensor(mm_sj.state)))

# reset
mm_v.reset()
mm_sj.reset()
check("reset 后 state 一致", mm_v.state == mm_sj.state == 0.)

# 嵌套模块 reset
class Outer_V(MemoryModule):
    def __init__(self):
        super().__init__()
        self.inner = TestMM_V()
    def single_step_forward(self, x):
        return self.inner.single_step_forward(x)

outer = Outer_V()
outer.single_step_forward(torch.tensor(5.0))
reset_net(outer)
check("嵌套 reset_net 后 inner.state == 0", outer.inner.state == 0.)


# ============================================================
print("\n=== 2.3 BaseNode vs neuron.BaseNode ===")

bn_v = BaseNode(v_threshold=1.0, v_reset=None, surrogate_function=v_sig,
                detach_reset=False, step_mode='s', backend='torch', store_v_seq=False)
bn_sj = sj_neuron.BaseNode(v_threshold=1.0, v_reset=None, surrogate_function=sj_sig,
                            detach_reset=False, step_mode='s', backend='torch', store_v_seq=False)

# 属性检查
for attr in ['v_threshold', 'v_reset', 'detach_reset', 'step_mode', 'backend', 'store_v_seq']:
    check(f"属性 {attr}", getattr(bn_v, attr) == getattr(bn_sj, attr),
          f"v={getattr(bn_v, attr)} sj={getattr(bn_sj, attr)}")

check("初始 v", bn_v.v == bn_sj.v == 0.)

# v_float_to_tensor
x_test = torch.randn(3, 5)
bn_v.v_float_to_tensor(x_test)
bn_sj.v_float_to_tensor(x_test)
check("v_float_to_tensor shape", bn_v.v.shape == bn_sj.v.shape == x_test.shape)
check("v_float_to_tensor 值", torch.equal(bn_v.v, bn_sj.v))

# 非零初始值
bn_v2 = BaseNode(v_reset=0.5)
bn_sj2 = sj_neuron.BaseNode(v_reset=0.5)
check("v_reset=0.5 初始 v", bn_v2.v == bn_sj2.v == 0.5)


# ============================================================
print("\n=== 2.4 reset_net vs functional.reset_net ===")

class NestA_V(MemoryModule):
    def __init__(self):
        super().__init__()
        self.register_memory('a', 0.)
        self.child = TestMM_V()
    def single_step_forward(self, x):
        pass

nest = NestA_V()
nest.a = 99.
nest.child.state = 42.
reset_net(nest)
check("reset_net 递归: a == 0", nest.a == 0.)
check("reset_net 递归: child.state == 0", nest.child.state == 0.)


# ============================================================
print("\n=== 2.5 Linear vs layer.Linear ===")

v_lin = v_layer.Linear(10, 5, bias=False, step_mode='s')
sj_lin = sj_layer.Linear(10, 5, bias=False, step_mode='s')

# 复制权重
sj_lin.weight.data = v_lin.weight.data.clone()

x_lin = torch.randn(3, 10)
check("forward 一致", torch.equal(v_lin(x_lin), sj_lin(x_lin)))
check("step_mode 属性", v_lin.step_mode == sj_lin.step_mode == 's')
check("state_dict keys", set(v_lin.state_dict().keys()) == set(sj_lin.state_dict().keys()))


# ============================================================
print("\n=== 2.6 集成: PLIFNode ===")

from atomic_ops.plif_node import PLIFNode

# 用 vendored Sigmoid 构造
plif = PLIFNode(dim=8, init_tau=2.0, v_threshold=0.5,
                surrogate_function=v_surrogate.Sigmoid(alpha=4.0))
# 用 spikingjelly Sigmoid 构造另一个
plif_sj = PLIFNode(dim=8, init_tau=2.0, v_threshold=0.5,
                   surrogate_function=sj_surrogate.Sigmoid(alpha=4.0))

# 复制权重
plif_sj.w.data = plif.w.data.clone()
plif_sj.v_th.data = plif.v_th.data.clone()

# 多步前向
torch.manual_seed(123)
inputs = [torch.randn(2, 8) for _ in range(5)]

plif.reset()
plif_sj.reset()

all_match = True
for i, inp in enumerate(inputs):
    out_v = plif(inp)
    out_sj = plif_sj(inp)
    if not torch.equal(out_v, out_sj):
        all_match = False
        print(f"    step {i}: max diff = {(out_v - out_sj).abs().max().item()}")

check("PLIFNode 多步前向 bit-exact", all_match)


# ============================================================
print("\n=== 2.7 集成: SelectivePLIFNode ===")

from atomic_ops.selective_plif import SelectivePLIFNode

splif_v = SelectivePLIFNode(surrogate_function=v_surrogate.Sigmoid(alpha=4.0))
splif_sj = SelectivePLIFNode(surrogate_function=sj_surrogate.Sigmoid(alpha=4.0))

torch.manual_seed(456)
batch, DN = 2, 16
splif_v.reset()
splif_sj.reset()

all_match_s = True
for i in range(5):
    x_s = torch.randn(batch, DN)
    beta = torch.sigmoid(torch.randn(batch, DN))
    alpha = torch.ones(batch, DN) * 0.5
    v_th = torch.ones(batch, DN) * 0.5

    spike_v = splif_v.single_step_forward(x_s, beta, alpha, v_th)
    spike_sj = splif_sj.single_step_forward(x_s, beta, alpha, v_th)
    if not torch.equal(spike_v, spike_sj):
        all_match_s = False
        print(f"    step {i}: max diff = {(spike_v - spike_sj).abs().max().item()}")
    # 也检查膜电位
    if not torch.equal(splif_v.v, splif_sj.v):
        all_match_s = False
        print(f"    step {i} v: max diff = {(splif_v.v - splif_sj.v).abs().max().item()}")

check("SelectivePLIFNode 多步 spike+V bit-exact", all_match_s)


# ============================================================
print("\n=== 2.8 Triton 快路径显式命中断言 ===")

from atomic_ops.parallel_scan import plif_parallel_forward, plif_rowparam_forward, _HAS_TRITON

if _HAS_TRITON and torch.cuda.is_available():
    # 打桩: monkey-patch _TritonPLIFForward.apply 加计数器
    from atomic_ops import parallel_scan as ps
    _orig_triton_fwd = ps._TritonPLIFForward.apply
    _orig_triton_rp = ps._TritonPLIFRowParamForward.apply
    triton_fwd_count = [0]
    triton_rp_count = [0]

    def _patched_fwd(*args, **kwargs):
        triton_fwd_count[0] += 1
        return _orig_triton_fwd(*args, **kwargs)

    def _patched_rp(*args, **kwargs):
        triton_rp_count[0] += 1
        return _orig_triton_rp(*args, **kwargs)

    ps._TritonPLIFForward.apply = staticmethod(_patched_fwd)
    ps._TritonPLIFRowParamForward.apply = staticmethod(_patched_rp)

    # 用 vendored Sigmoid 调用
    K, batch, D = 12, 2, 64
    beta = torch.rand(K, batch, D, device='cuda') * 0.5 + 0.5
    u = torch.randn(K, batch, D, device='cuda') * 0.1
    v_th = torch.ones(K, batch, D, device='cuda') * 0.5
    v_init = torch.zeros(batch, D, device='cuda')

    vs = v_surrogate.Sigmoid(alpha=4.0)

    # plif_parallel_forward
    triton_fwd_count[0] = 0
    spike, V_post, _ = plif_parallel_forward(beta, u, v_th, v_init, surrogate_function=vs)
    check("plif_parallel_forward 命中 Triton", triton_fwd_count[0] > 0,
          f"count={triton_fwd_count[0]}")

    # plif_rowparam_forward
    triton_rp_count[0] = 0
    beta_row = torch.rand(batch, D, device='cuda') * 0.5 + 0.5
    v_th_row = torch.ones(batch, D, device='cuda') * 0.5
    spike2, V_post2 = plif_rowparam_forward(beta_row, u, v_th_row, v_init, surrogate_function=vs)
    check("plif_rowparam_forward 命中 Triton", triton_rp_count[0] > 0,
          f"count={triton_rp_count[0]}")

    # 恢复
    ps._TritonPLIFForward.apply = _orig_triton_fwd
    ps._TritonPLIFRowParamForward.apply = _orig_triton_rp
else:
    print("  (跳过: 无 Triton 或无 CUDA)")


# ============================================================
print(f"\n{'='*50}")
print(f"结果: {passed} passed, {failed} failed")
if failed > 0:
    print("❌ 一致性验证未通过，不能进行 import 替换！")
    sys.exit(1)
else:
    print("✓ 全部通过，可以安全替换 spikingjelly import。")
