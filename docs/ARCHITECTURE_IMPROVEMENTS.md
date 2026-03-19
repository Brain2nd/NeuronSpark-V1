# NeuronSpark 架构改进方案

> 基于 Qwen3.5 (GatedDeltaNet + MoE) 架构分析，结合 SNN 特性推导的改进方向。
> 适用于当前 V1toV2 分支，下一轮预训练时实施。

## 参考架构：Qwen3.5 GatedDeltaNet

源码：`transformers/models/qwen3_next/modeling_qwen3_next.py`

```
层结构: 10 × (3×DeltaNet-MoE + 1×Attention-MoE) = 40 层
DeltaNet 递推: S[t] = (I - β·k·kᵀ)·S[t-1] + v·kᵀ   // 矩阵状态，选择性遗忘
因果卷积: Conv1d(kernel=4, groups=D) 在递推前预混合相邻 token
门控归一化: RMSNorm(x) * SiLU(z)
MoE: 256 专家取 8 + 1 共享
```

---

## 改进一：因果卷积（Causal Conv1d）

### 动机

Qwen3.5 和 Mamba 都在递推层前加了因果卷积（kernel_size=4）。作用是在做选择性状态更新之前，给每个 token 一个局部窗口的上下文预混合。

当前 NeuronSpark 每个 token 独立经过 6 条投影进入 PLIF，没有相邻 token 的预混合。虽然有"皮层侧向连接"（token t ← token t-1 平均脉冲），但只看前 1 个 token 且是固定的平均操作。

### 方案

在 SNNBlock 的 6 条投影之前加深度因果卷积：

```python
# snn_block.py __init__:
self.conv1d = nn.Conv1d(
    in_channels=D,
    out_channels=D,
    kernel_size=4,
    groups=D,          # 深度卷积，每通道独立
    padding=3,         # causal: 只看过去
    bias=False,
)

# forward_parallel:
# spike_in_seq: (TK, batch, D)
# 因果卷积预混合
conv_in = spike_in_seq.permute(1, 2, 0)        # (batch, D, TK)
conv_out = self.conv1d(conv_in)[:, :, :TK]     # causal: 截断未来
spike_in_seq = conv_out.permute(2, 0, 1)        # (TK, batch, D)
# 然后继续 6 条投影...
```

### 参数开销

- 参数量：D × kernel_size = 1024 × 4 = 4096 per layer
- 20 层总计：81,920 参数（<0.01% 的总参数量）
- 计算量：深度卷积，极小

### 推理时的状态

生成时需要维护 conv_state（最近 kernel_size-1=3 帧的历史），和 Mamba 一样。
当前 generate() 用 forward_parallel 处理 K 帧，conv_state 需要跨 token 传递。

### 注意事项

- 因果卷积作用在 TK 维度（K 帧连续），不是 token 维度
- kernel_size=4 意味着看当前帧 + 前 3 帧
- 如果 K=12，一个 token 的 12 帧内部可以互相看到（局部时序混合）
- 跨 token 边界时，下一个 token 的第 1 帧可以看到上一个 token 的最后 3 帧

---

## 改进二：β 初始化范围扩展

### 动机

当前 β ∈ [0.80, 0.99]，有效记忆长度 L_eff = 1/(1-β) ∈ [5, 100] 帧 ≈ [<1, 8] token。
即使最慢的神经元也只能记住 ~8 个 token，远不够 2048 的上下文。

Qwen3.5 的 DeltaNet 用 `A_log` 参数控制衰减，初始化为 Uniform(0, 16)，衰减率 = exp(-exp(A_log) × Δ)，可以非常接近 1。

### 方案

扩展 `snn_block.py` 的 `_initialize_parameters` 中 beta_values 范围：

```python
# 当前:
beta_values = torch.linspace(0.80, 0.99, N)    # N=8: [0.80, 0.83, ..., 0.99]

# 改进（N=16 时）:
beta_values = torch.linspace(0.80, 0.9999, N)  # N=16: [0.80, ..., 0.99, ..., 0.9999]
```

最后几组神经元成为真正的"长程记忆通道"：

| N 组 | β | L_eff (帧) | L_eff (token, K=12) |
|---|---|---|---|
| 1-4 | 0.80-0.90 | 5-10 | <1 |
| 5-8 | 0.90-0.95 | 10-20 | 1-2 |
| 9-12 | 0.95-0.99 | 20-100 | 2-8 |
| 13-16 | 0.99-0.9999 | 100-10000 | 8-833 |

### 配套修改

1. **V2 泄漏量输出补偿**：β=0.9999 的通道 (1-β)=0.0001，输出极微弱。
   W_out 的发放率均衡缩放（`out_scale_per_n = 1/√target_p_fire`）已有，
   但需要确认 target_p_fire 对极高 β 通道的设定是否合理。

2. **V_th 校准**：极高 β 通道的 V 累积非常慢，σ_V 也非常小。
   当前 V_th 校准公式 `σ_V = σ_I_base × √(1 - β^{2K})` 对 β=0.9999 时：
   `σ_V ≈ σ_I_base × √(1 - 0.9999^24) ≈ σ_I_base × 0.035`，极小。
   V_th 需要相应降低，否则这些神经元永远不发放。

3. **数值稳定性**：β^24576（2048 token × K=12）对 β=0.9999 是 exp(24576 × log(0.9999)) = exp(-2.46) ≈ 0.086。还在合理范围。但 β=0.99999 时就趋于 1，可能有精度问题（bf16 最小精度 ~1e-4）。

### 结论

β ∈ [0.80, 0.9999] 在 bf16 下安全，可以支持 ~800 token 的记忆跨度。
β > 0.9999 在 bf16 下和 1.0 无法区分，不可用。

---

## 改进三：N 扩展（状态扩展因子）

### 动机

当前 N=8，每个可见维度 8 个隐神经元，对应 8 个不同时间尺度。
DeltaNet 的状态是矩阵 (D_key × D_value)，信息容量远大于我们的向量 (D×N)。

增大 N 等价于：
- 更多时间尺度的通道（更精细的 β 频谱）
- 更大的隐状态容量（记住更多信息）
- 更丰富的选择性调制（W_β, W_α, W_th 的输出维度增大）

### 方案

N: 8 → 16

### 参数影响

受影响的参数（每层）：

| 参数 | 当前 (N=8) | N=16 | 增量 |
|---|---|---|---|
| W_in: D → D×N | 1024×8192 | 1024×16384 | +8M |
| W_β: D → D×N | 1024×8192 | 1024×16384 | +8M |
| W_α: D → D×N | 1024×8192 | 1024×16384 | +8M |
| W_th: D → D×N | 1024×8192 | 1024×16384 | +8M |
| W_out: D×N → D | 8192×1024 | 16384×1024 | +8M |
| b_β, b_α, b_th | 3×8192 | 3×16384 | +24K |
| **层合计** | ~42M | ~82M | +40M |

不受影响的：W_gate, W_skip (D→D), SNNFFN (D→D_ff→D)

20 层总计参数增量：+800M → 模型从 1.2B 变成 ~2.0B

### 显存影响

隐状态 V 和中间激活的 D×N 维度翻倍。
粗估：当前 38.7GB → N=16 约 45-46GB（接近 48GB 极限）。

需要实测确认是否放得下。可能需要降 D 或降 L 来补偿。

### 折中方案：N=12

| 配置 | 总参数 | 估计显存 |
|---|---|---|
| D=1024, N=8, L=20 | 1.2B | 38.7GB |
| D=1024, N=12, L=20 | ~1.6B | ~42GB |
| D=1024, N=16, L=20 | ~2.0B | ~46GB |
| D=896, N=16, L=20 | ~1.5B | ~40GB |

N=12 可能是甜点——在 48GB 内保持余量，同时比 N=8 多 50% 的隐状态容量。

### 和 β 扩展的协同

N 增大后，β 初始化范围自然可以更宽——更多分组覆盖从极快到极慢的完整频谱。
N=16 配合 β ∈ [0.80, 0.9999]，最慢的 4 组（N=13-16）专门做长程记忆。

---

## 实施计划

### 下一轮预训练前改动

1. `snn_block.py`:
   - `__init__` 加 `self.conv1d = nn.Conv1d(...)`
   - `_initialize_parameters` 扩展 `beta_values` 到 [0.80, 0.9999]
   - `forward_parallel` 在投影前加因果卷积

2. 训练脚本：
   - `--N` 默认值从 8 改为 12 或 16
   - 实测确认显存是否放得下

3. 生成脚本：
   - `generate()` 维护 conv_state 跨 token 传递

### 需要实测验证

- [ ] N=12 / N=16 在 4×4090D DDP batch=1 的实际显存
- [ ] 因果卷积对 TPS 的影响
- [ ] β ∈ [0.80, 0.9999] 的训练稳定性（极高 β 通道是否正常学习）
- [ ] 和当前 V2 泄漏量激活的兼容性（极高 β 的 (1-β)·V_post 是否过于微弱）
