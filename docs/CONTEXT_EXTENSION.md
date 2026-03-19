# SNN 上下文扩展方案：从 Mamba 长文本增程方法到 SNN 适配

> 基于 ICLR 2025 及最新预印本的 5 种 Mamba 上下文扩展方法，推导 NeuronSpark SNN 架构的适配方案。

## 1. 数学基础：SNN vs Mamba 的状态递推对比

### Mamba 状态方程
```
h[t] = Ā·h[t-1] + B̄·x[t]
Ā = exp(Δ·A)         // Δ: 离散化步长, A: 连续时间参数
```

### NeuronSpark SNN 状态方程
```
V_pre[t] = β(t)·V_post[t-1] + α(t)·I[t]          // 充电
s[t] = Θ(V_pre[t] - V_th(t))                       // 发放
V_post[t] = V_pre[t] - V_th(t)·s[t]                // 软重置
output = (1-β)·V_post                               // V2 泄漏量激活
```

### 核心映射

| Mamba | NeuronSpark SNN | 差异 |
|---|---|---|
| Ā = exp(Δ·A) | β(t) = sigmoid(W_β·x + b_β) | β 天然有界 (0,1)，无 Δ 的 OOD 问题 |
| 隐状态 h（线性递推） | 膜电位 V（有 spike-reset 非线性） | SNN 有断裂点 |
| 输出 = C·h（线性读出） | 输出 = (1-β)·V_post（泄漏量） | 高 β 通道输出被 (1-β) 压制 |
| 无 spike | spike + soft reset | 发放时 V 减去 V_th，历史记忆部分擦除 |

### SNN 特有的长文本崩溃机制

1. **累积衰减**：与 Mamba 相同，β^L → 0。但 SNN 还有 spike-reset 的额外记忆擦除。
2. **V_th 擦除效应**：每次发放，V_post = V_pre - V_th，累积的历史信息被切掉 V_th。高 β 全局通道长时间积累后突然发放，一次性丢失大量记忆。
3. **(1-β) 输出压制**：V2 模式下，全局通道（高 β）的输出被 (1-β) 因子天然压制。β=0.99 的通道输出只有 β=0.5 通道的 1/50。

---

## 2. 当前架构的记忆跨度分析

每个神经元的有效记忆长度：
```
L_eff(β) = 1/(1-β)    // 无发放时的指数衰减半衰期
```

当前 b_β 初始化范围 [1.4, 4.6] → β ∈ [0.80, 0.99]：

| β | L_eff (帧) | L_eff (token, K=12) | 占训练长度 2048 |
|---|---|---|---|
| 0.80 | 5 | <1 | <0.1% |
| 0.90 | 10 | <1 | <0.1% |
| 0.95 | 20 | ~2 | 0.1% |
| 0.99 | 100 | ~8 | 0.4% |

**结论**：即使最慢的神经元（β=0.99）也只能记住 ~8 个 token 的信息。当前架构在训练长度内就已是"短记忆"。要记住 2048 个 token，需要 β > 0.9998。

---

## 3. 方法一：β 偏置校准

### 来源
MambaExtend (ICLR 2025) — 校准离散化步长 Δt

### SNN 适配

**方案**：只调偏置 b_β，不动权重 W_β（保留输入依赖的选择性）

```
β_ext(t) = sigmoid(W_β·x[t] + b_β + Δb)
```

Δb 的推导：让平均累积衰减在目标长度下与训练长度一致

```
sigmoid(E[z] + Δb)^L_target = sigmoid(E[z])^L_train
Δb = logit(sigmoid(E[z])^(L_train/L_target)) - E[z]
```

其中 E[z] ≈ b_β（假设 E[W_β·x] ≈ 0）。

### 必须联动校准 α

β 增大后，V 稳态值 = α·I/(1-β) 暴增，发放率失控。必须同时缩小 α：

```
α_ext = α · (1-β_ext)/(1-β)
```

对 α(t) = softplus(W_α·x + b_α)，只改 b_α：

```
b_α_ext = b_α + log((1-β_ext)/(1-β))
```

### V_th 不动

α 联动校准后，稳态 V 不变，V_th 保持训练语义。

### 对选择性的影响

- 全局缩放 sigmoid 输入 z/s → 压制选择性（所有 β 趋向 0.5）— **不可取**
- 只加偏置 Δb → W_β·x 的差异完全保留，选择性不变 — **可取**

### 实施
- 免训练，推理时修改 b_β 和 b_α
- 每层一个 Δb，可用零阶优化在少量长文本上校准
- 实现改动：`snn_block.py` forward_parallel 中 b_beta += Δb

---

## 4. 方法二：全局通道保护

### 来源
LongMamba (ICLR 2025) — 全局通道 token 过滤

### SNN 特有的复杂性

**不能简单对应线性递推**，因为 SNN 有 spike-reset 非线性：

高 β 全局通道的行为周期：
1. 长时间亚阈值累积（V 缓慢增长）
2. V 超过 V_th → 发放
3. V_post = V_pre - V_th → 大量历史信息被 reset 擦除
4. 输出 = (1-β)·(V_pre - V_th) → 如果 V_pre 刚过阈值，输出 ≈ 0
5. 回到步骤 1

**核心矛盾**：spike-reset 是长程记忆的天敌。每次发放都切掉 V_th 量的历史信息。

### 提高 V_th 方案分析

V_th' = c·V_th (c>1)：
- ✓ 发放更难 → 减少废话 token 的记忆擦除
- ✗ 发放时擦除量 = V_th' > V_th → 每次记忆损失更大
- ✗ 重要 token 也更难触发发放 → 信息提取变少

**结论：单纯提高 V_th 不完备。**

### 减小 reset 系数 γ 方案

```
V_post = V_pre - γ·V_th·s    // γ < 1, 减少擦除量
```

γ = L_train/L_target → 长序列时更温和的 reset

- ✓ 保护长程记忆（擦除更少）
- ✗ V_post = V_pre - γ·V_th > V_pre - V_th → 发放后 V 起点更高 → 可能连续 burst
- 需要配合 β 校准防止 burst

### 最优方案：gate 复用做输入门控

SNNBlock 已有 gate = sigmoid(W_gate·x)，复用它控制全局通道状态更新：

```
// 识别全局通道: 按 β 排序，top 25% 为全局
global_mask = (β > β_threshold)

// 全局通道：gate 调制输入
V_global[t] = β·V_global[t-1] + gate[t]·α·I[t]

// 局部通道：正常更新
V_local[t] = β·V_local[t-1] + α·I[t]
```

gate 低的 token（W_gate 判为不重要）不污染全局通道膜电位。

- ✓ 不引入新参数（gate 已训练）
- ✓ 不改变发放判据（V_th 不动）
- ✓ 不改变 reset 机制
- ✗ 需要修改 forward_parallel 的 parallel scan 逻辑

### 实施
- 修改 `snn_block.py` 的 `forward_parallel`
- 全局通道的 u_hidden 乘以 gate_all
- 需要重新训练或至少 fine-tune

---

## 5. 方法三：逐层计算精简

### 来源
DeciMamba (ICLR 2025) — 深层丢弃不重要 token

### SNN 的约束

1. K 帧结构要求每个 token 有连续 K 帧，不能中间丢帧（破坏 parallel scan）
2. V2 模式下 spike=0 不代表贡献为零（(1-β)·V_post 仍非零）
3. 隐状态跨 token 连续，跳过 token 等价于只做自然衰减

### 可行方案：简化路径

不丢弃 token，对不重要 token 用轻量路径：

```
if ||(1-β)·V_post_input[t]||_2 < ε:
    fast path: V[t] = β_mean · V[t-1]   // 只做标量衰减，跳过 6 条 matmul
else:
    full path: 完整 SNNBlock 计算
```

- ✓ parallel scan 递推不断裂
- ✓ 副作用积极：跳过无关 token 减少全局通道污染
- ✓ fast path 只需 1 次标量乘法 vs full path 6 次 matmul
- ✗ 判据需要先计算 input neuron（~10% 开销）
- ✗ 不同 token 走不同路径，无法用统一的 parallel scan kernel

### 实施
- 推理优化，不影响训练
- 需要 benchmark 不同阈值 ε 下的速度/质量权衡

---

## 6. 方法四：β 谱重塑

### 来源
Mamba Modulation (arXiv 2024) — 状态转移矩阵特征值谱缩放

### 分析

SNN 的"A 矩阵"是对角的，每个神经元的 β 就是特征值。当前 β ∈ [0.80, 0.99]，最大 L_eff = 100 帧 ≈ 8 token。

要支持 2048 token 上下文：β > 1 - 1/(2048×12) = 0.99996
要支持 8192 token 上下文：β > 0.99999

### 推理时按 rank 差异化偏移

```
rank_i = 按 β 从小到大排名 / 总数   ∈ [0, 1]
Δw_i = rank_i · Δw_max · log(L_target/L_train)
w_ext_i = w_i + Δw_i
β_ext_i = sigmoid(w_ext_i)
```

rank 高的（慢）神经元获得更大 Δw，rank 低的（快）几乎不动。

- ✓ 比方法一更精细（差异化 vs 全局）
- ✓ 保留 β 谱的相对结构
- ✗ 仍需联动 α 校准
- ✗ 推理时需要知道每个神经元的 rank（需预计算并缓存）

### 训练时的根本解决

扩展 b_β 初始化范围：

```
当前: beta_values = torch.linspace(0.80, 0.99, N)   // L_eff 5~100
扩展: beta_values = torch.linspace(0.80, 0.9999, N)  // L_eff 5~10000
```

让最后几组神经元成为真正的长程记忆通道。

**注意**：β 极接近 1 时，(1-β)·V_post 极小，输出信号微弱。需要 W_out 的对应列权重放大来补偿——当前初始化已有 `out_scale_per_n = 1/√target_p_fire` 的发放率均衡，但可能不够。

### 实施
- 推理时校准：修改 `snn_block.py` 初始化的 β 分布
- 训练时解决：修改 `_initialize_parameters` 中的 `beta_values` 范围
- 下一轮预训练时可直接改

---

## 7. 方法五：双通道状态凝练

### 来源
ReMamba (arXiv 2024) — 两次前向提纯隐状态

### SNN 的重要性度量

**不用 spike count**（V2 模式下 spike rate 不代表输出贡献）。

可选度量：
- `||(1-β)·V_post[t]||_2`：输出 activation 幅度
- PonderNet halt 分布的熵 H(t)：高熵 = 复杂 = 重要

### 两次前向设计

```
Pass 1 (轻量, ~10% FLOPs):
  只运行 PLIFNode 前向（不含 SNNBlock 6 条投影）
  记录 importance[t] = ||(1-β)·V_post_input[t]||_2

  top_k = topk(importance, T // compression_ratio)

Pass 2 (完整前向):
  reset all V to 0
  时间压缩：将 top_k token 连续排列
  for t in sorted(top_k):
    run full forward with original x[t]
    V 只被重要 token 塑造

  处理 query tokens → 生成输出
```

### 时间压缩的合理性

Pass 2 把稀疏的重要 token 连续排列，token 间衰减从 β^gap 变为 β^1。

- ✓ 减少长间隔造成的记忆损失
- ✓ SelectivePLIF 的 β(t) 基于 token 内容计算，不依赖位置，语义正确
- ✗ 改变了 token 间的时间关系（原本相距 400 步的 token 现在相邻）
- ✗ 对于依赖位置的模式可能有影响（但 SNN 无位置编码，影响较小）

### 实施
- 推理时新增模式，不影响训练
- 需要修改 `generate` 函数
- 适用于长文本检索/QA，不适用于生成

---

## 8. 实施路线

### 阶段一：免训练推理校准（预训练完成后立即可做）

| 方法 | 改动 | 效果预期 |
|---|---|---|
| β 偏置校准 (§3) | b_β += Δb, b_α 联动 | 4-8× 外推 |
| β 谱拉伸 (§6 推理部分) | 按 rank 差异化 Δw | 比全局偏移更稳定 |

### 阶段二：下一轮训练时内置（改初始化 + 架构微调）

| 方法 | 改动 | 效果预期 |
|---|---|---|
| b_β 初始化扩展 (§6 训练部分) | beta_values linspace 扩到 0.9999 | 原生长程记忆通道 |
| gate 复用输入门控 (§4) | forward_parallel 修改 | 全局通道记忆保护 |
| reset 系数 γ 可学习 (§4) | V_post = V_pre - γ·V_th·s | 自适应记忆擦除 |

### 阶段三：推理优化（评估后按需实施）

| 方法 | 改动 | 效果预期 |
|---|---|---|
| 简化路径跳过 (§5) | 推理时低 norm token 用 fast path | 速度提升，隐式记忆保护 |
| 双通道凝练 (§7) | 两次前向 | 超长文本检索/QA |

---

## 参考文献

1. MambaExtend — *Training-Free Approach to Improve Long Context Extension of Mamba* (ICLR 2025)
2. LongMamba — *Enhancing Mamba's Long Context Capabilities via Training-Free Receptive Field Enlargement* (ICLR 2025)
3. DeciMamba — *Exploring the Length Extrapolation Potential of Mamba* (ICLR 2025)
4. Mamba Modulation — *On the Length Generalization of Mamba* (arXiv 2024)
5. ReMamba — *Equip Mamba with Effective Long-Sequence Modeling* (arXiv 2024)
