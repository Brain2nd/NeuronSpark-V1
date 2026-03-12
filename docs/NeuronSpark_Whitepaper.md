# NeuronSpark: 基于脉冲神经网络的大规模语言模型

## 工程算法白皮书 v2.0

---

## 目录

- [1. 摘要](#1-摘要)
- [2. 引言与动机](#2-引言与动机)
- [3. 整体架构](#3-整体架构)
- [4. 核心神经元模型](#4-核心神经元模型)
- [5. SNN 解码层](#5-snn-解码层)
- [6. PonderNet 动态时间步聚合](#6-pondernet-动态时间步聚合)
- [7. 训练稳定性机制](#7-训练稳定性机制)
- [8. Triton 并行扫描内核](#8-triton-并行扫描内核)
- [9. 训练基础设施](#9-训练基础设施)
- [10. 训练健康监控](#10-训练健康监控)
- [11. 推理与生成](#11-推理与生成)
- [12. 数据管线](#12-数据管线)
- [13. 实验与诊断体系](#13-实验与诊断体系)
- [附录 A: 超参数总表](#附录-a-超参数总表)
- [附录 B: 张量形状流转表](#附录-b-张量形状流转表)
- [附录 C: 文件结构](#附录-c-文件结构)
- [附录 D: 稀疏性分析与 ASIC 适配](#附录-d-稀疏性分析与-asic-适配)

---

## 1. 摘要

NeuronSpark 是一个 **100% 脉冲神经网络 (SNN) 语言模型**，当前版本含 447M 可训练参数（D=640, 20 层）。模型内部不含任何传统 ANN 组件——没有 Transformer 自注意力，没有标准 MLP，所有计算单元均为 PLIF (Parametric Leaky Integrate-and-Fire) 脉冲神经元。

核心创新为 **脉冲电流激活 (Spike Current Activation)**：神经元输出 $V_{th} \times \text{spike}$（稀疏值域 $\{0, V_{th}\}$），经线性投影后汇入连续残差流。这一机制使模型在保持 SNN 脉冲语义的同时，通过代理梯度 (Surrogate Gradient) 实现端到端可微训练，解决了传统 SNN 深层梯度消失问题。

模型采用 **PonderNet 动态时间步聚合**：每个 token 被展开为 $K$ 个 SNN 时间步，通过学习的几何分布停止概率自适应加权聚合，使简单 token 用少量步数、复杂 token 用更多步数，实现弹性计算资源分配。

---

## 2. 引言与动机

### 2.1 为什么用 SNN 做语言模型

传统 Transformer 语言模型的核心操作是全连接自注意力 (Self-Attention)，计算复杂度为 $O(n^2 d)$，且每层所有神经元均参与计算（稠密激活）。生物大脑则通过脉冲 (Spike) 进行稀疏事件驱动计算，能耗远低于稠密矩阵乘法。

NeuronSpark 探索一条完全不同的路径：

1. **脉冲驱动**：神经元仅在膜电位超过阈值时发放脉冲，输出稀疏
2. **时间动力学**：信息编码在脉冲时序中，而非静态激活值
3. **动态计算深度**：PonderNet 机制使不同 token 使用不同步数
4. **生物可解释性**：每个组件对应明确的神经科学概念（膜电位、阈值、衰减、突触）
5. **硬件适配潜力**：脉冲电流的稀疏性（实测发放率 4%~20%）可在 ASIC 上实现事件驱动计算

### 2.2 核心设计原则

**脉冲电流 + 连续残差流**：SNN 的关键挑战是二值脉冲的不可微性。NeuronSpark 的解决方案是让神经元输出**脉冲电流** $V_{th} \times \text{spike}$（而非原始脉冲），通过线性投影汇入连续值残差流。这样：

- 前向传播保持脉冲语义：输出值域为 $\{0, V_{th}\}$，稀疏且离散
- 反向传播通过代理梯度获得稠密梯度：$\frac{\partial \text{spike}}{\partial V_{pre}} \approx \alpha \sigma(\alpha x)(1 - \sigma(\alpha x))$
- 层间传递连续值 $h$（残差流），避免二值信号的深层衰减

**信号流分离**：残差流 $h$ 是连续值，在层间传递；脉冲电流是稀疏值，仅在每层内部（神经元输出 → 投影 → PonderNet 聚合）存在。两者通过 out_proj 投影和残差加法衔接。

---

## 3. 整体架构

### 3.1 三阶段前向传播

```
token_ids (batch, seq_len)
    │
    ▼ ── encode() ──────────────────────────────────────────────
    │  Embedding lookup: (batch, seq_len) → (batch, seq_len, D)
    │  Repeat K frames:  (batch, seq_len, D) → (TK, batch, D)
    │  其中 TK = seq_len × K
    ▼
    │ ── snn_forward() ─────────────────────────────────────────
    │  20 × SNNDecoderLayer (gradient checkpointing):
    │    ┌─ [SubLN Gain Clamp]
    │    ├─ [MPD-AGL: 自适应 surrogate α]
    │    │
    │    ├─ RMSNorm → PLIF → spike_current → SNNBlock
    │    │  → PonderNet K聚合 → out_proj → SubLN PostNorm → 残差
    │    │
    │    └─ RMSNorm → PLIF → spike_current → SNNFFN
    │       → PonderNet K聚合 → out_proj → SubLN PostNorm → 残差
    │
    │  输出: h_out (TK, batch, D), ponder_cost, ek_floor_cost
    ▼
    │ ── decode() ──────────────────────────────────────────────
    │  RMSNorm → output PLIF → spike_current
    │  → K帧均值聚合 → decode_proj → LateralInhibition
    │  → Embedding^T (权重绑定) → logits
    ▼
logits (batch, seq_len, vocab_size)
```

### 3.2 与 Transformer 的结构对标

| Transformer (Qwen3) | NeuronSpark SNN | 对应关系 |
|---|---|---|
| RMSNorm | RMSNorm | 完全相同（PyTorch 实现） |
| Self-Attention (QKV) | SNNBlock (6路投影 + SelectivePLIF + gate/skip) | 注意力 → 脉冲时间动力学 |
| SwiGLU MLP | SNNFFN (gate/up PLIF + down + skip) | SiLU → 脉冲电流门控 |
| 残差连接 | 残差连接 | 完全相同 |
| 无 | PonderNet 动态K聚合 | SNN 特有：时间步加权 |
| 无 | SubLN PostNorm (含 gain clamp) | SNN 特有：深层梯度控制 |
| Softmax(QK^T/√d)V | PLIF 膜电位积分 + 脉冲发放 | 注意力机制 → 神经元动力学 |

### 3.3 参数配置

| 参数 | 值 | 说明 |
|---|---|---|
| `vocab_size` | 6144 | 自训练 BPE 分词器 |
| `D` | 640 | 隐藏维度 |
| `N` | 8 | 状态扩展因子，隐神经元维度 = D×N = 5120 |
| `K` | 32 | 最大 SNN 时间步（PonderNet 动态） |
| `num_layers` | 20 | 解码层数 |
| `D_ff` | 1920 | FFN 中间层维度（3×D） |
| `seq_max_length` | 512 | 最大序列长度 |
| 总参数量 | ~447M | 全部可训练 |

---

## 4. 核心神经元模型

### 4.1 PLIF 神经元 (Parametric Leaky Integrate-and-Fire)

> 源码位置: `atomic_ops/plif_node.py`

PLIF 是模型的基本计算单元，每个维度拥有独立可学习的衰减率 $\beta$ 和阈值 $V_{th}$。

**动力学方程：**

$$V_{pre}[t] = \beta \cdot V_{post}[t-1] + (1 - \beta) \cdot x[t]$$

$$s[t] = \Theta(V_{pre}[t] - V_{th})$$

$$V_{post}[t] = V_{pre}[t] - V_{th} \cdot s[t] \quad \text{(软重置)}$$

其中：
- $\beta = \sigma(w) \in (0, 1)$，$w$ 为可学习权重（D 维）
- $V_{th}$ 为可学习阈值（D 维，每维独立）
- $\Theta(\cdot)$ 为 Heaviside 阶跃函数
- 软重置：发放后膜电位减去 $V_{th}$（而非清零），保留超阈值部分

**代理梯度 (Surrogate Gradient)：**

$$\frac{\partial s}{\partial V_{pre}} \approx \alpha \cdot \sigma(\alpha \cdot (V_{pre} - V_{th})) \cdot (1 - \sigma(\alpha \cdot (V_{pre} - V_{th})))$$

$\alpha$ 由 MPD-AGL 机制动态设定（见 §7.3），不再使用固定值。

**参数初始化：**

```python
# 衰减率 β：多时间尺度
init_w = -log(τ - 1)          # τ = 2.0 → w ≈ 0
w ~ N(init_w, 0.5)            # 覆盖快衰减和慢衰减

# 阈值 V_th：均匀扰动
v_th ~ U[0.5·v_threshold, 1.5·v_threshold]
```

**PLIF 在模型中的角色：**

| 位置 | 数量 | 维度 | V_th 默认值 | 功能 |
|---|---|---|---|---|
| 输入神经元 (input_neuron1/2) | 20层 × 2 = 40 | D=640 | 0.5 | Pre-LN 后脉冲化，实测发放率 ~20% |
| FFN gate/up 神经元 | 20层 × 2 = 40 | D_ff=1920 | 0.15 | 脉冲门控，实测发放率 ~5% |
| 输出神经元 | 1 | D=640 | 0.3 | 解码输出 |

### 4.2 SelectivePLIF 神经元 (动态参数 PLIF)

> 源码位置: `atomic_ops/selective_plif.py`

SelectivePLIF 是 SNNBlock 的核心，其参数（$\beta$、$\alpha$、$V_{th}$）**不是可学习参数**，而是由输入动态调制：

**动力学方程：**

$$V[t] = \beta(t) \cdot V[t-1] + \alpha(t) \cdot I[t]$$

$$s[t] = \Theta(V[t] - V_{th}(t))$$

$$V[t] \leftarrow V[t] - V_{th}(t) \cdot s[t]$$

**参数调制（由 SNNBlock 投影计算）：**

$$\beta(t) = \sigma(W_\beta \cdot x + b_\beta) \in (0, 1) \quad \text{(衰减率)}$$

$$\alpha(t) = \text{softplus}(W_\alpha \cdot x + b_\alpha) > 0 \quad \text{(输入增益)}$$

$$V_{th}(t) = |W_{th} \cdot x + b_{th}| + V_{th,min} > 0 \quad \text{(动态阈值)}$$

与固定参数 PLIF 的关键区别：
- 每个时间步的动力学参数由当前输入决定，实现**输入依赖的神经调制**
- $\alpha(t)$ 控制输入增益，替代固定 $(1-\beta)$ 系数
- $V_{th}(t)$ 动态阈值使神经元对不同输入有不同发放敏感度
- 实测发放率 ~4%（稀疏度高于固定参数 PLIF）

### 4.3 脉冲电流激活 (Spike Current Activation)

> 源码位置: `atomic_ops/__init__.py`

脉冲电流是连接离散 SNN 与连续残差流的桥梁：

**前向：**

$$\text{spike\_current} = V_{th} \odot \text{spike}$$

值域为 $\{0, V_{th}\}$，保持稀疏性。

**反向（自定义 autograd.Function）：**

$$\frac{\partial L}{\partial \text{spike}} = \frac{\partial L}{\partial \text{sc}} \cdot (V_{th} - \varepsilon)$$

$$\frac{\partial L}{\partial V_{th}} = \frac{\partial L}{\partial \text{sc}} \cdot \text{spike}$$

其中 $\varepsilon = 10^{-6}$ 防止与代理梯度中的 $V_{th}$ 项冲突。

**显存优化**：spike ∈ {0, 1} 在 forward 中存为 uint8（1 字节/元素），backward 时 cast 回计算 dtype，节省 50% spike 显存。

---

## 5. SNN 解码层

### 5.1 SNNBlock (自注意力等价层)

> 源码位置: `atomic_ops/snn_block.py`

SNNBlock 替代 Transformer 的自注意力层。核心思想：用 PLIF 神经元的**时间积分动力学**替代显式的注意力矩阵计算。

**信号流：**

```
spike_in (TK, batch, D)
    │
    ├── W_in:    D → D×N  ─────────── I[t] (主输入电流)
    ├── W_β:     D → D×N  → σ(·+b_β) ── β(t) (衰减调制)
    ├── W_α:     D → D×N  → softplus(·+b_α) ── α(t) (增益调制)
    ├── W_th:    D → D×N  → |·+b_th|+min ── V_th(t) (阈值调制)
    ├── W_gate:  D → D    → σ(·) ──── gate[t] (门控)
    └── W_skip:  D → D    ─────────── I_skip[t] (跳跃连接)
    │
    ▼
SelectivePLIF: V[t] = β(t)·V[t-1] + α(t)·I[t]
               s[t] = Θ(V[t] - V_th(t))
               V[t] -= V_th(t)·s[t]
    │
    ▼ spike_current = V_th(t) × s[t]
    │
    ▼ W_out: D×N → D
    │
    ▼ output = W_out(sc_hidden) ⊙ gate + I_skip
```

**显存优化**：6 条 D→DN 投影中，β、α、V_th 采用逐个计算 + 原地激活（`sigmoid_()`、`abs_()`），避免同时存在 6 个 (TK, batch, DN) 张量。峰值 DN 张量数从 8 降至 3。

**权重初始化细节：**

1. **$\beta$ 偏置初始化**（多时间尺度记忆）：
   $$\beta_{target} \in [0.80, 0.99] \quad (\text{N=8 个等间距时间尺度})$$
   $$b_\beta = \text{logit}(\beta_{target}) + \mathcal{N}(0, 0.1)$$

2. **$\alpha$ 偏置初始化**（初始增益 ≈ 1.0）：
   $$b_\alpha \sim \mathcal{N}(0.5413, 0.1) \quad (\text{softplus}(0.5413) \approx 1.0)$$

3. **$V_{th}$ 偏置初始化**（基于目标发放率校准）：
   $$\sigma_V = \sqrt{p/3} \cdot \sqrt{1 - \beta^{2K}}$$
   其中 $p=0.15$（保守估计输入发放率），$K=16$ 为参考步数。
   $$\text{target\_p\_fire} = \text{linspace}(0.25, 0.08, N)$$
   $$z = \sqrt{2} \cdot \text{erfinv}(2(1 - p_{fire}) - 1)$$
   $$b_{th} = \text{clamp}(\sigma_V \cdot z - V_{th,min}, \; \text{min}=0.05) + \mathcal{N}(0, 0.02)$$

4. **W_in 时间尺度缩放**：`W_in *= sqrt(1 - β²)`，使不同时间尺度的通道有匹配的初始驱动强度。

5. **W_out 发放率均衡缩放**：低发放率通道的输出权重放大，高发放率通道缩小，均衡各通道对输出的贡献。

### 5.2 SNNFFN (SwiGLU 等价层)

> 源码位置: `atomic_ops/snn_ffn.py`

SNNFFN 对标 Transformer 的 SwiGLU MLP，用脉冲电流门控替代 SiLU 激活：

**Transformer SwiGLU：**

$$\text{output} = W_{down}(\text{SiLU}(W_{gate} \cdot x) \odot W_{up} \cdot x)$$

**SNN 等价：**

$$\text{sc}_{gate} = V_{th} \cdot \text{PLIF}(W_{gate} \cdot x) \quad \text{(脉冲电流替代 SiLU)}$$

$$\text{sc}_{up} = V_{th} \cdot \text{PLIF}(W_{up} \cdot x)$$

$$\text{gated} = \text{sc}_{gate} \odot \text{sc}_{up}$$

$$\text{output} = W_{down} \cdot \text{gated} + W_{skip} \cdot x$$

**gated 的极端稀疏性**：gate 和 up 两路脉冲电流相乘后，稀疏度叠加。实测 gate 发放率 ~5%、up 发放率 ~5%，gated 有效密度仅 ~0.2%（独立发放假设下 5%×5%=0.25%）。这是 ASIC 硬件加速的关键机会——`down_proj(gated)` 的输入 99.8% 为零。

**计算优化**：gate 和 up 投影合并为单次矩阵乘法，两组 PLIF 神经元合并为单次 row-param 并行扫描：

```python
W_gate_up = torch.cat([gate_proj.weight, up_proj.weight], dim=0)  # (2D_ff, D)
I_gate_up = F.linear(flat, W_gate_up)  # 1 matmul instead of 2
# Row-param PLIF scan: (TK, batch, 2×D_ff) neurons in one kernel call
spike_merged, V_post = plif_rowparam_forward(beta_row, u_merged, vth_row, ...)
```

### 5.3 SNNDecoderLayer 完整结构

> 源码位置: `atomic_ops/snn_decoder_layer.py`

```
h (TK, batch, D) ─ 连续残差流
│
├─── SubLN Gain Clamp: block/ffn_post_norm.weight ∈ [gain_min, gain_max]
├─── MPD-AGL: 自适应设置 4 类神经元的 surrogate α
│
├─── 子层 1: SNNBlock ───────────────────────────────────────
│  h_norm = RMSNorm(h)                              # Pre-LN
│  sc = input_neuron1.parallel_scan(h_norm)          # PLIF → spike_current
│  [记录 _fr_input1 发放率]
│  cont = snn_block.forward_parallel(sc)             # SNNBlock 处理
│  [记录 _firing_rate_hidden 发放率]
│  frames = cont.view(seq_len, K, batch, D)          # 分 K 帧
│  combined, pc, ek, efc = PonderNet(frames)         # 动态 K 聚合
│  res = block_out_proj(combined)                    # 输出投影 (seq, batch, D)
│  res = block_post_norm(res)                        # SubLN PostNorm
│  h = h + res.broadcast_to(TK, batch, D)            # 残差
│
├─── 子层 2: SNNFFN ─────────────────────────────────────────
│  h_norm = RMSNorm(h)
│  sc = input_neuron2.parallel_scan(h_norm)
│  [记录 _fr_input2 发放率]
│  cont = snn_ffn.forward_parallel(sc)
│  [记录 _fr_gate, _fr_up, _fr_gated 发放率]
│  frames = cont.view(seq_len, K, batch, D)
│  combined, pc, ek, efc = PonderNet(frames)
│  res = ffn_out_proj(combined)
│  res = ffn_post_norm(res)
│  h = h + res.broadcast_to(TK, batch, D)
│
▼
h (TK, batch, D), ponder_cost, ek_floor_cost
```

---

## 6. PonderNet 动态时间步聚合

### 6.1 数学框架

SNN 的每个 token 被展开为 $K$ 个时间步。传统做法是均匀平均 K 帧输出，但这浪费了简单 token 的计算资源。PonderNet 引入学习的停止概率，使不同 token 有不同的有效计算步数。

**停止概率：**

$$p_k = \sigma(\text{halt\_proj}(\text{frame}_k)) \in (0, 1)$$

**生存概率（到第 $k$ 步还未停止）：**

$$S_k = \prod_{j=1}^{k-1}(1 - p_j)$$

**几何分布权重（恰好在第 $k$ 步停止）：**

$$\lambda_k = p_k \cdot S_k$$

**归一化权重：**

$$\hat{\lambda}_k = \frac{\lambda_k}{\sum_{j=1}^{K} \lambda_j}$$

**加权聚合：**

$$\text{output} = \sum_{k=1}^{K} \hat{\lambda}_k \cdot \text{frame}_k$$

**期望步数：**

$$E[K] = \sum_{k=1}^{K} k \cdot \hat{\lambda}_k$$

### 6.2 正则化项

**Ponder Cost**（鼓励用更少步数）：

$$\mathcal{L}_{ponder} = \text{mean}(E[K])$$

**E[K] Floor Cost**（防止坍缩到 $K=1$）：

$$\mathcal{L}_{floor} = \text{mean}(\text{ReLU}(\text{floor} - E[K])^2)$$

当 $E[K]$ 低于下界 `ek_floor`（默认 4.0）时产生二次惩罚，梯度流回 halt 参数使其降低停止概率。

### 6.3 Halt 参数初始化

```python
halt_proj.weight: Xavier uniform × 0.01      # 小权重
halt_proj.bias = -3.5                         # σ(-3.5) ≈ 0.029

# 效果：初始 p_halt ≈ 0.03 → 几何分布接近均匀
# λ_1/λ_K ≈ 1.5，所有帧近似等权
```

### 6.4 融合几何分布计算

> 源码位置: `atomic_ops/snn_decoder_layer.py`，`_fused_geometric_halt()`

将 sigmoid、clamp、log1p、cumsum、exp、normalize 6 个 PyTorch 操作封装为单函数。使用 log 空间数值稳定实现：

```python
def _fused_geometric_halt(halt_logits):
    halt_logits = halt_logits.clamp(-6.0, 6.0)  # bf16 安全
    p_halt = torch.sigmoid(halt_logits)
    log_1_minus_p = torch.log1p(-p_halt)
    # Exclusive cumsum: log_survive[:, k] = Σ_{j<k} log(1-p_j)
    log_survive = torch.zeros_like(log_1_minus_p)
    log_survive[:, 1:, :] = torch.cumsum(log_1_minus_p[:, :-1, :], dim=1)
    survive = torch.exp(log_survive)
    halt_weights = p_halt * survive
    halt_weights = halt_weights / (halt_weights.sum(dim=1, keepdim=True) + 1e-8)
    return halt_weights
```

**bf16 数值安全**：bf16 尾数仅 7 位，`sigmoid(6.3+)` 舍入为 1.0 → `log1p(-1.0) = -inf`。因此 clamp logits 到 $[-6, 6]$（`sigmoid(6) = 0.99609375` 在 bf16 下安全）。

**注意**：此函数不使用 `@torch.compile` 装饰器，因为 compile 会绕过 gradient checkpoint 的 pack/unpack hooks，导致每层泄漏 ~1 GB 显存。

---

## 7. 训练稳定性机制

### 7.1 问题诊断：深层梯度消失

20 层 SNN 的梯度比 $\|g_{L00}\| / \|g_{L19}\|$ 初始可达 12,530×，意味着浅层梯度是深层的万倍以上。

基于 57k 步训练日志 (fsdp_1faa7df) 的根因分析识别了两条问题链：

**问题链 A: SubLN gain 正反馈**
Pre-LN 模式下，浅层 $\|W_{out}\|$ 增长在反向传播中放大梯度 → 浅层更新更快 → $\|W_{out}\|$ 进一步增长 → 正反馈。L19 的 SubLN gain 增长至 L00 的数倍，加剧深层梯度饥饿。

**问题链 B: b_th 漂移 → MPD alpha 崩溃**
$b_{th}$ 在 10×LR + 0 weight_decay 下无约束增长（+207%）→ $V_{th}$ 持续漂移 → MPD-AGL 公式中 $\alpha = C / (\cdots \times V_{th})$ 被压至下限 → surrogate 有效区间过窄 → 深层梯度进一步消失。

### 7.2 SubLN Post-RMSNorm（含 Gain Clamp）

> 对应代码: `snn_decoder_layer.py`, `block_post_norm` / `ffn_post_norm`

在子层输出（out_proj 之后、残差之前）插入 RMSNorm，增益初始化为：

$$\text{gain} = (2 \cdot L)^{-1/2}$$

其中 $L$ 为总层数（20 层时 gain ≈ 0.158）。

**原理**：Jacobian $J_k \propto \text{gain} / \text{RMS}(f_l)$，当 $\|W_{out}\|$ 增长时，$\text{RMS}(f_l)$ 同步增长，自动抵消，实现增益控制。

**Gain Clamp（v2 新增）**：训练过程中 gain 可能偏离初始值过远，导致问题链 A 的正反馈。每次前向传播前将 gain 钳位至 $[\text{gain}_{init} \times 0.25, \; \text{gain}_{init} \times 3.0]$。

```python
with torch.no_grad():
    self.block_post_norm.weight.data.clamp_(self._gain_min, self._gain_max)
    self.ffn_post_norm.weight.data.clamp_(self._gain_min, self._gain_max)
```

### 7.3 MPD-AGL (膜电位分布自适应代理梯度)

> 对应代码: `snn_decoder_layer.py`, `_mpd_alpha()`

代理梯度 $sg(x) = \alpha \cdot \sigma(\alpha x) \cdot (1 - \sigma(\alpha x))$ 的有效区间 $\approx 2.2/\alpha$。若膜电位分布偏离此区间，绝大部分神经元的代理梯度 $\approx 0$，造成梯度消失。

**自适应公式：**

$$\alpha = \frac{C}{\sqrt{1 + \beta^2} \cdot \gamma \cdot V_{th}}$$

其中：
- $C = 4.0 \cdot \sqrt{1.25} \cdot 0.5 \approx 2.236$
- $\beta$：当前层 PLIF 衰减率均值
- $\gamma$：Pre-LN RMSNorm 权重绝对值均值（输入 scale）
- $V_{th}$：当前层阈值均值
- 钳位范围：$[2.0, 16.0]$（v2 将下界从 1.0 提升至 2.0，防止有效区间过窄）

**物理直觉**：
- $\beta$ 增大（强积分）→ 膜电位分布更宽 → $\alpha$ 减小，扩大有效区间
- $\gamma$ 增大（输入 scale 增大）→ 膜电位偏移更大 → $\alpha$ 减小
- $V_{th}$ 增大 → 膜电位离阈值更远 → $\alpha$ 减小

**逐层逐神经元类型设置**：每次前向传播前，根据当前参数值动态计算 $\alpha$，分别为 input1、hidden、input2、FFN 四类神经元独立设置。初始值约为：input ~4.0, hidden ~9.0, FFN ~13.3。

### 7.4 b_th L2 正则化（v2 新增）

> 对应代码: `model.py`, `_compute_b_th_reg_cost()`

$$\mathcal{L}_{b_{th}} = \frac{1}{L} \sum_{l=1}^{L} \text{mean}(b_{th}^{(l)2})$$

$b_{th}$ 在 neuron 参数组（10×LR, 0 weight_decay）下无约束增长，L2 正则化提供软回复力，防止 $V_{th}$ 远离初始校准分布 → 遏制问题链 B。

### 7.5 SNVR (层间权重范数方差正则化)

> 对应代码: `model.py`, `_compute_snvr_cost()`

$$\mathcal{L}_{SNVR} = \sum_{\text{type}} \text{Var}(\{\|W_{\text{type}}^{(l)}\|_F\}_{l=1}^{L})$$

对每种权重类型（W_in、W_out、W_gate、W_skip、ffn_gate、ffn_down、out_proj、ffn_out_proj），计算其在各层的 Frobenius 范数的方差。

**原理**：Jacobian 乘积 $\prod_l \|J_l\|$ 在层间范数不一致时指数级发散或收缩。SNVR 约束各层权重范数一致，使 Jacobian 乘积稳定。

### 7.6 Natural Gradient 补偿

> 对应代码: `model.py`, `compensate_modulation_gradients()`

sigmoid 和 softplus 在饱和区导数趋近 0，导致 $b_\beta$、$b_\alpha$ 等参数的有效学习率被压缩。

**补偿公式：**

$$g_{b_\beta} \leftarrow \frac{g_{b_\beta}}{\beta(1-\beta)} \quad \text{(sigmoid 导数的倒数)}$$

$$g_{b_\alpha} \leftarrow \frac{g_{b_\alpha}}{\sigma(b_\alpha)} \quad \text{(softplus 导数 = sigmoid)}$$

钳位分母防止除零：$\beta(1-\beta) \geq 0.01$，$\sigma(b_\alpha) \geq 0.1$。

$b_{th}$ 通过 $|\cdot|$ 变换为 $V_{th}$，导数为 $\pm 1$，无饱和问题，不需要补偿。

### 7.7 损失函数总览

$$\mathcal{L} = \mathcal{L}_{CE} + \lambda_{ponder} \cdot E[K] + \lambda_{floor} \cdot \mathcal{L}_{floor} + \lambda_{SNVR} \cdot \mathcal{L}_{SNVR} + \lambda_{b_{th}} \cdot \mathcal{L}_{b_{th}}$$

| 损失项 | 默认权重 | 功能 |
|---|---|---|
| $\mathcal{L}_{CE}$ | 1.0 | 交叉熵语言建模损失 |
| $E[K]$ | 0.01 | 鼓励减少不必要的计算步数 |
| $\mathcal{L}_{floor}$ | 0.1 | 防止 PonderNet 坍缩到 K=1 |
| $\mathcal{L}_{SNVR}$ | 0.01 | 层间权重范数方差一致性 |
| $\mathcal{L}_{b_{th}}$ | 0.01 | 遏制 b_th 漂移 → V_th 漂移 → MPD alpha 崩溃 |

---

## 8. Triton 并行扫描内核

### 8.1 PLIF 线性递归的并行求解

> 源码位置: `atomic_ops/parallel_scan.py`

PLIF 的核心递归 $V[k] = \beta \cdot V[k-1] + u[k]$ 是一个线性递归，可用前缀扫描 (Prefix Scan) 并行求解。

**三层后端：**

| 后端 | 使用条件 | 复杂度 | 实现 |
|---|---|---|---|
| Fused PLIF Triton kernel | CUDA + Sigmoid surrogate（默认） | O(K) 顺序扫描 | 单 kernel 融合 scan+spike+reset |
| Triton linear_recurrence | CUDA，非 Sigmoid 或无 surrogate | O(K) 列级并行 | 通用线性递归 |
| Hillis-Steele scan | CPU 回退 | O(K log K) | PyTorch 实现 |

### 8.2 Triton 融合 PLIF 内核

$K \leq 32$ 时，顺序扫描的寄存器效率高于 Hillis-Steele 并行扫描，因此实际使用顺序扫描 + 融合操作。

**前向内核 `_fused_plif_fwd_kernel`：**

```
每个 program 处理 (batch_idx, dim_idx) 的 K 步：
for k in range(K):
    V_pre = β[k] × V_post_prev + u[k]
    spike = (V_pre > V_th[k]) as uint8      # 节省 50% 存储
    V_post = V_pre - V_th[k] × spike        # 软重置
    store spike[k], V_post[k]

所有 K 步在寄存器中完成，一次 kernel launch。
```

**反向内核 `_fused_plif_bwd_kernel`：**

```
反向累积：从 k=K-1 到 k=0
for k in range(K-1, -1, -1):
    重算 V_pre = β[k] × V_post[k-1] + u[k]
    sg = α × σ(α(V_pre - V_th[k])) × (1 - σ(α(V_pre - V_th[k])))

    # 代理梯度 α 从 surrogate_function.alpha 读取（MPD-AGL 动态设定）
    grad_u[k], grad_V_prev 按链式法则计算
    累积 grad_β, grad_V_th
```

### 8.3 Row-Parameter 优化

> `_fused_plif_fwd_rowparam_kernel` / `_fused_plif_bwd_rowparam_kernel`

当 $\beta$、$V_{th}$ 在 $K$ 步内不变（PLIFNode 情况），将它们作为 **row parameter** 仅加载一次到寄存器：

```
# 标准 kernel：每步加载 β[k], V_th[k], u[k] → 3 次内存读取/步
# Row-param kernel：β, V_th 一次加载，每步仅读 u[k] → 1 次内存读取/步
# 节省约 40% 内存带宽
```

适用于：input_neuron1/2, output_neuron, SNNFFN gate/up neuron（均为固定参数 PLIFNode）。

### 8.4 LateralInhibition Triton 融合

> 源码位置: `atomic_ops/lateral_inhibition.py`

LateralInhibition（数学等价于 RMSNorm + 可学习 gain）使用 Triton 融合内核加速：

```
PyTorch 路径：cast_fp32 → pow(2) → mean → rsqrt → mul(gain)  [5 kernel launch]
Triton 路径：  _li_fwd_kernel(x, gain, eps)                    [1 kernel launch]
```

前向+反向各 1 次 kernel launch，反向内核重算前向中间值（避免保存 rrms 到 ctx），节省显存。

**注意**：Pre-LN 分支中的 `RMSNorm` 和 SubLN `PostNorm` 均使用 PyTorch 原生实现（无 Triton），仅输出端的 `LateralInhibition` 有 Triton 加速。

---

## 9. 训练基础设施

### 9.1 优化器配置

采用 AdamW，3 组参数分别设置学习率和权重衰减：

| 参数组 | 学习率 | Weight Decay | 包含参数 |
|---|---|---|---|
| 标准衰减 | $2 \times 10^{-4}$ | 0.1 | 投影权重 (W_in, W_out, W_gate, W_skip, ffn_gate/up/down/skip, out_proj) |
| 无衰减 | $2 \times 10^{-4}$ | 0.0 | RMSNorm gain, Embedding, halt_proj |
| 神经元 ×10 | $2 \times 10^{-3}$ | 0.0 | PLIF w, V_th, b_beta, b_alpha, b_th |

**学习率调度：**

$$\text{lr}(t) = \text{lr}_{min} + \frac{1}{2}(1 + \cos(\pi \cdot r)) \cdot (\text{lr}_0 - \text{lr}_{min})$$

其中 $r = \frac{t - t_{warmup}}{t_{total} - t_{warmup}}$，warmup 阶段线性增长至 $\text{lr}_0$。

| 参数 | 值 |
|---|---|
| 基础学习率 | $2 \times 10^{-4}$ |
| 最小学习率 | $2 \times 10^{-5}$ |
| Warmup 步数 | 500 |
| 梯度裁剪 | 1.0 |

### 9.2 分布式训练 (FSDP)

> 源码位置: `train_fsdp.py`

采用 PyTorch FSDP (Fully Sharded Data Parallel)，按 `SNNDecoderLayer` 为单位分片：

```python
auto_wrap_policy = transformer_auto_wrap_policy(
    transformer_layer_cls={SNNDecoderLayer}
)
```

**混合精度：**
- 计算精度：bfloat16
- 归约精度：float32（AllReduce 前）
- 参数精度：bfloat16

**梯度累积优化：**

```python
is_boundary = (step + 1) % accumulation_steps == 0
sync_ctx = nullcontext() if is_boundary else model.no_sync()
with sync_ctx:
    loss.backward()  # no_sync 期间跳过 AllReduce
```

**基准训练配置 (fsdp_1faa7df)：**

```bash
torchrun --nproc_per_node=4 --master_port=29700 train_fsdp.py \
  --D 640 --D_ff 1920 --num_layers 20 --K 32 \
  --batch_size 4 --accumulation_steps 1 --log_interval 10 \
  --dashboard_dir runs/fsdp_1faa7df
# Effective batch size: 4/gpu × 4 gpus × accum 1 = 16
```

### 9.3 Checkpoint 管理

```python
# 自动保存
save_interval = 500 steps

# Checkpoint 内容 (FULL_STATE_DICT 模式，兼容单卡加载)
{
    'model_state_dict':  全部模型参数,
    'optimizer_state':   AdamW 完整状态 (m, v, step),
    'step':              当前步数,
    'epoch':             当前 epoch,
    'best_loss':         历史最优损失,
    'tokens_seen':       已处理 token 总数,
    'model_config':      模型超参数 (用于重建),
}
```

FSDP checkpoint 通过 `FSDP.state_dict_type(model, FULL_STATE_DICT)` 收集完整参数到 rank 0 CPU 后保存，可直接用于单卡推理。

---

## 10. 训练健康监控

> 源码位置: `dashboard.py`

### 10.1 SNNDashboard 架构

基于 TensorBoard 的全方位 SNN 训练监控，仅 rank 0 启用记录（其余 rank 为 no-op），两级频率：

**每步日志 (log_interval=10 步)：**
- 训练标量：loss, ppl, lr, TPS, tokens_seen, ponder_cost, b_th_reg_cost
- 参数监控：weight_norm, grad_norm, update_ratio（每个参数）
- 神经元动力学：$\beta$, $\alpha$, $V_{th}$ 的语义值（sigmoid/softplus 变换后）
- PonderNet：E[K] min/max（逐层）, halt 权重 norm + 偏置
- SubLN：block/ffn post_norm gain（逐层）
- MPD-AGL：surrogate $\alpha$（逐层逐神经元类型）
- 发放率：input1/input2/hidden/gate/up/gated（逐层 6 类）
- 层间梯度健康：grad ratio (max/min), grad_max, grad_min
- 健康综合指标（见下）

**检查点日志 (save_interval=500 步)：**
- 权重/梯度直方图（全部可训练参数）
- Sigmoid/softplus 导数均值（诊断 Natural Gradient 补偿效果）

### 10.2 健康综合评分

`_log_health()` 方法计算综合健康指数 `health/score` ∈ [0, 1]（1=健康），由 4 个维度加权：

| 维度 | 权重 | 指标 | 健康条件 |
|---|---|---|---|
| SubLN gain 均衡 | 0.30 | gain_ratio (max/min), gain_gini | ratio→1 为佳，≥5 告警 |
| MPD alpha 分布 | 0.25 | mpd_floor_rate, mpd_alpha_min | 触底（α=2.0）比例低为佳 |
| 层间梯度均衡 | 0.25 | grad_gini, grad_share/layer_XX | gini→0 为均匀 |
| 神经元活性 | 0.20 | fr_dead (<0.1%), fr_saturated (>80%) | 无死/饱和神经元 |

**独立指标（方便定位问题根源）：**
- `health/gain_ratio`, `health/gain_gini`
- `health/mpd_floor_rate`, `health/mpd_alpha_min`
- `health/grad_gini`, `grad_share/layer_XX`（逐层梯度份额）
- `health/fr_dead_neurons`, `health/fr_saturated_neurons`, `health/fr_min`, `health/fr_max`

### 10.3 FSDP 梯度处理

多卡 FSDP 下 `summon_full_params` 只聚合参数，不聚合梯度。因此训练循环在 `optimizer.step()` 之前不调用 `cache_grad_norms`（目前未启用梯度缓存），在 `optimizer.step()` 之后通过 `summon_full_params(writeback=False, rank0_only=True)` 聚合参数后记录 log。

---

## 11. 推理与生成

### 11.1 标准自回归生成

> 源码位置: `model.py`, `generate()`

```
1. Prefill（并行处理 prompt）：
   prompt → encode → K帧 → 20层前向 → decode → logits
   采样首个新 token

2. 自回归循环：
   for each new token:
     encode(token) → K帧
     20层前向（神经元 V 状态跨 token 保持）
     decode → logits → 采样

   神经元膜电位 V 不重置，在 token 间持续演化。
   这使模型能通过 V 的时间动态编码跨 token 上下文。
```

**采样策略：**
- Temperature 缩放
- Top-k 过滤

### 11.2 自投机解码 (Self-Speculative Decoding)

> 源码位置: `model.py`, `generate_speculative()`

利用 SNN 的动态 K 特性实现**无需额外 draft 模型**的投机解码：

```
Phase 1: Draft（K_draft=4, 即 K_full 的 12.5%）
  ├── 保存所有神经元 V 状态检查点
  ├── 设置 model.K = 4
  └── 贪心生成 lookahead=5 个 draft token

Phase 2: Verify（K_full=32）
  ├── 恢复 V 状态到检查点
  ├── 设置 model.K = 32
  ├── 并行前向验证 [next_token, draft_0, ..., draft_k-1]
  ├── 比较 argmax(draft_logits) vs argmax(verify_logits)
  ├── 接受匹配前缀
  └── 从 verify_logits[n_accept] 采样 recovery token

Phase 3: V 状态重建
  ├── 全部接受: verify 的 V 状态已正确
  └── 部分接受: 恢复 V 状态，重跑已接受 tokens 重建正确 V 状态
```

**加速原理**：K=4 的 draft 仅消耗 K=32 的 12.5% 计算；并行验证 k+1 个 token 的开销 ≈ 1 个 token。若 draft 全部命中，k+1 个 token 的总计算远小于逐 token 生成。

**正确性保证**：贪心模式下 draft 与 verify 使用相同模型，argmax 确定性匹配，输出与标准 AR 完全一致。

---

## 12. 数据管线

### 12.1 预训练数据

**Seq-Monkey 数据处理 (`scripts/deal_dataset.py`)：**

```
输入: Seq-Monkey 长文本 JSONL
    {"text": "一段很长的中文文本..."}

处理: 按 512 字符切分
    split_text(text, chunk_size=512)

输出: 切分后的 JSONL
    {"text": "512字符块"}
```

**大规模数据预处理 (`scripts/prepare_data.py`)：**

```
输入: SkyPile-150B (620GB, 437 个 JSONL 文件)

处理: 多进程 tokenize → flat binary
    1. tokenizer.encode(text) → token_ids (uint32)
    2. 追加写入 train_tokens.bin
    3. 断点续传（记录已处理文件）

输出:
    data/processed/train_tokens.bin   # 训练集
    data/processed/val_tokens.bin     # 验证集 (2M tokens)
```

### 12.2 数据集类

**PretrainDataset (`dataset.py`)：**

```python
# 随机访问 via 字节偏移量索引
__init__: 扫描 JSONL 文件记录每行起始字节偏移 → self._offsets
__getitem__: seek(offset[i]) → 读一行 → tokenize → shift → mask

# 输出: X[:-1], Y[1:], loss_mask (padding=0)
```

**SFTDataset：**

```python
# ChatML 格式对话
# Loss mask: 仅在 assistant 回复 token 上计算损失
# <|im_start|>assistant\n ... <|im_end|> 区间标记为 1
```

### 12.3 分词器

自训练 BPE 分词器，vocab_size = 6144，训练脚本 `scripts/train_tokenizer.py`。

---

## 13. 实验与诊断体系

### 13.1 SNNDashboard 训练监控

见 [第 10 节](#10-训练健康监控)。

### 13.2 实验脚本体系

| 脚本 | 功能 |
|---|---|
| `exp/verify_halt_grad.py` | 验证几何分布函数梯度正确性 |
| `exp/bench_spark_opt.py` | 基准性能测试（forward+backward） |
| `exp/bench_compile_time.py` | Triton JIT / torch.compile 编译开销 |
| `exp/bench_max_tps.py` | 最大训练吞吐量测试 |
| `exp/estimate_max_scale.py` | 最大可训练模型规模评估 |
| `exp/estimate_full_mem.py` | 完整训练显存评估（含优化器+数据集） |

---

## 附录 A: 超参数总表

### A.1 模型架构

| 参数 | 符号 | 默认值 | 说明 |
|---|---|---|---|
| 隐藏维度 | D | 640 | 可见维度 |
| 状态扩展 | N | 8 | 隐神经元 = D×N = 5120 |
| 最大时间步 | K | 32 | PonderNet 动态 |
| 层数 | L | 20 | 解码层数 |
| FFN 维度 | D_ff | 1920 | 3×D |
| 词表大小 | V | 6144 | BPE |
| 序列长度 | S | 512 | 最大输入长度 |

### A.2 神经元参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| input_neuron init_tau | 2.0 | 初始时间常数 |
| input_neuron v_threshold | 0.5 | 输入神经元阈值 |
| ffn_neuron v_threshold | 0.15 | FFN gate/up 阈值（低阈值 → 较高发放率） |
| output_neuron v_threshold | 0.3 | 输出神经元阈值 |
| v_th_min | 0.1 | SelectivePLIF 最小动态阈值 |
| surrogate alpha | MPD-AGL 动态 | 初始 ~4.0 (input), ~9.0 (hidden), ~13.3 (FFN) |

### A.3 训练参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| learning_rate | 2e-4 | 基础学习率 |
| neuron_lr_mult | 10.0 | 神经元参数学习率倍率 |
| weight_decay | 0.1 | 投影权重 AdamW 衰减 |
| warmup_iters | 500 | 线性预热步数 |
| grad_clip | 1.0 | 梯度范数裁剪 |
| batch_size | 4 | 每卡 batch size |
| accumulation_steps | 1 | 梯度累积步数 |
| ponder_weight | 0.01 | PonderNet 正则化权重 |
| ek_floor | 4.0 | E[K] 下界 |
| ek_floor_weight | 0.1 | E[K] 下界惩罚权重 |
| snvr_weight | 0.01 | SNVR 正则化权重 |
| b_th_reg_weight | 0.01 | b_th L2 正则化权重 |

---

## 附录 B: 张量形状流转表

以 batch=4, seq_len=512, D=640, N=8, K=32, D_ff=1920 为例：

| 阶段 | 张量 | 形状 | dtype | 大小 |
|---|---|---|---|---|
| 输入 | token_ids | (4, 512) | int64 | 16 KB |
| Embedding | h_embed | (4, 512, 640) | fp32 | 5 MB |
| Repeat K | h_repeated | (16384, 4, 640) | bf16 | 80 MB |
| 层内 block_norm | h_norm | (16384, 4, 640) | bf16 | 80 MB |
| 输入 PLIF spike_current | sc | (16384, 4, 640) | bf16 | 80 MB |
| SNNBlock W_in → u_hidden | u_hidden | (16384, 4, 5120) | bf16 | 640 MB |
| SNNBlock beta_all | beta_all | (16384, 4, 5120) | bf16 | 640 MB |
| PLIF scan spike | s_hidden | (16384, 4, 5120) | uint8 | 320 MB |
| PLIF scan V_post | V_post | (16384, 4, 5120) | bf16 | 640 MB |
| SNNBlock output | I_total | (16384, 4, 640) | bf16 | 80 MB |
| PonderNet frames | frames | (512, 32, 4, 640) | bf16 | 80 MB |
| PonderNet halt_logits | halt_logits | (512, 32, 4) | bf16 | 128 KB |
| PonderNet combined | combined | (512, 4, 640) | bf16 | 2.5 MB |
| 输出 logits | logits | (4, 512, 6144) | bf16 | 24 MB |

---

## 附录 C: 文件结构

```
NeuronSpark-V1/
│
├── model.py                    主模型: SNNLanguageModel
│                                 encode / snn_forward / decode
│                                 generate / generate_speculative
│                                 get_param_groups / compensate_modulation_gradients
│                                 _compute_snvr_cost / _compute_b_th_reg_cost
│
├── atomic_ops/                 核心 SNN 计算组件
│   ├── __init__.py               spike_current_activation (_SpikeCurrentFn)
│   ├── plif_node.py              PLIFNode: 固定参数 PLIF 神经元
│   ├── selective_plif.py         SelectivePLIFNode: 动态参数 PLIF 神经元
│   ├── snn_block.py              SNNBlock: 自注意力等价层 (6路投影 + SelectivePLIF)
│   ├── snn_ffn.py                SNNFFN: SwiGLU 等价 FFN (gate/up PLIF + down + skip)
│   ├── snn_decoder_layer.py      SNNDecoderLayer: 完整解码层
│   │                               PonderNet / MPD-AGL / SubLN / Gain Clamp
│   │                               _fused_geometric_halt (PyTorch)
│   ├── parallel_scan.py          Triton PLIF 并行扫描内核
│   │                               标准 fused / row-param / linear_recurrence
│   │                               Hillis-Steele CPU fallback
│   ├── rms_norm.py               RMSNorm (PyTorch 实现)
│   ├── lateral_inhibition.py     LateralInhibition (Triton fused fwd/bwd)
│   └── fp16_codec.py             fp16 编解码工具
│
├── dataset.py                  数据集: PretrainDataset / SFTDataset
├── dashboard.py                SNNDashboard: TensorBoard 训练监控 + 健康评分
│
├── train.py                    单 GPU 预训练
├── train_ddp.py                多 GPU DDP 预训练
├── train_fsdp.py               多 GPU FSDP 预训练 (推荐)
├── train_spark.py              DGX Spark 单 GPU 预训练 (UMA 优化)
│
├── sft.py                      单 GPU SFT 微调
├── sft_ddp.py                  多 GPU DDP SFT
├── sft_fsdp.py                 多 GPU FSDP SFT
│
├── generate_sample.py          推理/文本生成 (含自投机解码)
├── snn_wrapper.py              SNN 模型封装
│
├── scripts/
│   ├── prepare_data.py           SkyPile JSONL → binary tokens 预处理
│   ├── deal_dataset.py           Seq-Monkey 文本切分
│   ├── train_tokenizer.py        BPE 分词器训练
│   ├── init_from_qwen3.py        从 Qwen3 初始化权重
│   └── download_dataset.sh       数据下载
│
├── exp/                        实验与诊断脚本
│   ├── verify_halt_grad.py       几何分布函数梯度验证
│   ├── bench_spark_opt.py        性能基准测试
│   ├── bench_compile_time.py     编译开销测试
│   ├── bench_max_tps.py          最大吞吐量测试
│   ├── estimate_max_scale.py     最大模型规模评估
│   └── estimate_full_mem.py      完整训练显存评估
│
├── tokenizer_snn/              自训练 BPE 分词器文件
├── checkpoints/                训练 checkpoint
├── runs/                       TensorBoard 日志
└── docs/                       文档
```

---

## 附录 D: 稀疏性分析与 ASIC 适配

### D.1 前向传播中的稀疏性

SNN 模型的核心优势在于脉冲驱动的稀疏性。以下分析基于 fsdp_1faa7df 训练日志的实测发放率：

| 操作 | 输入类型 | 实测发放率 | 占层 FLOPs | ASIC 加速潜力 |
|---|---|---|---|---|
| SNNBlock 6路投影 (W_in/W_β/W_α/W_th/W_gate/W_skip) | spike_current | ~20% (input neuron) | ~43.6% | **高**: 稀疏 matmul |
| SNNBlock W_out 投影 | spike_current (hidden) | ~4% | ~7.3% | **极高**: 96% 输入为零 |
| SNNFFN gate+up 投影 | spike_current | ~20% (input neuron) | ~26.2% | **高**: 稀疏 matmul |
| SNNFFN down_proj 投影 | gated (sc_gate × sc_up) | ~0.2% | ~13.1% | **极高**: 99.8% 输入为零 |
| SNNFFN skip_proj 投影 | spike_current | ~20% | ~4.4% | **高**: 稀疏 matmul |

**汇总**：层内 94.5% 的 matmul FLOPs 的输入具有稀疏结构，在稀疏硬件上可实现理论 5~500× 加速（取决于发放率和硬件稀疏粒度）。

### D.2 当前阶段约束

模型尚未完成收敛训练，因此：
- 所有修改必须**零表达能力损失**：不改变模型数学行为，仅改变计算方式
- 稀疏 matmul 是数学等价的（跳过零值乘法，结果相同）→ **可安全实施**
- 架构级稀疏化（如删减投影路径、降低 N）会降低表达能力 → **当前不允许**

### D.3 非稀疏计算

以下操作为连续值输入的稠密计算，不具有稀疏加速潜力：
- PLIF parallel scan（β·V + u，连续值递推）
- PonderNet halt_proj + 几何分布（连续值 → 概率）
- RMSNorm（连续值归一化）
- Embedding lookup / decode_proj / LateralInhibition（连续值操作）

这些操作仅占总 FLOPs 的 ~5.5%。

---

*NeuronSpark Project — CC BY-NC-SA 4.0*
