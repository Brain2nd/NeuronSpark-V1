# NeuronSpark V2.5 架构与脑结构的对应关系

> 供神经科学专家校验的技术文档。
> 梳理每个计算组件的数学定义、在模型中的功能角色、以及对应的脑区/神经机制。

---

## 1. 整体架构概览

```
输入 token → Embedding → repeat K 帧
  → 24 层交替堆叠:
      18 层 SNN Decoder Layer（皮层局部回路）
       6 层 SNN-Attention Layer（丘脑-海马系统）
  → 输出神经元 → 侧向抑制 → 词表投影 → logits
```

配比 3:1（每 4 层中 3 层局部回路 + 1 层全局记忆）。该比例参考 Qwen3.5 的 DeltaNet:Attention 配比，并非基于脑区体积比设计。

---

## 2. 基础神经元：PLIF（Parametric Leaky Integrate-and-Fire）

### 数学定义

```
V_pre[t] = β · V_post[t-1] + (1-β) · x[t]       充电（漏积分）
s[t]     = Θ(V_pre[t] - V_th)                     发放（阈值比较）
V_post[t] = V_pre[t] - V_th · s[t]                软重置
output   = (1-β) · V_post[t]                       输出（膜电位泄漏量）
```

### 可学习参数

| 参数 | 含义 | 值域 |
|---|---|---|
| β = sigmoid(w) | 膜时间常数（衰减率） | (0, 1)，D 维，每个神经元独立 |
| V_th | 发放阈值 | R+，D 维，每个神经元独立 |

### 生物对应

| 计算元素 | 脑中对应 |
|---|---|
| V（膜电位） | 神经元胞体的膜电位 |
| β（衰减率） | 膜时间常数 τ = 1/(1-β)，由离子通道特性决定 |
| (1-β)·x（充电项） | 突触后电位（PSP）的积分 |
| V_th（阈值） | 动作电位触发阈值（~-55mV） |
| soft reset（V -= V_th·s） | 动作电位后的相对不应期 |
| (1-β)·V_post（输出） | 突触前释放的神经递质量，与膜电位相关的分级释放 |

### 校验问题

1. 输出使用膜电位泄漏量 (1-β)·V_post 而非二值 spike，这对应生物系统中的什么？
   - 我们的解释：类似于分级电位（graded potential），如视网膜光感受器的输出
   - 或：突触小泡释放概率与膜电位的连续关系
2. β 在训练中由数据驱动优化，是否对应离子通道的自适应修饰（如磷酸化调节）？

---

## 3. SelectivePLIF：输入依赖的动态参数神经元

### 数学定义

```
β(t) = sigmoid(W_β · x[t] + b_β)                 动态衰减率
α(t) = softplus(W_α · x[t] + b_α)                动态写入增益
V_th(t) = V_th_min + |W_th · x[t] + b_th|         动态阈值

V[t] = β(t) · V[t-1] + α(t) · I[t]
s[t] = Θ(V[t] - V_th(t))
V[t] -= V_th(t) · s[t]
```

### 生物对应

| 动态参数 | 脑中对应 |
|---|---|
| β(t) 输入依赖的衰减率 | 神经调制（如乙酰胆碱/去甲肾上腺素改变膜时间常数） |
| α(t) 输入依赖的增益 | 突触增益调制（如 NMDA 受体的电压依赖性） |
| V_th(t) 输入依赖的阈值 | 阈值的快速适应（如 Na+ 通道的失活/恢复动力学） |
| W_β, W_α, W_th（6 条并行投影） | 不同类型的突触输入调制不同膜特性 |

### 校验问题

1. β, α, V_th 同时由同一个输入 x 调制，这在生物中是否合理？
   - 不同调制通常由不同的神经递质系统（胆碱能、去甲肾上腺素能、多巴胺能）独立控制
   - 我们的设计中，不同投影矩阵 W_β, W_α, W_th 是否足够模拟这种独立性？
2. 6 条并行投影路径对应什么？是否可对应树突不同区域的独立输入？

---

## 4. SNNBlock：隐状态空间计算单元

### 结构

```
x(D 维) → 因果卷积(kernel=4) → 6 条并行投影:
  ├─ W_in(D→D×N)    → I[t]（主输入电流）
  ├─ W_β(D→D×N)     → β(t)（衰减率调制）
  ├─ W_α(D→D×N)     → α(t)（增益调制）
  ├─ W_th(D→D×N)    → V_th(t)（阈值调制）
  ├─ W_gate(D→D)    → gate（输出门控）
  └─ W_skip(D→D)    → skip（残差跳跃）

SelectivePLIF(I, β, α, V_th) → V_post(D×N 维)

W_out(D×N→D) · V_post ⊙ gate + skip → 输出(D 维)
```

### N = 8（状态扩展因子）

每个可见维度扩展为 N=8 个隐神经元，各有不同的 β 初始化 ∈ [0.80, 0.99]，
覆盖从快（τ=5 帧）到慢（τ=100 帧）的多时间尺度。

### 因果卷积

```
Conv1d(D, D, kernel_size=4, groups=D, causal)
```

每个通道独立看前 3 帧 + 当前帧的局部上下文。

### 生物对应

| 计算元素 | 脑中对应 |
|---|---|
| SNNBlock 整体 | 皮层微柱（cortical minicolumn）的局部回路 |
| N=8 个多时间尺度神经元 | 架构设计为 N=16，因 GPU 显存限制实际训练用 N=8。β∈[0.80,0.99] 的多时间尺度分布是有意设计，N 的具体值由硬件决定 |
| W_in（主输入投影） | 丘脑皮层投射（thalamocortical projection）到 L4 |
| W_β/W_α/W_th（调制投影） | 来自高级皮层的反馈调制（top-down modulation） |
| W_gate（输出门控） | L5 锥体神经元的输出门控（basal ganglia 通路） |
| W_skip（残差跳跃） | 直接通路（feedforward bypass），不经过时序处理 |
| W_out（D×N → D 投影） | 微柱的输出汇聚（readout from multiple cell types） |
| 因果卷积(k=4) | 树突的局部时空整合（dendritic integration over ~4 个突触延迟） |

### 校验问题

1. N=8 个并行的不同 β 神经元，是否对应皮层微柱内不同细胞类型的不同时间常数？
2. 6 条并行投影是否有过多？皮层微柱的实际输入模态数量是多少？
3. gate 机制（sigmoid 门控输出）在皮层回路中有直接对应吗？

---

## 5. SNNFFN：脉冲前馈网络

### 结构

```
x → gate_proj(D→D_ff) → PLIFNode_gate → V_post_gate
x → up_proj(D→D_ff)   → PLIFNode_up   → V_post_up

gated = (1-β_g)·V_post_gate × (1-β_u)·V_post_up    // 双通路乘法门控

down_proj(D_ff→D)(gated) + skip_proj(D→D)(x) → 输出
```

### 生物对应

| 计算元素 | 脑中对应 |
|---|---|
| 双通路乘法门控 | 树突的乘法交互（dendritic multiplication），两条输入在树突上的非线性整合 |
| gate × up 结构 | 类似 SiLU/SwiGLU 的生物实现：一条通路控制增益，另一条传递信号 |
| D_ff > D 的扩展-压缩 | 皮层中间层（如 L2/3）的神经元数量 > 输入/输出层 |
| skip 连接 | 直接兴奋性传递（monosynaptic excitation） |

### 校验问题

1. 树突的乘法计算是否有足够的实验证据支持？
   - 参考：Poirazi et al. 2003, "Pyramidal Neuron as a Two-Layer Neural Network"
2. gate × up 的双通路结构是否对应 ON-OFF 通路（如视觉系统的 ON-center / OFF-center）？

---

## 6. PonderNet：自适应计算步数

### 数学定义

```
对每个 token 的 K=12 帧 SNN 输出:

p_k = σ(halt_proj(frame_k))               每步的停止概率
S_k = ∏_{j<k} (1 - p_j)                   生存概率
λ_k = p_k · S_k                           几何分布权重
λ̂_k = λ_k / Σ λ_k                         归一化

output = Σ_k λ̂_k · frame_k                加权聚合
E[K] = Σ_k k · λ̂_k                        期望步数
```

简单 token 早停（E[K]≈1），复杂 token 多想（E[K]≈10-11）。

### 生物对应

| 计算元素 | 脑中对应 |
|---|---|
| 自适应步数 | 皮层振荡中的可变处理周期：简单刺激引起少量 gamma 振荡周期，复杂刺激引起更多周期 |
| halt 信号 | 前额叶对处理完成度的监控（metacognition），类似 anterior cingulate cortex (ACC) 的冲突/完成检测 |
| 几何分布聚合 | 类似 evidence accumulation (漂移扩散模型)，累积够就停 |
| K=12 上限 | 由 GPU 显存约束决定（原设计 K=32，因 48GB 显存限制降至 12）。巧合落在 Theta-Gamma 耦合理论预测的 7-12 范围内，但非有意设计 |

### 校验问题

1. K=12 帧是否对应 gamma 振荡的 cycle 数？一个 gamma burst 通常有多少个 cycle？
2. PonderNet 的停止概率学习是否有对应的神经机制？ACC 的活动是否可以建模为这样的 halt 信号？

---

## 7. SNNDecoderLayer：完整的皮层局部回路

### 结构（双子层）

```
子层 1: RMSNorm → PLIFNode → SNNBlock → PonderNet聚合 → out_proj → 残差中心化 → 残差
子层 2: RMSNorm → PLIFNode → SNNFFN  → PonderNet聚合 → out_proj → 残差中心化 → 残差
```

### 生物对应

| 计算元素 | 脑中对应 |
|---|---|
| 双子层结构 | 皮层回路的两阶段处理：L4→L2/3（接收→关联）和 L2/3→L5（关联→输出） |
| RMSNorm（侧向抑制） | 抑制性中间神经元（PV+ interneurons）的 divisive normalization |
| 输入 PLIFNode | 兴奋性中间神经元（stellate cells in L4），将连续信号转为脉冲编码 |
| 残差中心化（h -= mean(h)） | 适应（adaptation），消除 DC 偏移，维持群体活动平衡 |
| 残差连接（h = h + sublayer） | 皮层层间的跳跃连接（如 L2/3 直接到 L5 的投射） |
| out_proj 的缩放初始化 | 突触的初始弱连接，通过学习增强（Hebbian-like） |

---

## 8. SNN-Attention Layer：丘脑-海马记忆系统

### 结构

```
子层 1: SNN-Attention
  h → K帧平均聚合 → token 级表示
  → 投影 q, k, v
  → RoPE(q, k)                              // 相对位置编码
  → PLIFNode gate（spike 门控写入判定）
  → M[T] = Σ_{t≤T} gate[t] · k[t] · v[t]ᵀ  // 无衰减累积
  → output[t] = q[t]ᵀ · M[t]                // content-based 分离查询
  → RMSNorm → out_proj → 残差

子层 2: SNNFFN（同 SNNDecoderLayer）
```

### 生物对应

| 计算元素 | 脑中对应 |
|---|---|
| SNN-Attention 整体 | 海马体 CA1-CA3 回路的快速联想记忆 |
| M 矩阵（无衰减累积） | 海马体 CA3 的自联想网络（autoassociative network），突触权重通过 Hebbian 规则快速修改 |
| gate（PLIFNode spike 门控） | 海马体门控机制：只有"新颖"或"重要"的信息被编码（novelty detection by dentate gyrus） |
| k·vᵀ 外积写入 | Hebbian 学习规则：pre × post → 突触增强。k 是 pre-synaptic pattern，v 是 post-synaptic pattern |
| q·M 查询 | 模式补全（pattern completion）：给一个部分线索 q，从联想记忆 M 中恢复完整模式 |
| 无衰减（β_M = 1） | 海马体的记忆持续时间（minutes to hours），远长于皮层局部回路的 β 衰减 |
| RoPE 位置编码 | 海马体的位置细胞（place cells）/ 网格细胞（grid cells），编码空间/序列位置 |
| K 帧平均 → token 级操作 | 海马体操作在比皮层更慢的时间尺度（theta rhythm ~4-8Hz vs gamma ~30-100Hz） |
| 3:1 配比（18 SNN : 6 Attention） | 皮层面积远大于海马体，大部分处理是局部的，少量需要全局记忆 |

### 校验问题

1. M 矩阵的无衰减累积是否对应海马体 CA3 的突触可塑性？CA3 的突触增强持续时间是？
2. spike 门控写入是否对应齿状回（dentate gyrus）的稀疏编码/新颖性检测？
3. q·M 的线性读出是否过于简化了 CA1 的模式补全过程？
4. RoPE 作为位置编码放在"海马层"是否合理？海马的位置/网格细胞确实编码序列位置。
5. 3:1 的皮层:海马比例在解剖学上是否有依据？

---

## 9. 层间信息流：连续残差流

### 设计

```
h[l+1] = h[l] + sublayer(h[l])
```

层间传递连续值 h（不是 spike），仅在 SNN 子层内部使用 spike-reset 动力学。

### 生物对应

| 计算元素 | 脑中对应 |
|---|---|
| 连续残差流 | 皮层白质中的长程轴突传导（graded potential propagation via myelinated axons） |
| 层内 spike 动力学 | 灰质内的局部回路处理（local circuit processing with action potentials） |
| 残差连接 | 跨区域的直接投射（bypass connections between cortical areas） |

### 校验问题

1. 层间用连续值传递而非 spike，是否可解释为长程轴突中的模拟信号传导？
2. 或者：每层的输出可理解为群体编码（population rate code），而非单神经元 spike？

---

## 10. 输出层：侧向抑制 + 竞争选择

### 结构

```
h → output_neuron（PLIFNode）→ K帧平均 → decode_proj → LateralInhibition → Embedding^T → logits
```

### 生物对应

| 计算元素 | 脑中对应 |
|---|---|
| output_neuron | 运动皮层 / Broca 区的输出神经元 |
| LateralInhibition | 输出层的抑制性回路，实现赢者通吃（winner-take-all）竞争 |
| Embedding^T（tied head） | 反向投射：从内部表征到词汇空间的映射，类似语言产出中的词汇选择 |
| softmax(logits) | 词汇间的互斥竞争选择 |

---

## 11. 多时间尺度层级总结

| 时间尺度 | 组件 | 对应脑区 | 频率范围 |
|---|---|---|---|
| 最快（帧级） | PLIF 动力学，K=12 帧/token | 皮层局部回路，gamma 振荡 | ~30-100 Hz |
| 中等（token 级） | PonderNet 聚合，SNN-Attention | 海马 theta 节律 | ~4-8 Hz |
| 最慢（序列级） | 残差流累积，M 矩阵持久化 | 工作记忆，前额叶持续活动 | ~0.1-1 Hz |

---

## 12. 训练算法的生物对应

| 训练组件 | 算法 | 生物对应 |
|---|---|---|
| 替代梯度（surrogate gradient） | 前向 Θ(x)，反向 sigmoid'(αx) | STDP 的连续近似：spike timing → 突触修改 |
| Adam 优化器 | 自适应学习率 | 突触标签假说（synaptic tagging）：不同突触有不同的可塑性阈值 |
| neuron_lr_mult = 10× | 神经元参数更高学习率 | 内在可塑性（intrinsic plasticity）比突触可塑性更快 |
| compensate_modulation_gradients | sigmoid 饱和区梯度补偿 | 稳态可塑性（homeostatic plasticity）：维持神经元在最佳工作区间 |
| 残差中心化 | h -= mean(h) | 突触缩放（synaptic scaling）：全局增益调整维持活动平衡 |

---

## 13. 待校验的核心问题清单

1. **泄漏量输出 (1-β)·V_post 的生物学解释**：分级电位？突触释放概率？还是群体发放率编码？
2. **6 条并行调制投影**：是否对应不同的神经调制系统？数量是否合理？
3. **N=8**：由显存约束决定（设计值 N=16）。β∈[0.80,0.99] 的多时间尺度是有意设计，N 的值是硬件限制。
4. **无衰减联想记忆 M**：海马 CA3 的突触增强是否可近似为无衰减？时间尺度？
5. **spike 门控写入**：齿状回的稀疏编码是否是 spike-based gating 的生物基础？
6. **3:1 配比**：参考 Qwen3.5 的 DeltaNet:Attention 比例和整数倍约束，非脑区体积比设计。
7. **K=12**：由显存约束决定，巧合落在 Theta-Gamma 范围。是否值得将来有意锁定在生物学合理区间？
8. **PonderNet 的元认知停止**：ACC 的冲突检测信号是否可建模为 halt probability？
9. **连续残差流**：层间不用 spike 传递，用群体编码/分级电位，是否有实验支持？
10. **RoPE 在海马层**：位置/网格细胞编码序列位置的假说是否被广泛接受？

---

## 14. 训练过程观测数据（实证佐证）

> 以下数据来自 V2.5 架构的实际预训练（1.16B 参数，17M 条中英双语数据，4×RTX4090D DDP）。
> Step 0 → 17640+，约 51M tokens 已处理。

### 14.1 神经元时间常数的自发分化

训练过程中，SelectivePLIF 的 β(t) 和 V_th(t) 出现了**浅层-深层分化**，且无任何人工约束（纯数据驱动）：

| 参数 | 浅层（L0-L2）变化 | 深层（L16-L22）变化 | 生物学解读 |
|---|---|---|---|
| block_beta_t | 微降 -0.6~-0.9% | 降 -1.1~-1.2% | 深层衰减更快 → 更短的时间常数 → 对瞬时特征更敏感 |
| block_vth_t | 升 +9% | 升 +17~21% | 深层阈值升高更多 → 更高的选择性 → 只响应强信号 |
| block_vth_t std | 升 +43% | 升 +68~79% | 深层维度间 V_th 分化更剧烈 → 功能特化更强 |
| block_alpha_t | 降 -2.4% | 降 -3.9~-4.4% | 深层写入增益缩小 → 更"谨慎"地整合信息 |

**生物学类比**：这和皮层层级处理的观测一致——初级感觉皮层（V1）神经元响应快、阈值低、选择性弱；高级联合皮层（PFC）神经元响应慢、阈值高、选择性强。模型自发学到了这种层级分化，没有人工设计。

### 14.2 SNNFFN 的深层加速现象

FFN 子层的 gate/up 神经元 β 出现**深层加速下降**：

| 层 | ffn_gate_beta 变化 | ffn_up_beta 变化 |
|---|---|---|
| L0-L2 | -0.7~-1.1% | -0.7~-1.2% |
| L8-L10 | -1.7~-2.6% | -1.7~-2.7% |
| L16-L18 | -5.1~-5.5% | -5.1~-5.6% |
| L20-L22 | **-6.0~-6.1%** | **-6.0~-6.0%** |

深层 FFN 神经元的 β 降幅是浅层的 **8 倍**。深层 FFN 正在向"快速响应"模式演化——缩短记忆、增强对当前输入的敏感度。

**生物学类比**：高级皮层的非线性整合（如 PFC 的工作记忆维护和快速切换）需要更快的突触动力学。

### 14.3 V_th 的维度间分化（功能特化）

所有层的 V_th std 都在大幅上升（+43% ~ +79%），意味着同一层内不同神经元的阈值在分化：

- 部分神经元 V_th 升高 → 稀疏发放 → 专门响应特定模式
- 部分神经元 V_th 降低 → 频繁发放 → 提供背景活动/基线信号

**这对应皮层中的稀疏编码理论**（Olshausen & Field, 1996）：信息由少量高选择性神经元的活动表示，大部分神经元保持沉默。训练过程自发涌现了这种稀疏化。

### 14.4 PonderNet 的自适应计算分配

PonderNet 从初始的 uniform 分布（E[K]≈2，所有 token 等步数）演化为高度自适应的分配：

| 度量 | 初始值 | 训练后 |
|---|---|---|
| ek_min（最简单 token） | 2.0 | 1.0（第 1 步就停） |
| ek_max（最复杂 token） | 2.0 | 10-11（接近 K=12 上限） |
| ponder_cost（平均步数） | 2.0 | 4.2 |

模型学会了：简单 token（常见词、标点）用 1 步处理，复杂 token（专业术语、数学符号、罕见词）用 10+ 步反复处理。

**生物学类比**：这和大脑的"快速通道 vs 慢速通道"一致。快速通路（如腹侧视觉通路的 feedforward sweep，~100ms）处理简单/熟悉的刺激；慢速通路（涉及反馈回路和 recurrent processing，~300-500ms）处理复杂/模糊的刺激。PonderNet 的 E[K] 分布量化了这种自适应。

### 14.5 halt 权重的学习曲线

halt_proj 的权重从初始化的 ~0.014 增长到 0.3-1.0（增长 20-70 倍），其中：

- FFN halt 增长更快（涨 50-70 倍）：FFN 子层需要更强的停止判断
- Block halt 增长较慢（涨 20-35 倍）：Block 子层的停止决策相对温和

**生物学类比**：ACC（前扣带回皮层）在学习过程中增强了对"处理完成度"的监控能力。FFN 对应的"关联处理"比 Block 对应的"时序处理"需要更明确的完成信号。

### 14.6 梯度流的自然均衡化

梯度 Gini 系数从 0.61（高度不均衡）下降到 0.43-0.47（中度），说明深层逐渐开始承担更多学习负担。

grad_share 的分布：
- Layer 0（最浅）：25-45%（主导但在下降）
- SNN-Attention 层（L3, L7, L11, L15, L19, L23）：~1%（稳定偏低）
- 其他 SNN 层：3-10%（相对均匀）

**SNN-Attention 层梯度份额稳定在 ~1%** 看似很低，但这些层的参数量也只占总参数的 ~3%。按参数归一化后，SNN-Attention 层的学习效率（grad_share / param_share）和 SNN 层接近。

### 14.7 β 多样性的持续增强

所有层的 beta_std 都在上升（+4.4% ~ +12.3%），说明同一层内不同神经元的时间常数在持续分化——快的更快、慢的更慢。

**这对应大脑发育中的突触修剪和功能特化**（synaptic pruning）：初始时所有神经元相似（β std 小），随着学习，不同神经元分化出不同的功能角色（β std 增大）。训练过程中 β 多样性的增强是模型"成熟"的标志。

### 14.8 gini 波动与数据难度的关联

当 loss 突然降低（碰到"简单" batch）时，grad_gini 会短暂飙升（>0.5），然后回落。分析发现：

- gini 升高时，layer_00 的梯度份额从 ~25% 跳到 37-45%
- 同时 PonderNet 的 ek_min 降到 1.0（更多 token 早停）
- loss 回升后 gini 迅速回落

**生物学类比**：这类似于大脑对"熟悉刺激"和"新奇刺激"的不同处理模式。熟悉刺激主要由低级皮层快速处理（浅层梯度占主导），新奇刺激需要全层级参与（梯度更均匀分布）。模型的 gini 波动量化了这种切换。

### 14.9 epileptic_rate 的缓慢下降

静态估算的 epileptic_rate 从 0.44 下降到 0.37（-16%），说明随着 V_th 升高和 β 调整，部分原本"过度兴奋"的神经元正在被调节到合理的工作区间。

**生物学类比**：这对应**稳态可塑性（homeostatic plasticity）**——神经网络通过调整兴奋性/抑制性平衡，将整体活动水平维持在功能最优区间。模型通过学习 V_th 和 β 自发实现了这种稳态调节。

### 14.10 layer_output_norm_ratio 的增长

各层输出范数比从 1.0 增长到 1.40，说明不同层的输出幅度在分化。这是正常的——不同层学习了不同抽象级别的特征，其输出幅度自然不同。

**关键**：ratio 增长是单调缓慢的（1.0 → 1.4），没有突然跳变。如果突然跳到 >3 则提示梯度爆炸风险。当前 1.4 是健康的分化。
