# v3 Architecture Decisions Log

**状态**：讨论中（持续更新）
**Owner**: Zhengzheng Tang
**最后更新**：2026-04-22

本文档记录 v3 架构讨论里**已达成一致**的决策，和**仍待决**的问题。实现细节另开 spec 文件，这里只记结论 + 理由。

---

## 总体框架

- **模型规模**：保留 1.16B 级（D=1024, 24 层, tied embed）。不做 MoE —— 1B 规模上不值得。
- **稀疏轴**：
  1. **时间维稀疏（v3 PonderNet 改造）** —— 每 token 动态 k_t 步
  2. **特征维稀疏（脉冲激活 = 0 跳 GEMM 行）** —— SNNFFN 中利用 spike=0
  - 两轴正交，可以叠加。
- **分支策略**：v3 分支已清理到 HF-native（见下一节）。未来所有架构改动都直接编辑 `neuronspark/modeling_neuronspark.py`（2969 行单文件）。

---

## Tokenizer（已交付，代码已落地）

- **方案**：Qwen3-1.7B tokenizer 过滤非 EN/ZH 语言 token → 128 387 vocab
- **保留**：Qwen3 全部 26 个 special token（含 `<think>/</think>`、`<tool_call>`、FIM、repo 全家桶）
- **pad**：复用 `<|endoftext|>`，不单独加 `<|pad|>`（对齐 Qwen 实践）
- **bos**：无
- **Embedding**：131.5 M，tied，占 1.16B 的 11.4%

**产物：**
- `tokenizer_v3/`（tokenizer.json + tokenizer_config.json）
- `scripts/tokenizer/build_v3_tokenizer.py`（可复跑）
- `scripts/tokenizer/verify_v3_tokenizer.py`（220MB 语料横评）

---

## PonderNet 改造（已锁定设计，未实现）

### 核心 claim

**"k 步数 = 每 token 计算配额 = PLIF 膜电位积分深度"** 是同一个变量的三种读法，对标 Mamba 的输入相关 Δ_t。v3 让它真正 **per-token 学习**，而不是 V2.5 的固定 K=12。

### 关键架构变化

| | V2.5 | v3 |
|---|---|---|
| 决策时机 | 跑完每步后看 halt_proj(h_k) → 动态停 | **跑 PLIF 前**，k_predictor(input_emb) 一次性输出 k_t |
| 决策依据 | 当前 step 的 hidden state | **input token embedding**（Mamba-Δ 风格）|
| 推理实际算几步 | K=12 全跑 | 只跑 k_t 步，**真省 FLOP** |
| Output 构造 | sum_k λ_k · h_k（加权平均）| **h_{k_t}（单一终止状态）** |

**决定性理由**：V2.5 的 halt_proj(h_k) 必须先算完步 k 才能决定是否停，相当于已经付了成本 → **零推理节省**。v3 把决策移到 PLIF 之前，k_predictor 一次前馈就得 k_t，这才是真的稀疏。

### 训练方案：Scheme X（dense train, sparse infer）

**训练：**
```
k_logits = k_predictor(input_emb)              # 一次前馈
for k in 1..K:                                  # 训练时全 K 顺序算（状态依赖物理必需）
    V_k, h_k = PLIF_step(...)
h_stack = stack(h_1..h_K)

# Gumbel-Softmax Straight-Through
gumbel = -log(-log(rand(K)))
y_soft = softmax((k_logits + gumbel) / T)
y_hard = one_hot(argmax(y_soft))
y_st = y_hard + y_soft - y_soft.detach()
output = (y_st · h_stack).sum(K_axis)          # forward = h_{k_t}

loss = CE(lm_head(output), target)             # 标准 CE，无架构耦合
loss.backward()                                 # 梯度流: k_predictor 经 y_soft, PLIF 经 h_{k_t} 链
```

**推理：**
```
k_t = argmax(k_predictor(input_emb))
for k in 1..k_t:                                # 只跑 k_t 步
    V_k, h_k = PLIF_step(...)
return h_{k_t}
```

**为什么选 X 不选 REINFORCE（方案 Y）：**
- X: 训练 FLOP = 全 K（一次性浪费），推理 FLOP = k_t/K 比例（永久收益）
- Y: 训练推理都稀疏，但 REINFORCE 方差高不稳定
- 训练是一次性成本，推理是每次部署每次节省，X 划算

### 反坍缩三层防线

`k_predictor` 在深 k 位置的 logits 有塌缩风险（若总选小 k）。三层保护：

1. **Gumbel 温度退火** T: 2.0 → 0.3
   - 训练初期 T 高 → k_t 均匀分布 → k_predictor 每个 k 位置都拿到梯度
   - 训练后期 T 低 → 硬选 → 与推理对齐
2. **Gradient-free bias balancing**（DeepSeek-V3 式）
   - 每 step 后统计 `usage_k`，`bias_k -= lr_bias · (usage_k - 1/K)`
   - 不进 autograd 图，不干扰主 loss
   - 长期保证跨 k 多样性
3. **Forced exploration** ε=0.05
   - 每 token 5% 概率强制随机 k_t
   - **结构性兜底**：不论如何，每个 k 位置每 1/ε 个 sample 必被选中一次

### k_predictor 架构

```python
class KPredictor(nn.Module):
    def __init__(self, D, K, hidden=None):
        hidden = hidden or D // 4
        self.net = nn.Sequential(
            nn.Linear(D, hidden), nn.SiLU(),
            nn.Linear(hidden, K),
        )
        self.bias = nn.Parameter(torch.zeros(K), requires_grad=False)

    def forward(self, input_emb):
        return self.net(input_emb) + self.bias   # (..., K) logits
```

### 参数预算

每层一个 k_predictor：D → D/4 → K = 1024 × 256 + 256 × 12 ≈ 266 K params/layer × 24 = **6.4 M 参数**（~0.5% 开销）。

---

## 脉冲激活稀疏（未实现，需先 bench）

**目标**：SNNFFN 中利用 `gate_spike × up_spike` 自然产生的零行，稀疏化 `down_proj` GEMM。

**代价/收益**：
- Triton kernel 开销 vs dense GEMM 的本征高效率
- 需先 profile **真实发放率**，曲线上稀疏率 > X% 才真省
- 预计 X 在 50-60%（经验值）

**执行顺序**：
1. 先写 `scripts/bench_spike_sparse.py`（0.5 天）
   - 合成 spike tensor 控制零率 20-80%
   - 对比：dense bf16 GEMM / 稀疏 Triton / 理论下限
   - 画 sparsity × speedup 曲线
2. 同时 profile V2.5 训练中真实发放率
3. bench 出来后：
   - 若曲线 + 真实稀疏率支持 → 写真 Triton kernel，集成到 SNNFFN.down_proj
   - 若不支持 → 方案作废，稀疏信号要从别处拿（加稀疏正则激活函数）

**作用域决策**：先只做 `SNNFFN.down_proj` 一个位置（ROI 最高）。如果成功再扩到 gate/up_proj。

---

## 仍待决议题

### 1. v3 Pretrain 数据（详见 `docs/v3_pretrain_design.md` 的 §9）

Token 预算 10/25/80B、EN/ZH 比例、代码多语种、context 长度、pretrain 混合方案等 8 个开放决策点。

### 2. PonderNet 细节

- Gumbel 温度 schedule：linear / cosine / exponential？
- T_init、T_final 具体值？（上面 2.0 → 0.3 是初步估算）
- bias_balancing 的 `lr_bias` 取值？
- forced exploration ε 的退火（early 高 → late 低）？
- k_predictor hidden size 是否 D/4 够用？
- K 是否保留 12？还是调整（例如 K=8 更激进降 FLOP 上限）？

### 3. 脉冲稀疏 kernel

- bench 结果出来之前先不定。

### 4. 训练超参

- ponder cost β（奖励低 k_t）要不要加？
- ε-decay schedule

---

## v3 PLIF 神经元生物化改造 (2026-04-23, 已实现)

### 动机

V2.5 传层间信号是 `(1-β)·V_post` "膜电位泄漏量" —— 几乎永不为零 (~100% 稠密)。
脉冲稀疏 bench (Stage 2) 证明：这种架构下"spike=0 跳 GEMM 行"的思路不成立。

用户提出生物学修正：**神经元发放的实际电流 = 超阈量 (V_pre - V_th)，离子流直到 V 降回 V_th**。
这与 IF/LIF 神经元模型的传统"配额发放 (每次减 V_th)"不同，更符合钠通道打开的物理机制。

### 具体改动 (neuronspark/modeling_neuronspark.py)

**数学**：

```
V_pre[k]  = β · V_post[k-1] + u[k]            (充电, 与 V2.5 相同)
output[k] = max(V_pre[k] - V_th, 0)            (★新: 超阈电流, ReLU, 天然稀疏)
V_post[k] = V_pre[k] - output[k] = min(V_pre[k], V_th)   (★新: 降到阈值)
```

**相比 V2.5 变化**：
- 输出从 `(1-β)·V_post` (连续幅度泄漏，稠密) → `max(V_pre - V_th, 0)` (超阈电流，稀疏)
- Soft reset 从 `V_post = V_pre - V_th·spike` → `V_post = min(V_pre, V_th)`
- Surrogate Sigmoid 梯度替换为精确 ReLU 梯度 (无近似, 训练更稳)

**生物语义**：
- 神经元发放 = 超过阈值触发钠通道 → 离子流出量 = V_pre - V_th
- 离子持续流出直到 V 降回 V_th (而非配额地减 V_th)
- 未发放 (V_pre ≤ V_th) 时无电流流出，output = 0

### 实测稀疏度 (小模型 D=128 K=6 4 层, untrained)

| Tensor | 严格零比例 | 说明 |
|---|---:|---|
| gate_out (PLIF gate 输出) | 89% | 发放率 ~11% |
| up_out (PLIF up 输出) | 89% | 同上 |
| gate × up | **98.5%** | 两个 sparse 乘积, 进 down_proj 的输入 |

down_proj GEMM 输入 98% 稀疏 → Stage 3 写 Triton sparse kernel 时有真实加速空间。

### 涉及代码位置

- 2 个 Triton forward kernel + 2 个 backward kernel (`_fused_plif_fwd/bwd_kernel` / `_fused_plif_fwd/bwd_rowparam_kernel`)
- 2 个 autograd Function (`_TritonPLIFForward` / `_TritonPLIFRowParamForward`)
- `plif_parallel_forward` / `plif_rowparam_forward` Python wrappers (含 CPU fallback)
- `PLIFNode.forward` (single_step)
- `SelectivePLIFNode.single_step_forward`
- `SNNBlock.forward_parallel` (`output_hidden` → W_out，之前传 V_post_hidden)
- `SNNFFN.forward_parallel` + `single_step_forward`
- `SNNDecoderLayer._input_neuron_parallel`
- `SNNAttentionDecoderLayer._input_neuron_parallel` + `_gate_neuron_parallel`
- `SNNLanguageModel._output_neuron_parallel`

### 回归测试 (通过)

- CUDA forward + backward (non-DS): ✓
- DS engine wrap + forward/backward/step: ✓
- update_ponder_bias under DS: ✓
- save_pretrained / from_pretrained round-trip: ✓
- Gumbel RNG determinism: ✓

### 待观察

- 训练初期稀疏率会怎么演化 (初始化随机, 后期训练可能稳在 30-70%)
- V_th 和 β 初始化需要调以获得合理初始发放率 (当前偏高 89%)
- bf16 下 abs_mean ≈ 1e-3 量级, 需留意是否有 underflow 风险

---

## 已放弃方案（避免回头讨论）

- ❌ **MoE / Spike-gated MoE**：1B 规模不值得
- ❌ **V2.5 样式的 halt_proj(h_k) 动态停**：等于跑完 K 才决定，零推理收益
- ❌ **Output = weighted_sum_truncated(h, λ, k_t)**：仍然是加权平均，不是单一终止状态
- ❌ **REINFORCE 训练**：方差高，收敛难
- ❌ **Soft aggregation + entropy reg**：虽然简单，但训练推理不对齐（训练加权混合、推理单步）
- ❌ **Gumbel-ST + halt_proj(h_k)**：决策依赖 iteration 输出，等于不省推理
- ❌ **蒸馏 GLM / 闭源模型黑盒**：公开 R1-distill 数据已足够，成本不划算
- ❌ **训练 tokenizer 从头**：Qwen3 filter 是最优解

---

## Time 维 Triton 现状（核对清楚避免再混淆）

- 现有 `_fused_plif_fwd_rowparam_kernel` **不是** Mamba 式 parallel scan
- 时间维（K 或 TK）是 **sequential loop**（kernel 内部 `for k in range(K)`）
- 加速来源：
  1. Fused ops（多个 PyTorch op 合成 1 次 kernel 启动）
  2. V 在寄存器驻留，不走 HBM
  3. β/v_th row-param 一次性加载
- **神经动力学完全精确，无任何近似**（代码注释 line 396: "Exact computation — sequential scan IS the ground truth"）
- 曾经有过 3-phase 近似实现（linear scan + spike fixed-point iteration + correction），已废弃换成 fused sequential

**v3 是否上 Mamba 式 parallel scan（log 深度）**：暂不做。原因：
- 当前 sequential fused kernel 每步已极快
- v3 推理 k_t 早停天然和 sequential loop 合拍（直接改循环上限即可真省 FLOP）
- parallel scan 要处理 PLIF 非线性 spike + soft reset，实现复杂且可能被迫近似
- 如未来 profile 显示 TK=24576 下 sequential 成为训练瓶颈再考虑

---

## 工程原则：功能先行，优化有凭据

**规则：**
1. **先实现最小功能版**（正确但不优化）—— 架构改动落地，确保 loss 下降、梯度通路正确
2. **优化前做 baseline 测量**（latency / memory / throughput）
3. **每次只做一个优化**，优化后**必须再测量**，用数据确认有效才保留
4. **优化无效或回归的立刻回退**，不允许堆叠未验证的优化

**不允许：**
- 未经 bench 就同时上多个优化
- "应该有收益"的臆断（必须实测）
- 为了追代码"整洁"而做没收益的重构

**适用范围：** 所有 v3 架构改动（PonderNet、脉冲稀疏 kernel、k_t 早停集成、Mamba parallel scan 等）。

---

## 实现排期（新版：功能先 → bench → 优化）

### Stage 1: 最小功能版（correctness only，不优化）

| # | 任务 | 预计 | 交付标准 |
|---:|---|---|---|
| 1.1 | v3 PonderNet spec 文档 | 0.5 天 | 算法伪码 + 边界 case 列举 |
| 1.2 | `modeling_neuronspark.py` 改造：加 KPredictor / Gumbel-ST / bias balancing buffer | 2-3 天 | 单元测试：前向通，梯度全流通，bias 更新正确 |
| 1.3 | `train_pretrain.py` 对接：温度退火 + post-step bias update hook | 0.5 天 | 冒烟 100 step loss 非 NaN |
| 1.4 | **Baseline bench**：V2.5 vs v3 stage-1 的 latency / memory / loss 下降曲线对比 | 0.5 天 | `scripts/bench_v3_vs_v2.py` 输出报告 |

### Stage 2: 脉冲稀疏可行性验证

| # | 任务 | 预计 | 触发条件 |
|---:|---|---|---|
| 2.1 | `scripts/bench_spike_sparse.py`：合成稀疏率曲线 + 实测当前发放率 | 0.5 天 | Stage 1 完成 |
| 2.2 | 决策：基于 bench 数据选层次 1/2/3（见前文）或 abandon | 0.5 天 | 2.1 出数据 |

### Stage 3: 按需优化（每次一个，bench 验证）

| # | 任务 | 预计 | 触发条件 | 验证 |
|---:|---|---|---|---|
| 3.1 | k_t 早停集成到 PLIF kernel（改 loop 上限或加 mask）| 0.5-1 天 | 默认做 | 推理 latency 前后对比 |
| 3.2 | Bitmap spike 输出（层次 2 脉冲稀疏）| 1-2 天 | 2.2 选层次 2 | bench GEMM latency 前后对比 |
| 3.3 | Mega-kernel PLIF+FFN 融合（层次 3）| 3-5 天 | 3.2 仍不够 | 同上 |
| 3.4 | Mamba parallel scan | 3-5 天 | sequential loop 成训练瓶颈 | long-TK latency 前后 |

**每个 3.x 任务必须：** bench 前后数字 → 有收益才 merge；无收益或回归 → revert。

### Stage 4: v3 预训练数据 pipeline（另一条线，独立推进）

| # | 任务 | 预计 |
|---:|---|---|
| 4.1 | 8 个数据组成决策点敲定（见 `v3_pretrain_design.md` §9）| 0.5 天 |
| 4.2 | `scripts/v3_data/` 全套实现（download / filter / dedup / tokenize / merge）| 4-5 天 |
| 4.3 | H100 上跑通完整 pipeline 出 25B token pretrain mix | 2-3 天 wallclock |

**并行建议：** Stage 1-3 和 Stage 4 完全独立，可同时推进。
