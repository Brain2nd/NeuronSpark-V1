# V4 Early-Stop Design — 前向 + 反向完整规格

**Status**: spec (pre-implementation)
**Branch**: v4
**Goal**: 让 PonderNet 的 per-token 动态 k_t 真正省 FLOP（推理只跑 k_t 步），且训练/推理语义一致、bit-exact 等价。修正 V2.5/v3 架构里破坏早停的所有耦合点。

---

## 0. 为什么 v3 架构早停不成立（已确认）

实测「真截断 K=8」与「全 K=12」输出不同。根因：被丢弃的 frame k_t+1..K **不是因果无关的** —— 它们通过两类机制泄漏进下一 token：

1. **SNNBlock.conv1d**（causal kernel=4，沿 TK 轴）—— token t+1 的 frame 0 卷积窗口含 token t 的尾 3 帧
2. **PLIF 膜电位连续递推**（沿 TK 轴）—— 6 处 over-TK 的 PLIF 状态（`input_neuron1/2`、`snn_block.hidden_neuron`、`snn_ffn.gate_neuron/up_neuron`、模型级 `output_neuron`）都把第 K 帧的膜电位携带给下一 token
3. **decode 的 `v_out.view(seq,K,..).mean(dim=1)`** —— 输出对全 K 帧取平均

安全（无需改）：残差流（K 冗余）、SNNAttentionDecoderLayer 的 attention 子层（全 token 级）、emb 展开、KPredictor bias。

---

## 1. 核心改动

### 1.1 删 conv1d（Mamba-3 路线）

依据 Mamba-3 (ICLR 2026, §3.4/§4.2)：external short conv 被「exp-trapezoidal 离散化 + B,C bias」内化，加回 short conv「does not improve performance; in fact, it slightly worsens it」。我们直接删 `SNNBlock.conv1d` + `conv_state`。conv 的「局部上下文」作用由 attention 子层 + 残差流承担。

**影响**：`SNNBlock.__init__` 删 `self.conv1d` / `register_memory('conv_state')`；`forward_parallel` 删 Phase 0；`single_step_forward` 删 conv；`get_param_groups` 删 conv1d 覆盖。

### 1.2 PLIF 递推改成「per-token K-frame，inter-token 携带第 k_t 帧」

设 token t 的 halt 帧 = index `k_t ∈ {0,..,K-1}`（"k_t 步" = 计算 frame 0..k_t）。

**前向（训练，跑全 K）**：
```
for token t:
    v0_t = inter-token carried state        # token 0: = neuron.v (上次 forward 末态); token t>0: = v_{k_{t-1}}^{t-1}
    # 跑全 K 帧（梯度需要）
    for k = 0..K-1:  v_{k}^t = β · v_{k-1}^t + u_k^t ; ... (PLIF spike/reset)
    output_t = output_{k_t}^t               # 残差用第 k_t 帧
    v_carry_t = v_{k_t}^t                   # 携带给 t+1 用第 k_t 帧的膜电位
neuron.v = v_carry_{T-1}                     # 末态供下次 forward
```

**前向（推理，只跑 k_t）**：同上，但内层只跑 `k = 0..k_t`（frame > k_t 不算）。因为 frame > k_t 既不进 output_t 也不进 v_carry_t，**训练（跑全 K）和推理（跑 k_t）的 output_t / v_carry_t 完全相同** → bit-exact 等价。

**inter-token 递推（PLIF 线性，可并行 scan）**：
PLIF 是 v_init 的线性函数：`v_{k}^t = β^{k+1} · v0_t + drift_{k}^t`，其中 `drift_{k}^t = Σ_{j=0}^{k} β^{k-j} u_j^t`（先不计 spike/reset，见 §1.5）。
inter-token：`v0_{t+1} = v_{k_t}^t = β^{k_t+1} · v0_t + drift_{k_t}^t`。
这是 token 维的一阶线性递推（T 步），可用 associative scan 并行：每 token 贡献 `(decay_t, drift_t) = (β^{k_t+1}, drift_{k_t}^t)`，scan 得所有 `v0_t`，再并行重算各 token 的 K 帧。

**实现层面**：
- 训练用「分段 PLIF kernel」：输入 `(T, K, ...)` + 每 token 的 `k_t`（T 维 int）+ 初始 `v_init`（batch 维）；kernel 内：先按上式 scan 出 `v0_t`，再并行算 `(T, K, ...)` 的全帧输出 + gather 第 k_t 帧的 v 作为 `v_carry`。返回 `(output_all (T,K,..), v_carry (batch,..))`。
- 推理（autoregressive，1 token/步）：直接跑 k_t 帧，简单。

### 1.3 forward reorder：先算 k_t

当前：`input_neuron + snn_block 跑全 K → 之后 _ponder_aggregate_v3 选 k_t`。
改成：
```
h_token = h.view(seq,K,b,D)[:, 0]                    # token-level input (K 份相同, 取第 0 帧)
k_logits_block = block_k_predictor(h_token)
k_t_block = pick(k_logits_block)                      # 训练: Gumbel-ST 采样; 推理: argmax
                                                       # → (T, b), 值 ∈ {0..K-1}
v_in_block = input_neuron1_segmented(block_norm(h), k_t_block)   # 用 k_t 做分段递推
cont_block, v_carry_block = snn_block_segmented(v_in_block, k_t_block)
output_block = gather(cont_block, k_t_block)          # (T, b, D)
# 残差中心化 + repeat_interleave(K) + 加回 h (h 仍 K 冗余)
```
FFN 子层同理（用 `ffn_k_predictor` → `k_t_ffn`，独立于 block 子层的 k_t）。

`input_neuron1/2`、`snn_block.hidden_neuron`、`snn_ffn.gate_neuron/up_neuron` 的 `.v` 都改成「携带第 k_t 帧」（block 子层用 k_t_block，FFN 子层用 k_t_ffn）。

### 1.4 decode 的 K 聚合

`v_out.view(seq,K,b,D).mean(dim=1)` 改：output_neuron 也走 k_predictor → k_t_out → 只跑 k_t_out 帧 → 取第 k_t_out 帧（或对 0..k_t_out 帧 mean）。**或**：output_neuron 固定 K（它是最后一步、占比 ~1/25，省不省 FLOP 影响小），但那样它仍携带 v_K → 仍耦合 → 还是得改成携带 v_{k_t_out}。**决策**：output_neuron 加一个 `output_k_predictor`，与层一致处理。

### 1.5 spike/reset 与线性 scan 的关系（关键技术点）

PLIF 含软重置：`v_pre = β·v + u; output = max(v_pre - v_th, 0); v = v_pre - output = min(v_pre, v_th)`。这**不是纯线性**（有 `min/max`）。所以 §1.2 的「associative scan」不能直接套用（drift 公式假设了线性）。

两个选项：
- **(A) 分段 kernel 直接顺序跑**：训练时仍跑全 K，但 kernel 在每个 token 边界 reset v_init 为「上 token 第 k_{t-1} 帧的 v」。这要求 inter-token 是顺序依赖 —— 但可以两遍：第一遍跑全 TK（连续递推，和现在一样）拿到所有帧的 V_post；第二遍**不重算**，只 gather：token t 的 inter-token 输入应该是 V_post[token t-1, frame k_{t-1}]，而连续递推给的是 V_post[token t-1, frame K-1]。**这两者不同**，所以第一遍的连续递推结果对 token t≥1 是错的（用了错的 v0_t）。→ 必须真的分段重跑。
- **(B) 分段 kernel + 串行 token scan**：kernel 内对 token 维串行（T 次迭代），每次：从 v0_t 跑 K 帧（或推理时 k_t 帧）的 PLIF（含 spike/reset），存全帧 V_post + output，取第 k_t 帧 v 作为 v0_{t+1}。token 内的 K 帧可并行（其实不行，PLIF 帧间也是递推 —— 但 K=12 很小，串行 12 步可接受；token 维 T 串行 + 帧维 K 串行 = O(TK) 串行，和现在的连续 TK 递推同复杂度，只是分段了）。

**结论**：用 (B)。kernel 改成「双层循环：外层 token (T)，内层 frame (k_t+1 或 K)」，外层迭代间传递「上 token 第 k_t 帧的 v」。这和现在的单层 TK 循环相比，只是把 inter-token 的传递从「第 K 帧」改成「第 k_t 帧」，复杂度不变。训练全 K：内层跑满 K，但传递第 k_t 帧。推理：内层跑 k_t。

### 1.6 反向传播

新 kernel 的 backward：
- 训练跑全 K，所以 K 帧的 PLIF 动力学梯度都在（和现在一样，∂output/∂v_pre = s exact ReLU，软重置的子梯度照旧）
- 新增梯度路径：`v_carry_t = v_{k_t}^t` → 影响 token t+1 → ... 这条 inter-token 路径的梯度要正确反传。由于 k_t 是离散选择（Gumbel-ST），`v_{k_t}^t` 的 gather 用 `y_st`（hard forward / soft backward）→ 梯度通过 y_soft 流回 k_predictor + 通过 frame 维流回各帧的 v。
- 具体：`v_carry_t = Σ_k y_st_t[k] · v_k^t`（和 output 的 gather 同构）。backward：`∂v_carry_t/∂v_k^t = y_st_t[k]`，`∂v_carry_t/∂(k_logits_t)` 经 y_soft。kernel 内把这个 gather 显式实现（forward gather + backward scatter），或在 kernel 外用 PyTorch 的 `gather` + STE 包装（更稳，先这样）。

**实现策略**：kernel 只负责「给定每 token v0_t，跑 K 帧 PLIF，返回全帧 output + 全帧 V_post」。inter-token 的 v0_t scan + gather 在 PyTorch 层做（用 `torch.cumprod`/手写 token 维循环 + autograd 自动反传）。这样 kernel 改动最小（只是 v_init 从单个 batch-tensor 变成 (T, batch) 的 per-token tensor，外层多一个 token 循环），反向靠 PyTorch autograd + 现有 PLIF kernel backward。

---

## 2. 训练循环改动

- `_ponder_aggregate_v3` 拆成两步：(a) `pick_k_t(k_logits)` 在 forward reorder 里提前调；(b) gather output + v_carry。
- `update_bias` hook：不变（仍用 `_last_y_hard_xxx`）。
- 温度退火：不变。
- `ponder_weight`：可开始进 loss（鼓励小 k_t）—— V4 stage 2 再说，先设 0。

## 3. 配置 / 兼容

- `config.K` 含义不变（K_max=12）。
- safetensors：删了 conv1d → 旧 v3 ckpt 不能 load 到 V4（少 conv 权重 + PLIF 语义变）→ **V4 必须重新预训练**（已确认接受）。
- `from_pretrained` 的 RoPE buffer 重填 / 混合精度逻辑：不变（attention 层的 RoPE 不动）。

## 4. 验证

1. **训练 forward == 推理 forward（同 k_t）**：构造固定 k_t（绕过随机），训练模式跑全 K + gather，推理模式跑 k_t，断言 output_t / v_carry_t / logits 逐元素相等（这是 bit-exact 早停的核心保证）。
2. **梯度数值检查**：小模型 + `torch.autograd.gradcheck` 验证新 kernel 的 backward（含 inter-token v_carry 路径）。
3. **conv 删除前后 forward 差异**：量化删 conv 对一个固定 batch 的影响（仅供参考，删 conv 是 Mamba-3 验证过的）。
4. **小规模预训练 smoke**：4090 跑 ~500 step，看 loss 下降正常、E[K] 分布合理、无 NaN。

## 5. 分阶段实现（避免 big-bang）

- **P1**：删 conv1d + conv_state（所有相关文件）。测：forward 能跑、loss 能算。
- **P2**：forward reorder —— k_predictor 提前，`_ponder_aggregate_v3` 拆 pick/gather。结构变但仍跑全 K、仍携带第 K 帧（行为暂时不变，纯重构）。测：forward 数值与 P1 一致。
- **P3**：PLIF 携带改第 k_t 帧（6 处）+ 分段 kernel（v_init 变 per-token）。测：训练 forward == 推理 forward（验证 1）。
- **P4**：decode 的 K 聚合 + output_neuron k_predictor。测：同上 + 端到端 forward。
- **P5**：backward gradcheck + 小规模预训练 smoke。
- **P6**：full 预训练。

每阶段 commit + 测试，不混在一起。
