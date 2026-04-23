# v3 PonderNet Specification

**Status**: spec (pre-implementation)
**Owner**: Zhengzheng Tang
**Last updated**: 2026-04-22
**Depends on**: `docs/v3_architecture_decisions.md`

本文档精确定义 v3 PonderNet 改造的算法、张量维度、梯度路径、边界 case。先拍 spec 再动代码。

---

## 1. 设计目标（一句话）

**让 K 维从"固定 12 步"变成"per-token 由输入决定的动态 k_t 步"**，推理时只跑 k_t 步（真省 FLOP），训练时跑全 K 为反向通路稳定。

和 Mamba 的 input-dependent Δ_t 同构：per-token compute allocation by learned predictor.

---

## 2. 组件定义

### 2.1 `KPredictor` 模块

**Location**: `neuronspark/modeling_neuronspark.py`（集成在 SNNDecoderLayer 内部，每层一个实例）

**接口**：
```python
class KPredictor(nn.Module):
    def __init__(self, D: int, K: int, hidden: int | None = None):
        super().__init__()
        h = hidden or D // 4
        self.net = nn.Sequential(
            nn.Linear(D, h),
            nn.SiLU(),
            nn.Linear(h, K),
        )
        # Gradient-free balancing bias (DeepSeek-V3 风格)
        self.register_buffer("bias", torch.zeros(K))
        # Running usage stats for bias update
        self.register_buffer("_usage_ema", torch.full((K,), 1.0 / K))

    def forward(self, input_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_emb: (..., D) — per-token embedding (未经 PLIF 展开)
        Returns:
            logits: (..., K) — halt-step distribution logits (含 bias)
        """
        return self.net(input_emb) + self.bias

    @torch.no_grad()
    def update_bias(self, y_hard: torch.Tensor, lr_bias: float, ema_decay: float = 0.99):
        """
        每 training step 后调用, gradient-free 更新 bias.

        Args:
            y_hard: (..., K) — 本 step 的 one-hot 选择 (aggregated over 所有 token)
            lr_bias: bias 更新学习率 (典型 1e-3)
            ema_decay: usage EMA 衰减系数
        """
        # 当前 batch 的 usage
        usage = y_hard.mean(dim=tuple(range(y_hard.dim() - 1)))  # (K,)
        # EMA 平滑避免 noise
        self._usage_ema.mul_(ema_decay).add_(usage, alpha=1 - ema_decay)
        # 偏差 = 实际使用 - 均匀目标
        diff = self._usage_ema - (1.0 / self._usage_ema.shape[0])
        # 欠使用 (diff<0) → bias 加；过使用 (diff>0) → bias 减
        self.bias.add_(-lr_bias * diff)
```

**参数预算**：D=1024, h=256, K=12
- 第一层：1024 × 256 = 262 K
- 第二层：256 × 12 = 3 K
- 合计 ~265 K / layer × 24 = **6.4 M 参数**（占 1.16B 模型的 0.55%）

### 2.2 温度退火 schedule

**位置**：`SNNLanguageModel` 持有 `self.ponder_T`（标量），train_pretrain.py 每 step 更新。

**schedule**（可 CLI 配置）：
```python
T(step) = T_final + (T_init - T_final) * exp(-step / tau)
# 默认: T_init=2.0, T_final=0.3, tau = 0.1 * total_steps
# 早期 T≈2.0 (强随机), 末期 T≈0.3 (接近硬选)
```

### 2.3 Forced exploration 概率

**参数**：`ε_explore ∈ [0, 0.1]`，默认 0.05。
**退火**：可选线性衰减到 0.01（末期减少噪声）。

---

## 3. Forward 算法（训练）

### 3.1 张量维度约定

- `B`：batch size
- `T`：sequence length (training seq)
- `K`：每 token 最大 PLIF 步数（v3 保留 = 12）
- `D`：hidden size = 1024
- `L`：层数 = 24

### 3.2 SNNDecoderLayer.forward_parallel（v3 新版，per-layer）

```python
def forward_parallel(self, h_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        h_seq: (TK, B, D) — 全 T·K 帧输入 (V2.5 兼容布局)

    Returns:
        output: (TK, B, D) — 标准 TK 输出 (向下游层保持接口不变)
        ponder_cost: scalar — 平均 k_t (用于 log, 不一定进 loss)
        k_t_onehot_hard: (T, B, K) — 本层本次选择的 one-hot (用于 bias update)
    """
    TK, B, D = h_seq.shape
    K, D_ = self.K, D
    T = TK // K

    # ==== Step 1: 从 input 预测 k_t 分布 ====
    # 只用每 token 的 "原始" input embedding (第 k=0 帧代表 token 原位)
    h_input = h_seq.view(T, K, B, D)[:, 0]      # (T, B, D) — token-level input
    k_logits = self.k_predictor(h_input)        # (T, B, K) — 含 bias

    # ==== Step 2: Gumbel-Softmax Straight-Through ====
    # 训练时按温度退火+force exploration 采样
    if self.training:
        gumbel = -torch.log(-torch.log(torch.rand_like(k_logits) + 1e-9) + 1e-9)
        logits_noisy = (k_logits + gumbel) / self.ponder_T
        y_soft = F.softmax(logits_noisy, dim=-1)                # (T, B, K)

        # Forced exploration: ε 概率替换 y_hard 为随机 one-hot
        if self.eps_explore > 0:
            random_k = torch.randint(0, K, (T, B), device=h_seq.device)
            explore_mask = (torch.rand(T, B, device=h_seq.device) < self.eps_explore)
            y_hard = F.one_hot(y_soft.argmax(-1), K).float()   # (T, B, K)
            y_hard_rand = F.one_hot(random_k, K).float()
            y_hard = torch.where(
                explore_mask.unsqueeze(-1), y_hard_rand, y_hard
            )
        else:
            y_hard = F.one_hot(y_soft.argmax(-1), K).float()

        # STE: forward=hard, backward=soft
        y_st = y_hard + y_soft - y_soft.detach()               # (T, B, K)
    else:
        # 推理：纯 argmax，无 gumbel, 无 exploration
        y_soft = F.softmax(k_logits, dim=-1)
        y_hard = F.one_hot(y_soft.argmax(-1), K).float()
        y_st = y_hard  # 推理不需要 ST

    # ==== Step 3: 全 K forward (PLIF 顺序展开)，不因 k_t 早停 ====
    # 维持现有 snn_block + snn_ffn 实现不变
    # 调用现有 Triton kernel (_fused_plif_fwd_rowparam_kernel)
    cont_block = self.snn_block.forward_parallel(h_seq)       # (TK, B, D)
    h_after_block = h_seq + cont_block                         # 残差
    v_in2 = self._input_neuron_parallel(self.input_neuron2, self.ffn_norm(h_after_block))
    cont_ffn = self.snn_ffn.forward_parallel(v_in2)           # (TK, B, D)

    # ==== Step 4: 用 y_st 按 k_t hard-pick 聚合 ====
    # 取 ffn 子层的 K 帧输出, 每 token 只保留 k_t 帧位置
    frames = cont_ffn.view(T, K, B, D)                         # (T, K, B, D)
    # y_st: (T, B, K) → (T, K, B, 1) for broadcast
    y_st_b = y_st.permute(0, 2, 1).unsqueeze(-1)               # (T, K, B, 1)
    output_token = (y_st_b * frames).sum(dim=1)                # (T, B, D) — h_{k_t}

    # 广播回 TK 布局（向下游层保持接口）
    output = output_token.unsqueeze(1).expand(T, K, B, D).reshape(TK, B, D)

    # ==== Step 5: ponder cost (log 用, 可选加入 loss) ====
    k_expected = (y_st.detach() * torch.arange(1, K + 1, device=h_seq.device).float()).sum(-1)
    ponder_cost = k_expected.mean()

    return output, ponder_cost, y_hard.detach()  # y_hard for bias update
```

### 3.3 关键决策：**每层独立 k_t** vs 整栈共享 k_t

**决策：每层独立 k_t。** 理由：
- 不同层承担的计算密度不同（早层提取 local feature，深层做抽象）
- 同一 token 在层 l 可能 k=3 就够，在层 l+1 可能 k=8
- 每层独立 k_predictor 允许这种差异化
- 参数开销 6.4 M 可接受

---

## 4. Forward 算法（推理）

### 4.1 真早停版本

```python
def forward_parallel_inference(self, h_seq):
    TK, B, D = h_seq.shape
    T = TK // K

    h_input = h_seq.view(T, K, B, D)[:, 0]
    k_logits = self.k_predictor(h_input)         # (T, B, K)
    # 推理无 Gumbel, 纯 argmax
    k_t = k_logits.argmax(dim=-1)                # (T, B), 值域 [0, K-1]
    k_t_plus_1 = k_t + 1                          # 实际步数 [1, K]
    max_k_t = k_t_plus_1.max().item()

    # 只跑 max_k_t 步的 PLIF (Triton kernel K 参数改为 max_k_t)
    # batch 内 token 早停差异靠下游 mask 处理
    h_seq_truncated = h_seq.view(T, K, B, D)[:, :max_k_t].reshape(T * max_k_t, B, D)
    cont_block = self.snn_block.forward_parallel(h_seq_truncated)
    ...

    # 聚合时 per-token 取各自 k_t 位置
    frames = cont_ffn.view(T, max_k_t, B, D)
    gather_idx = k_t.unsqueeze(1).unsqueeze(-1).expand(T, 1, B, D)  # (T, 1, B, D)
    output_token = frames.gather(dim=1, index=gather_idx).squeeze(1)  # (T, B, D)
    ...
```

**FLOP 节省**：理想状况 E[k_t] / K ≈ 30-50%，实测为准。

### 4.2 Stage 1 简化：推理先不做真早停

Stage 1 为控制改动，**推理也跑全 K**（和训练一样），只是用 argmax（无 Gumbel / exploration）做 gather：

```python
# stage-1 inference (no real early-stop)
y_hard = F.one_hot(k_logits.argmax(-1), K).float()
y_st = y_hard
# 后续聚合和训练相同，只是少了 Gumbel 和 exploration
```

真早停留给 Stage 3.1 任务单独上线，配合 bench。

---

## 5. Bias 更新 hook（训练循环）

### 5.1 数据流

```
train_pretrain.py:
    for step, batch in enumerate(loader):
        loss, aux_info = engine(batch)   # aux_info 含 per-layer y_hard
        engine.backward(loss)
        engine.step()

        # === Bias update (gradient-free, 每 step 调用) ===
        with torch.no_grad():
            for layer in engine.module.snn.layers:
                if hasattr(layer, 'k_predictor'):
                    layer.k_predictor.update_bias(
                        y_hard=aux_info[layer_idx],
                        lr_bias=args.lr_bias,
                        ema_decay=0.99,
                    )

        # === Temperature annealing ===
        engine.module.snn.ponder_T = temperature_schedule(step, args.total_steps)
```

### 5.2 超参默认

| 参数 | 默认值 | 说明 |
|---|---|---|
| `K` | 12 | 保留 V2.5 一致 |
| `T_init` | 2.0 | Gumbel 初温 |
| `T_final` | 0.3 | Gumbel 末温 |
| `tau` | 0.1 × total_steps | 退火时常数 |
| `eps_explore` | 0.05 | 强制均匀采样概率 |
| `lr_bias` | 1e-3 | bias 更新步长 |
| `bias_ema_decay` | 0.99 | usage EMA 衰减 |
| `ponder_weight` | 0.0 | ponder_cost 进 loss 的权重（stage 1 先设 0） |

---

## 6. 梯度路径验证

### 6.1 Forward 关键张量

- `k_logits` (T, B, K)：可导，来自 `k_predictor(h_input)`
- `y_soft` (T, B, K)：可导，softmax(logits + gumbel) / T
- `y_hard` (T, B, K)：不可导，one_hot(argmax)
- `y_st = y_hard + y_soft - y_soft.detach()`：forward 值 = y_hard，backward 梯度 = y_soft
- `output = Σ_k y_st · h_k`：output 对 y_st 可导，y_st 对 y_soft 可导
- `loss = CE(lm_head(output), target)`

### 6.2 Backward 检查点

| 梯度目标 | 路径 | 预期 |
|---|---|---|
| `k_predictor.net` params | `loss → output → y_st → y_soft → k_logits → net` | ✓ 有梯度 |
| `k_predictor.bias` buffer | 不在 autograd 图 | ✓ 只经 update_bias hook |
| PLIF β/α/v_th/W_in/W_out | `loss → output → h_{k_t} → V_{k_t..1} → PLIF params` | ✓ 有梯度，顺序反传 |
| embed_tokens / lm_head | 标准 CE 路径 | ✓ |

### 6.3 反坍缩三层机制作用点

1. **Gumbel 温度**：对 `y_soft` 的熵增，强制探索不同 k_t
2. **Bias buffer**：gradient-free，每 step 外推 k_logits 多样性
3. **Forced exploration**：少数样本直接替换 y_hard 为随机 k，保底

---

## 7. 边界 case 列表

| case | 处理 |
|---|---|
| batch 内某 token 的 k_t=K（最深）| Forward 照常跑全 K，无区别 |
| batch 全部 token 的 k_t=1（最浅）| 推理 max_k_t=1 可能导致 Triton kernel 退化；需要 K_min=1 支持 |
| bias drift 极端（某 k 的 bias=-inf）| 加 clamp bias ∈ [-5, 5] |
| Gumbel 噪声数值爆（rand=0 → log(0)）| epsilon=1e-9 防数值问题 |
| 推理 batch=1 单 token 生成 | 同训练逻辑，k_logits 仍为 (1, 1, K) |
| k_predictor 初始化 | net 标准 init，bias 初始 zeros |
| Multi-layer 之间 k_t 不一致 | OK，每层独立，下游层照常接收 (TK, B, D) |
| 保存 ckpt | bias / usage_ema 作为 buffer 随模型保存 |
| V2.5 ckpt 加载 v3 模型 | k_predictor 缺失 → 随机初始化，需 warmup 若干步 |

---

## 8. 兼容性

### 8.1 对 HF API 的影响

- `NeuronSparkForCausalLM.forward`：不变（内部 `self.snn(input_ids)` 调用 v3 PonderNet）
- `from_pretrained` / `save_pretrained`：自动带 k_predictor 参数 + bias buffer
- `generate`：推理走 argmax 路径（stage 1 全 K，stage 3.1 启用真早停）

### 8.2 对训练脚本的影响

- `train_pretrain.py`：需加 CLI: `--ponder_T_init`, `--ponder_T_final`, `--ponder_tau`, `--eps_explore`, `--lr_bias`
- bias update hook：在 `engine.step()` 之后调用
- 温度更新：在 forward 之前 `model.snn.ponder_T = schedule(step)`

### 8.3 对配置的影响

`NeuronSparkConfig` 加字段：
```python
@dataclass
class NeuronSparkConfig:
    ...  # 原有字段
    # v3 PonderNet additions
    k_predictor_hidden: int = 256     # KPredictor 隐层维度
    ponder_T_init: float = 2.0
    ponder_T_final: float = 0.3
    eps_explore: float = 0.05
    # 以下 runtime 状态不进 config
```

---

## 9. 测试计划（Stage 1.2 交付时必测）

1. **单元测试** `tests/test_kpredictor.py`：
   - forward output shape (T, B, K)
   - bias 加法正确
   - update_bias 增加欠使用项，减少过使用项
   - clamp bounds 有效
2. **梯度测试** `tests/test_ponder_gradient.py`：
   - k_predictor.net params 有非零梯度
   - PLIF params 有非零梯度
   - y_hard 不在 autograd 图 (assertion)
3. **正向数值测试** `tests/test_ponder_forward.py`：
   - 强制 k_t=K（bias 全负无穷）→ 输出等价于 V2.5 最后一帧
   - 强制 k_t=0（bias 置某 k 全正无穷）→ 输出等于 h_k
4. **冒烟训练**：100 step `train_pretrain.py`，loss 单调非 NaN

---

## 10. 显式不做（Stage 1 范围外）

- ❌ 真早停推理（在 Stage 3.1）
- ❌ 脉冲稀疏集成（在 Stage 3.2-3.3）
- ❌ Mamba parallel scan（在 Stage 3.4，条件触发）
- ❌ 跨层 k_t 共享（决策已定独立）
- ❌ 动态 K（K 本身变化，非 k_t 变化）
- ❌ 多头 k_predictor（单头足够）
