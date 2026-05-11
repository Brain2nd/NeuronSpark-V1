# V4 P3b — Segmented PLIF Kernel 设计（forward + backward）

**Status**: spec (pre-implementation)
**Branch**: v4
**目的**: PLIF 递推从「连续 TK 单段」改成「per-token K 帧、token 边界把运行膜电位 patch 成上 token 第 k_{t-1} 帧的 V_post」。这是 V4 早停的核心机制。复杂度仍 O(TK)/列，一次 kernel launch。

---

## 0. 符号

- `T` = token 数, `K` = K_max（每 token 最大帧数）, `TK = T*K`
- `num_cols` = batch × hidden（kernel grid 的列维；hidden 对 input_neuron 是 D，对 snn_block hidden 是 DN，对 gate/up 是 D_ff，对 output_neuron 是 D）
- `u[i, c]`：第 i 帧（i = t*K + k，t = i//K，k = i%K）第 c 列的「已缩放输入」（rowparam: `(1-β)·x`；non-rowparam: 投影后的 I）
- `β[c]`（rowparam，per-column 常数）或 `β[i, c]`（non-rowparam / selective，per-frame data-dependent）
- `vth[c]`（rowparam）或 `vth[i, c]`（non-rowparam）
- `k_t_cols[t, c] ∈ {0..K-1}`：token t、列 c 的 halt 帧 index。由层级 `k_predictor` 的 argmax/Gumbel 得到 `k_t (T, batch)`，再 expand 到 `(T, num_cols)`（`k_t_cols[t, c] = k_t[t, c // hidden]`）。**kernel 外预计算后传入**（kernel 不需知道 `hidden`）。
- `v_init[c]`：本次 forward call 的初始膜电位（训练里 `functional.reset_net` 每次重置为 0；`generate_cached` 里 = 上次 forward call 的 `v_carry`）
- `training: bool`：训练（跑全 K 帧）/ 推理（早停只跑 0..k_t）

## 1. Forward 递推

per column c（grid over num_cols）：
```
v = v_init[c]
for t in 0..T-1:
    if t > 0:
        # token 边界 patch: v ← 上 token 第 k_{t-1} 帧的 V_post（已在上一轮存好）
        v = V_post[(t-1)*K + k_t_cols[t-1, c], c]
    k_halt = k_t_cols[t, c]
    for k in 0..K-1:
        i = t*K + k
        v_pre = β·v + u[i, c]                    # β: β[c] (rowparam) 或 β[i,c] (non-rowparam)
        out   = max(v_pre - vth, 0.0)            # vth: vth[c] 或 vth[i,c]
        v     = v_pre - out                      # = min(v_pre, vth)，软重置
        output[i, c] = out
        V_post[i, c] = v
        if (not training) and k == k_halt:
            break                                # 推理早停：第 k_t 帧之后不算（FLOP 节省）
# v_carry[c] = 序列最后一个 token 第 k_t 帧的 V_post（detach，给下次 forward call）
v_carry[c] = V_post[(T-1)*K + k_t_cols[T-1, c], c]
```

要点：
- **token 边界 patch 是唯一新增逻辑**：每 K 帧一次额外 load。其余与现有 `_fused_plif_fwd_*_kernel` 完全一致。
- **训练跑全 K**：内层 k-loop 跑满 K（梯度需要：层级 `y_st` 的软路径给所有 K 帧梯度）。
- **推理早停**：内层 k-loop 在 `k == k_halt` 时 break。frame k_t+1..K-1 不算 → uninitialized garbage，但下游只读 frame k_t（层级 gather 是 one-hot at k_t）+ 下 token 的 patch 读 frame k_{t-1}（已算）→ 安全。
- **bit-exact 早停保证**：训练（跑全 K）和推理（跑 k_t）的 `output[t*K + k_t[t], c]` 和 patch 链 `V_post[t*K + k_t[t], c]` 完全相同（frame > k_t 既不进 output 的下游路径也不进 patch 链）→ forward 输出逐元素相等。

## 2. Backward

上游梯度：`g_out[i, c] = ∂L/∂output[i, c]`（层级 STE gather 给：训练时所有 K 帧都有，权重 y_soft；frame k_t 那个有 y_hard 的硬权重的前向贡献但梯度也是 y_soft）。`V_post` 没有直接下游消费者作梯度入口（v_carry 是 detach 的），但 `V_post` 在 forward 里有两类**内部**消费者，反向都要算：
1. **同 token 内下一帧**：`v_pre[i+1] = β·V_post[i] + u[i+1]`（i, i+1 同 token）
2. **下 token 的 patch**：`v_init(token t) = V_post[(t-1)*K + k_{t-1}, c]`（即 frame k_{t-1} → token t 的 frame 0）

反向递推（per column c，**逆序**遍历 token / 帧）：
```
acc_v = 0.0                          # ∂L/∂(当前位置的 v / v_pre 之间的累积)
acc_β = 0.0; acc_vth = 0.0
g_v_init = 0.0
# 注意: V_post[i] = v_pre[i] - out[i] = min(v_pre[i], vth[i])
#       out[i] = relu(v_pre[i] - vth[i])
#       s[i] = 1 if (v_pre[i] > vth[i]) else 0   (spike indicator)
#       ∂out/∂v_pre = s ; ∂out/∂vth = -s
#       ∂V_post/∂v_pre = (1-s) ; ∂V_post/∂vth = s
for t in T-1 .. 0:
    for k in K-1 .. 0:
        i = t*K + k
        s = (v_pre[i] > vth[i])     # 需 recompute v_pre 或前向存 v_pre；现有 bwd kernel 是 recompute
        # 进入这一帧的总 ∂L/∂V_post[i]:
        g_Vpost_i = acc_v_from_next_frame_within_token   # 同 token 内 frame i+1 传回的 (= β[i+1] * g_v_pre[i+1]，见现有 kernel)
        if k == k_t_cols[t, c]:
            g_Vpost_i += g_v_init_of_next_token          # 下 token frame 0 的 patch 传回的（在外层维护）
        # g_out[i] 来自层级:
        g_out_i = g_out[i, c]
        # V_post[i] = v_pre[i] - out[i] ; out[i] = relu(v_pre[i] - vth[i])
        # ∂L/∂v_pre[i] = g_Vpost_i * (1-s) + g_out_i * s        (链：V_post 经 (1-s)，out 经 s)
        #              = g_Vpost_i + s*(g_out_i - g_Vpost_i)
        g_v_pre_i = g_Vpost_i + s*(g_out_i - g_Vpost_i)
        # ∂L/∂vth[i] += g_Vpost_i * s + g_out_i * (-s) = s*(g_Vpost_i - g_out_i)
        acc_vth += s*(g_Vpost_i - g_out_i)              # rowparam: 累加到 grad_vth_row[c]；non-rowparam: 写 grad_vth[i,c]
        # v_pre[i] = β[i]·v[i-1] + u[i]
        # ∂L/∂u[i] = g_v_pre_i
        grad_u[i, c] = g_v_pre_i
        # ∂L/∂β[i] += g_v_pre_i * v[i-1]  （v[i-1] = V_post[i-1] 或 patch 值；non-rowparam 写 grad_β[i,c]，rowparam 累加 grad_β_row[c]）
        v_prev = (V_post[i-1] if k>0 else (V_post[(t-1)*K + k_t_cols[t-1,c]] if t>0 else v_init[c]))
        acc_β += g_v_pre_i * v_prev                     # rowparam 累加；non-rowparam 写
        # ∂L/∂v[i-1] = g_v_pre_i * β[i]  → 传给上一帧（k>0）或上 token 的 patch 来源（k==0, t>0）或 v_init（k==0, t==0）
        if k > 0:
            acc_v_from_next_frame_within_token = g_v_pre_i * β[i]   # 传给 frame i-1（同 token）
        else:  # k == 0
            if t > 0:
                # 传给「上 token 第 k_{t-1} 帧的 V_post」—— 外层在处理上 token 的 frame k_{t-1} 时把这个加进 g_Vpost
                g_v_init_of_next_token = g_v_pre_i * β[i]
                acc_v_from_next_frame_within_token = 0.0   # 上 token 的 frame K-1 不从这里收（它收 0，因为 token 内 frame K-1 是死帧除非 K-1==k_t）
            else:
                g_v_init = g_v_pre_i * β[i]
# 输出: grad_u (TK, num_cols), grad_β (rowparam: (num_cols,); non-rowparam: (TK, num_cols)),
#       grad_vth (同 β), grad_v_init (num_cols,)
```

要点：
- **与现有 `_fused_plif_bwd_*_kernel` 的唯一差别**：在 token 边界（k==0, t>0），「传给上一帧」的梯度不是传给 frame i-1（那是上 token 的 frame K-1），而是传给「上 token 的 frame k_{t-1}」—— 外层用一个 `g_patch[t]` 数组承接，在逆序到上 token 的 frame k_{t-1} 时把它加进该帧的 `g_Vpost`。
- **训练里 frame k_t+1..K-1 的梯度**：它们 `g_out[i]` 来自 y_soft（小），且它们的 `g_Vpost` 来自 frame i+1（在 token 内），最终汇到 u/β/vth。这是训练的正常梯度，不影响推理（推理不反向）。
- **recompute vs store**：现有 kernel 反向时 recompute `v_pre`（重跑前向）。segmented 版同理 recompute，但 recompute 也要带 token 边界 patch（用 forward 存的 V_post 或重跑）。最简单：forward 存 `V_post` 全帧（已经存了），backward recompute `v_pre[i] = β[i]·v_prev + u[i]`，其中 `v_prev` 按上面的规则取（同 token frame i-1 的 V_post / 上 token frame k_{t-1} 的 V_post / v_init）。所以 backward 只需 `u`, `β`, `vth`, `V_post`, `v_init`, `k_t_cols`, `g_out`，不需额外存 `v_pre`。

## 3. 集成（哪些神经元用哪个 kernel）

- **rowparam segmented kernel**（β/vth per-column 常数）：`input_neuron1`、`input_neuron2`、`snn_ffn.gate_neuron`、`snn_ffn.up_neuron`、`output_neuron`
- **non-rowparam (selective) segmented kernel**（β/vth per-frame data-dependent）：`snn_block.hidden_neuron`

接口（autograd.Function 包装，正向调 fwd kernel、反向调 bwd kernel）：
```python
output, V_post, v_carry = SegmentedPLIFRowParam.apply(beta_row, u, vth_row, v_init, k_t_cols, training)
# u: (TK, num_cols); beta_row/vth_row/v_init: (num_cols,); k_t_cols: (T, num_cols) int; training: bool
# output/V_post: (TK, num_cols); v_carry: (num_cols,) detached
```
层级用法（以 SNNDecoderLayer block 子层为例）：
```python
k_logits = self.block_k_predictor(h)                 # (T, b, K)
y_st, y_hard = _ponder_v3_pick(k_logits, ...)        # (T, b, K)
k_t = y_hard.argmax(-1)                              # (T, b) int  ← 用 y_hard（含 exploration）的 argmax
k_t_cols_in  = k_t.repeat_interleave(D,  dim=1).reshape(T, b*D)    # input_neuron: hidden=D
k_t_cols_hid = k_t.repeat_interleave(DN, dim=1).reshape(T, b*DN)   # hidden_neuron: hidden=DN
# input_neuron (rowparam):
u_in = (1 - beta_in) * h_normed_flat                 # (TK, b*D)  ... wait — h_normed 是 token-level (T,b,D),
                                                     #   需 expand K 份: h_normed.unsqueeze(1).expand(T,K,b,D).reshape(TK,b*D) * scale
out_in, Vpost_in, vcarry_in = SegmentedPLIFRowParam.apply(beta_in_row, u_in, vth_in_row, v_init_in, k_t_cols_in, self.training)
input_neuron1.v = vcarry_in   # (b*D,) → reshape (b, D)
# snn_block 用 out_in (reshape 回 (TK,b,D)) 做投影 + non-rowparam segmented kernel for hidden_neuron, 同样传 k_t_cols_hid
# gather: frames = out_block.view(T,K,b,D); combined = (y_st.permute(0,2,1).unsqueeze(-1) * frames).sum(1)
```
注意：**input_neuron 和 snn_block.hidden_neuron 用同一个 k_t**（都是 block 子层的 k_t^block）；gate/up_neuron 用 k_t^ffn；output_neuron 用 k_t^out（P4 给它加 k_predictor）。

## 4. 验证（必须通过才进 P4）

1. **bit-exact 早停**：构造固定 k_t（绕过 Gumbel/exploration，直接喂 y_hard），分别跑 `training=True`（全 K）和 `training=False`（早停），断言 `output[gather at k_t]`、`v_carry`、层级 logits 逐元素相等（fp32 下严格 ==，bf16 下 ~0）。
2. **gradcheck**：小 num_cols + 小 T,K，`torch.autograd.gradcheck(SegmentedPLIFRowParam.apply, (beta, u, vth, v_init, k_t_cols, True))`（fp64）—— 验证 fwd/bwd 一致，含 token 边界 patch 路径。同理 non-rowparam 版。
3. **退化检查**：当所有 `k_t == K-1`（即从不早停），segmented kernel 的输出应与现有连续 kernel **逐 bit 相同**（patch 值 = frame K-1 = 连续递推自然结果）。这是「不破坏现有行为」的回归测试。
4. **端到端**：小模型 forward + backward + generate_cached，loss/grad finite，E[K] 分布合理。

## 5. 实现顺序

- P3b-1: 写 `SegmentedPLIFRowParam` (fwd + bwd kernel) + autograd.Function。测：退化检查（all k_t=K-1 == 现有 rowparam kernel）+ gradcheck。
- P3b-2: 写 `SegmentedPLIFSelective` (non-rowparam, fwd + bwd) for snn_block.hidden_neuron。测：退化检查 + gradcheck。
- P3b-3: 集成进 `_input_neuron_parallel`、`snn_block.forward_parallel`、`snn_ffn.forward_parallel`（layer 传 k_t_cols 下去）。output_neuron 暂保持 P2 行为（P4 改）。测：bit-exact 早停 + 端到端。
- P3b-4: commit。
