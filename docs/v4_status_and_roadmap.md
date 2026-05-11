# V4 — Status & Roadmap

**Branch**: `v4` (基于 `v3` @ 4e21ccf 分出；备份 tag `v3-backup-pre-stage3.1` 指向 v3 当时状态)
**核心目标**: 让 PonderNet 的 per-token 动态 k_t 真正省 FLOP/显存（推理只跑 k_t 步），修正 V2.5/v3 架构里破坏早停的所有耦合点，且训练/推理语义一致。

---

## 1. 已实现（V4 当前 = commit 4632dc3）

### 1.1 删 conv1d（Mamba-3 路线）
- `SNNBlock` 的 V2.5 因果卷积（`conv1d`, kernel=4, depthwise）被删。它沿 TK 轴卷，把相邻 token 的 K 帧块耦合起来 → 破坏早停。Mamba-3 (ICLR 2026) 证明 external short conv 可被递推内化、加回反而略降。局部上下文由 SNNAttentionDecoderLayer + 残差流承担。
- 同步删 `conv_state`、`get_param_groups` 的 conv1d 组、`utils/param_groups.py` 的 conv1d 路由。

### 1.2 残差流 → token-level (T, B, D)
- **实测验证**（`tests/verify_residual_redundant.py`）：v3 残差流的 K 维是严格逐 bit 冗余（fp32/bf16、每层、K 轴偏差 = 0.000e+00）。
- V4 把残差流改为 `(T, B, D)` token-level；K 展开只在 SNN block / SNNFFN / output_neuron 内部局部进行。去掉所有 `view(seq,K)` / `repeat_interleave(K)` / attn 的 `mean(dim=1)→expand(K)` 往返。数值零偏差。
- `encode()` 返回 `(seq_len, batch, D)`；各层 `forward_parallel` 收发 `(T, B, D)`；SNNAttentionDecoderLayer 的 attn 子层原生 token 级。

### 1.3 forward reorder + segmented PLIF（携带第 k_t 帧）
- **判据**：「需要携带第 k_t 帧」= 沿 TK（per-token-K-frame）轴递推的 PLIF —— 不是「所有 PLIF」也不是「按选择性/非选择性区分」。SNNAttentionDecoderLayer 的 `gate_neuron` 沿 token 轴递推、没有 K 维 → 豁免；`M_state`/`pos_offset`（token 级）、`KPredictor.bias`/`_usage_ema`（全局 stats）也豁免。
- **6 处 over-TK PLIF 状态**全改成「携带第 k_t 帧的膜电位」（旧的 `plif_*_v2` 返回第 K 帧）：`input_neuron1` / `input_neuron2`（rowparam）、`snn_block.hidden_neuron`（**selective**，最关键 —— SSM-state 类比）、`snn_ffn.gate_neuron` / `up_neuron`（rowparam，merged）、`output_neuron`（rowparam）。各子层用各自的 k_t（block 子层 k_t^block、FFN 子层 k_t^ffn、output 读出 k_t^out）。
- **forward reorder**：`k_predictor` 提前到 SNN block 之前（早停时 k_t 必须在跑帧前已知）；`_ponder_aggregate_v3` 拆成 `pick_k`（`_ponder_v3_pick`）+ gather（STE：forward=y_hard, backward=y_soft）。
- **output_neuron 加自己的 `output_k_predictor`**（统一原则）；`decode` 的 `mean(dim=1) over K` → 改成取第 k_t^out 帧的 V_post（与层的 gather 同构）。`set_ponder_*` / `update_ponder_bias` 也覆盖 output 路径。

### 1.4 segmented PLIF —— PyTorch reference + Triton 融合 kernel
- **PyTorch reference**（`_segmented_plif_rowparam_pytorch` / `_segmented_plif_selective_pytorch`）：per-token K-frame 递推 + token 边界把运行膜电位 patch 成上 token 第 k_{t-1} 帧的 V_post。autograd 自动反向。CPU fallback + 测试基准。
- **Triton 融合 kernel**（`_segmented_plif_fwd/bwd_rowparam_kernel` + `_*_selective_kernel`，`_SegmentedPLIF{RowParam,Selective}` autograd.Function）：runtime token 循环 + per-lane gather-load 做 token 边界 patch；`IS_TRAINING` constexpr（训练全 K / 推理早停到 batch 内 max(k_t)+1）；backward 逆序递推（`g_patch` 退化成单寄存器标量，因逆序遍历下跨 token 梯度只存活 ≤K+1 帧）。
- **dispatch**：`segmented_plif_rowparam` / `segmented_plif_selective` 在 CUDA+Triton 时走 kernel，否则 PyTorch reference。**层级代码不变**。

### 1.5 推理早停覆盖范围（含投影矩阵）
- 初版只让 PLIF 递推早停；投影矩阵（W_in/W_out/W_gate_up/etc.）仍对全 K 帧做 matmul（input_neuron 返回 dense (TK,...) 带垃圾尾帧）。已修正：每层算 `K_act = max(k_t)+1`（eval）/ `K`（train），repeat-to-K / input_neuron / 投影 / PLIF / frame view / `y_st` gather（切 `[:K_act]`，eval 时 y_st one-hot 无损）全部对 K_act 帧做。output_neuron 同。
- **实测**（D=1024, N=16, 12 层, seq=128）：eval forward k_t=11(12帧) 80.9ms / k_t=5(6帧) 59.1ms / k_t=2(3帧) 51.0ms / k_t=0(1帧) 49.4ms → **E[K]≈4.5 时 ~1.3x 加速**，极端 1.64x。activation 显存：862MB(12帧) → 455(6) → 248(3) → 112(1) → **E[K]≈4.5 时 activation 省 ~57%**（总显存这个规模下省 ~18%，参数占大头；长序列相对省更多）。**训练不变**（K_act=K=12，梯度需要全帧；梯度 checkpoint 已在）。

### 1.6 验证（已全过）
- `tests/test_segmented_plif.py`（8 个）：退化检查（all k_t=K-1 == 现有连续 kernel）/ 早停 bit-exact（gather@k_t out/V_post/v_carry 差 0.0e+00）/ fp64 gradcheck（PyTorch reference）/ rowparam+selective kernel fwd+bwd vs reference（~1e-7）。
- `tests/test_v4_end_to_end.py`：grad-flow（全参数 finite 非零梯度）+ overfit smoke（loss 下降，无 NaN，E[K] sane）。
- `tests/verify_residual_redundant.py`：残差 K 轴偏差 = 0。
- 4090 上 P5.5-5：133.5M 模型（D=512, 12 层, K=12, seq=512）overfit smoke loss 9.09→0.0073，kernel 生效（_seg_rowparam_call=20100, _seg_selective_call=5400），E[K] range [1,12]，无 NaN，479ms/step。

### 1.8 神经元设计 v4.1 —— 量子化释放 (quantal) + AHP (后超极化)，可配置

**动机**（讨论结论）：v3 默认的 bio-ReLU `output = relu(V_pre - v_th)` + 软重置 `V_post = min(V_pre, v_th)` 在「保留膜内历史上下文」上削得太狠 —— 一旦发放，膜被钉回 v_th，「这个神经元累积了多强的上下文」这个量从膜里被抹掉（只活在 output 流里）。**硬重置 (V→0) 完全不可取**（膜电位是递归状态、承载历史上下文）—— 不讨论。改进方向：**量子化释放** —— 发放只释放固定一份 `v_th`，超阈的**剩余余量留在膜里**：
```
spike = 𝟙[V_pre > v_th]                     # 硬发放 (forward); sigmoid surrogate (α) 提供反向梯度
output = v_th · spike                        # 释放固定一份 v_th (= 「V_th × spike」, 与 spike_current 语义对齐)
V_post = V_pre - v_th·spike - ahp·spike      # 剩余余量 (V_pre - v_th) 留膜里; AHP 额外下压 ahp
```
强输入 → 一串 spike (rate code, 串长 ∝ V_pre/v_th)；膜保留「还没释放成整份的余量」。**代价**：output 是阶跃 → 反向必须用 surrogate (sigmoid'(α·(V_pre-v_th)))，不像 supra 是精确 ReLU 梯度。

**AHP（后超极化 / 不应期）**：发放后膜额外下压 `ahp·𝟙[V_pre>v_th]`（ahp = per-channel 可学习参数, 初值小 ≈0）。**不导致梯度消失** —— 膜递推的 per-step 雅可比 `∂V_post/∂V_pre` 在 supra 模式下 `= 1 - s_hard`（与加 AHP 前相同，AHP·s_hard 用零梯度的硬指示子）；AHP 通过降发放率反而让长程膜梯度更通畅（少一些「发放步 = 梯度屏障」）。唯一副作用：AHP 太大时膜被恢复曲线主导 → 输入对输出贡献变小（信号衰减，非时序梯度消失）→ ahp 初值小、可学习/可 data-dependent 缓解。AHP 与现有 selective `v_th(t)` 功能分工：v_th(t) = 慢的输入条件阈值调制，AHP = 快的事件触发不应期。

**实现（可配置, config 默认 = v3 的 supra/no-ahp，向后兼容）**：
- `NeuronSparkConfig`: `spike_mode ∈ {"supra","quantal"}`、`use_ahp: bool`、`ahp_init: float`、`surrogate_alpha: float`。
- `modeling_neuronspark.py`: 模块级 `NEURON_SPIKE_MODE/USE_AHP/...`（`SNNLanguageModel.__init__` 从 config 设, 构造层之前）；`PLIFNode`/`SelectivePLIFNode` 据此创建 per-channel `ahp` 参数 + 记 `spike_mode/surrogate_alpha`；`_spike_step()` 是 fire+reset 的统一 PyTorch 实现（supra = 精确 ReLU；quantal = straight-through `s = sig + (s_hard - sig).detach()`，forward 值 = 硬指示子, 反向梯度 = sigmoid'）。
- Triton: `_segmented_plif_fwd/bwd_{rowparam,selective}_kernel` 加 `SPIKE_QUANTAL: tl.constexpr` + `ALPHA` runtime + `AHP_ROW_ptr` (per-channel, 全 0 = 无 AHP) + `GRAD_AHP_ROW_ptr`。backward 量子化梯度：`g_surr = α·σ·(1-σ)`；`∂out/∂V_pre = v_th·g_surr`；`∂V_post/∂V_pre = 1 - (v_th+ahp)·g_surr`；`∂out/∂v_th = s - v_th·g_surr`；`∂V_post/∂v_th = (v_th+ahp)·g_surr - s`；`∂V_post/∂ahp = -s`（out 与 AHP 项共用同一个 surrogate-graded s）。
- 覆盖 6 处 segmented PLIF（input_neuron1/2、hidden_neuron[selective]、ffn gate/up[merged]、output_neuron）+ `PLIFNode.forward`/`single_step_forward` 单步路径。SNNAttentionDecoderLayer 的 token 级 `_gate_neuron_parallel`（非 segmented）暂未切（dead-ish, 跟早停无关）—— 待 quantal 确认后补。
- 测试（`tests/test_segmented_plif.py`，全过）：supra+AHP fp64 gradcheck（rowparam+selective，含 ahp 参数）；quantal(+ahp) 早停 bit-exact；Triton kernel vs PyTorch reference fwd+bwd（quantal+ahp、supra+ahp、rowparam+selective，~1e-4 / grad ~2e-3）。`tests/test_v4_end_to_end.py` 4 配置 overfit smoke（loss 下降、grad finite、无 NaN；quantal 的 grad norm 小 ~3x，因 output ≈ v_th·s ≈ 0.05·s 信号幅度小）。

**待办**：① 消融 {supra / supra+AHP / quantal / quantal+AHP}（4090 四卡并行，同 P-A 那套，比 loss / E[K] / 发放率）→ 决定是否把 config 默认翻成 quantal+AHP。② quantal 若胜：W_out 初始化 / LR 可能要随 output 幅度变小调；补 token 级 gate_neuron 的 spike form；重训。

### 1.7 设计文档
- `docs/v4_early_stop_design.md` —— 早停修正的总规格（哪些耦合点、怎么改）。
- `docs/v4_segmented_plif_kernel_design.md` —— segmented PLIF 的 forward/backward 数学 + Triton kernel 实现细则（§6）。
- 本文档 —— status & roadmap。

---

## 2. 已知限制 / 现状结论

- **推理加速 ~1.3x（典型 E[K]）** —— modest。原因：有个 ~49ms 的「k_t 无关地板」（3 个 attention 层 token 级·k_t 无关 + decode/lm_head + ~1000 个 kernel launch 开销/forward + 1-帧 SNN 不可避免成本 + 11+ 个 `.item()` 同步）占推理成本 ~60%。早停只能省「SNN block 的 K 帧计算」那 ~40%（matmul + PLIF）。对比 v3（dynamic K 对 FLOP 毫无作用），是进步但不戏剧性。
- **训练成本不变** —— 早停是推理-only 设计；训练跑全 K（梯度需要）。
- **必须重新预训练** —— V4 删了 conv、改了 PLIF 携带语义、残差布局、加了 output_k_predictor → 旧 v3 ckpt 不兼容。

---

## 3. 规划 / 已排除

### P-B（降级 —— 用户指示优先级靠后）：CUDA graphs / kernel 融合
- 砍 ~1000 个 kernel launch 开销/forward（「k_t 无关地板」大头）。CUDA graphs 把整个 forward 捕成一个 graph → 每步 ~零 launch 开销 → 地板砍约一半 → 早停相对收益再变大。短序列 decode 时这是最大杠杆（FLOP 小、launch 开销主导）。注：也是 SNN block 单 token wall-clock 的真瓶颈所在（segmented PLIF kernel memory-bound + 一堆小 kernel），不是投影 matmul（见下 P-A 失败结论）。

### P-C（待定）：full 预训练（V4 从头训）
- 大资源投入（多卡、大数据、几天）。需定：在哪训、规模、数据、K_max。V4 必须重训（架构改了）。

### 不做（已排除）

**P-A 低秩化 SNN block 投影 —— 试过, 不可取, 已 revert（commit 90087fd + bfb8f95 revert 了 b978ba9 + 2b80dd9）。**
- 改法曾是：`W = A·B`（r≪min(in,out)）对 W_in/W_beta_x/W_alpha_x/W_th_x/W_out（SNNBlock）+ gate/up/down（SNNFFN）；不动 K 步串行选择性动力学。
- **ablation 实测**（D=512, N=16, 12 层, seq=512, batch=4, 1500 step, SFT 数据切片, 4 卡并行）：
  | rank | params | final_loss(last100) | ms/step | vs full |
  |---|---|---|---|---|
  | full | 289.6M | 7.290 | 1206 | baseline |
  | r=256 | 196.4M | 7.408 | 1099 (~9%快) | +0.118 |
  | r=128 | 139.2M | 7.429 | 1049 (~13%快) | +0.139 |
  | r=64 | 110.5M | 7.423 | 1038 (~14%快) | +0.133 |
- **结论 / 为什么不可取**：
  1. 低秩 **+0.13 loss（≈ ppl 高 ~14%）** vs 满秩 —— 有质量损失, 不是「无损」。
  2. **r=64≈r=128≈r=256**（差异在噪声内）→ gap 不是低秩瓶颈造成的, 是**参数量差异**（满秩 290M vs 低秩 110-200M）。即低秩只是「换个更小的模型」, 不是「同模型更省」。
  3. **速度只省 ~10-14%**（远不是预期的「matmul ×10 少」）—— 因为 SNN block 单 token wall-clock 的大头是 **segmented PLIF kernel**（memory-bound, DN=8192 维 × K 步, 跟 proj rank 无关），不是投影 matmul。投影低秩只省了那 ~50-60% 的投影 FLOP, 对总 wall-clock 只 ~10-14%。
  → 「想 290M 模型快」: 瓶颈是 PLIF kernel（→ CUDA 工程 / 减小 N, 都不是「架构层面、不碰旋钮、不碰动力学」的可取选项）。低秩对此无帮助。

**其它已排除（破坏神经动力学）**：
- 「投影一次」（K 步递推输入变常数 → 退化成 settling 曲线 → 废 SNN 串行动力学）。
- 连续 Δ_t（消掉 K 维 → 1 步 → 就不是 SNN 了，跟「唯一创新是 Mamba→BioSSM」冲突）。
- 调旋钮（N / K_max / attention interval）不算架构改动；K_max=8 那种只省训练、推理早停已覆盖。

---

## 4. 文件 / 测试索引
- 核心代码：`neuronspark/modeling_neuronspark.py`（segmented PLIF 函数 + Triton kernel + autograd.Function + 层级集成 + output_neuron k_predictor）
- 测试：`tests/test_segmented_plif.py`（8）、`tests/test_v4_end_to_end.py`（2）、`tests/verify_residual_redundant.py`
- 设计：`docs/v4_early_stop_design.md`、`docs/v4_segmented_plif_kernel_design.md`、本文档
- 备份：tag `v3-backup-pre-stage3.1`（v3 改动前状态）
