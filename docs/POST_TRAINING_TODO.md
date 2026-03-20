# 训练完成后、上下文增程前的待办事项

## 1. 归一化层统一为 LateralInhibition（侧向抑制）

当前状态：模型中同时存在 RMSNorm 和 LateralInhibition，数学等价但参数名不同。

- RMSNorm（参数名 `weight`）：用于所有层的 Pre-LN（block_norm, ffn_norm, attn_norm, attn_out_norm）
- LateralInhibition（参数名 `gain`）：仅用于输出层

统一方案：全部替换为 LateralInhibition
- 有 Triton kernel 加速
- SNN 框架术语一致（divisive normalization, Carandini & Heeger 2012）
- 需要 checkpoint key 映射：`*.weight` → `*.gain`（对所有 norm 层）

注意：训练期间不能改，会破坏 checkpoint 兼容性。

## 2. health/score 指标修正

当前 score 公式依赖假的 epileptic_rate 估算（硬编码 α·I=0.2），导致得分虚低。
训练完后重新设计 score 公式，或移除 epileptic_rate 的权重。

## 3. 上下文增程实验

参见 `docs/CONTEXT_EXTENSION.md`：
- β 偏置校准（免训练推理时加 Δb）
- SNN-Attention 层 RoPE 插值（PI/NTK/YaRN 修改 rope_base）
- 联想记忆层写入门控阈值调整
