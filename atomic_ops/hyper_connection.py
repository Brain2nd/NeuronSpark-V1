"""
HyperConnection: 多流超连接（身份跳跃 + 动态聚合/分配）

多流残差连接:
  x_{l+1} = x_l + H_post ⊗ f(H_pre @ x_l)

设计原理:
  - 身份跳跃连接: 梯度高速公路，Jacobian = I，所有方向梯度无衰减
  - H_pre: 动态聚合 n 流 → 1 维子层输入（softmax + 方差重归一化）
  - H_post: 动态分配子层输出 → n 流（2·sigmoid ∈ (0,2)）
  - 流间信息交换通过计算路径（H_pre 聚合 → 子层 → H_post 分配）

  注: H_res (Birkhoff 双随机混合) 已移除。实验表明 H_res 的 n×n 矩阵
  在 skip 路径上引入非身份 Jacobian，非均匀方向特征值 λ₂ = 0.893，
  40 个子层后衰减 0.893^40 ≈ 0.011，导致深层梯度消失/浅层梯度爆炸。
  身份跳跃连接在所有方向保持完整梯度流，与 SubLN 互补。

动态 H 矩阵（输入依赖，条件信号 = vec(x_l) 展平 nD 维）:
  c = RMSNorm(vec(x_l))                                    ∈ R^{nD}
  H_pre  = softmax(α_pre · (c @ θ_pre) + b_pre) / √(Σw²)  ∈ R^n, 方差保持
  H_post = 2·σ(α_post · (c @ θ_post) + b_post)            ∈ (0,2)^n

SNN 适配:
  - H_pre 用 softmax 替代 sigmoid: 凸组合保证权重和 = 1
  - 方差重归一化 ÷√(Σw_i²): 防止凸组合压缩方差导致膜电位低于阈值

初始化:
  θ = 0, α = 0.01
  b_pre = 0 → softmax(0) = 1/n（均匀聚合）
  b_post ≈ 0 + noise → 2·sigmoid ≈ 1.0 ± 5%（打破流间对称）
"""

import math

import torch
import torch.nn as nn

from .rms_norm import RMSNorm


def sinkhorn_log(logits: torch.Tensor, num_iters: int = 20) -> torch.Tensor:
    """Log 域 Sinkhorn-Knopp: 投影到双随机矩阵集（Birkhoff 多面体）。

    数学: 交替行列归一化 ≡ 交替 KL 投影到行/列约束集。
    收敛保证: 任何正矩阵经有限步 Sinkhorn 收敛到双随机矩阵。

    数值稳定性: bf16 下 logsumexp 精度不足，使用 float32 计算。

    Args:
        logits: (..., n, n) — 未归一化 log 域输入
        num_iters: 迭代次数（n≤8 时 20 次足够收敛到 <1e-6 误差）

    Returns:
        (..., n, n) — 双随机矩阵（非负，行列和均为 1）
    """
    dtype = logits.dtype
    logits = logits.float()
    for _ in range(num_iters):
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)  # 行归一化
        logits = logits - torch.logsumexp(logits, dim=-2, keepdim=True)  # 列归一化
    return torch.exp(logits).to(dtype)


class _SinkhornProjection(torch.autograd.Function):
    """Sinkhorn 投影的自定义前向/反向。

    前向: 完整 Sinkhorn 迭代（20 步），保证输出为双随机矩阵。
    反向: 切空间投影梯度，避免迭代 Jacobian 的指数衰减。

    反向梯度推导:
      在双随机矩阵流形 DS_n 的切空间 T_H 上，切向量满足行列和均为 0。
      对任意梯度 G = ∂L/∂H，投影到 T_H:
        P(G) = G - r·1^T - 1·c^T + μ·1·1^T
      其中 r_i = mean_j(G_ij), c_j = mean_i(G_ij), μ = mean_ij(G_ij)。
      链式法则: ∂L/∂logits ≈ H ⊙ P(∂L/∂H)  (exp 的 Jacobian × 切空间投影)
    """

    @staticmethod
    def forward(ctx, logits, num_iters):
        dtype = logits.dtype
        M = logits.detach().float()
        for _ in range(num_iters):
            M = M - torch.logsumexp(M, dim=-1, keepdim=True)
            M = M - torch.logsumexp(M, dim=-2, keepdim=True)
        result = torch.exp(M)
        # 保存 H（切空间投影用）和 logits（梯度传播到输入）
        ctx.save_for_backward(result)
        ctx.dtype = dtype
        return result.to(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        H, = ctx.saved_tensors  # float32
        grad = grad_output.float()

        # exp 的 Jacobian: ∂exp(M)/∂M = diag(exp(M)) → element-wise 乘 H
        Hg = H * grad

        # 切空间投影: 减去行均值、列均值，加回全局均值
        # 保证投影后行和 = 0, 列和 = 0
        row_mean = Hg.mean(dim=-1, keepdim=True)
        col_mean = Hg.mean(dim=-2, keepdim=True)
        total_mean = Hg.mean(dim=(-2, -1), keepdim=True)
        grad_logits = Hg - row_mean - col_mean + total_mean

        return grad_logits.to(ctx.dtype), None


def sinkhorn_projection(logits: torch.Tensor, num_iters: int = 20) -> torch.Tensor:
    """Sinkhorn 投影（自定义反向，避免梯度衰减）。

    前向: 完整 Sinkhorn → 双随机矩阵。
    反向: 切空间投影梯度 → 有效梯度传播。

    Args:
        logits: (..., n, n) — 未归一化输入
        num_iters: Sinkhorn 迭代次数

    Returns:
        (..., n, n) — 双随机矩阵
    """
    return _SinkhornProjection.apply(logits, num_iters)


class HyperConnection(nn.Module):
    """多流超连接模块（单子层用）。

    使用 pre()/post() 分离接口，不包装子层本身:
      x_pre, cache = hc.pre(x)        # n 流 → 1 维子层输入
      f_out = sublayer(x_pre)          # 子层处理（不变）
      x_out = hc.post(x, f_out, cache) # 身份跳跃 + 分配 → n 流

    身份跳跃: x_{l+1} = x_l + H_post ⊗ f(H_pre @ x_l)
    梯度高速公路: ∂x_{l+1}/∂x_l = I + O(gain)，所有方向梯度保持。

    参数量开销（n=4, D=896）: ~33K/instance, 2 instances/layer, 20 layers → ~1.3M（0.15%）

    Args:
        n: 流数量（推荐 4）
        D: 隐藏维度
        sinkhorn_iters: 保留兼容性（不再使用）
        alpha_init: 动态 H 矩阵初始缩放（越小越接近静态初始值）
    """

    def __init__(self, n: int, D: int, sinkhorn_iters: int = 20,
                 alpha_init: float = 0.01):
        super().__init__()
        self.n = n
        self.D = D

        # 输入归一化（条件信号 = vec(x_l) 展平 nD 维，保留完整流间信息）
        self.norm = RMSNorm(n * D)

        # H_pre: 聚合权重 — softmax + 方差重归一化（SNN 阈值兼容）
        self.theta_pre = nn.Parameter(torch.zeros(n * D, n))
        self.b_pre = nn.Parameter(torch.zeros(n))  # softmax(0) = 1/n
        self.alpha_pre = nn.Parameter(torch.tensor([alpha_init]))

        # H_post: 分配权重 — 子层输出如何分配到各流
        # b_post 加小噪声打破流间对称
        self.theta_post = nn.Parameter(torch.zeros(n * D, n))
        self.b_post = nn.Parameter(0.1 * torch.randn(n))  # 2·sigmoid(~0) ≈ 1.0 ± 5%
        self.alpha_post = nn.Parameter(torch.tensor([alpha_init]))

    def _compute_H(self, x):
        """从 n 流输入计算 H_pre 和 H_post 矩阵（动态，输入依赖）。

        Args:
            x: (*, n, D) — n 流输入

        Returns:
            H_pre:  (*, 1, n) — 聚合权重（softmax + 方差重归一化）
            H_post: (*, n)    — 分配权重
        """
        # vec(x_l): 展平 n 流为 nD 维条件信号，保留完整流间区分信息
        x_flat = x.flatten(-2)  # (*, n, D) → (*, nD)
        x_norm = self.norm(x_flat)  # (*, nD)

        # H_pre: softmax → 凸组合 + 方差重归一化（SNN 阈值兼容）
        # softmax 保证 Σw=1; ÷√(Σw²) 补偿方差压缩: Var(x_pre) = σ²
        # 初始 w=1/n 时 renorm=√n，被下游 RMSNorm 吸收
        pre_logits = self.alpha_pre * (x_norm @ self.theta_pre) + self.b_pre
        H_pre_w = torch.softmax(pre_logits, dim=-1)  # (*, n), Σ=1
        renorm = torch.rsqrt((H_pre_w * H_pre_w).sum(dim=-1, keepdim=True).clamp(min=1e-8))
        H_pre = (H_pre_w * renorm).unsqueeze(-2)  # (*, 1, n) 融合方差重归一化

        # H_post: 2·σ(·) → (0, 2)^n，子层输出的分配强度
        post_logits = self.alpha_post * (x_norm @ self.theta_post) + self.b_post
        H_post = 2.0 * torch.sigmoid(post_logits)  # (*, n)

        return H_pre, H_post

    def pre(self, x):
        """聚合 n 流 → 子层输入。

        Args:
            x: (*, n, D) — n 流残差

        Returns:
            x_pre: (*, D) — 子层输入（n 流加权聚合）
            H_cache: tuple — 缓存的 H_post，传给 post()
        """
        H_pre, H_post = self._compute_H(x)
        # (*, 1, n) @ (*, n, D) → (*, 1, D) → squeeze → (*, D)
        x_pre = torch.matmul(H_pre, x).squeeze(-2)
        return x_pre, H_post

    def post(self, x, f_out, H_cache):
        """身份跳跃 + 输出分配 → n 流。

        x_{l+1} = x_l + H_post ⊗ f(x_pre)

        身份跳跃连接保证 ∂x_{l+1}/∂x_l = I（所有方向无衰减）。
        H_post 控制子层输出分配到各流的强度 ∈ (0, 2)。

        Args:
            x: (*, n, D) — 原始 n 流输入
            f_out: (*, D) — 子层输出
            H_cache: H_post tensor — 从 pre() 返回的分配权重

        Returns:
            (*, n, D) — 更新后的 n 流残差
        """
        H_post = H_cache

        # 输出分配: (*, n) ⊗ (*, D) → (*, n, D)
        z = torch.einsum("...n,...d->...nd", H_post, f_out)

        return x + z
