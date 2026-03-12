"""
HyperConnection: 流形约束超连接（DeepSeek mHC, arXiv 2512.24880）

替代标准残差连接 x + f(x) 为多流残差混合:
  x_{l+1} = H_res @ x_l + H_post ⊗ f(H_pre @ x_l)

核心保证:
  - H_res 约束在 Birkhoff 多面体（双随机矩阵集）
  - 双随机矩阵谱范数 ≤ 1，对乘法封闭 → 梯度不爆炸的代数保证
  - ∏_l ‖H_res^(l)‖ ≤ 1 — 无论网络多深，残差混合不放大信号
  - Sinkhorn-Knopp 投影: log 域交替行列归一化

动态 H 矩阵（输入依赖，条件信号 = vec(x_l) 展平 nD 维）:
  c = RMSNorm(vec(x_l))                                    ∈ R^{nD}
  H_pre  = σ(α_pre · (c @ θ_pre) + b_pre)                 ∈ (0,1)^n
  H_post = 2·σ(α_post · (c @ θ_post) + b_post)            ∈ (0,2)^n
  H_res  = Sinkhorn(α_res · (c @ θ_res) + b_res)          ∈ DS^{n×n}

初始化（精确等价标准残差，streams 相同时）:
  θ = 0, α = 0.01
  b_pre = logit(1/n) → sigmoid = 1/n → H_pre @ x = x
  b_post ≈ 0 + noise → 2·sigmoid ≈ 1.0 ± 5%（打破流间对称）
  b_res = I → Sinkhorn(I) 对角占优（streams 相同时乘以行和=1 → 等价 I）

Sinkhorn 梯度:
  直接反传 Sinkhorn 迭代会导致梯度指数衰减（Jacobian 谱半径 < 1）。
  使用 SinkhornProjection 自定义反向: 前向完整 Sinkhorn 投影，
  反向用切空间投影梯度（隐式微分的一阶近似）。
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
    """流形约束超连接模块（单子层用）。

    使用 pre()/post() 分离接口，不包装子层本身:
      x_pre, cache = hc.pre(x)        # n 流 → 1 维子层输入
      f_out = sublayer(x_pre)          # 子层处理（不变）
      x_out = hc.post(x, f_out, cache) # 残差混合 + 分配 → n 流

    参数量开销（n=4, D=896）: ~90K/instance, 2 instances/layer, 20 layers → ~3.6M（0.4%）

    Args:
        n: 流数量（推荐 4）
        D: 隐藏维度
        sinkhorn_iters: Sinkhorn 迭代次数
        alpha_init: 动态 H 矩阵初始缩放（越小越接近静态初始值）
    """

    def __init__(self, n: int, D: int, sinkhorn_iters: int = 20,
                 alpha_init: float = 0.01):
        super().__init__()
        self.n = n
        self.D = D
        self.sinkhorn_iters = sinkhorn_iters

        # 输入归一化（条件信号 = vec(x_l) 展平 nD 维，保留完整流间信息）
        self.norm = RMSNorm(n * D)

        # H_pre: 聚合权重 — 哪些流参与子层输入
        self.theta_pre = nn.Parameter(torch.zeros(n * D, n))
        self.b_pre = nn.Parameter(torch.full((n,), -math.log(n - 1)))  # sigmoid → 1/n
        self.alpha_pre = nn.Parameter(torch.tensor(alpha_init))

        # H_post: 分配权重 — 子层输出如何分配到各流
        # b_post 加小噪声打破流间对称（加速 H_res 梯度涌现）
        self.theta_post = nn.Parameter(torch.zeros(n * D, n))
        self.b_post = nn.Parameter(0.1 * torch.randn(n))  # 2·sigmoid(~0) ≈ 1.0 ± 5%
        self.alpha_post = nn.Parameter(torch.tensor(alpha_init))

        # H_res: 残差混合矩阵 — 双随机约束，谱范数 ≤ 1
        self.theta_res = nn.Parameter(torch.zeros(n * D, n * n))
        self.b_res = nn.Parameter(torch.eye(n))  # Sinkhorn(I) → 对角占优
        self.alpha_res = nn.Parameter(torch.tensor(alpha_init))

    def _compute_H(self, x):
        """从 n 流输入计算三个 H 矩阵（动态，输入依赖）。

        Args:
            x: (*, n, D) — n 流输入

        Returns:
            H_pre:  (*, 1, n) — 聚合权重
            H_post: (*, n)    — 分配权重
            H_res:  (*, n, n) — 双随机混合矩阵
        """
        # vec(x_l): 展平 n 流为 nD 维条件信号，保留完整流间区分信息
        x_flat = x.flatten(-2)  # (*, n, D) → (*, nD)
        x_norm = self.norm(x_flat)  # (*, nD)

        # H_pre: σ(·) → (0, 1)^n，聚合各流的相对贡献
        pre_logits = self.alpha_pre * (x_norm @ self.theta_pre) + self.b_pre
        H_pre = torch.sigmoid(pre_logits).unsqueeze(-2)  # (*, 1, n)

        # H_post: 2·σ(·) → (0, 2)^n，子层输出的分配强度
        post_logits = self.alpha_post * (x_norm @ self.theta_post) + self.b_post
        H_post = 2.0 * torch.sigmoid(post_logits)  # (*, n)

        # H_res: Sinkhorn → 双随机矩阵，谱范数 ≤ 1 保证梯度不爆炸
        n = self.n
        res_logits = self.alpha_res * (x_norm @ self.theta_res)  # (*, n*n)
        shape = res_logits.shape[:-1] + (n, n)
        res_logits = res_logits.view(shape) + self.b_res  # (*, n, n)
        H_res = sinkhorn_projection(res_logits, self.sinkhorn_iters)  # (*, n, n)

        return H_pre, H_post, H_res

    def pre(self, x):
        """聚合 n 流 → 子层输入。

        Args:
            x: (*, n, D) — n 流残差

        Returns:
            x_pre: (*, D) — 子层输入（n 流加权聚合）
            H_cache: tuple — 缓存的 H 矩阵，传给 post()
        """
        H_pre, H_post, H_res = self._compute_H(x)
        # (*, 1, n) @ (*, n, D) → (*, 1, D) → squeeze → (*, D)
        x_pre = torch.matmul(H_pre, x).squeeze(-2)
        return x_pre, (H_pre, H_post, H_res)

    def post(self, x, f_out, H_cache):
        """残差混合 + 输出分配 → n 流。

        x_{l+1} = H_res @ x_l + H_post ⊗ f(x_pre)

        Args:
            x: (*, n, D) — 原始 n 流输入
            f_out: (*, D) — 子层输出
            H_cache: tuple — 从 pre() 返回的 H 矩阵

        Returns:
            (*, n, D) — 更新后的 n 流残差
        """
        _, H_post, H_res = H_cache

        # 残差混合: (*, n, n) @ (*, n, D) → (*, n, D)
        h_res = torch.matmul(H_res, x)

        # 输出分配: (*, n) ⊗ (*, D) → (*, n, D)
        z = torch.einsum("...n,...d->...nd", H_post, f_out)

        return h_res + z
