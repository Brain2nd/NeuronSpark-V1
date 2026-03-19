"""
SNNAssociativeMemoryLayer: 基于短时突触可塑性的脉冲联想记忆

数学定义:
  写入: M[t] = β_M · M[t-1] + write_gate[t] · k[t] · v[t]ᵀ
  读出: output[t] = q[t]ᵀ · M[t]

所有控制信号由 PLIFNode 产生:
  x → PLIFNode_k → (1-β_k)·V_post → W_k → k
  x → PLIFNode_v → (1-β_v)·V_post → W_v → v
  x → PLIFNode_q → (1-β_q)·V_post → W_q → q
  x → PLIFNode_g → spike → write_gate (spike 门控: 只有发放时才写入)

生物对应: 短时突触可塑性 (Short-Term Synaptic Plasticity)
参考: Qwen3.5 GatedDeltaNet, DeltaNet 线性注意力
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import base, surrogate

from .plif_node import PLIFNode
from .rms_norm import RMSNorm
from .parallel_scan import plif_rowparam_forward


class SNNAssociativeMemoryLayer(base.MemoryModule):
    """
    SNN 联想记忆层 — spike 驱动的矩阵状态读写。

    替代标准注意力的全局感知层，O(n) 复杂度，O(1) 状态大小。

    Args:
        D: 可见维度
        D_key: 键/查询维度 (控制记忆"槽位"数)
        D_value: 值维度 (每个槽位存储的信息量)
        num_memory_groups: 多时间尺度记忆组数 (不同 β_M)
        beta_M_range: 记忆衰减率范围 (min, max)
        num_layers: 总层数 (用于输出缩放)
        activation_mode: v1/v2 激活模式
    """

    def __init__(
        self,
        D: int,
        D_key: int = 128,
        D_value: int = 128,
        num_memory_groups: int = 3,
        beta_M_range: tuple = (0.95, 0.9999),
        num_layers: int = 1,
        activation_mode: str = 'v2',
    ):
        super().__init__()
        self.D = D
        self.D_key = D_key
        self.D_value = D_value
        self.num_memory_groups = num_memory_groups
        self.activation_mode = activation_mode

        # ====== 输入 PLIFNode（产生 spike 驱动的 k/v/q/gate） ======
        self.neuron_k = PLIFNode(dim=D, init_tau=2.0, v_threshold=0.5)
        self.neuron_v = PLIFNode(dim=D, init_tau=2.0, v_threshold=0.5)
        self.neuron_q = PLIFNode(dim=D, init_tau=2.0, v_threshold=0.3)
        self.neuron_gate = PLIFNode(dim=D, init_tau=2.0, v_threshold=0.8)  # 高阈值：严格筛选

        # ====== 投影 ======
        self.W_k = nn.Linear(D, D_key, bias=False)
        self.W_v = nn.Linear(D, D_value, bias=False)
        self.W_q = nn.Linear(D, D_key, bias=False)
        self.W_out = nn.Linear(D_value * num_memory_groups, D, bias=False)

        # ====== 输入/输出归一化 ======
        self.norm = RMSNorm(D)

        # ====== 多时间尺度记忆衰减率 β_M ======
        beta_M_values = torch.linspace(beta_M_range[0], beta_M_range[1], num_memory_groups)
        self.register_buffer('beta_M', beta_M_values)  # (num_memory_groups,)

        # ====== 记忆矩阵 M: (num_groups, D_key, D_value) ======
        self.register_memory('M', 0.)

        # ====== 初始化 ======
        self._init_weights(num_layers)

    def _init_weights(self, num_layers):
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_q.weight)
        # 输出投影缩放（GPT-2 style）
        std = 0.02 / math.sqrt(2 * num_layers)
        nn.init.normal_(self.W_out.weight, std=std)

    def _neuron_parallel(self, neuron, x):
        """PLIFNode parallel scan 前向，返回激活值。"""
        TK, batch, D = x.shape
        beta = neuron.beta
        u = (1.0 - beta) * x

        v_init = neuron.v
        if isinstance(v_init, float):
            v_init = torch.zeros(batch, D, device=x.device, dtype=x.dtype)

        beta_row = beta.unsqueeze(0).expand(batch, D).contiguous()
        v_th_row = neuron.v_th.unsqueeze(0).expand(batch, D).contiguous()

        spike, V_post = plif_rowparam_forward(
            beta_row, u, v_th_row, v_init,
            surrogate_function=neuron.surrogate_function,
        )
        neuron.v = V_post[-1].detach()

        if self.activation_mode == 'v2':
            return (1.0 - beta) * V_post, spike
        return V_post, spike

    def forward_parallel(self, h):
        """
        并行前向传播。

        Args:
            h: (TK, batch, D) — 连续值输入

        Returns:
            output: (TK, batch, D) — 联想记忆输出
        """
        TK, batch, D = h.shape
        h_normed = self.norm(h)

        # ====== 1. PLIFNode 产生 k/v/q 激活 + gate spike ======
        act_k, _ = self._neuron_parallel(self.neuron_k, h_normed)
        act_v, _ = self._neuron_parallel(self.neuron_v, h_normed)
        act_q, _ = self._neuron_parallel(self.neuron_q, h_normed)
        _, spike_gate = self._neuron_parallel(self.neuron_gate, h_normed)
        # spike_gate: (TK, batch, D), 二值

        # ====== 2. 投影到 key/value/query 空间 ======
        k = self.W_k(act_k.reshape(TK * batch, D)).reshape(TK, batch, self.D_key)
        v = self.W_v(act_v.reshape(TK * batch, D)).reshape(TK, batch, self.D_value)
        q = self.W_q(act_q.reshape(TK * batch, D)).reshape(TK, batch, self.D_key)

        # ====== 3. Write gate: spike 均值作为标量门控 ======
        # 对 D 维取均值得到标量门控 (TK, batch, 1)
        write_gate = spike_gate.mean(dim=-1, keepdim=True)  # (TK, batch, 1)

        # ====== 4. 归一化 k (L2 norm，稳定外积) ======
        k = F.normalize(k, dim=-1)

        # ====== 5. 递推更新多组记忆矩阵 M ======
        # M: (num_groups, batch, D_key, D_value)
        if isinstance(self.M, float):
            self.M = torch.zeros(
                self.num_memory_groups, batch, self.D_key, self.D_value,
                device=h.device, dtype=h.dtype,
            )

        outputs = []
        for t in range(TK):
            k_t = k[t]  # (batch, D_key)
            v_t = v[t]  # (batch, D_value)
            q_t = q[t]  # (batch, D_key)
            g_t = write_gate[t]  # (batch, 1)

            # 外积: k_t · v_t^T → (batch, D_key, D_value)
            kv_outer = k_t.unsqueeze(-1) * v_t.unsqueeze(-2)  # (batch, D_key, D_value)
            kv_gated = g_t.unsqueeze(-1) * kv_outer  # (batch, D_key, D_value)

            # 更新所有记忆组
            group_outputs = []
            for g in range(self.num_memory_groups):
                self.M[g] = self.beta_M[g] * self.M[g] + kv_gated
                # 读出: q · M → (batch, D_value)
                out_g = torch.bmm(q_t.unsqueeze(1), self.M[g]).squeeze(1)  # (batch, D_value)
                group_outputs.append(out_g)

            # 拼接多组输出
            out_t = torch.cat(group_outputs, dim=-1)  # (batch, D_value * num_groups)
            outputs.append(out_t)

        # (TK, batch, D_value * num_groups)
        output = torch.stack(outputs, dim=0)

        # ====== 6. 输出投影 ======
        output = self.W_out(output.reshape(TK * batch, -1)).reshape(TK, batch, D)

        # detach M 防止梯度穿越 token 边界
        self.M = self.M.detach()

        return output
