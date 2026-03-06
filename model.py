"""
SNNLanguageModel: SNN 隐状态空间语言模型（全膜电位 + 动态 K）

架构（三段式）：
  model.encode(token_ids)    → h_seq           # 输入: embed → repeat K 次（可微分）
  model.snn_forward(h_seq)   → h_out, pc       # SNN 核心: 20 层，全膜电位 + 动态 K 聚合
  model.decode(h_out, seq)   → logits          # 输出: output_neuron(V_post) → K帧mean → proj → logits

核心设计：
  1. 全膜电位：所有神经元输出 V_post 而非 spike
  2. 动态 K：PonderNet 自适应停止，不同 token 不同有效步数
     - 每层每子层学习 halt_proj(D→1)，从 SNN 输出逐步计算停止概率
     - 几何分布权重加权聚合，替代 uniform mean
     - ponder_cost 正则化鼓励早停

数学原理见 SNN_SELECTIVE_STATE_SPACE.md。
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import surrogate
from torch.utils.checkpoint import checkpoint

from atomic_ops import SNNDecoderLayer, spike_current_activation
from atomic_ops.plif_node import PLIFNode
from atomic_ops.rms_norm import RMSNorm
from atomic_ops.parallel_scan import plif_rowparam_forward
from atomic_ops.snn_decoder_layer import _mpd_alpha
# fp16_encode/fp16_decode 已移除: 全膜电位架构不需要 spike 编解码
from atomic_ops.lateral_inhibition import LateralInhibition


@dataclass
class SNNModelOutput:
    """模型输出容器，对齐教程 CausalLMOutputWithPast 接口。"""
    last_loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    ponder_cost: Optional[torch.Tensor] = None  # 动态 K: 平均期望步数
    ek_floor_cost: Optional[torch.Tensor] = None  # E[K] 下界惩罚
    snvr_cost: Optional[torch.Tensor] = None  # 层间权重谱范数方差正则化


class SNNLanguageModel(nn.Module):
    """
    从零训练的 SNN 隐状态空间语言模型（parallel scan）。

    Args:
        vocab_size: 词表大小（默认 6144，自训练 BPE）
        D: 可见维度
        N: 状态扩展因子
        K: 每 token 最大 SNN 时间步数（K_max）。PonderNet 动态决定有效步数 ∈ [1, K]。
           K 越大 → 复杂 token 可用更多步数，但计算量和显存线性增长。
        num_layers: SNN 解码层数
        D_ff: FFN 中间层维度
        v_th_min: 动态阈值下限
    """

    def __init__(
        self,
        vocab_size: int = 6144,
        D: int = 1024,
        N: int = 8,
        K: int = 32,
        num_layers: int = 20,
        D_ff: int = 3072,
        v_th_min: float = 0.1,
        ek_floor: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.D = D
        self.N = N
        self.K = K
        self.num_layers = num_layers
        self.D_ff = D_ff

        # ====== Embedding + Norm（全部可训练）======
        self.embed_tokens = nn.Embedding(vocab_size, D)
        self.norm = LateralInhibition(D)

        # ====== 解码投影 ======
        self.decode_proj = nn.Linear(D, D)

        # ====== 输出 RMSNorm + 输出神经元 ======
        self.output_norm = RMSNorm(D)
        self.output_neuron = PLIFNode(
            dim=D,
            init_tau=2.0,
            v_threshold=0.3,
            surrogate_function=surrogate.Sigmoid(alpha=4.0),
        )

        # ====== SNN Decoder Layers ======
        self.layers = nn.ModuleList([
            SNNDecoderLayer(
                D=D, N=N, D_ff=D_ff, v_th_min=v_th_min,
                ffn_v_threshold=0.15,
                K=K,
                num_layers=num_layers,
                layer_idx=i,
                ek_floor=ek_floor,
            )
            for i in range(num_layers)
        ])

        self._init_weights()

    def _reset_neurons(self):
        """直接重置所有神经元膜电位，替代 functional.reset_net。

        functional.reset_net 遍历模块树，遇到 FSDP wrapper 时无法识别
        MemoryModule 子类，产生大量 WARNING 且可能触发不一致的 FSDP 通信。
        直接赋值 v=0. 绕过模块树遍历，FSDP 安全。
        """
        for layer_module in self.layers:
            layer_module.input_neuron1.v = 0.
            layer_module.input_neuron2.v = 0.
            layer_module.snn_block.hidden_neuron.v = 0.
            layer_module.snn_ffn.gate_neuron.v = 0.
            layer_module.snn_ffn.up_neuron.v = 0.
        self.output_neuron.v = 0.

    def _reset_layer_neurons(self, layer_module):
        """重置单个层的所有神经元膜电位。"""
        layer_module.input_neuron1.v = 0.
        layer_module.input_neuron2.v = 0.
        layer_module.snn_block.hidden_neuron.v = 0.
        layer_module.snn_ffn.gate_neuron.v = 0.
        layer_module.snn_ffn.up_neuron.v = 0.

    def _init_weights(self):
        """初始化所有可训练权重（从零训练）。"""
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.decode_proj.weight)
        nn.init.zeros_(self.decode_proj.bias)

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        """输入边界：token_ids → 连续值序列。

        Embedding lookup，每 token 重复 K 次作为 SNN 时间步输入。
        梯度可通过 embedding 直接反传。

        Returns: (seq_len*K, batch, D), 连续值
        """
        emb = self.embed_tokens(token_ids)       # (batch, seq_len, D)
        batch, seq_len, D = emb.shape
        # 每 token 重复 K 次: (batch, seq_len, D) → (batch, seq_len*K, D) → (TK, batch, D)
        emb_k = emb.unsqueeze(2).expand(-1, -1, self.K, -1).reshape(batch, seq_len * self.K, D)
        return emb_k.permute(1, 0, 2).contiguous()  # (TK, batch, D)

    def snn_forward(self, spike_seq: torch.Tensor):
        """SNN 核心：spike_seq → (h_out, ponder_cost, ek_floor_cost)。

        纯 SNN 层计算，带梯度检查点。
        每层返回 (h, ponder_cost, ek_floor_cost)，作为 checkpoint 输出保留梯度图。

        Returns:
            h: (seq_len*K, batch, D), 连续值
            total_ponder_cost: scalar, 所有层平均期望步数
            total_ek_floor_cost: scalar, 所有层 E[K] 下界惩罚均值
        """
        h = spike_seq
        ponder_costs = []
        ek_floor_costs = []

        def _layer_forward(layer_mod, x):
            self._reset_layer_neurons(layer_mod)
            return layer_mod(x)  # 通过 __call__ 触发 FSDP 参数聚合钩子

        for layer_module in self.layers:
            h, pc, efc = checkpoint(
                _layer_forward, layer_module, h,
                use_reentrant=False,
            )
            ponder_costs.append(pc)
            ek_floor_costs.append(efc)

        total_ponder_cost = sum(ponder_costs) / len(ponder_costs)
        total_ek_floor_cost = sum(ek_floor_costs) / len(ek_floor_costs)
        return h, total_ponder_cost, total_ek_floor_cost

    def _output_neuron_parallel(self, h: torch.Tensor) -> torch.Tensor:
        """输出 PLIF 神经元的 parallel scan 前向：连续 h → 脉冲电流。

        Args:
            h: (TK, batch, D) 连续值（SNN 最后一层输出）

        Returns:
            spike_current: (TK, batch, D) 脉冲电流（V_th * spike，稀疏激活值）
        """
        TK, batch, D = h.shape

        # MPD-AGL: 输出神经元 surrogate alpha 自适应（前置 output_norm）
        with torch.no_grad():
            go = self.output_norm.weight.data.abs().mean().item()
            bo = self.output_neuron.beta.mean().item()
            vo = self.output_neuron.v_th.abs().mean().item()
            self.output_neuron.surrogate_function.alpha = _mpd_alpha(bo, vo, go)

        beta = self.output_neuron.beta  # (D,)
        u = self.output_neuron.alpha * h  # PLIF: u = α · x（解耦积分强度）

        v_init = self.output_neuron.v
        if isinstance(v_init, float):
            v_init = torch.zeros(batch, D, device=h.device, dtype=h.dtype)

        beta_row = beta.unsqueeze(0).expand(batch, D).contiguous()
        v_th_row = self.output_neuron.v_th.unsqueeze(0).expand(batch, D).contiguous()

        spike, V_post = plif_rowparam_forward(
            beta_row, u, v_th_row, v_init,
            surrogate_function=self.output_neuron.surrogate_function,
        )
        del u  # Triton PLIF ctx 不保存 u，可安全释放

        self.output_neuron.v = V_post[-1].detach()
        del V_post
        return spike_current_activation(spike, v_th_row.unsqueeze(0))  # 脉冲电流作为激活值

    def decode(self, h_out: torch.Tensor, seq_len: int) -> torch.Tensor:
        """输出边界：连续 h → 输出神经元(V_post) → K 帧聚合 → logits。

        梯度流: loss → logits → norm → decode_proj → K帧mean
                → V_post(output_neuron) → h_out → SNN layers

        Returns: (batch, seq_len, vocab_size)
        """
        h_out = self.output_norm(h_out)                    # RMSNorm: 控制 scale
        v_out = self._output_neuron_parallel(h_out)    # (TK, batch, D), V_post 膜电位
        # K 帧聚合: (TK, batch, D) → (seq_len, K, batch, D) → mean → (seq_len, batch, D)
        decoded = v_out.view(seq_len, self.K, -1, self.D).mean(dim=1)
        decoded = decoded.permute(1, 0, 2)                 # (batch, seq_len, D)
        h = self.decode_proj(decoded)                      # (batch, seq_len, D)
        h = self.norm(h)                                   # (batch, seq_len, D)
        return F.linear(h, self.embed_tokens.weight)       # (batch, seq_len, vocab)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 50,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        自回归生成（SNN 神经元状态跨 token 连续维护）。

        1. Prefill: forward_parallel 并行处理 prompt，建立所有神经元 V 状态
        2. Autoregressive: 逐 token 生成，每 token 用 forward_parallel 处理 K 帧
           复用 Triton parallel scan kernel，神经元 V 状态跨 token 连续传递

        Args:
            prompt_ids: (batch, prompt_len) token IDs
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度（<=0 = greedy）
            top_k: top-k 采样（None/0 = 不限制）
            eos_token_id: 遇到此 token 停止生成

        Returns:
            (batch, prompt_len + generated_len) 完整序列
        """
        batch, prompt_len = prompt_ids.shape

        # 重置所有神经元（新序列的初始条件 V=0，直接赋值，FSDP 安全）
        self._reset_neurons()

        # ====== Prefill: parallel 处理整个 prompt ======
        h_seq = self.encode(prompt_ids)  # (prompt_len*K, batch, D), 连续值
        h = h_seq
        for layer_module in self.layers:
            h, _, _ = layer_module.forward_parallel(h)  # 推理忽略 ponder_cost
        # 此时所有层的所有神经元 .v 状态 = prompt 末尾状态

        logits = self.decode(h, prompt_len)

        # 采样第一个新 token
        next_token = self._sample(logits[:, -1, :], temperature, top_k)
        generated = [next_token]

        # ====== Autoregressive: 逐 token，forward_parallel 处理 K 帧 ======
        for _ in range(max_new_tokens - 1):
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            # 编码单 token → K 帧连续值（复用 encode）
            frames = self.encode(next_token)  # (K, batch, D)

            # K 帧通过 SNN — 不 reset，神经元 .v 跨 token 连续传递
            h = frames
            for layer_module in self.layers:
                h, _, _ = layer_module.forward_parallel(h)

            logits = self.decode(h, 1)

            next_token = self._sample(logits[:, -1, :], temperature, top_k)
            generated.append(next_token)

        return torch.cat([prompt_ids, torch.cat(generated, dim=1)], dim=1)

    def _sample(self, logits: torch.Tensor, temperature: float = 1.0, top_k: int = None) -> torch.Tensor:
        """从 logits 采样（temperature + top-k）。

        Returns: (batch, 1)
        """
        if temperature <= 0:
            return logits.argmax(dim=-1, keepdim=True)
        logits = logits / temperature
        if top_k is not None and top_k > 0:
            top_k = min(top_k, logits.size(-1))
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float('-inf')
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def forward(
        self,
        token_ids: torch.Tensor,
        target_ids: torch.Tensor = None,
    ) -> SNNModelOutput:
        """
        前向传播（全膜电位 + 动态 K）。

        encode → h_seq               # 输入（embed repeat K 次，可微分）
        snn_forward → h_out, pc      # SNN 核心（全膜电位 + 动态 K 聚合）
        decode → logits              # 输出（V_post → K帧mean → proj → logits）

        梯度流:
          embed_tokens → repeat K → SNN layers(V_post + 动态K)
            → output_neuron(V_post) → K帧mean → decode_proj → logits(tied head)
          ponder_cost: 动态 K 正则化，鼓励用更少步数处理简单 token
        """
        batch, seq_len = token_ids.shape

        # 重置所有神经元状态（直接赋值，FSDP 安全）
        self._reset_neurons()

        # 三段式
        spike_seq = self.encode(token_ids)            # 输入边界
        h_out, ponder_cost, ek_floor_cost = self.snn_forward(spike_seq)  # SNN 核心
        logits = self.decode(h_out, seq_len)          # 输出边界

        # SNVR: 层间权重范数方差正则化（可微分，梯度流回权重）
        snvr_cost = self._compute_snvr_cost()

        if target_ids is not None:
            logits_flat = logits.reshape(-1, self.vocab_size)
            targets_flat = target_ids.reshape(-1)
            self.last_loss = F.cross_entropy(
                logits_flat, targets_flat,
                ignore_index=0, reduction='none',
            )
            return SNNModelOutput(
                last_loss=self.last_loss,
                ponder_cost=ponder_cost,
                ek_floor_cost=ek_floor_cost,
                snvr_cost=snvr_cost,
            )

        return SNNModelOutput(logits=logits, ponder_cost=ponder_cost,
                              ek_floor_cost=ek_floor_cost, snvr_cost=snvr_cost)

    def _compute_snvr_cost(self) -> torch.Tensor:
        """Spectral Norm Variance Regularization: 惩罚同类权重跨层 Frobenius 范数方差。

        Jacobian 谱范数与权重范数正相关。若各层同类权重范数不一致
        (浅层大、深层小)，∏‖J_k‖ 必然指数发散 → 梯度消失/爆炸。
        惩罚范数方差 → 迫使各层 ‖J_k‖ 趋近统一 → 梯度流保持。

        Returns:
            scalar — 所有权重类型的范数方差之和
        """
        norms_by_type = {}
        for layer_module in self.layers:
            block = layer_module.snn_block
            ffn = layer_module.snn_ffn
            for name, param in [
                ('W_in', block.W_in.weight),
                ('W_out', block.W_out.weight),
                ('W_gate', block.W_gate.weight),
                ('W_skip', block.W_skip.weight),
                ('ffn_gate', ffn.gate_proj.weight),
                ('ffn_down', ffn.down_proj.weight),
                ('out_proj', layer_module.block_out_proj.weight),
                ('ffn_out_proj', layer_module.ffn_out_proj.weight),
            ]:
                norms_by_type.setdefault(name, []).append(param.norm())

        total_var = torch.zeros(1, device=self.layers[0].block_out_proj.weight.device,
                                dtype=torch.float32).squeeze()
        for norms in norms_by_type.values():
            stacked = torch.stack(norms)
            total_var = total_var + stacked.var()
        return total_var

    def compensate_modulation_gradients(self, max_comp: float = 100.0):
        """
        Natural Gradient 补偿: sigmoid/softplus 饱和补偿。

          β = sigmoid(b_beta), sigmoid 在高 β 区（β=0.99, sigmoid'=0.01）梯度衰减 100x。
          补偿: grad /= activation'(b)，等价于在 β/α 空间做梯度下降。

        注意: 原 Phase 2（层间梯度均衡）已移除。几何均值归一化会将所有层的
        b_beta/b_alpha/b_th 梯度强制相同，破坏层间梯度多样性。
        层间梯度差异由 SubLN Post-RMSNorm 自然平衡。

        调用时机: optimizer.step() 之前、clip_grad_norm_ 之前。

        Args:
            max_comp: 补偿因子上限（防止极端值导致不稳定）
        """
        for layer_module in self.layers:
            block = layer_module.snn_block

            # b_beta: sigmoid 饱和补偿
            # sigmoid'(z) = sigmoid(z) · (1 - sigmoid(z)) = β · (1-β)
            if block.b_beta.grad is not None:
                with torch.no_grad():
                    beta = torch.sigmoid(block.b_beta.data)
                    sigmoid_deriv = (beta * (1.0 - beta)).clamp(min=1.0 / max_comp)
                    block.b_beta.grad.div_(sigmoid_deriv)

            # b_alpha: softplus 补偿（较温和，softplus'(z) = sigmoid(z)）
            if block.b_alpha.grad is not None:
                with torch.no_grad():
                    softplus_deriv = torch.sigmoid(block.b_alpha.data).clamp(min=0.1)
                    block.b_alpha.grad.div_(softplus_deriv)

            # b_th: |·| 导数为 ±1，无衰减，不需要补偿

    def get_param_groups(self) -> dict[str, list[nn.Parameter]]:
        """
        按功能分组的可训练参数。
        """
        groups = {
            'embedding': [self.embed_tokens.weight],
            'norm': [self.norm.gain],
            'decode': list(self.decode_proj.parameters()),
            # 输出神经元
            'output_neuron': [self.output_neuron.w, self.output_neuron.w_gain, self.output_neuron.v_th],
            # RMSNorm（Pre-LN 分支归一化）
            'rms_norms': [self.output_norm.weight],
            # 残差流组件
            'residual_projs': [],
            'input_neurons': [],
            # 动态 K: 停止投影
            'halt_projs': [],
            # SNNBlock 参数
            'W_in': [],
            'W_beta': [],
            'W_alpha': [],
            'W_th': [],
            'W_gate': [],
            'W_skip': [],
            'W_out': [],
            'b_beta': [],
            'b_alpha': [],
            'b_th': [],
            'block_output_neuron': [],
            # SNNFFN 参数
            'ffn_gate_proj': [],
            'ffn_up_proj': [],
            'ffn_down_proj': [],
            'ffn_skip_proj': [],
            'ffn_neurons': [],
        }

        for layer_module in self.layers:
            block = layer_module.snn_block
            ffn = layer_module.snn_ffn

            # 残差流组件
            groups['residual_projs'].extend([
                layer_module.block_out_proj.weight,
                layer_module.ffn_out_proj.weight,
            ])
            groups['input_neurons'].extend([
                layer_module.input_neuron1.w,
                layer_module.input_neuron1.w_gain,
                layer_module.input_neuron1.v_th,
                layer_module.input_neuron2.w,
                layer_module.input_neuron2.w_gain,
                layer_module.input_neuron2.v_th,
            ])
            groups['rms_norms'].extend([
                layer_module.block_norm.weight,
                layer_module.ffn_norm.weight,
                layer_module.block_post_norm.weight,
                layer_module.ffn_post_norm.weight,
            ])

            # 动态 K: 停止投影参数
            groups['halt_projs'].extend(list(layer_module.block_halt.parameters()))
            groups['halt_projs'].extend(list(layer_module.ffn_halt.parameters()))

            # SNNBlock 参数
            groups['W_in'].append(block.W_in.weight)
            groups['W_beta'].extend([block.W_beta_x.weight])
            groups['W_alpha'].extend([block.W_alpha_x.weight])
            groups['W_th'].extend([block.W_th_x.weight])
            groups['W_gate'].append(block.W_gate.weight)
            groups['W_skip'].append(block.W_skip.weight)
            groups['W_out'].append(block.W_out.weight)
            groups['b_beta'].append(block.b_beta)
            groups['b_alpha'].append(block.b_alpha)
            groups['b_th'].append(block.b_th)

            # SNNFFN 参数
            groups['ffn_gate_proj'].append(ffn.gate_proj.weight)
            groups['ffn_up_proj'].append(ffn.up_proj.weight)
            groups['ffn_down_proj'].append(ffn.down_proj.weight)
            groups['ffn_skip_proj'].append(ffn.skip_proj.weight)
            groups['ffn_neurons'].extend([
                ffn.gate_neuron.w, ffn.gate_neuron.w_gain, ffn.gate_neuron.v_th,
                ffn.up_neuron.w, ffn.up_neuron.w_gain, ffn.up_neuron.v_th,
            ])

        return groups
