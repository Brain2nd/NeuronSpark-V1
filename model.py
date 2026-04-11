"""
SNNLanguageModel: SNN 隐状态空间语言模型（全膜电位 + 动态 K）

架构（三段式）：
  model.encode(token_ids)    → h_seq           # 输入: embed → repeat K 次（可微分）
  model.snn_forward(h_seq)   → h_out, pc       # SNN 核心: num_layers 层，全膜电位 + 动态 K 聚合
  model.decode(h_out, seq)   → logits          # 输出: output_neuron(V_post) → K帧mean → proj → logits

核心设计：
  1. 膜电位泄漏量：PLIFNode 输出 (1-β)·V_post（泄漏量），自然强调快响应神经元
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
from atomic_ops.snn_base import functional, surrogate
from torch.utils.checkpoint import checkpoint

from atomic_ops import SNNDecoderLayer, SNNAttentionDecoderLayer
from atomic_ops.plif_node import PLIFNode
from atomic_ops.rms_norm import RMSNorm
from atomic_ops.parallel_scan import plif_rowparam_forward
from atomic_ops.lateral_inhibition import LateralInhibition


@dataclass
class SNNModelOutput:
    """模型输出容器，对齐教程 CausalLMOutputWithPast 接口。"""
    last_loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    ponder_cost: Optional[torch.Tensor] = None  # 动态 K: 平均期望步数


class SNNLanguageModel(nn.Module):
    """
    从零训练的 SNN 隐状态空间语言模型（parallel scan）。

    Args:
        vocab_size: 词表大小（默认 64000）
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
        vocab_size: int = 64000,
        D: int = 1024,
        N: int = 8,
        K: int = 12,
        num_layers: int = 24,
        D_ff: int = 3072,
        v_th_min: float = 0.1,
        activation_mode: str = 'v2',
        memory_layer_interval: int = 4,  # 0=禁用联想记忆层
        D_key: int = 128,
        D_value: int = 128,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.D = D
        self.N = N
        self.K = K
        self.num_layers = num_layers
        self.D_ff = D_ff
        self.activation_mode = activation_mode
        self.memory_layer_interval = memory_layer_interval
        self.v_th_min = v_th_min
        self.D_key = D_key
        self.D_value = D_value

        # ====== Embedding + Norm（全部可训练）======
        self.embed_tokens = nn.Embedding(vocab_size, D)
        self.norm = LateralInhibition(D)

        # ====== 解码投影 ======
        self.decode_proj = nn.Linear(D, D, bias=False)

        # ====== 输出 RMSNorm + 输出神经元 ======
        self.output_norm = RMSNorm(D)
        self.output_neuron = PLIFNode(
            dim=D,
            init_tau=2.0,
            v_threshold=0.3,
            surrogate_function=surrogate.Sigmoid(alpha=4.0),
        )

        # ====== 混合层栈: SNN Decoder + SNN-Attention Decoder ======
        # 每 memory_layer_interval 层插入 1 层 SNN-Attention 解码层
        # 例: interval=4, 24 层 → 层 3,7,11,15,19,23 为 SNN-Attention 层
        self.layers = nn.ModuleList()
        self.layer_types = []  # 'snn' or 'memory'
        for i in range(num_layers):
            if memory_layer_interval > 0 and (i + 1) % memory_layer_interval == 0:
                self.layers.append(SNNAttentionDecoderLayer(
                    D=D, N=N, D_ff=D_ff,
                    D_key=D_key, D_value=D_value,
                    v_th_min=v_th_min,
                    ffn_v_threshold=0.15,
                    K=K,
                    num_layers=num_layers,
                    layer_idx=i,
                    activation_mode=activation_mode,
                ))
                self.layer_types.append('memory')
            else:
                self.layers.append(SNNDecoderLayer(
                    D=D, N=N, D_ff=D_ff, v_th_min=v_th_min,
                    ffn_v_threshold=0.15,
                    K=K,
                    num_layers=num_layers,
                    layer_idx=i,
                    activation_mode=activation_mode,
                ))
                self.layer_types.append('snn')

        self._init_weights()

    def _init_weights(self):
        """初始化所有可训练权重（从零训练）。"""
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.decode_proj.weight)

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

    def snn_forward(self, h_seq: torch.Tensor):
        """SNN 核心：h_seq → (h_out, ponder_cost)。

        纯 SNN 层计算，带梯度检查点。
        每层返回 (h, ponder_cost)，ponder_cost 作为 checkpoint 输出保留梯度图。

        Returns:
            h: (seq_len*K, batch, D), 连续值
            total_ponder_cost: scalar, 所有层平均期望步数
        """
        h = h_seq
        ponder_costs = []

        def _layer_forward(layer_mod, x):
            functional.reset_net(layer_mod)
            return layer_mod.forward_parallel(x)  # 统一返回 (h, ponder_cost)

        for layer_module in self.layers:
            h, pc = checkpoint(
                _layer_forward, layer_module, h,
                use_reentrant=False,
            )
            ponder_costs.append(pc)

        total_ponder_cost = sum(ponder_costs) / len(ponder_costs)
        return h, total_ponder_cost

    def _output_neuron_parallel(self, h: torch.Tensor) -> torch.Tensor:
        """输出 PLIF 神经元的 parallel scan 前向：连续 h → 膜电位泄漏量。

        Args:
            h: (TK, batch, D) 连续值（SNN 最后一层输出）

        Returns:
            leak: (TK, batch, D) 膜电位泄漏量 (1-β)·V_post
        """
        TK, batch, D = h.shape
        input_dtype = h.dtype

        beta = self.output_neuron.beta.to(input_dtype)
        u = (1.0 - beta) * h

        v_init = self.output_neuron.v
        if isinstance(v_init, float):
            v_init = torch.zeros(batch, D, device=h.device, dtype=input_dtype)

        beta_row = beta.unsqueeze(0).expand(batch, D).contiguous()
        v_th_row = self.output_neuron.v_th.to(input_dtype).unsqueeze(0).expand(batch, D).contiguous()

        spike, V_post = plif_rowparam_forward(
            beta_row, u, v_th_row, v_init,
            surrogate_function=self.output_neuron.surrogate_function,
        )

        self.output_neuron.v = V_post[-1].detach()
        if self.activation_mode == 'v2':
            return ((1.0 - beta) * V_post).to(input_dtype)
        return V_post.to(input_dtype)

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
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
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
            top_p: nucleus 采样阈值（1.0 = 不限制）
            repetition_penalty: 重复惩罚（1.0 = 无惩罚，>1.0 = 惩罚重复）
            eos_token_id: 遇到此 token 停止生成

        Returns:
            (batch, prompt_len + generated_len) 完整序列
        """
        batch, prompt_len = prompt_ids.shape

        # 重置所有神经元（新序列的初始条件 V=0）
        for layer_module in self.layers:
            functional.reset_net(layer_module)
        functional.reset_net(self.output_neuron)

        # ====== Prefill: parallel 处理整个 prompt ======
        h_seq = self.encode(prompt_ids)  # (prompt_len*K, batch, D), 连续值
        h = h_seq
        for layer_module in self.layers:
            h, _ = layer_module.forward_parallel(h)
        # 此时所有层的所有神经元 .v 状态 = prompt 末尾状态

        logits = self.decode(h, prompt_len)

        # 已生成的 token ID 集合（用于 repetition_penalty）
        generated_ids = prompt_ids.clone()

        # 采样第一个新 token
        next_token = self._sample(logits[:, -1, :], temperature, top_k, top_p,
                                  repetition_penalty, generated_ids)
        generated = [next_token]
        generated_ids = torch.cat([generated_ids, next_token], dim=1)

        # ====== Autoregressive: 逐 token，forward_parallel 处理 K 帧 ======
        for _ in range(max_new_tokens - 1):
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            # 编码单 token → K 帧连续值（复用 encode）
            frames = self.encode(next_token)  # (K, batch, D)

            # K 帧通过 SNN — 不 reset，神经元 .v 跨 token 连续传递
            h = frames
            for layer_module in self.layers:
                h, _ = layer_module.forward_parallel(h)

            logits = self.decode(h, 1)

            next_token = self._sample(logits[:, -1, :], temperature, top_k, top_p,
                                      repetition_penalty, generated_ids)
            generated.append(next_token)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

        return torch.cat([prompt_ids, torch.cat(generated, dim=1)], dim=1)

    def _sample(self, logits: torch.Tensor, temperature: float = 1.0,
                top_k: int = None, top_p: float = 1.0,
                repetition_penalty: float = 1.0,
                generated_ids: torch.Tensor = None) -> torch.Tensor:
        """从 logits 采样（temperature + repetition_penalty + top-k + top-p）。

        Returns: (batch, 1)
        """
        if temperature <= 0:
            return logits.argmax(dim=-1, keepdim=True)

        # Repetition penalty: 对已出现的 token 降低概率
        if repetition_penalty != 1.0 and generated_ids is not None:
            for b in range(logits.size(0)):
                prev_ids = generated_ids[b].unique()
                score = logits[b, prev_ids]
                # 正 logit 除以 penalty（降低），负 logit 乘以 penalty（更负）
                logits[b, prev_ids] = torch.where(
                    score > 0, score / repetition_penalty, score * repetition_penalty
                )

        logits = logits / temperature

        # Top-k
        if top_k is not None and top_k > 0:
            top_k = min(top_k, logits.size(-1))
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float('-inf')

        # Top-p (nucleus sampling)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # 移除累积概率超过 top_p 的 token（保留第一个超过的）
            sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[sorted_mask] = float('-inf')
            logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

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

        # 重置所有神经元状态
        for layer_module in self.layers:
            functional.reset_net(layer_module)
        functional.reset_net(self.output_neuron)

        # 三段式
        h_seq = self.encode(token_ids)                # 输入边界
        h_out, ponder_cost = self.snn_forward(h_seq)  # SNN 核心 + ponder cost
        logits = self.decode(h_out, seq_len)          # 输出边界

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
            )

        return SNNModelOutput(logits=logits, ponder_cost=ponder_cost)

    def compensate_modulation_gradients(self, max_comp: float = 100.0):
        """
        Natural Gradient 补偿（两阶段）。

        Phase 1: Sigmoid/softplus 饱和补偿
          β = sigmoid(b_beta), sigmoid 在高 β 区（β=0.99, sigmoid'=0.01）梯度衰减 100x。
          补偿: grad /= activation'(b)，等价于在 β/α 空间做梯度下降。

        Phase 2: 层间梯度均衡
          残差链反向传播每层放大 ~1.17×，num_layers 层累积显著梯度比。
          深层选择性参数（b_beta/b_alpha/b_th）梯度被压制，无法有效学习。
          修复: 将每层调制参数梯度 norm 归一化到所有层的几何均值。

        调用时机: scaler.unscale_(optimizer) 之后、clip_grad_norm_ 之前。

        Args:
            max_comp: 补偿因子上限（防止极端值导致不稳定）
        """
        # ====== Phase 1: Sigmoid/softplus 饱和补偿 ======
        for layer_module in self.layers:
            if not hasattr(layer_module, 'snn_block'):
                continue  # 联想记忆层无调制参数
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

        # ====== Phase 2: 层间梯度均衡 ======
        # 残差链 h = h + sublayer(h) 的反向路径 ∂h_{l+1}/∂h_l = I + ∂sublayer/∂h_l
        # 每层放大 ~1.17×, 深层累积显著梯度比 → L0 梯度远大于 L_{num_layers-1}
        # 用几何均值归一化每层调制参数梯度 norm，消除残差放大效应
        with torch.no_grad():
            for param_name in ['b_beta', 'b_alpha', 'b_th']:
                norms = []
                params_list = []
                for layer_module in self.layers:
                    if not hasattr(layer_module, 'snn_block'):
                        continue
                    p = getattr(layer_module.snn_block, param_name)
                    if p.grad is not None:
                        n = p.grad.norm().item()
                        if n > 1e-12:
                            norms.append(n)
                            params_list.append(p)

                if len(norms) >= 2:
                    # 几何均值: exp(mean(log(norms))) — 对数尺度均衡，不受极端值影响
                    log_mean = sum(math.log(n) for n in norms) / len(norms)
                    geo_mean = math.exp(log_mean)
                    for p, n in zip(params_list, norms):
                        scale = geo_mean / n
                        scale = max(min(scale, max_comp), 1.0 / max_comp)
                        p.grad.mul_(scale)

    def get_param_groups(self) -> dict[str, list[nn.Parameter]]:
        """
        按功能分组的可训练参数。
        """
        groups = {
            'embedding': [self.embed_tokens.weight],
            'norm': [self.norm.gain],
            'decode': list(self.decode_proj.parameters()),
            # 输出神经元
            'output_neuron': [self.output_neuron.w, self.output_neuron.v_th],
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

        for layer_module, layer_type in zip(self.layers, self.layer_types):
            if layer_type == 'memory':
                # SNNAttentionDecoderLayer: attn 子层 + ffn 子层
                groups['residual_projs'].extend([
                    layer_module.qkv_proj.weight,
                    layer_module.attn_out_proj.weight,
                    layer_module.ffn_out_proj.weight,
                ])
                groups['input_neurons'].extend([
                    layer_module.gate_neuron.w, layer_module.gate_neuron.v_th,
                    layer_module.input_neuron2.w, layer_module.input_neuron2.v_th,
                ])
                groups['rms_norms'].extend([
                    layer_module.attn_norm.weight,
                    layer_module.attn_out_norm.weight,
                    layer_module.ffn_norm.weight,
                ])
                groups['halt_projs'].extend(list(layer_module.ffn_halt.parameters()))
                # SNNFFN 参数
                ffn = layer_module.snn_ffn
                groups['ffn_gate_proj'].append(ffn.gate_proj.weight)
                groups['ffn_up_proj'].append(ffn.up_proj.weight)
                groups['ffn_down_proj'].append(ffn.down_proj.weight)
                groups['ffn_skip_proj'].append(ffn.skip_proj.weight)
                groups['ffn_neurons'].extend([
                    ffn.gate_neuron.w, ffn.gate_neuron.v_th,
                    ffn.up_neuron.w, ffn.up_neuron.v_th,
                ])
                continue

            block = layer_module.snn_block
            ffn = layer_module.snn_ffn

            # 残差流组件
            groups['residual_projs'].extend([
                layer_module.block_out_proj.weight,
                layer_module.ffn_out_proj.weight,
            ])
            groups['input_neurons'].extend([
                layer_module.input_neuron1.w,
                layer_module.input_neuron1.v_th,
                layer_module.input_neuron2.w,
                layer_module.input_neuron2.v_th,
            ])
            groups['rms_norms'].extend([
                layer_module.block_norm.weight,
                layer_module.ffn_norm.weight,
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
                ffn.gate_neuron.w, ffn.gate_neuron.v_th,
                ffn.up_neuron.w, ffn.up_neuron.v_th,
            ])

        return groups
