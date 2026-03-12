"""
SNNLanguageModel: SNN 隐状态空间语言模型（mHC 多流残差 + 脉冲电流激活 + 动态 K）

架构（三段式）：
  model.encode(token_ids)    → h_streams          # 输入: embed → n 流扩展
  model.snn_forward(h_streams) → h_out, pc        # SNN 核心: 20 层，mHC 多流残差 + 动态 K 聚合
  model.decode(h_out, seq)   → logits             # 输出: n 流聚合 → K 帧 output_neuron → proj → logits

核心设计：
  1. mHC 多流残差: H_res ∈ Birkhoff 多面体，谱范数 ≤ 1，梯度不爆炸的代数保证
  2. 脉冲电流激活：神经元输出 V_th * spike（稀疏），经投影后汇入 n 流残差
  3. 动态 K：PonderNet 自适应停止，不同 token 不同有效步数
  4. 层间状态: (seq_len, batch, n, D) — n 流残差（n=4 推荐）

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
# fp16_encode/fp16_decode 已移除: 脉冲电流架构不需要 spike 编解码
from atomic_ops.lateral_inhibition import LateralInhibition


@dataclass
class SNNModelOutput:
    """模型输出容器，对齐教程 CausalLMOutputWithPast 接口。"""
    last_loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    ponder_cost: Optional[torch.Tensor] = None  # 动态 K: 平均期望步数
    ek_floor_cost: Optional[torch.Tensor] = None  # E[K] 下界惩罚
    b_th_reg_cost: Optional[torch.Tensor] = None  # b_th L2 正则化（遏制 V_th 漂移）


class SNNLanguageModel(nn.Module):
    """
    从零训练的 SNN 隐状态空间语言模型（mHC 多流残差 + parallel scan）。

    Args:
        vocab_size: 词表大小（默认 6144，自训练 BPE）
        D: 可见维度
        N: 状态扩展因子
        K: 每 token 最大 SNN 时间步数（K_max）。PonderNet 动态决定有效步数 ∈ [1, K]。
        num_layers: SNN 解码层数
        D_ff: FFN 中间层维度
        v_th_min: 动态阈值下限
        n_hc_streams: mHC 流数量（推荐 4，Birkhoff 多面体约束）
        sinkhorn_iters: Sinkhorn 迭代次数（20 对 n≤8 足够收敛）
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
        n_hc_streams: int = 4,
        sinkhorn_iters: int = 20,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.D = D
        self.N = N
        self.K = K
        self.num_layers = num_layers
        self.D_ff = D_ff
        self.n_hc_streams = n_hc_streams

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
                n_hc_streams=n_hc_streams,
                sinkhorn_iters=sinkhorn_iters,
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

    def _save_v_states(self):
        """保存所有神经元膜电位状态（自投机解码用）。

        Returns:
            dict: 所有神经元 V 状态的克隆副本，~657KB/batch (bf16)
        """
        def _clone(v):
            return v.clone() if isinstance(v, torch.Tensor) else v

        states = {}
        for i, layer in enumerate(self.layers):
            states[f'l{i}_in1'] = _clone(layer.input_neuron1.v)
            states[f'l{i}_in2'] = _clone(layer.input_neuron2.v)
            states[f'l{i}_hidden'] = _clone(layer.snn_block.hidden_neuron.v)
            states[f'l{i}_gate'] = _clone(layer.snn_ffn.gate_neuron.v)
            states[f'l{i}_up'] = _clone(layer.snn_ffn.up_neuron.v)
        states['output'] = _clone(self.output_neuron.v)
        return states

    def _restore_v_states(self, states):
        """恢复所有神经元膜电位状态。"""
        def _clone(v):
            return v.clone() if isinstance(v, torch.Tensor) else v

        for i, layer in enumerate(self.layers):
            layer.input_neuron1.v = _clone(states[f'l{i}_in1'])
            layer.input_neuron2.v = _clone(states[f'l{i}_in2'])
            layer.snn_block.hidden_neuron.v = _clone(states[f'l{i}_hidden'])
            layer.snn_ffn.gate_neuron.v = _clone(states[f'l{i}_gate'])
            layer.snn_ffn.up_neuron.v = _clone(states[f'l{i}_up'])
        self.output_neuron.v = _clone(states['output'])

    def _set_K(self, new_K):
        """切换 SNN 时间步数 K（所有层 + 模型顶层）。"""
        self.K = new_K
        for layer in self.layers:
            layer.K = new_K

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
        """输入边界：token_ids → n 流残差。

        Embedding lookup，扩展到 n 流（所有流初始值相同）。
        mHC 初始化保证 H_pre @ [emb]*n = emb，等价于标准残差起点。

        Returns: (seq_len, batch, n, D), 连续值 n 流
        """
        emb = self.embed_tokens(token_ids)       # (batch, seq_len, D)
        batch, seq_len, D = emb.shape
        n = self.n_hc_streams
        # 扩展到 n 流: (batch, seq_len, D) → (batch, seq_len, 1, D) → (batch, seq_len, n, D)
        emb_n = emb.unsqueeze(2).expand(-1, -1, n, -1)
        # 转置: (batch, seq_len, n, D) → (seq_len, batch, n, D)
        return emb_n.permute(1, 0, 2, 3).contiguous()

    def snn_forward(self, h_streams: torch.Tensor):
        """SNN 核心：h_streams → (h_out, ponder_cost, ek_floor_cost)。

        纯 SNN 层计算，带梯度检查点。
        每层接收/返回 (seq_len, batch, n, D) n 流残差。
        层内展开到 TK 做 K 帧时间动力学，PonderNet 聚合回 token 级。

        Returns:
            h_streams: (seq_len, batch, n, D), n 流残差
            total_ponder_cost: scalar, 所有层平均期望步数
            total_ek_floor_cost: scalar, 所有层 E[K] 下界惩罚均值
        """
        h = h_streams
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
            h: (TK, batch, D) 连续值（展开后的 K 帧输入）

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
        u = (1.0 - beta) * h  # PLIF: u = (1-β) · x

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

    def decode(self, h_streams: torch.Tensor, seq_len: int) -> torch.Tensor:
        """输出边界：n 流残差 → 聚合 → K 帧 output_neuron(spike_current) → logits。

        流程:
          1. n 流均值聚合 → (seq_len, batch, D)
          2. 展开到 TK: (seq_len*K, batch, D)
          3. output_norm → output_neuron(spike_current) → K 帧均值
          4. decode_proj → LateralInhibition → tied embedding → logits

        Returns: (batch, seq_len, vocab_size)
        """
        # 1. n 流均值聚合
        h_collapsed = h_streams.mean(dim=2)  # (seq_len, batch, D)

        # 2. 展开到 TK
        batch = h_streams.shape[1]
        TK = seq_len * self.K
        h_tk = h_collapsed.unsqueeze(1).expand(-1, self.K, -1, -1).reshape(
            TK, batch, self.D)

        # 3. 输出神经元处理
        h_tk = self.output_norm(h_tk)                     # RMSNorm: 控制 scale
        v_out = self._output_neuron_parallel(h_tk)    # (TK, batch, D), 脉冲电流
        # K 帧聚合: (TK, batch, D) → (seq_len, K, batch, D) → mean → (seq_len, batch, D)
        decoded = v_out.view(seq_len, self.K, batch, self.D).mean(dim=1)
        decoded = decoded.permute(1, 0, 2)                 # (batch, seq_len, D)

        # 4. 投影 + 侧抑制 + tied head
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
        h_streams = self.encode(prompt_ids)  # (prompt_len, batch, n, D)
        h = h_streams
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

            # 编码单 token → n 流残差
            frames = self.encode(next_token)  # (1, batch, n, D)

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

    @torch.no_grad()
    def generate_speculative(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 50,
        eos_token_id: Optional[int] = None,
        K_draft: int = 4,
        lookahead: int = 5,
    ) -> torch.Tensor:
        """
        自投机解码：同一模型用低 K 做 Draft，高 K 做 Verify，加速推理。

        原理：
          1. Draft 阶段: K_draft（默认 4）逐 token 生成 `lookahead` 个候选
          2. Verify 阶段: K_full（原始 K）一次 forward_parallel 批量验证
          3. 接受匹配的 token + recovery token，回滚 V 状态到正确位置

        Args:
            prompt_ids: (batch, prompt_len)
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_k: top-k 采样
            eos_token_id: 停止 token
            K_draft: Draft 阶段 SNN 时间步数
            lookahead: 每轮 Draft 生成的候选 token 数

        Returns:
            (batch, prompt_len + generated_len) 完整序列
        """
        batch, prompt_len = prompt_ids.shape
        K_full = self.K
        assert K_draft < K_full, f"K_draft({K_draft}) 必须小于 K_full({K_full})"

        self._reset_neurons()

        # ====== Prefill（K_full） ======
        h = self.encode(prompt_ids)
        for layer in self.layers:
            h, _, _ = layer.forward_parallel(h)
        logits = self.decode(h, prompt_len)
        next_token = self._sample(logits[:, -1, :], temperature, top_k)
        generated = [next_token]

        n_accepted = 0
        n_drafted = 0

        while len(generated) < max_new_tokens:
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            k = min(lookahead, max_new_tokens - len(generated))
            if k <= 0:
                break

            # ====== 保存 V 状态 ======
            saved_v = self._save_v_states()

            # ====== Draft: K_draft 逐 token 贪心生成 k 个候选 ======
            self._set_K(K_draft)
            draft_tokens = []
            current = next_token
            for _ in range(k):
                h_d = self.encode(current)
                for layer in self.layers:
                    h_d, _, _ = layer.forward_parallel(h_d)
                logits_d = self.decode(h_d, 1)
                # Draft 始终贪心，最大化接受率
                tok = logits_d[:, -1, :].argmax(dim=-1, keepdim=True)
                draft_tokens.append(tok)
                current = tok
                if eos_token_id is not None and (tok == eos_token_id).all():
                    break

            k_actual = len(draft_tokens)
            n_drafted += k_actual

            # ====== 恢复 V 状态 + 切回 K_full ======
            self._restore_v_states(saved_v)
            self._set_K(K_full)

            # ====== Verify: 一次 forward_parallel 处理 k+1 tokens ======
            verify_ids = torch.cat([next_token] + draft_tokens, dim=1)
            h_v = self.encode(verify_ids)
            for layer in self.layers:
                h_v, _, _ = layer.forward_parallel(h_v)
            verify_logits = self.decode(h_v, k_actual + 1)

            # ====== 接受/拒绝: argmax 比较 ======
            target_preds = verify_logits[:, :k_actual, :].argmax(dim=-1)
            draft_ids = torch.cat(draft_tokens, dim=1)
            matches = (target_preds == draft_ids)

            any_mismatch = (~matches).any(dim=1)
            first_mismatch = (~matches).float().argmax(dim=1)
            n_accept_per_batch = torch.where(
                any_mismatch, first_mismatch,
                torch.tensor(k_actual, device=matches.device, dtype=first_mismatch.dtype),
            )
            n_accept = int(n_accept_per_batch.min().item())
            n_accepted += n_accept

            # ====== 提交已接受的 token + recovery ======
            accepted = draft_tokens[:n_accept]
            generated.extend(accepted)
            recovery = self._sample(verify_logits[:, n_accept, :], temperature, top_k)

            # EOS 检查
            eos_hit = False
            if eos_token_id is not None:
                for i, tok in enumerate(accepted):
                    if (tok == eos_token_id).all():
                        generated = generated[:len(generated) - len(accepted) + i + 1]
                        eos_hit = True
                        break
            if eos_hit:
                break

            generated.append(recovery)

            # ====== 重建 V 状态 ======
            if n_accept == k_actual:
                next_token = recovery
            else:
                self._restore_v_states(saved_v)
                if n_accept > 0:
                    rebuild_ids = torch.cat([next_token] + accepted, dim=1)
                else:
                    rebuild_ids = next_token
                h_r = self.encode(rebuild_ids)
                for layer in self.layers:
                    h_r, _, _ = layer.forward_parallel(h_r)
                _ = self.decode(h_r, rebuild_ids.shape[1])
                next_token = recovery

            if eos_token_id is not None and (recovery == eos_token_id).all():
                break

        # 截断到 max_new_tokens
        generated = generated[:max_new_tokens]

        # 统计
        total = len(generated)
        if n_drafted > 0:
            rate = n_accepted / n_drafted
            print(f"[Speculative] {total} tokens, "
                  f"接受率 {n_accepted}/{n_drafted}={rate:.1%}, "
                  f"K_draft={K_draft}, K_full={K_full}, lookahead={lookahead}")

        if not generated:
            return prompt_ids
        return torch.cat([prompt_ids, torch.cat(generated, dim=1)], dim=1)

    def forward(
        self,
        token_ids: torch.Tensor,
        target_ids: torch.Tensor = None,
    ) -> SNNModelOutput:
        """
        前向传播（mHC 多流残差 + 脉冲电流激活 + 动态 K）。

        encode → h_streams             # 输入（embed → n 流扩展）
        snn_forward → h_out, pc        # SNN 核心（mHC 多流残差 + 动态 K 聚合）
        decode → logits                # 输出（n 流聚合 → spike_current → K帧mean → proj → logits）

        梯度流:
          embed_tokens → n 流 → SNN layers(mHC + spike_current + 动态K)
            → n 流聚合 → output_neuron(spike_current) → K帧mean → decode_proj → logits(tied head)
          ponder_cost: 动态 K 正则化，鼓励用更少步数处理简单 token
        """
        batch, seq_len = token_ids.shape

        # 重置所有神经元状态（直接赋值，FSDP 安全）
        self._reset_neurons()

        # 三段式
        h_streams = self.encode(token_ids)                    # 输入边界
        h_out, ponder_cost, ek_floor_cost = self.snn_forward(h_streams)  # SNN 核心
        logits = self.decode(h_out, seq_len)                  # 输出边界

        # b_th L2 正则化（遏制 10×LR 下 V_th 漂移）
        b_th_reg_cost = self._compute_b_th_reg_cost()

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
                b_th_reg_cost=b_th_reg_cost,
            )

        return SNNModelOutput(logits=logits, ponder_cost=ponder_cost,
                              ek_floor_cost=ek_floor_cost,
                              b_th_reg_cost=b_th_reg_cost)

    def _compute_b_th_reg_cost(self) -> torch.Tensor:
        """b_th L2 正则化：遏制 V_th 漂移（10×LR + 0 weight_decay 下 b_th 无约束）。

        b_th 通过 |·|+v_th_min 变换为 V_th(t)。10×LR 加速学习但无 weight decay 约束，
        导致 b_th 绝对值持续增长 → V_th 漂移 → MPD alpha 崩溃。
        L2 正则化提供软回复力，防止 b_th 远离初始分布。

        Returns:
            scalar — 所有层 b_th 的均方均值
        """
        total = torch.zeros(1, device=self.layers[0].block_out_proj.weight.device,
                            dtype=torch.float32).squeeze()
        for layer_module in self.layers:
            total = total + layer_module.snn_block.b_th.pow(2).mean()
        return total / len(self.layers)

    def compensate_modulation_gradients(self, max_comp: float = 100.0):
        """
        Natural Gradient 补偿: sigmoid/softplus 饱和补偿。

          β = sigmoid(b_beta), sigmoid 在高 β 区（β=0.99, sigmoid'=0.01）梯度衰减 100x。
          补偿: grad /= activation'(b)，等价于在 β/α 空间做梯度下降。

        调用时机: optimizer.step() 之前、clip_grad_norm_ 之前。

        Args:
            max_comp: 补偿因子上限（防止极端值导致不稳定）
        """
        for layer_module in self.layers:
            block = layer_module.snn_block

            # b_beta: sigmoid 饱和补偿
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
            # mHC 超连接参数
            'hc_params': [],
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

            # mHC 超连接参数
            for hc in [layer_module.block_hc, layer_module.ffn_hc]:
                groups['hc_params'].extend([
                    hc.norm.weight,
                    hc.theta_pre, hc.b_pre, hc.alpha_pre,
                    hc.theta_post, hc.b_post, hc.alpha_post,
                    hc.theta_res, hc.b_res, hc.alpha_res,
                ])

        return groups
