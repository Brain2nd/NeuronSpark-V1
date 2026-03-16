# NeuronSpark V3: Bio-Inspired Hybrid Language Model

## Algorithm Whitepaper

---

## 1. Overview

NeuronSpark V3 is a hybrid language model that replaces conventional sequence mixers (Mamba / Self-Attention) with **Bio-inspired Spiking State Space Modules (BioSSM)**, while retaining Mixture-of-Experts (MoE) for channel mixing and sparse Grouped-Query Attention (GQA) for global context. The architecture processes tokens through a heterogeneous stack of 52 layers with three distinct block types arranged in a fixed interleaving pattern.

### 1.1 Design Philosophy

The core principle is **spike current + continuous residual flow**: spiking neurons output spike current $V_{th} \times s$ (a continuous, sparse amplitude-modulated signal), which is projected and accumulated into a continuous residual stream. Layers transmit continuous hidden states $\mathbf{h} \in \mathbb{R}^D$, not binary spikes. This solves deep-layer gradient vanishing while preserving SNN spike semantics.

### 1.2 Architecture Summary

| Component | Count | Function |
|-----------|-------|----------|
| BioSSM (S) | 23 layers | Sequence mixing via spiking state space |
| MoE (E) | 23 layers | Channel mixing via sparse expert routing |
| Attention (*) | 6 layers | Global context via grouped-query attention |
| Total | 52 layers | Heterogeneous decoder stack |

Layer pattern:

```
S E S E S * E S E S E S * E S E S E S * E S E S E S * E S E S E S E S * E S E S E S E S E
```

---

## 2. BioSSM Layer

The BioSSM layer is the core innovation, replacing Mamba as the sequence mixer. Each BioSSM layer converts continuous hidden states to binary spikes, processes them through a selective spiking state space, and aggregates across dynamic timesteps via PonderNet.

### 2.1 Overall Data Flow

```
Input:  h ∈ R^(T × B × D)          [continuous hidden states]

Step 1: Pre-LN RMSNorm
Step 2: Input PLIF Neuron           → spikes ∈ {0,1}^(TK × B × D)
Step 3: SNNBlock (7 projections)    → spike_current ∈ R^(TK × B × D)
Step 4: Reshape to (T × K × B × D)
Step 5: PonderNet Aggregation       → aggregated ∈ R^(T × B × D)
Step 6: Output Projection + Residual

Output: h_out = h + out_proj(aggregated)
```

Where:
- $T$ = sequence length
- $B$ = batch size
- $D$ = hidden dimension (e.g. 2688)
- $K$ = maximum SNN timesteps per token (e.g. 16)

### 2.2 Input PLIF Neuron

The input neuron converts continuous hidden states into binary spike trains by repeating each token $K$ times and applying a Parametric Leaky Integrate-and-Fire (PLIF) neuron.

**Expansion**: Each token's hidden state $\mathbf{x}_t$ is repeated $K$ times along a new time axis:

$$\mathbf{x}_{t,k} = \mathbf{x}_t \quad \text{for } k = 1, \ldots, K$$

Yielding shape $(T \cdot K, B, D)$.

**PLIF dynamics** (per-dimension, $d = 1, \ldots, D$):

$$\beta_d = \sigma(w_d) \in (0, 1)$$

$$V_d[n] = \beta_d \cdot V_d[n-1] + (1 - \beta_d) \cdot x_d[n]$$

$$s_d[n] = \Theta(V_d[n] - v_{th,d})$$

$$V_d[n] \leftarrow V_d[n] - v_{th,d} \cdot s_d[n] \quad \text{(soft reset)}$$

Where:
- $w_d$ is a learnable parameter, $\beta_d = \sigma(w_d)$ is the membrane decay rate
- $v_{th,d}$ is a learnable firing threshold, initialized $\sim U[0.25, 0.75]$
- $\Theta(\cdot)$ is the Heaviside step function (forward), approximated by sigmoid surrogate (backward)
- $V_d[0] = 0$ (reset per sequence)

**Output**: Binary spike tensor $\mathbf{s} \in \{0, 1\}^{TK \times B \times D}$

**Implementation**: Fused Triton parallel scan kernel (`plif_rowparam_forward`) computes the entire sequence in a single pass with $O(T \cdot K)$ work and $O(\log(T \cdot K))$ depth.

### 2.3 SNNBlock: Selective Spiking State Space

The SNNBlock is the computational core, analogous to Mamba's selective SSM. It processes binary spikes through **7 independent projections** with input-dependent dynamic neuron parameters.

#### 2.3.1 Projection Architecture

Given input spikes $\mathbf{s} \in \{0,1\}^{TK \times B \times D}$:

| Projection | Dimension | Activation | Purpose |
|-----------|-----------|------------|---------|
| $W_{in}$ | $D \to D \cdot N$ | — | Input current to hidden neurons |
| $W_\beta$ | $D \to D \cdot N$ | $\sigma(\cdot + b_\beta)$ | Dynamic decay rate $\beta(t) \in (0,1)$ |
| $W_\alpha$ | $D \to D \cdot N$ | $\text{softplus}(\cdot + b_\alpha)$ | Dynamic write gain $\alpha(t) \in \mathbb{R}^+$ |
| $W_{th}$ | $D \to D \cdot N$ | $|\cdot + b_{th}| + v_{th,min}$ | Dynamic threshold $V_{th}(t) > 0$ |
| $W_{gate}$ | $D \to D$ | $\sigma(\cdot)$ | Output gating |
| $W_{skip}$ | $D \to D$ | — | Skip connection (residual) |
| $W_{out}$ | $D \cdot N \to D$ | — | Hidden→visible projection |

Where $N$ is the state expansion factor (default 4), yielding $D \times N$ hidden neurons per layer.

#### 2.3.2 Selective PLIF Dynamics

The hidden neurons use a **SelectivePLIFNode** with input-dependent parameters (no learnable parameters of its own):

$$\mathbf{V}[n] = \boldsymbol{\beta}(n) \odot \mathbf{V}[n-1] + \boldsymbol{\alpha}(n) \odot \mathbf{I}[n]$$

$$\mathbf{s}_h[n] = \Theta(\mathbf{V}[n] - \mathbf{V}_{th}(n))$$

$$\mathbf{V}[n] \leftarrow \mathbf{V}[n] - \mathbf{V}_{th}(n) \odot \mathbf{s}_h[n]$$

Where all operations are element-wise over $D \cdot N$ dimensions.

**Key difference from standard PLIF**: $\beta(t)$, $\alpha(t)$, $V_{th}(t)$ are computed per-timestep from the input, making the neuron **selective** — its temporal dynamics adapt to the input content, analogous to Mamba's selective SSM parameterization.

#### 2.3.3 Spike Current Output

The hidden neurons output **spike current**, not binary spikes:

$$\mathbf{sc}[n] = \mathbf{V}_{th}(n) \odot \mathbf{s}_h[n]$$

This is a continuous, sparse signal: zero where no spike fires, $V_{th}$ where a spike fires. The amplitude carries information about the neuron's dynamic threshold at the moment of firing, enabling continuous gradient flow through the binary spike event.

#### 2.3.4 Gated Output Combination

$$\mathbf{I}_{out} = W_{out} \cdot \mathbf{sc}$$

$$\mathbf{output} = \mathbf{I}_{out} \odot \sigma(W_{gate} \cdot \mathbf{s}) + W_{skip} \cdot \mathbf{s}$$

The gate modulates the hidden neuron output by input spike activity; the skip connection provides a direct path from input to output.

#### 2.3.5 Parameter Initialization

The initialization is calibrated to produce desired firing rates and stable dynamics:

**Decay bias $b_\beta$**: Logit-spaced over $\beta \in [0.80, 0.99]$ for $N$ expansion channels:

$$b_{\beta,i} = \log\frac{\beta_i}{1 - \beta_i} + \mathcal{N}(0, 0.1), \quad \beta_i = 0.80 + \frac{i}{N-1}(0.99 - 0.80)$$

**Threshold bias $b_{th}$**: Calibrated for target firing rates $p_{fire} \in [0.25, 0.08]$ (higher for fast channels, lower for slow):

$$\sigma_I = \sqrt{p_{assumed} / 3}, \quad \sigma_V = \sigma_I \cdot \sqrt{1 - \beta^{2K}}$$

$$b_{th} = \max(0.05, \sigma_V \cdot z_{score} - v_{th,min})$$

Where $z_{score} = \sqrt{2} \cdot \text{erfinv}(2(1-p_{fire})-1)$.

**Output weight $W_{out}$**: Scaled by $1/\sqrt{p_{fire}}$ to compensate for firing rate sparsity.

### 2.4 PonderNet Adaptive Aggregation

Each token's $K$ timestep frames are aggregated with learned, position-dependent weights following a geometric distribution (PonderNet):

#### 2.4.1 Halt Probability

A per-frame halt probability is computed from the frame content:

$$p_k = \sigma(\text{clamp}(\text{halt\_proj}(\mathbf{f}_k), -6, 6)), \quad k = 1, \ldots, K$$

Where $\text{halt\_proj}: \mathbb{R}^D \to \mathbb{R}$ is a linear projection.

#### 2.4.2 Geometric Distribution Weights

$$\lambda_k = p_k \prod_{j=1}^{k-1}(1 - p_j)$$

Computed in log-space for numerical stability:

$$\log \lambda_k = \log p_k + \sum_{j=1}^{k-1} \log(1 - p_j)$$

Normalized: $\hat{\lambda}_k = \lambda_k / \sum_{k'} \lambda_{k'}$

#### 2.4.3 Weighted Aggregation

$$\mathbf{h}_{agg}[t] = \sum_{k=1}^{K} \hat{\lambda}_k[t] \cdot \mathbf{f}_k[t]$$

Different tokens use different effective step counts: a simple token may halt at $K=3$, while a complex token may use all $K=16$ steps.

#### 2.4.4 Regularization Costs

**Expected steps** (per position):

$$E[K]_t = \sum_{k=1}^{K} k \cdot \hat{\lambda}_k[t]$$

**Ponder cost** (encourage early halting):

$$\mathcal{L}_{ponder} = \frac{1}{T} \sum_t E[K]_t$$

**E[K] floor penalty** (prevent collapse to $K=1$):

$$\mathcal{L}_{floor} = \frac{1}{T} \sum_t \text{ReLU}(K_{floor} - E[K]_t)^2$$

### 2.5 Output Projection and Residual

$$\mathbf{h}_{out} = \mathbf{h} + W_{out\_proj} \cdot \mathbf{h}_{agg}$$

Where $W_{out\_proj} \in \mathbb{R}^{D \times D}$ is initialized with GPT-2 style depth scaling:

$$W_{out\_proj} \sim \mathcal{N}\left(0, \frac{0.02}{\sqrt{2 \cdot L}}\right)$$

With $L$ = total number of layers.

### 2.6 MPD-AGL Adaptive Surrogate Gradient

The surrogate gradient width $\alpha$ is dynamically computed based on neuron parameters:

$$\alpha = \frac{C}{\sqrt{1 + \bar{\beta}^2} \cdot \bar{\gamma} \cdot \bar{V}_{th}}, \quad C \approx 2.236$$

$$\alpha \in [2.0, 16.0]$$

Where:
- $\bar{\beta} = \text{mean}(\sigma(b_\beta))$: average decay rate
- $\bar{V}_{th} = \text{mean}(|b_{th}| + v_{th,min})$: average threshold
- $\bar{\gamma} = \text{mean}(W_{norm})$: average RMSNorm weight

This ensures the surrogate gradient adapts to the neuron's operating point: high threshold or high decay → sharper surrogate; low threshold → smoother surrogate.

---

## 3. Mixture-of-Experts (MoE) Layer

### 3.1 Architecture

Each MoE layer consists of:
- **Router**: Sigmoid top-k routing with EMA load balancing
- **32 Routed Experts**: Gated MLP in latent space ($D_{latent} = 256$)
- **1 Shared Expert**: Gated MLP in full dimension ($D$)

### 3.2 Routing

Given input $\mathbf{h} \in \mathbb{R}^{N_{tokens} \times D}$:

$$\text{logits} = \mathbf{h} \cdot W_{router}^T + b_{ema} \in \mathbb{R}^{N_{tokens} \times 32}$$

$$\text{scores} = \sigma(\text{logits})$$

Top-$k$ selection ($k=4$): For each token, select the 4 experts with highest scores.

$$\hat{w}_i = \frac{w_i}{\sum_{j \in \text{top-k}} w_j} \cdot r_{scale}$$

Where $r_{scale} = 2.5$ is the routed scaling factor.

### 3.3 EMA Router Bias

An exponential moving average tracks expert load:

$$\text{load}_i^{(t)} = (1 - \eta) \cdot \text{load}_i^{(t-1)} + \eta \cdot f_i$$

Where $f_i$ is the fraction of tokens routed to expert $i$, $\eta = 10^{-3}$.

Bias correction pushes load toward uniformity:

$$b_{ema,i} = 10 \cdot (\bar{\text{load}} - \text{load}_i)$$

### 3.4 Latent Space Expert Computation

Routed experts operate in a compressed latent space for efficiency:

$$\mathbf{z} = W_{down} \cdot \mathbf{h} \in \mathbb{R}^{256} \quad \text{(latent projection)}$$

For each activated expert $i$:

$$\mathbf{x}_{gate}, \mathbf{x}_{linear} = \text{split}(W_{fc1}^{(i)} \cdot \mathbf{z})$$

$$\mathbf{y}^{(i)} = W_{fc2}^{(i)} \cdot (\text{sc}(\mathbf{x}_{gate}) \odot \mathbf{x}_{linear})$$

Where $\text{sc}(\cdot)$ is the **spike current activation**: $\text{sc}(x) = V_{th} \cdot \Theta(x - V_{th})$ with $V_{th} = 0.3$.

$$\mathbf{z}_{routed} = \sum_{i \in \text{top-k}} \hat{w}_i \cdot \mathbf{y}^{(i)}$$

$$\mathbf{h}_{routed} = W_{up} \cdot \mathbf{z}_{routed} \in \mathbb{R}^D \quad \text{(latent→full)}$$

### 3.5 Shared Expert

Operates in full $D$-dimensional space with gating:

$$\mathbf{h}_{shared} = \text{SharedExpert}(\mathbf{h})$$

$$g = \sigma(W_{gate} \cdot \mathbf{h})$$

$$\mathbf{output} = \mathbf{h}_{routed} + g \odot \mathbf{h}_{shared}$$

### 3.6 Load Balance Loss

$$\mathcal{L}_{lb} = N_{experts} \sum_{i=1}^{N_{experts}} f_i \cdot P_i \cdot \alpha_{aux}$$

Where $f_i$ = fraction of tokens to expert $i$, $P_i$ = mean router probability for expert $i$, $\alpha_{aux} = 10^{-4}$.

---

## 4. Grouped-Query Attention (GQA) Layer

### 4.1 Architecture

6 attention layers are sparsely interleaved in the 52-layer stack, providing global context that BioSSM's recurrent dynamics cannot capture.

| Parameter | Value |
|-----------|-------|
| Query heads | 8 |
| KV heads | 2 (GQA 4:1 ratio) |
| Head dimension | 128 |
| Positional encoding | None |

### 4.2 No Positional Encoding

BioSSM's recursive dynamics implicitly encode position through membrane potential history. The attention layers do not apply RoPE or any explicit positional encoding, relying on the positional information already present in the residual stream from BioSSM processing.

### 4.3 Forward Pass

$$\mathbf{Q} = \mathbf{h} \cdot W_Q \in \mathbb{R}^{B \times T \times 8 \times 128}$$

$$\mathbf{K} = \mathbf{h} \cdot W_K \in \mathbb{R}^{B \times T \times 2 \times 128}$$

$$\mathbf{V} = \mathbf{h} \cdot W_V \in \mathbb{R}^{B \times T \times 2 \times 128}$$

GQA expansion: $\mathbf{K}, \mathbf{V}$ are repeated 4 times along the head axis to match 8 query heads.

$$\text{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} + \mathbf{M}_{causal}\right)\mathbf{V}$$

$$\mathbf{output} = W_O \cdot \text{concat}(\text{heads}) \in \mathbb{R}^{B \times T \times D}$$

Output projection uses GPT-2 depth scaling: $W_O \sim \mathcal{N}(0, 0.02/\sqrt{2L})$.

---

## 5. Multi-Token Prediction (MTP)

### 5.1 Architecture

A single MTP head predicts the token at offset $+2$ (one step beyond the standard next-token prediction), sharing the LM head weights.

### 5.2 Computation

For prediction offset $\delta$ ($\delta = 1$ for the default single MTP head):

$$\mathbf{h}_{trunc} = \mathbf{h}[:, :-({\delta+1})] \in \mathbb{R}^{B \times S \times D}$$

$$\mathbf{e}_{target} = \text{Embed}(\text{target\_ids}[:, \delta:-1]) \in \mathbb{R}^{B \times S \times D}$$

Separate normalization (Nemotron design):

$$\mathbf{combined} = W_{combine} \cdot [\text{RMSNorm}_h(\mathbf{h}_{trunc}) \| \text{RMSNorm}_e(\mathbf{e}_{target})]$$

Gated MLP trunk with $\text{ReLU}^2$ activation:

$$\mathbf{x}_{gate}, \mathbf{x}_{linear} = \text{split}(W_{fc1} \cdot \mathbf{combined})$$

$$\mathbf{trunk} = W_{fc2} \cdot (\text{ReLU}(\mathbf{x}_{gate})^2 \odot \mathbf{x}_{linear})$$

$$\text{logits}_{mtp} = \text{RMSNorm}_{out}(\mathbf{trunk}) \cdot W_{lm\_head}^T$$

$$\mathcal{L}_{mtp} = \text{CE}(\text{logits}_{mtp}, \text{target\_ids}[:, \delta+1:])$$

---

## 6. Complete Forward Pass

### 6.1 End-to-End Data Flow

```
token_ids: (B, T)
    ↓
Embedding: (B, T, D)
    ↓
52 × NeuronSparkBlock:
    ├── BioSSM (S):  h → Pre-LN → PLIF → K-expand → SNNBlock → PonderNet → out_proj → h + Δ
    ├── MoE   (E):  h → Pre-LN → Router → Experts(latent) + Shared → out_proj → h + Δ
    └── Attn  (*):  h → Pre-LN → Q/K/V → SDPA(causal) → O_proj → h + Δ
    ↓
RMSNorm: (B, T, D)
    ↓
LM Head: (B, T, vocab_size)
    ↓
logits → CE loss + MTP loss + ponder_cost + ek_floor_cost + lb_loss
```

### 6.2 Loss Function

$$\mathcal{L} = \mathcal{L}_{CE} + w_{mtp} \cdot \mathcal{L}_{mtp} + w_{ponder} \cdot \mathcal{L}_{ponder} + w_{floor} \cdot \mathcal{L}_{floor} + \mathcal{L}_{lb}$$

| Loss | Weight | Purpose |
|------|--------|---------|
| $\mathcal{L}_{CE}$ | 1.0 | Next-token prediction |
| $\mathcal{L}_{mtp}$ | 1.0 | Multi-token prediction |
| $\mathcal{L}_{ponder}$ | 0.01 | Encourage early PonderNet halting |
| $\mathcal{L}_{floor}$ | 0.1 | Prevent PonderNet collapse to $K=1$ |
| $\mathcal{L}_{lb}$ | $10^{-4}$ per layer | MoE load balance |

---

## 7. Parallel Scan Implementation

All PLIF neuron computations use fused Triton kernels for GPU-efficient parallel scanning.

### 7.1 Forward Scan

The recurrence $V[n] = \beta \cdot V[n-1] + u[n]$ is computed via prefix-sum decomposition:

For a sequence of length $L$, the scan operates in $O(L)$ work with $O(\log L)$ parallel depth by associating pairs $(a_i, b_i)$ where $a_i = \beta_i$ and $b_i = u_i$, composed as:

$$(a_2, b_2) \circ (a_1, b_1) = (a_2 \cdot a_1, a_2 \cdot b_1 + b_2)$$

The spike generation and soft reset are fused into the same kernel, avoiding intermediate memory writes.

### 7.2 Backward Accumulation

The reverse-mode gradient uses accumulation over the scanned sequence, also implemented as a fused Triton kernel with surrogate gradient computation inline.

### 7.3 Surrogate Gradient

The Heaviside function $\Theta(x)$ uses a sigmoid surrogate in the backward pass:

$$\frac{\partial s}{\partial V} \approx \frac{\alpha}{2} \cdot \text{sech}^2\left(\frac{\alpha (V - V_{th})}{2}\right) = \alpha \cdot \sigma(\alpha(V-V_{th})) \cdot (1 - \sigma(\alpha(V-V_{th})))$$

Where $\alpha$ is either fixed or adaptively computed via MPD-AGL (Section 2.6).

---

## 8. Model Configurations

### 8.1 Distillation Configuration (Nemotron-3-Nano-30B scale)

| Parameter | Value |
|-----------|-------|
| Hidden dimension $D$ | 2688 |
| State expansion $N$ | 4 |
| Max SNN timesteps $K$ | 16 |
| Number of layers | 52 |
| BioSSM layers | 23 |
| MoE layers | 23 |
| Attention layers | 6 |
| Vocabulary size | 131,072 |
| Attention heads | 8 (Q) / 2 (KV) |
| Head dimension | 128 |
| Routed experts | 32 |
| Experts per token | 4 |
| $v_{th,min}$ | 0.1 |
| $E[K]$ floor | 4.0 |
| Total parameters | ~28B (excluding replaced Mamba) |
| BioSSM parameters | ~3.8B |

### 8.2 Standalone Configuration (Original design)

| Parameter | Value |
|-----------|-------|
| Hidden dimension $D$ | 1024 |
| State expansion $N$ | 8 |
| Max SNN timesteps $K$ | 16 |
| Number of layers | 40 |
| Vocabulary size | 6,144 |
| Total parameters | ~874M |

---

## 9. Key Innovations

### 9.1 Spike Current as Continuous Output

Unlike conventional SNNs that output binary spikes, BioSSM outputs **spike current** $V_{th} \times s$. This provides:
- Continuous amplitude modulation through the dynamic threshold
- Sparse activation (zero where no spike)
- Smooth gradient flow via the surrogate function

### 9.2 Selective Dynamics

Following Mamba's selective SSM design, all neuron parameters ($\beta$, $\alpha$, $V_{th}$) are **input-dependent**, computed from separate linear projections of the input. This allows the neuron to:
- Focus on relevant features (high $\alpha$)
- Maintain or forget memory (high/low $\beta$)
- Adjust sensitivity (dynamic $V_{th}$)

### 9.3 PonderNet Adaptive Computation

Different tokens require different amounts of computation. PonderNet learns a per-position halting distribution over $K$ timesteps, allowing the model to allocate more SNN timesteps to complex tokens and fewer to simple ones.

### 9.4 Latent-Space MoE Routing

Experts compute in a compressed 256-dimensional latent space ($D_{latent} = D/4$), reducing per-expert computation by $16\times$ compared to full-dimension experts. A shared expert operates in full dimension to capture common patterns.

### 9.5 Spike Current Activation in MoE

MoE experts use spike current activation instead of conventional SiLU/ReLU:

$$\text{sc}(x) = V_{th} \cdot \Theta(x - V_{th}), \quad V_{th} = 0.3$$

This maintains the spiking paradigm throughout the architecture while providing the gating behavior needed for expert computation.
