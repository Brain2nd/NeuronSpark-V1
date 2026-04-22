# v3 Pretrain Dataset Design & Execution Plan

**Branch**: v3
**Status**: 设计中，待 H100 落地
**Last updated**: 2026-04-22
**Owner**: Zhengzheng Tang

---

## 0. TL;DR

v3 预训练数据重做，相对 V2.5 的核心改动：

1. **Tokenizer 换成 v3-128K**（filtered Qwen3），全部数据**从原文 retokenize**
2. **混入合成指令 + CoT 数据**（约 10-20%），pretrain 阶段就激活 `<think>/</think>` 推理模式
3. **加代码语料**（约 10-15%），让 v3-tokenizer 的代码压缩红利真正兑现
4. **英文 + 中文比例可控**（不再像 V2.5 那样磁盘堆中文实际吃英文）

目标规模：**20-30 B tokens**（比 V2.5 的 1.4B 大 15-20 倍），在 8×H100 约 **3-5 天**训完（D=1024 / 24 层 / K=12 / seq_len=2048，step 约 1.5-2M tokens/s）。

---

## 1. 背景：V2.5 的经验教训

| 问题 | V2.5 现状 | v3 对策 |
|---|---|---|
| Token 预算太小 | 1.4B tokens 无法让任何 ~1B 模型"学好" | 20B+ tokens，给基本能力机会 |
| 中文磁盘满但实际英文为主 | SkyPile 620GB 但采样比例低 | 明确声明目标 token 配比（EN:ZH ≈ 1:1 或可调） |
| 代码完全缺失 | 0 代码 token | 10-15% code 补强 |
| 数学从零 | gsm8k 0% | 5% 数学 CoT 数据 |
| Thinking token 从未训练 | `<\|thinking\|>/<\|/thinking\|>` 零信号 | 混入 R1-distill，pretrain 阶段就见过 `<think>/</think>` |
| Tokenizer 对长尾差 | 自训 64K，code 压缩只 3.20 B/tok | v3-128K 代码 4.48 B/tok (+40%) |

---

## 2. 设计目标

### 2.1 能力优先级

1. **双语 base-LM 基础**：EN/ZH 通用文本连贯性
2. **推理模式预激活**：大量 `<think>...</think>` 分布性训练，让 thinking token 在 pretrain 末就不是 OOV
3. **数学 token efficiency**：数学 CoT 数据密集但精准
4. **代码可读写**：不求 HumanEval 爆表，但至少能 tokenize + 生成结构正确的代码片段
5. **知识密度**：合成教科书（Cosmopedia）做知识压缩的 backbone

### 2.2 明确不做

- 多模态（vision/image）
- 非 EN/ZH 语言（JA/KO/AR/RU 等）
- 长上下文（>2048 暂不做，final phase 再考虑 YaRN 外推）
- 海量 web 无过滤堆量（我们不是 Llama-3 规模，走质胜于量）

### 2.3 三档 token 预算（对应不同信心 / 资源）

| 档位 | Tokens | 8×H100 wallclock | 用途 |
|---|---:|---:|---|
| **Min** | 10 B | ~1.5 天 | 小幅风险验证，确认管线通畅 |
| **Target** | **25 B** | **3.5 天** | **主推方案**，对齐 Phi-1/2 经验 |
| **Max** | 80 B | ~11 天 | 若目标是 Llama-1 / Mamba-1.4B 水平直接打 |

文档下文以 **Target = 25B tokens** 为基准，其余按比例缩放。

---

## 3. 数据源清单与配比

### 3.1 本地已有（raw text，可直接 retokenize）

| 数据集 | 位置 | 原始大小 | 语言 | 类型 | v3-tokens 估算 |
|---|---|---:|---|---|---:|
| SkyPile-150B | `data/SkyPile-150B/data/*.jsonl` | 620 GB | ZH | web | ~180 B raw |
| chinese_fineweb | `data/chinese_fineweb_max_tokens_100B/` (已 tokenize, 需拿回原文) | 373 GB | ZH | web | (已 NS-64K 预 tokenize, 要重获原文) |
| seq-monkey | `data/seq-monkey/*.jsonl` | 69 GB | ZH | 通用语料 | ~20 B raw |
| cosmopedia | `data/pretrain_raw/cosmopedia/data/*/` | 86 GB | EN | 合成教科书 | ~25 B raw |
| fineweb-edu | `data/pretrain_raw/fineweb-edu/sample/10BT/` | 27 GB | EN | 过滤 edu web | ~8 B raw |
| openwebmath | `data/pretrain_raw/openwebmath/data/` | 26 GB | EN | 数学 web | ~7 B raw |
| belle_math | `data/pretrain_raw/belle_math/` | 126 MB | ZH | 数学 SFT | ~40 M raw |
| r1_bootstrap (6 集) | `data/r1_bootstrap/` | 4.5 GB | EN+ZH | R1-distill CoT | ~1.4 B raw |

**raw 总量**：~240 B tokens 级别，远超 25B 目标 —— 可以"采样保持配比"而不是"吃满"。

### 3.2 待下载

| 数据集 | 估算大小 | 语言 | 类型 | 备注 |
|---|---:|---|---|---|
| bigcode/the-stack-dedup (Python/JS/Go/Rust/C++ subset) | ~50 GB subset | code | 多语种代码 | 全集 3 TB，取子集 |
| AI-MO/NuminaMath-CoT | 1.2 GB | EN+ZH | 数学 CoT | 860k 样本 |
| HuggingFaceFW/fineweb (EN) | 选 20-30 GB sample | EN | 通用 web | edu-filter 外的补充 |
| allenai/peS2o | 50 GB sample | EN | 学术论文 | 提升知识密度（可选） |

### 3.3 Target 25B tokens 的推荐 mix

| 类别 | 权重 | Tokens | 来源（采样策略） |
|---|---:|---:|---|
| 英文 web | 28% | 7.0 B | fineweb-edu 全吃 (~8B) + cosmopedia web_samples_v1/v2 各采 15GB |
| 中文 web | 22% | 5.5 B | SkyPile 采 20GB + seq-monkey 采 5GB |
| 合成教科书 | 15% | 3.75 B | cosmopedia stanford + openstax + stories + wikihow + khanacademy |
| 代码 | 15% | 3.75 B | the-stack Python 2B + JS 0.5B + C++ 0.5B + Go/Rust/Java 各 0.2B + markdown 0.3B |
| **R1-distill CoT** | **10%** | **2.5 B** | 全 6 集 repeat ×1.8 (~1.4B × 1.8) |
| 数学 | 7% | 1.75 B | openwebmath 全吃 (~7B 取 1.5B) + NuminaMath-CoT 0.25B |
| 中文专业 | 3% | 0.75 B | belle_math + coig-cqia + chinese-simpleqa (`data/raw/`) + cosmopedia 中文段 |

合计 25.0 B。

#### 对应 Min=10B 档

按同比例：EN web 2.8B / ZH web 2.2B / 合成 1.5B / 代码 1.5B / CoT 1.0B / 数学 0.7B / 中文专业 0.3B

#### 对应 Max=80B 档

放量，CoT 重复采样到 4x（约 5.6B），代码加权到 20%。

---

## 4. Pipeline 架构

```
RAW SOURCES (parquet/jsonl)
    │
    ▼
┌─────────────────────────────┐
│  stage 1: language filter   │  fasttext/langdetect 保留 EN/ZH
│  stage 2: quality filter    │  length / repetition / boilerplate
│  stage 3: dedup             │  MinHash-LSH (Jaccard 0.85)
│  stage 4: format normalize  │  R1-distill -> ChatML wrapped
└─────────────────────────────┘
    │
    ▼
CLEANED TEXT (per-source parquet)
    │
    ▼
┌─────────────────────────────┐
│  stage 5: tokenize (v3-128K)│  并行 shard，输出 uint32 .bin
│  stage 6: shard + shuffle   │  global shuffle by document
│  stage 7: weighted sample   │  按 mix ratio 生成 flat stream
└─────────────────────────────┘
    │
    ▼
FINAL: data/v3_pretrain_mix/
       shard_00000.bin .. shard_NNNNN.bin
       shard_00000.bos.idx .. (doc-boundary index)
       manifest.json (mix ratios, sources, token counts)
```

---

## 5. 各 stage 设计细节

> **注**：本版不做 eval decontamination。Pretrain 源（SkyPile / fineweb-edu / cosmopedia / the-stack）是大规模 web + 合成，直接命中 benchmark test 文本的概率极低；我们也不把 benchmark train split 混入 pretrain（V2.5 那种做法只在 SFT 阶段）。日后若要报"decontaminated 零样本"指标，可补一次性扫描或在 final 阶段再加。

### 5.1 语言过滤

- **工具**：`fasttext`（lid.176 model）+ `cld3` 双保险
- **规则**：
  - 中文 doc：`lang_score >= 0.9` 且 `zh char ratio >= 0.3`
  - 英文 doc：`lang_score >= 0.85` 且 `ascii ratio >= 0.85`
  - 其他：直接丢（除非是代码/数学源）
- **代码源**：不过滤语言（the-stack 自带 programming language label）

### 5.2 质量过滤

**启发式（借鉴 C4 / FineWeb）**：
- 文档长度 ≥ 128 token（太短丢）
- 文档长度 ≤ 100 KB（太长切分）
- 重复行比例 < 30%
- 重复段落比例 < 20%
- top-1 频词占比 < 12%
- bullet/list ratio（整篇都是列表）< 60%
- URL / 邮箱 / 电话 密度（SEO 垃圾）< 15%

**可选 perplexity filter**（二阶段）：
- 用 Qwen3-0.6B-Base 算 ppl
- 丢弃 ppl > 10K 的（乱码）和 ppl < 5 的（极低熵，可能是重复模板）
- 成本：8×H100 跑一遍全集约 1 天

### 5.3 去重

**两级**：
1. **URL/ID 去重**（快）：同 URL / doc_id 只留一个
2. **MinHash-LSH**（准）：
   - shingles：5-gram 词级
   - MinHash：128 hash functions
   - LSH：Jaccard threshold 0.85
   - 使用 `datasketch` 库或 HuggingFace `text_dedup`

**跨源去重**：SkyPile 和 seq-monkey 可能有重合，fineweb-edu 和 cosmopedia web 可能有重合，必须全局去重。

### 5.4 R1-distill 格式归一化

六个数据集 schema 各异，统一成 ChatML：

```
<|im_start|>user
{问题}<|im_end|>
<|im_start|>assistant
<think>
{推理轨迹}
</think>
{最终答案}<|im_end|>
```

字段映射：

| 数据集 | user | thinking | answer |
|---|---|---|---|
| OpenThoughts / Bespoke | `conversations[0]["value"]` 去 "return within \boxed{}" 前缀 | 从 `conversations[1]["value"]` 提取 `<\|begin_of_thought\|>...<\|end_of_thought\|>` | 剩余部分 |
| QwQ-LongCoT | `problem` | `qwq` 全文作为 thinking | 从 qwq 尾部抽 "Final Answer" 段 |
| Chinese-R1-110k | `input` | `reasoning_content` | `content` |
| s1K-1.1 | `question` | `deepseek_thinking_trajectory` | `deepseek_attempt` |
| LIMO | `question` | 无显式 thinking, 把 `solution` 塞入 thinking | `answer` |

**重要**：生成的文本作为 **pretrain** 使用（无 loss mask，整段算 loss），不是 SFT。

### 5.5 Tokenize

- **Tokenizer**：`tokenizer_v3/`（v3-128K filtered Qwen3）
- **分片**：每文档独立 encode，`<|endoftext|>` (id=128361) 作为文档分隔符
- **存储**：
  - `.bin` 文件：uint32 数组，连续 token IDs
  - `.bos.idx` 文件：uint64 数组，每文档在 .bin 中的起始 offset（bytes）
- **shard 大小**：每 shard 约 10 GB（~2.5B tokens at uint32）
- **目录**：`data/v3_pretrain_mix/shards/shard_NNNNN.bin`

这个格式兼容我们现有的 `train_ds.py` / `PretrainDataset`（在 V2.5 已验证）。

### 5.6 全局 shuffle + 权重采样

- **文档级 shuffle**：每个 source 内部 shuffle，然后按 mix 权重构建"虚拟索引流"
- **流式采样**：训练时每 step 从 manifest 按权重拿 batch，epoch 概念弱化
- **固定 seed**：`SEED_PRETRAIN_V3 = 42`，可复现

---

## 6. 输出产物

```
data/v3_pretrain_mix/
├── manifest.json               # mix config, token counts, sources, seed
├── dedup_report.json           # 去重前后 token 数
├── quality_report.json         # 质量过滤统计
├── shards/
│   ├── shard_00000.bin
│   ├── shard_00000.bos.idx
│   ├── ...
│   └── shard_09999.bin
└── tokenizer_v3/               # 硬链接或复制 tokenizer，避免训练时找不到
```

`manifest.json` 示例：

```json
{
  "version": "v3-target-25B",
  "total_tokens": 25_000_000_000,
  "tokenizer": "tokenizer_v3/ (sha256: ...)",
  "seed": 42,
  "created_utc": "2026-05-??T??:??:??Z",
  "mix": {
    "en_web":      {"tokens": 7.0e9, "weight": 0.28, "sources": [...]},
    "zh_web":      {"tokens": 5.5e9, "weight": 0.22, "sources": [...]},
    "synthetic":   {"tokens": 3.75e9, "weight": 0.15, "sources": [...]},
    "code":        {"tokens": 3.75e9, "weight": 0.15, "sources": [...]},
    "r1_distill":  {"tokens": 2.5e9, "weight": 0.10, "sources": [...]},
    "math":        {"tokens": 1.75e9, "weight": 0.07, "sources": [...]},
    "zh_pro":      {"tokens": 0.75e9, "weight": 0.03, "sources": [...]}
  },
  "dedup_jaccard": 0.85
}
```

---

## 7. 脚本清单

每个脚本独立可运行、可 resume（checkpoint 写 `.done` 文件跳过已完成 shard）。

| # | 脚本 | 输入 | 输出 | 预计时间 |
|---:|---|---|---|---:|
| 1 | `scripts/v3_data/download_v3.py` | 配置 list | `data/r1_bootstrap/` 等 | 2-5 h（含 the-stack） |
| 2 | `scripts/v3_data/language_filter.py` | raw jsonl / parquet | `data/v3_staging/<src>/lang.jsonl` | 4-6 h |
| 3 | `scripts/v3_data/quality_filter.py` | stage 2 out | `data/v3_staging/<src>/qual.jsonl` | 6-8 h |
| 4 | `scripts/v3_data/minhash_dedup.py` | stage 3 out | `data/v3_staging/<src>/final.jsonl` | 10-14 h |
| 5 | `scripts/v3_data/format_r1_distill.py` | r1_bootstrap/ | `data/v3_staging/r1/chatml.jsonl` | 1 h |
| 6 | `scripts/v3_data/tokenize_shard.py` | cleaned jsonl | `.bin` + `.bos.idx` | 6-10 h (parallel) |
| 7 | `scripts/v3_data/weighted_merge.py` | per-source shards | `data/v3_pretrain_mix/shards/` | 3-4 h |
| 8 | `scripts/v3_data/verify_mix.py` | final shards | 报告：token 数 / 配比 / 健康检查 | 30 min |

**全管线 wallclock 估算（8×H100, 24 CPU cores）**：**~1.5-2.5 天**（主要瓶颈在 MinHash dedup 和 quality filter 的 CPU 工作）。

---

## 8. 健康检查（stage 8 verify_mix）

每项必须 PASS：

1. **Token 总数** = manifest 声明值 ± 2%
2. **Mix 比例** = 声明值 ± 1pp
3. **Round-trip**：随机抽 100 文档，decode 回文本，与源比对（identity）
4. **Tokenizer 一致性**：全部 token ID 在 [0, 128386]
5. **Shard 完整性**：`.bos.idx` 与 `.bin` offset 对齐，无断裂
6. **可训性冒烟**：用 2 shards 跑 100 step `train_ds.py`，loss 不 NaN/Inf

---

## 9. 开放决策点（执行前必须定）

| # | 决策 | 选项 | 默认 | 需要确认 |
|---:|---|---|---|---|
| 1 | Token 预算 | 10B / 25B / 80B | 25B | ✓ |
| 2 | EN/ZH 比例 | 1:1 / 1:0.7 / 1:1.3 | **1:0.78（28+15 : 22+3=43:25）** | ✓ |
| 3 | Mix 方案 | Target mix（§3.3）vs Phi-style 激进 | §3.3 Target | ✓ |
| 4 | 代码语言选择 | Python-only / 多语种 | **多语种**（Python 主导 2B + 其他 1.75B） | ✓ |
| 5 | Context length | 2048 / 4096 | **2048**（和 V2.5 一致，避免架构改动） | ✓ |
| 6 | 质量 perplexity filter | 要 / 不要 | **不要**（成本高且 FineWeb-edu 已预筛） | ✓ |
| 7 | R1-distill 是否重复采样 | 1x / 1.8x / 4x | **1.8x** (让 CoT 达 10% 占比) | ✓ |
| 8 | chinese_fineweb 预 tokenize 数据怎么处理 | 原文重获 / 丢弃 | **丢弃**（SkyPile + seq-monkey 已够 ZH web） | ✓ |

---

## 10. H100 执行计划

### 10.1 磁盘预算

```
raw data (已有):        ~850 GB
download (the-stack子集 + NuminaMath): +60 GB
staging (dedup/filter/normalize 中间文件): ~600 GB peak
v3_pretrain_mix final: ~100 GB (25B tokens × 4B uint32)
───────────────────────────────────
峰值需求: ~1.6 TB
```

H100 pod 本地盘需至少 **2 TB NVMe**，或挂对象存储。

### 10.2 并行度

- **下载**：8 并发（HF token rate limit）
- **language filter / quality filter**：CPU-bound，`multiprocessing.Pool(ncpus)`
- **MinHash**：`datasketch` 单机多核；大数据集可分 bucket 并行
- **Tokenize**：per-shard 并行，每进程独占 tokenizer 实例（避免 tokenizers 库线程锁）
- **Weighted merge**：单进程 I/O-bound

### 10.3 Resume 策略

每个 stage 写 `stage_N_done.json` 记录已完成 shard 列表。崩溃后 rerun 同脚本自动跳过。

### 10.4 审计日志

每个脚本输出 `<stage>_log.jsonl`（每处理一个文件一行），包含：
- 输入/输出路径
- 原始 token 数 / 处理后 token 数
- 丢弃数 + 原因分布
- 耗时

---

## 11. 里程碑

| Milestone | 交付物 | 完成标志 |
|---|---|---|
| **M0**: 文档评审 | 本文件 + 用户确认 §9 所有决策 | 用户 ACK |
| **M1**: 脚本开发 | scripts/v3_data/*.py 全套 | 本地 dry-run 通过（1GB 子集） |
| **M2**: H100 管线跑通 | 完整 25B pretrain_mix | verify_mix.py 全 PASS |
| **M3**: 冒烟训练 | 100 step 训练不炸 | loss 曲线单调下降 + dashboard 无警告 |
| **M4**: 正式 pretrain | 25B tokens 跑完 | checkpoint + 指标对比 V2.5 |

---

## 12. 风险登记

| 风险 | 影响 | 对策 |
|---|---|---|
| chinese_fineweb 已预 tokenize 过 NS-64K，原文难获 | 丢失 373 GB 中文语料 | 用 SkyPile + seq-monkey 顶上（~690GB ZH raw 绝对够） |
| R1-distill 重复采样 1.8x 可能过拟合某类模式 | Thinking token 分布偏斜 | 采样前先 shuffle + 限制同一 prompt 只允许 1 个 CoT 实例 |
| The-stack 下载子集带 license 复杂 | 法律风险 | 只取 permissive subset（`bigcode/the-stack-dedup` `license_selection=True`） |
| 不做 decontamination，pretrain 可能偶遇 eval test | benchmark 数字虚高 | 默认不查；若后续发布论文，补一次 offline scan 报"contamination rate %" |
| 8×H100 的 NVMe 不足 2 TB | staging 阶段崩盘 | 提前和机房 / Runpod 确认；或分阶段串行跑 |
| tokenizer_v3 落地后发现 bug | 25B 全部重做 | 这份设计评审阶段就要覆盖 tokenizer round-trip 100% EXACT（已在 `scripts/tokenizer/verify_v3_tokenizer.py` 验证） |

---

## 13. 参考文献

- FineWeb / FineWeb-edu: Penedo et al. 2024
- Llama-3 data mixing: Llama Team 2024
- Phi-3 synthetic pretrain: Abdin et al. 2024
- DeepSeek-R1 distillation: DeepSeek-AI 2025
- OpenThoughts project: Bespoke Labs 2025
- MinHash-LSH dedup: Broder 1997, adapted in `text_dedup`
- SkyPile-150B: Skywork 2023
- Cosmopedia: HuggingFaceTB 2024
- The-Stack-v1-dedup: BigCode 2023
