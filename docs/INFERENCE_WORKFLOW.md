# 推理工作流（H200 → 本地）—— 必读

## 背景：H200 训练不能动

- H200 (`/data/NeuronSpark-V1-v3/`) 上的 `neuronspark/modeling_neuronspark.py` 与 `neuronspark/configuration_neuronspark.py` **不能改**：训练在跑，重启即丢进度。
- H200 上这两份文件是**老 buggy 版本**（带已知错误的 `SNNLanguageModel.generate()` + 错的 mixed-precision 加载）。训练用不到 `generate()`，所以 bug 对训练没影响。
- ckpt 保存时**不会带 modeling code**，HF 仓库 `Brain2nd/NeuronSpark-V3-1.1B-Pretrain` 上的 modeling/config 是历史上某次上传时连带传上去的旧版本，已经过期。

## 本地是唯一正确的代码源

`/home/dgxspark/Desktop/NeuronSpark-V1/neuronspark/modeling_neuronspark.py` + `configuration_neuronspark.py` 是**修过 inference bug 的正确版**。两个 fix 必须同时存在：

1. **`SNNLanguageModel.generate` 必须删除（或不被调用）**
   - 旧 bug：保留 `snn.generate`，prefill 后单 token 推 K 帧、跨 token 维护 PLIF `.v`。这与训练 forward 的 `reset_net()` 语义不一致 → 实际跑出来 token-level 流畅但语义破碎、混语种、乱日期数字。
   - 正确：让 HF `GenerationMixin.generate()` 接管，每步用全序列重算 forward（无 KV cache，慢但正确）。
2. **`NeuronSparkConfig` 必须含**
   ```python
   self.num_hidden_layers = num_layers   # HF generation 期望此字段
   self.use_cache = False                # 阻止 HF 试图建 DynamicCache
   ```
   缺这两行 HF 会试图构造 KV cache → 报错或行为异常。
3. **`from_pretrained` 用 per-tensor 混合精度**：矩阵 bf16 / 神经元 (`.w`/`.v_th`/`.b_beta`/`.b_alpha`/`.b_th`) fp32 / `k_predictor.bias`+`._usage_ema` fp32 / RoPE buffer 重填。对齐训练 `utils/param_groups.promote_neuron_params_fp32`。

## 标准流程（每次从 H200 取新 ckpt 后必走一遍）

```
1. H200 ckpt → HF private repo 上传 (只 weights，不要 modeling code)
   - 在 H200 上 upload_large_folder 整个 ckpt_stepN/ → Brain2nd/NeuronSpark-V3-1.1B-Pretrain
   - 包含: model.safetensors, deepspeed/, config.json (训练 config), generation_config.json,
           training_state.pth, latest, zero_to_fp32.py
   - 不含: modeling_neuronspark.py / configuration_neuronspark.py (HF 上的本来就是旧版，留它就行)

2. 本地下载 (ignore deepspeed/ training_state.pth latest)
   snapshot_download(repo_id, local_dir='checkpoints_hf_v3_stepN/',
                     ignore_patterns=['deepspeed/*', 'training_state.pth', 'latest'])

3. 【关键修复步骤 — 不做就乱码】
   cp neuronspark/modeling_neuronspark.py     checkpoints_hf_v3_stepN/modeling_neuronspark.py
   cp neuronspark/configuration_neuronspark.py checkpoints_hf_v3_stepN/configuration_neuronspark.py

4. 推回 HF (覆盖旧 modeling/config)，让 HF 仓库的 trust_remote_code 拉到正确版
   api.upload_file(path_or_fileobj='neuronspark/modeling_neuronspark.py',
                   path_in_repo='modeling_neuronspark.py', repo_id=...)
   api.upload_file(path_or_fileobj='neuronspark/configuration_neuronspark.py',
                   path_in_repo='configuration_neuronspark.py', repo_id=...)

5. 本地推理（多温度多超参对比）
   python generate.py --checkpoint checkpoints_hf_v3_stepN/ --mode pretrain \
       --prompt "..." --temperature {0, 0.5, 0.8, 1.0} --top_p {0.9, 0.95, 1.0} ...
```

## 验证 fix 生效的 sanity check

修复前后 greedy 输出对比（step 84000）：

| prompt | 修前（snn.generate bug） | 修后（HF GenerationMixin） |
|---|---|---|
| `The capital of France is` | "in French-speaking cultures that are the man..." | "the capital of the French Republic. The French government has been in power since 1945..." |
| `1+1=` | "20月2000000" | "2\n- 1+3=4\n" |
| `中华人民共和国的首都是` | "和朋友觉得这个话题对2020月31..." | "中国共产党党员、中国共产党党员..." |

**症状识别**：
- 输出含**乱日期数字** ("20月20...", "1520月2月20")、**无理由切换语种**、**单字流畅但跨 token 语义崩溃** = snn.generate path 命中 bug
- 输出**长程一致** + **答得上事实** + 连续重复也是连续 token 的重复 = HF GenerationMixin 正确路径

## 教训（写给未来的我）

- **永远先 md5 比对** `local neuronspark/` vs `下载 ckpt/` 的 modeling 与 config。不一致 = 必须用本地版覆盖下载版后再推理。
- **不要把"输出乱"先归因到模型欠训**。先排查推理路径 bug：greedy + 简单事实题（首都、1+1）可在 step 84000 这种早期阶段就看出 fix 是否生效。
- **H200 上的代码是过期的，本地 repo 才是 source of truth**。不要反过来。
- 上传 ckpt 到 HF 时只覆盖 weights，**不要带 modeling code 一起传**——除非你刚 commit 了新版且确定要替换。否则容易把 HF 上的旧 buggy 版"按原样"留下来给下一次推理踩坑。
