#!/bin/bash
# H100 SFT v2 启动脚本
#
# 从 HF 下数据 + 起始 ckpt, 用 DeepSpeed ZeRO-2 在 8x H100 上跑 SFT v2.
#
# 前置 (开机后 1-2 分钟):
#   cd /workspace/NeuronSpark-V1 && git fetch origin V2.5 && git reset --hard origin/V2.5
#   export HF_ENDPOINT=https://hf-mirror.com  # 如果 H100 网络不通直连 HF, 用 mirror
#   export HF_TOKEN=<你的 token>               # 如果需要写入权限
#
# 使用:
#   bash scripts/sft_v2_h100.sh               # 从头启动
#   bash scripts/sft_v2_h100.sh --resume ...  # 断点续训

set -e

# ================ 配置 ================
DATA_REPO='Brain2nd/NeuronSpark-SFT-v2-Mix'
BASE_REPO='Brain2nd/NeuronSpark-V2.5-SFT'  # 起始 ckpt (hf_step7000)
DATA_DIR='data/sft_v2_mix'
BASE_CKPT_DIR='checkpoints_sft/hf_step7000'
OUT_DIR='checkpoints_sft_v2'

# 训练超参 (3 epoch, seq_len 2048, 8x H100 bs=2 accum=8 → global 128)
EPOCHS=3
BATCH_SIZE=2
ACCUM_STEPS=8
MAX_LEN=2048
LR=5e-5
WARMUP=200       # 294k/(128)≈2300step/epoch, 3ep=7000, warmup 200 ~3%
NEURON_LR_MULT=10

# ================ 1. 下数据 ================
if [ ! -d "$DATA_DIR" ]; then
    echo "[$(date)] Downloading $DATA_REPO → $DATA_DIR"
    huggingface-cli download "$DATA_REPO" --repo-type dataset --local-dir "$DATA_DIR"
else
    echo "[$(date)] $DATA_DIR 已存在, 跳过下载"
fi

# ================ 2. 下起始 ckpt ================
if [ ! -f "$BASE_CKPT_DIR/model.safetensors" ]; then
    echo "[$(date)] Downloading $BASE_REPO → $BASE_CKPT_DIR"
    mkdir -p "$BASE_CKPT_DIR"
    huggingface-cli download "$BASE_REPO" --local-dir "$BASE_CKPT_DIR"
else
    echo "[$(date)] $BASE_CKPT_DIR 已存在, 跳过下载"
fi

# ================ 3. DeepSpeed config ================
# (8x H100 ZeRO-2 bf16 + 神经元 fp32 master)
cat > configs/ds_config_sft_v2.json <<'DSEOF'
{
  "train_batch_size": 128,
  "train_micro_batch_size_per_gpu": 2,
  "gradient_accumulation_steps": 8,
  "gradient_clipping": 1.0,
  "bf16": {"enabled": true},
  "zero_optimization": {
    "stage": 2,
    "contiguous_gradients": true,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "allgather_bucket_size": 5e8
  },
  "steps_per_print": 100,
  "wall_clock_breakdown": false
}
DSEOF

# ================ 4. 训练 ================
mkdir -p logs "$OUT_DIR" runs

# 命名规范: {kind}_{git_short_hash}, 对齐仓库旧 runs (ddp_ec4964b / sft_7f1cd8e)
GIT_HASH=$(git rev-parse --short HEAD)
RUN_NAME="sft_v2_${GIT_HASH}"
echo "[$(date)] Run name: $RUN_NAME"

deepspeed --num_gpus=8 sft_ds.py \
    --pretrained_ckpt "$BASE_CKPT_DIR" \
    --sft_data_path "$DATA_DIR" \
    --out_dir "$OUT_DIR" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --accumulation_steps $ACCUM_STEPS \
    --max_length $MAX_LEN \
    --learning_rate $LR \
    --warmup_iters $WARMUP \
    --neuron_lr_mult $NEURON_LR_MULT \
    --log_interval 50 \
    --save_interval 500 \
    --dashboard_dir "runs/${RUN_NAME}" \
    --deepspeed \
    --deepspeed_config configs/ds_config_sft_v2.json \
    "$@" \
    2>&1 | tee "logs/${RUN_NAME}.log"
