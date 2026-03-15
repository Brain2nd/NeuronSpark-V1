#!/bin/bash
# ==============================================================================
# NeuronSpark v3: 三阶段 Mamba → BioSSM 蒸馏 (4×48GB FSDP)
#
# Stage 1 (对齐):  短序列, 强 cosine alignment, BioSSM 学习模仿 Mamba
# Stage 2 (过渡):  中序列, 平衡 CE 和 alignment, 逐步独立
# Stage 3 (自主):  全序列, CE 为主, BioSSM 独立语言建模
#
# 用法:
#   # 本地 JSONL (默认)
#   bash scripts/run_distill_v3.sh
#
#   # 自定义 teacher 和数据
#   bash scripts/run_distill_v3.sh \
#     --teacher /path/to/Nemotron-3-Nano-30B \
#     --data /path/to/data.jsonl
#
#   # HuggingFace 数据集 (本地已下载)
#   bash scripts/run_distill_v3.sh \
#     --data /data/my_hf_dataset \
#     --data_type huggingface \
#     --text_column content
#
#   # 单阶段调试
#   bash scripts/run_distill_v3.sh --stage 1
#
# 监控:
#   tensorboard --logdir runs/distill_v3 --port 6006
# ==============================================================================
set -euo pipefail

# ======================== 默认配置 ========================

# 模型 & 数据
TEACHER_PATH="/home/dgxspark/Desktop/Nemotron-3-Super-Research/HuggingFace-Models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
DATA_PATH="data/seq-monkey/seq_monkey_datawhale.jsonl"
DATA_TYPE="jsonl"
TEXT_COLUMN="text"
DATA_SPLIT="train"
DATA_SUBSET=""
TOKENIZER_PATH=""   # 留空 = 使用 teacher 的 tokenizer

# 输出
OUT_DIR="checkpoints_distill_v3"
LOG_BASE="runs/distill_v3"

# 硬件
NGPU=4

# BioSSM 超参
N=4              # 状态扩展因子
K=16             # 最大 SNN 时间步
V_TH_MIN=0.1
EK_FLOOR=4.0

# 训练
BATCH=1          # per-GPU (30B 模型, VRAM 紧张)
ACCUM=16         # 梯度累积 → effective batch = BATCH × NGPU × ACCUM = 64
GRAD_CLIP=1.0
NEURON_LR=10.0
WEIGHT_DECAY=0.1

# SNN 正则
PONDER_W=0.01
EK_FLOOR_W=0.1

# 日志
SAVE_INTERVAL=500
LOG_INTERVAL=10
NUM_WORKERS=4

# 控制
RUN_STAGE=""     # 留空 = 跑全部 3 阶段, 设为 1/2/3 只跑单阶段
RESUME=""        # 手动指定恢复 checkpoint

# ======================== 参数解析 ========================
while [[ $# -gt 0 ]]; do
    case $1 in
        --teacher)       TEACHER_PATH="$2";  shift 2 ;;
        --data)          DATA_PATH="$2";     shift 2 ;;
        --data_type)     DATA_TYPE="$2";     shift 2 ;;
        --text_column)   TEXT_COLUMN="$2";   shift 2 ;;
        --data_split)    DATA_SPLIT="$2";    shift 2 ;;
        --data_subset)   DATA_SUBSET="$2";   shift 2 ;;
        --tokenizer)     TOKENIZER_PATH="$2"; shift 2 ;;
        --out_dir)       OUT_DIR="$2";       shift 2 ;;
        --log_dir)       LOG_BASE="$2";      shift 2 ;;
        --ngpu)          NGPU="$2";          shift 2 ;;
        --batch)         BATCH="$2";         shift 2 ;;
        --accum)         ACCUM="$2";         shift 2 ;;
        --N)             N="$2";             shift 2 ;;
        --K)             K="$2";             shift 2 ;;
        --stage)         RUN_STAGE="$2";     shift 2 ;;
        --resume)        RESUME="$2";        shift 2 ;;
        --save_interval) SAVE_INTERVAL="$2"; shift 2 ;;
        *)
            echo "未知参数: $1" >&2
            exit 1 ;;
    esac
done

export TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas

echo "============================================================"
echo "NeuronSpark v3 三阶段蒸馏: Mamba → BioSSM"
echo "  Teacher:    $TEACHER_PATH"
echo "  Data:       $DATA_PATH (type=$DATA_TYPE, col=$TEXT_COLUMN)"
echo "  GPUs:       $NGPU"
echo "  Batch:      ${BATCH}/gpu × ${NGPU} × accum ${ACCUM} = $((BATCH * NGPU * ACCUM))"
echo "  BioSSM:     N=$N, K=$K"
echo "  Output:     $OUT_DIR"
echo "  Monitor:    tensorboard --logdir $LOG_BASE"
echo "============================================================"

mkdir -p "$OUT_DIR"

# ======================== 工具函数 ========================

find_latest_ckpt() {
    local latest
    latest=$(ls -t "$OUT_DIR"/distill_v3_step*.pth 2>/dev/null | head -1)
    if [ -z "$latest" ]; then
        echo "ERROR: 未找到 checkpoint in $OUT_DIR" >&2
        exit 1
    fi
    echo "$latest"
}

# 构建数据参数 (所有 stage 共用)
build_data_args() {
    local args="--data_path $DATA_PATH --data_type $DATA_TYPE --text_column $TEXT_COLUMN"
    args="$args --data_split $DATA_SPLIT"
    if [ -n "$DATA_SUBSET" ]; then
        args="$args --data_subset $DATA_SUBSET"
    fi
    if [ -n "$TOKENIZER_PATH" ]; then
        args="$args --tokenizer_path $TOKENIZER_PATH"
    fi
    echo "$args"
}

run_stage() {
    local stage=$1
    local alpha=$2
    local beta=$3
    local lr=$4
    local max_len=$5
    local warmup=$6
    local batch=$7
    local accum=$8
    local extra_args="${9:-}"

    local log_dir="${LOG_BASE}/stage${stage}"
    local resume_arg=""

    # 恢复逻辑: 手动指定 > 自动找上一阶段
    if [ -n "$RESUME" ] && [ "$stage" -eq "${RUN_STAGE:-$stage}" ]; then
        resume_arg="--resume $RESUME"
        echo "  恢复自 (手动): $RESUME"
    elif [ "$stage" -gt 1 ]; then
        local ckpt
        ckpt=$(find_latest_ckpt)
        resume_arg="--resume $ckpt"
        echo "  恢复自 (自动): $ckpt"
    fi

    local eff=$((batch * NGPU * accum))
    echo ""
    echo "============================================================"
    echo "  Stage $stage: α=$alpha β=$beta lr=$lr max_len=$max_len"
    echo "  Batch: ${batch}/gpu × ${NGPU} × accum ${accum} = ${eff}"
    echo "  TensorBoard: $log_dir"
    echo "============================================================"

    local data_args
    data_args=$(build_data_args)

    torchrun --nproc_per_node="$NGPU" train_distill_v3.py \
        --teacher_path "$TEACHER_PATH" \
        $data_args \
        --N "$N" --K "$K" --v_th_min "$V_TH_MIN" --ek_floor "$EK_FLOOR" \
        --alpha_ce "$alpha" --beta_hidden "$beta" \
        --learning_rate "$lr" --max_length "$max_len" \
        --warmup_iters "$warmup" \
        --batch_size "$batch" --accumulation_steps "$accum" \
        --grad_clip "$GRAD_CLIP" --neuron_lr_mult "$NEURON_LR" \
        --weight_decay "$WEIGHT_DECAY" \
        --ponder_weight "$PONDER_W" --ek_floor_weight "$EK_FLOOR_W" \
        --out_dir "$OUT_DIR" --log_dir "$log_dir" \
        --save_interval "$SAVE_INTERVAL" --log_interval "$LOG_INTERVAL" \
        --num_workers "$NUM_WORKERS" \
        --epochs 1 \
        $resume_arg $extra_args

    echo ""
    echo "  Stage $stage 完成! checkpoint: $(find_latest_ckpt)"
}

# ======================== 三阶段执行 ========================
#
# 通讯优化: 增大 batch_size 减少 micro-steps → 线性减少冻结层 all-gather 次数
# effective batch = batch × NGPU × accum = 64 (恒定)
#
# 52 冻结层 × micro-steps 次 all-gather/optimizer_step:
#   Stage 1: 4×4×4=64,  micro=4  → 208 all-gathers (-75% vs accum=16)
#   Stage 2: 2×4×8=64,  micro=8  → 416 all-gathers (-50%)
#   Stage 3: 1×4×16=64, micro=16 → 832 all-gathers (显存限制)
#
#                           α    β    lr    seq   warm batch accum
#   Stage 1 (对齐):        0.3  2.0  3e-4  512   200  4     4
#   Stage 2 (过渡):        0.7  0.5  2e-4  1024  100  2     8
#   Stage 3 (自主):        1.0  0.1  1e-4  2048  100  1     16   --no_curriculum
#

should_run() {
    [ -z "$RUN_STAGE" ] || [ "$RUN_STAGE" = "$1" ]
}

if should_run 1; then
    #                α    β    lr    seq  warm batch accum
    run_stage 1     0.3  2.0  3e-4  512  200  4     4
fi

if should_run 2; then
    run_stage 2     0.7  0.5  2e-4  1024 100  2     8
fi

if should_run 3; then
    run_stage 3     1.0  0.1  1e-4  2048 100  1     16  "--no_curriculum"
fi

# ======================== 完成 ========================
echo ""
echo "============================================================"
echo "蒸馏完成!"
if ls "$OUT_DIR"/distill_v3_step*.pth &>/dev/null; then
    echo "  最终 checkpoint: $(find_latest_ckpt)"
fi
echo "  查看训练曲线: tensorboard --logdir $LOG_BASE"
echo "============================================================"
