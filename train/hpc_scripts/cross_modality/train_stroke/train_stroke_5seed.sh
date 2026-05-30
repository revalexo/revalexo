#!/bin/bash
# =============================================================================
# 5-seed training: Healthy -> Stroke cross-modality benchmark.
#
# Trains the IMU-only student on the 7 healthy older adults (video available
# only during training) and tests on the 10 stroke subjects baked into the
# stroke config's test_subjects. One run per seed; checkpoints land under
# OUTPUT_BASE/<config>/seed_<seed>/ for the eval scripts to consume.
#
# Note: the stroke and sarcopenic configs share identical train_subjects and
# val_split_ratio, so these checkpoints are the same weights produced by
# train_sarcopenic; only the test_subjects (and thus the test metrics) differ.
#
# Usage:
#   bash train_stroke_5seed.sh                 # all configs, 5 seeds each
#   bash train_stroke_5seed.sh <config_name>   # a single config, 5 seeds
# Override GPU:
#   CUDA_VISIBLE_DEVICES=0 bash train_stroke_5seed.sh
# Override repo location (auto-detected from this script otherwise):
#   REPO_DIR=/path/to/revalexo/train bash train_stroke_5seed.sh
# =============================================================================

set -e

# Repo root is three levels up: hpc_scripts/cross_modality/train_stroke/<script>.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

CONFIG_DIR="configs/train/revalexo_healthy_stroke_cross_modality"
OUTPUT_BASE="outputs/train/revalexo_healthy_stroke_cross_modality"
LOG_DIR="${REPO_DIR}/logs/train_stroke_5seed"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "${LOG_DIR}"
cd "${REPO_DIR}"

# All seven cross-modality configs (six benchmark methods plus the
# contrastive-pretrained + FitNets combination). Pass a single config name as
# the first argument to run just that one.
ALL_CONFIGS=(
    "deepconvlstm_acc_gyro"                                  # baseline (IMU-only, no transfer)
    "deepconvlstm_acc_gyro_contrastive_pretrained"           # contrastive pretraining (CP)
    "deepconvlstm_acc_gyro_kd_resnet50_dcl"                  # vanilla KD
    "deepconvlstm_acc_gyro_kd_fitnets"                       # FitNets KD
    "deepconvlstm_acc_gyro_kd_crd_membank"                   # CRD (memory bank) KD
    "deepconvlstm_acc_gyro_kd_nkd"                           # NKD
    "deepconvlstm_acc_gyro_contrastive_pretrained_kd_fitnets" # CP init + FitNets KD
)
if [ -n "${1:-}" ]; then
    CONFIGS=("$1")
else
    CONFIGS=("${ALL_CONFIGS[@]}")
fi

SEEDS=(0 1 2 3 42)

for config in "${CONFIGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        log_file="${LOG_DIR}/${config}__seed${seed}.log"
        run_log_dir="${OUTPUT_BASE}/${config}/seed_${seed}"

        echo "============================================"
        echo "[TRAIN-STROKE] config=${config} seed=${seed}"
        echo "               gpu=${CUDA_VISIBLE_DEVICES}  out=${run_log_dir}"
        echo "               start=$(date)"
        echo "============================================"

        python train.py \
            --config "${CONFIG_DIR}/${config}.yaml" \
            --seed "${seed}" \
            --log-dir "${run_log_dir}" \
            2>&1 | tee "${log_file}"

        exit_code=${PIPESTATUS[0]}
        if [ ${exit_code} -ne 0 ]; then
            echo "[TRAIN-STROKE] FAILED: ${config} seed=${seed} (exit ${exit_code})" \
                | tee -a "${LOG_DIR}/summary.log"
        else
            echo "[TRAIN-STROKE] DONE: ${config} seed=${seed}" \
                | tee -a "${LOG_DIR}/summary.log"
        fi
        echo "               finished=$(date)" | tee -a "${LOG_DIR}/summary.log"
        echo ""
    done
done

echo "[TRAIN-STROKE] All runs complete. See ${LOG_DIR}/summary.log"
