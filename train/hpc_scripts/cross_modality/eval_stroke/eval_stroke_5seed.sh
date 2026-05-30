#!/bin/bash
# =============================================================================
# 5-seed evaluation: Healthy -> Stroke cross-modality benchmark.
#
# Evaluates each seed's best checkpoint from train_stroke against the stroke
# test set (test_subjects baked into the stroke config). Run AFTER
# train_stroke_5seed.sh has produced checkpoints.
#
# Note: because the stroke and sarcopenic configs share identical
# train_subjects, the train_sarcopenic checkpoints are equivalent weights; to
# reuse them instead, set CKPT_BASE to the sarcopenic outputs directory.
#
# Usage:
#   bash eval_stroke_5seed.sh                 # all configs, 5 seeds each
#   bash eval_stroke_5seed.sh <config_name>   # a single config, 5 seeds
# Choose which checkpoint to evaluate (default: best):
#   CKPT_TYPE=best  bash eval_stroke_5seed.sh  # best_model.pt (best val metric)
#   CKPT_TYPE=final bash eval_stroke_5seed.sh  # last epoch (highest checkpoint_epoch_*.pt)
# Override GPU:
#   CUDA_VISIBLE_DEVICES=0 bash eval_stroke_5seed.sh
# Override repo location (auto-detected from this script otherwise):
#   REPO_DIR=/path/to/revalexo/train bash eval_stroke_5seed.sh
# =============================================================================

set -e

# Repo root is three levels up: hpc_scripts/cross_modality/eval_stroke/<script>.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

CONFIG_DIR="configs/train/revalexo_healthy_stroke_cross_modality"
CKPT_BASE="outputs/train/revalexo_healthy_stroke_cross_modality"
OUTPUT_BASE="outputs/train/revalexo_healthy_stroke_cross_modality"
LOG_DIR="${REPO_DIR}/logs/eval_stroke_5seed"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "${LOG_DIR}"
cd "${REPO_DIR}"

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

# Which checkpoint to evaluate: "best" (best_model.pt) or "final" (last epoch).
CKPT_TYPE="${CKPT_TYPE:-best}"
if [ "${CKPT_TYPE}" != "best" ] && [ "${CKPT_TYPE}" != "final" ]; then
    echo "ERROR: CKPT_TYPE must be 'best' or 'final' (got '${CKPT_TYPE}')" >&2
    exit 1
fi

for config in "${CONFIGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        log_file="${LOG_DIR}/${config}__seed${seed}__${CKPT_TYPE}.log"
        output_dir="${OUTPUT_BASE}/${config}/seed_${seed}/eval_stroke_${CKPT_TYPE}"
        # Most recent training run directory for this config/seed.
        run_dir=$(ls -dt "${CKPT_BASE}/${config}/seed_${seed}"/*/ 2>/dev/null | head -1)
        run_dir="${run_dir%/}"
        if [ "${CKPT_TYPE}" = "best" ]; then
            ckpt="${run_dir}/best_model.pt"
            [ -f "${ckpt}" ] || ckpt=""
        else
            # Final epoch = highest-numbered checkpoint_epoch_*.pt in the run dir.
            ckpt=$(ls "${run_dir}"/checkpoint_epoch_*.pt 2>/dev/null \
                   | sed -E 's#.*/checkpoint_epoch_([0-9]+)\.pt$#\1 &#' \
                   | sort -n | tail -1 | cut -d' ' -f2-)
        fi

        if [ -z "${ckpt}" ] || [ ! -f "${ckpt}" ]; then
            echo "[EVAL-STROKE] SKIP: ${config} seed=${seed} (no ${CKPT_TYPE} checkpoint under ${CKPT_BASE}/${config}/seed_${seed}/)" \
                | tee -a "${LOG_DIR}/summary.log"
            continue
        fi

        echo "============================================"
        echo "[EVAL-STROKE] config=${config} seed=${seed} ckpt_type=${CKPT_TYPE}"
        echo "              ckpt=${ckpt}"
        echo "              out=${output_dir}"
        echo "============================================"

        set +e
        python evaluate.py \
            --config "${CONFIG_DIR}/${config}.yaml" \
            --checkpoint "${ckpt}" \
            --seed "${seed}" \
            --output-dir "${output_dir}" \
            2>&1 | tee "${log_file}"
        rc=${PIPESTATUS[0]}
        set -e

        if [ "${rc}" -ne 0 ]; then
            echo "[EVAL-STROKE] FAILED rc=${rc}: ${config} seed=${seed}" \
                | tee -a "${LOG_DIR}/summary.log"
        else
            echo "[EVAL-STROKE] DONE: ${config} seed=${seed}" \
                | tee -a "${LOG_DIR}/summary.log"
        fi
    done
done

echo ""
echo "[EVAL-STROKE] All evals complete. See ${LOG_DIR}/summary.log"
