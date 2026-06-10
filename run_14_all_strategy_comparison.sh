#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_SCRIPT="${SCRIPT_DIR}/compare_14_all_inference_strategies.py"
ADAPTER_DIR="${ADAPTER_DIR:-${SCRIPT_DIR}/kanana_wntl_20260407_002343}"
BASE_MODEL="/shared/home/aif/hf_models/kanana"
TEST_PATH="${TEST_PATH:-${SCRIPT_DIR}/aes_dataset_mtl/test_14_all.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/strategy_comparison_results}"

RUN_ID="${RUN_ID:-kanana_wntl_14_all_strategy_comparison_$(date +%Y%m%d_%H%M%S_KST)}"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_ID}"
LOG_FILE="${OUTPUT_DIR}/run.log"

DEVICE_ID="${DEVICE_ID:-0}"
MAX_M="${MAX_M:-50}"
CHUNK_M="${CHUNK_M:-10}"
TOP_K="${TOP_K:-9}"
TEMPERATURE="${TEMPERATURE:-0.7}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-3072}"
SEED="${SEED:-42}"
LIMIT="${LIMIT:-}"

mkdir -p "${OUTPUT_DIR}"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=========================================="
echo "14-all strategy comparison"
echo "Started at: $(date)"
echo "Output dir: ${OUTPUT_DIR}"
echo "Log file:   ${LOG_FILE}"
echo "Device:     ${DEVICE_ID}"
echo "MAX_M:      ${MAX_M}"
echo "CHUNK_M:    ${CHUNK_M}"
echo "TOP_K:      ${TOP_K}"
echo "Temp:       ${TEMPERATURE}"
echo "Seq len:    ${MAX_SEQ_LENGTH}"
echo "=========================================="

export PYTHONUNBUFFERED=1

EXTRA_ARGS=()
if [[ -n "${LIMIT}" ]]; then
    EXTRA_ARGS+=(--limit "${LIMIT}")
fi

python3 "${PYTHON_SCRIPT}" \
    --adapter_dir "${ADAPTER_DIR}" \
    --base_model_name "${BASE_MODEL}" \
    --test_path "${TEST_PATH}" \
    --output_root "${OUTPUT_ROOT}" \
    --output_dir "${OUTPUT_DIR}" \
    --tag "kanana_wntl_14_all_strategy_comparison" \
    --device_id "${DEVICE_ID}" \
    --max_m "${MAX_M}" \
    --chunk_m "${CHUNK_M}" \
    --top_k "${TOP_K}" \
    --temperature "${TEMPERATURE}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --max_seq_length "${MAX_SEQ_LENGTH}" \
    --seed "${SEED}" \
    --fallback_score 5 \
    --xtick_step 5 \
    "${EXTRA_ARGS[@]}"

echo "=========================================="
echo "Finished at: $(date)"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=========================================="
