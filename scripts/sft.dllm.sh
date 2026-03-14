#!/usr/bin/env bash
set -euo pipefail

export WANDB_PROJECT=dllm-sft

DATA_DIR="${DATA_DIR:-datasets}"
DATASET_NAME="${DATASET_NAME:-commongen}"
TOP_K="${TOP_K:-5}"
RERANKED="${RERANKED:-False}"   # True/False
DLLM="${DLLM:-llada}"           # llada/dream
NUM_PROCESSES="${NUM_PROCESSES:-2}"

LLADA_MODEL="${LLADA_MODEL:-GSAI-ML/LLaDA-8B-Instruct}"
DREAM_MODEL="${DREAM_MODEL:-Dream-org/Dream-v0-Instruct-7B}"

DLLM_LC="$(echo "${DLLM}" | tr '[:upper:]' '[:lower:]')"
RERANKED_LC="$(echo "${RERANKED}" | tr '[:upper:]' '[:lower:]')"

case "${DLLM_LC}" in
  llada) MODEL_NAME_OR_PATH="${LLADA_MODEL}" ;;
  dream) MODEL_NAME_OR_PATH="${DREAM_MODEL}" ;;
  *)
    echo "Unsupported DLLM='${DLLM}'. Use 'llada' or 'dream'." >&2
    exit 1
    ;;
esac

if [[ "${RERANKED_LC}" == "true" ]]; then
  DATA_TAG="augmented_top${TOP_K}_reranked"
else
  DATA_TAG="augmented_top${TOP_K}"
fi

TRAIN_JSONL_PATH="${DATA_DIR}/${DATASET_NAME}/train_${DATA_TAG}.jsonl"
VALID_JSONL_PATH="${DATA_DIR}/${DATASET_NAME}/valid_${DATA_TAG}.jsonl"
TEST_JSONL_PATH="${DATA_DIR}/${DATASET_NAME}/test_${DATA_TAG}.jsonl"

RUN_TAG="${DLLM_LC}_sft_${DATASET_NAME}_${DATA_TAG}"
PREPROCESSED_DIR="${DATA_DIR}/preprocessed_dir/${RUN_TAG}"
OUTPUT_DIR="/data2/fyb/dllm/${MODEL_NAME_OR_PATH##*/}/sft/${RUN_TAG}"

echo "Backend: ${DLLM_LC}"
echo "Model: ${MODEL_NAME_OR_PATH}"
echo "Train file: ${TRAIN_JSONL_PATH}"
echo "Valid file: ${VALID_JSONL_PATH}"
echo "Test file: ${TEST_JSONL_PATH}"
echo "Output dir: ${OUTPUT_DIR}"

uv run accelerate launch \
  --config_file scripts/accelerate_configs/ddp.yaml --num_processes "${NUM_PROCESSES}" \
  run_dllm_sft.py \
  --sft_backend "${DLLM_LC}" \
  --train_jsonl_path "${TRAIN_JSONL_PATH}" \
  --valid_jsonl_path "${VALID_JSONL_PATH}" \
  --preprocessed_dir "${PREPROCESSED_DIR}" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 8 \
  --load_in_4bit True \
  --lora True \
  --r 16 \
  --lora_alpha 32 \
  --gradient_checkpointing True
