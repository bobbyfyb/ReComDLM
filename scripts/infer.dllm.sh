#!/usr/bin/env bash
set -euo pipefail

DLLM="${DLLM:-llada}" # llada|dream
DATASET_NAME="${DATASET_NAME:-commongen}"
TOP_K="${TOP_K:-10}"
RERANKED="${RERANKED:-False}" # True|False
BATCH_SIZE="${BATCH_SIZE:-32}"

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

RUN_TAG="${DLLM_LC}_sft_${DATASET_NAME}_${DATA_TAG}"
ADAPTER_PATH="${ADAPTER_PATH:-/data2/fyb/dllm/${MODEL_NAME_OR_PATH##*/}/sft/${RUN_TAG}/checkpoint-final}"
SPLIT_JSONL_PATH="${SPLIT_JSONL_PATH:-datasets/${DATASET_NAME}/test_${DATA_TAG}.jsonl}"
OUTPUT_JSONL_PATH="${OUTPUT_JSONL_PATH:-outputs/${DLLM_LC}/${DATASET_NAME}/${RUN_TAG}_test.jsonl}"

echo "Backend: ${DLLM_LC}"
echo "Model: ${MODEL_NAME_OR_PATH}"
echo "Adapter: ${ADAPTER_PATH}"
echo "Input: ${SPLIT_JSONL_PATH}"
echo "Output: ${OUTPUT_JSONL_PATH}"

uv run python run_dllm_inference.py \
  --infer_backend "${DLLM_LC}" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --adapter_path "${ADAPTER_PATH}" \
  --split_jsonl_path "${SPLIT_JSONL_PATH}" \
  --batch_size "${BATCH_SIZE}" \
  --output_jsonl_path "${OUTPUT_JSONL_PATH}"
