#!/usr/bin/env bash
set -euo pipefail

DLLM="${DLLM:-llada}" # llada|dream
DATASET_NAME="${DATASET_NAME:-commongen}"
TOP_K="${TOP_K:-5}"
RERANKED="${RERANKED:-False}" # True|False
PPL_MODEL="${PPL_MODEL:-openai-community/gpt2-large}"

DLLM_LC="$(echo "${DLLM}" | tr '[:upper:]' '[:lower:]')"
RERANKED_LC="$(echo "${RERANKED}" | tr '[:upper:]' '[:lower:]')"

case "${DLLM_LC}" in
  llada|dream) ;;
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
INPUT_JSONL="${INPUT_JSONL:-outputs/${DLLM_LC}/${DATASET_NAME}/${RUN_TAG}_test.jsonl}"
OUTPUT_METRICS="${OUTPUT_METRICS:-outputs/${DLLM_LC}/${DATASET_NAME}/${RUN_TAG}_test.metrics.json}"

echo "Backend: ${DLLM_LC}"
echo "Input: ${INPUT_JSONL}"
echo "Output: ${OUTPUT_METRICS}"

uv run python run_evaluate.py \
  --input "${INPUT_JSONL}" \
  --output "${OUTPUT_METRICS}" \
  --lowercase \
  --distinct-n 4 \
  --ppl-model "${PPL_MODEL}"
