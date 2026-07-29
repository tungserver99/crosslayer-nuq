#!/usr/bin/env bash
set -euo pipefail
set -x

# One-shot runner for LNQ + EndLoss sequential cross-layer GuidedQuant on
# Llama-3.2-1B with C4 calibration, followed by standalone perplexity evaluation.
# Defaults: 3-bit, C4 calibration with 128 samples / 2048 tokens.
#
# Override with environment variables if needed:
#   MODEL_NAME, BITS, NUM_GROUPS, MODE, CACHE_DIR, EVAL_CACHE_DIR,
#   EVAL_METHOD, EVAL_STRIDE, EVAL_DTYPE, NUM_ITERATIONS, CD_CYCLES,
#   RANDOM_STATE, OVERWRITE, SIGNED_GRAD_FLUSH_EVERY

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.2-1B}"
BITS="${BITS:-3}"
NUM_GROUPS="${NUM_GROUPS:-4}"
MODE="${MODE:-pack}"
DATASET="${DATASET:-c4}"
SEQ_LEN="${SEQ_LEN:-2048}"
NUM_EXAMPLES="${NUM_EXAMPLES:-128}"
NUM_ITERATIONS="${NUM_ITERATIONS:-3}"
CD_CYCLES="${CD_CYCLES:-4}"
RANDOM_STATE="${RANDOM_STATE:-42}"
OVERWRITE="${OVERWRITE:-0}"
SIGNED_GRAD_FLUSH_EVERY="${SIGNED_GRAD_FLUSH_EVERY:-8}"
CACHE_DIR="${CACHE_DIR:-cache}"
EVAL_CACHE_DIR="${EVAL_CACHE_DIR:-dataset_cache}"
EVAL_METHOD="${EVAL_METHOD:-block}"
EVAL_STRIDE="${EVAL_STRIDE:-512}"
EVAL_DTYPE="${EVAL_DTYPE:-fp16}"

# Colab often renders carriage-return progress updates as new lines.
# Keep tqdm enabled, but refresh less often so logs stay readable.
export TQDM_MININTERVAL="${TQDM_MININTERVAL:-10}"
export TQDM_MAXINTERVAL="${TQDM_MAXINTERVAL:-30}"
export TQDM_POSITION="${TQDM_POSITION:--1}"
export SIGNED_GRAD_FLUSH_EVERY

MODEL_BASENAME="${MODEL_NAME##*/}"
SIGNED_GRADIENTS_DIR="${CACHE_DIR}/signed_gradients/${MODEL_BASENAME}-${DATASET}_s${NUM_EXAMPLES}_blk${SEQ_LEN}_g${NUM_GROUPS}"
PACKED_MODEL_DIR="${CACHE_DIR}/layerwise_packed/endlossxl-layerwise-${MODEL_BASENAME}-w${BITS}-${DATASET}_s${NUM_EXAMPLES}_blk${SEQ_LEN}_g${NUM_GROUPS}_iter${NUM_ITERATIONS}_cd${CD_CYCLES}"
PPL_JSON="${PACKED_MODEL_DIR}/ppl_${EVAL_METHOD}.json"
PPL_TAG="llama32_1b_endlossxl_guidedquant_${BITS}bit_${DATASET}_${NUM_EXAMPLES}_${SEQ_LEN}_g${NUM_GROUPS}_iter${NUM_ITERATIONS}_cd${CD_CYCLES}_${EVAL_METHOD}"

quantize_overwrite_args=()
layerwise_overwrite_args=()
if [[ "${OVERWRITE}" == "1" || "${OVERWRITE}" == "true" ]]; then
  quantize_overwrite_args=(--overwrite_tokens --overwrite_gradients --overwrite_quantize --overwrite_pack)
  layerwise_overwrite_args=(--overwrite_hessians --overwrite_quantize --overwrite_pack)
fi

python quantize.py "${MODEL_NAME}" \
  --seed_precision "${BITS}" \
  --parent_precision "${BITS}" \
  --dataset "${DATASET}" \
  --seq_len "${SEQ_LEN}" \
  --num_examples "${NUM_EXAMPLES}" \
  --num_groups "${NUM_GROUPS}" \
  --mode "${MODE}" \
  --cache_dir "${CACHE_DIR}" \
  --random_state "${RANDOM_STATE}" \
  --signed_gradients_path "${SIGNED_GRADIENTS_DIR}" \
  "${quantize_overwrite_args[@]}"

python endloss_crosslayer_nuq.py "${MODEL_NAME}" \
  --seed_precision "${BITS}" \
  --dataset "${DATASET}" \
  --seq_len "${SEQ_LEN}" \
  --num_examples "${NUM_EXAMPLES}" \
  --num_groups "${NUM_GROUPS}" \
  --num_iterations "${NUM_ITERATIONS}" \
  --cd_cycles "${CD_CYCLES}" \
  --mode "${MODE}" \
  --cache_dir "${CACHE_DIR}" \
  --random_state "${RANDOM_STATE}" \
  --signed_gradients_path "${SIGNED_GRADIENTS_DIR}" \
  "${layerwise_overwrite_args[@]}"

if [[ ! -d "${PACKED_MODEL_DIR}" ]]; then
  echo "Packed EndLoss cross-layer GuidedQuant model directory not found: ${PACKED_MODEL_DIR}" >&2
  exit 1
fi

eval_args=(
  --model-path "${PACKED_MODEL_DIR}"
  --datasets wikitext2 c4
  --seqlen "${SEQ_LEN}"
  --method "${EVAL_METHOD}"
  --dtype "${EVAL_DTYPE}"
  --cache-dir "${EVAL_CACHE_DIR}"
  --out-json "${PPL_JSON}"
  --tag "${PPL_TAG}"
)

if [[ "${EVAL_METHOD}" == "sliding" ]]; then
  eval_args+=(--stride "${EVAL_STRIDE}")
fi

python eval_ppl.py "${eval_args[@]}"

echo "Done."
echo "Packed model: ${PACKED_MODEL_DIR}"
echo "PPL results: ${PPL_JSON}"

