#!/bin/bash
#SBATCH --job-name=qwen3vl-sbbench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=h100-ferranti
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=00:25:00
#SBATCH --output=logs/infer_%j.log
#SBATCH --error=logs/infer_%j.err

# ============================================================
# vLLM server runs inside Singularity container (GPU + CUDA)
# Client runs with your normal Python (just needs aiohttp/pandas)
#
# Resume after crash: just re-submit this script.
#
# Usage:
#   mkdir -p logs results
#   sbatch run_job.slurm
# ============================================================

cd /weka/eickhoff/esx139/thinking_models
source .venv/bin/activate

cd /weka/eickhoff/esx139/thinking_models/vllm_files

set -euo pipefail

# ========================
#  PATHS — CHANGE THESE
# ========================

SIF="/weka/eickhoff/esx139/thinking_models/vllm.sif"
INPAINTED_DIR="/weka/eickhoff/esx139/flux_inpainting/flux_klein/inpainting_results/Race/Workplace" 
ID_TO_INDEX="/weka/eickhoff/esx139/inpainting/experiments/id_to_index.json"
CLASS_CAT="5"
ORIGINAL_IMAGES_DIR=""
HF_CACHE="/weka/eickhoff/esx139/.cache/huggingface"

# ========================
#  EXPERIMENT CONFIG
# ========================

MODEL="Qwen/Qwen3-VL-8B-Thinking"
MAX_MODEL_LEN=16384
GPU_MEM_UTIL=0.92

USE_INPAINTED="true"                       # "true" or "false"
OUTPUT_PREFIX="results/cat_5/qwen8B/workplace/sbbench_inpainted"  # change per experiment

CONCURRENCY=32
MAX_TOKENS=8192
TIMEOUT=300
RETRIES=2
LIMIT=""

# ========================
#  ENVIRONMENT
# ========================

cd /weka/eickhoff/esx139/thinking_models
source .venv/bin/activate
cd /weka/eickhoff/esx139/thinking_models/vllm_files

mkdir -p "$HF_CACHE" logs results

PORT=$(python3 -c "import random; print(random.randint(10000, 60000))")
SERVER_URL="http://127.0.0.1:${PORT}"

echo "============================================"
echo "Job ID:       ${SLURM_JOB_ID:-interactive}"
echo "Node:         $(hostname)"
echo "GPUs:         ${CUDA_VISIBLE_DEVICES:-not set}"
echo "Model:        $MODEL"
echo "Port:         $PORT"
echo "SIF:          $SIF"
echo "Inpainted:    $USE_INPAINTED"
echo "Output:       $OUTPUT_PREFIX"
echo "============================================"

# ========================
#  START VLLM IN CONTAINER
# ========================

echo "[$(date +%H:%M:%S)] Starting vLLM server ..."

singularity exec --nv \
    --bind /weka:/weka \
    --env HF_HOME="${HF_CACHE}" \
    --env TRANSFORMERS_CACHE="${HF_CACHE}" \
    --env HUGGINGFACE_HUB_CACHE="${HF_CACHE}" \
    "$SIF" \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --port "$PORT" \
        --host 127.0.0.1 \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --reasoning-parser deepseek_r1 \
        --dtype bfloat16 \
        --trust-remote-code \
        --no-enable-log-requests \
    &

VLLM_PID=$!
echo "[$(date +%H:%M:%S)] vLLM PID: $VLLM_PID"

cleanup() {
    echo "[$(date +%H:%M:%S)] Killing vLLM (PID $VLLM_PID) ..."
    kill $VLLM_PID 2>/dev/null || true
    pkill -P $VLLM_PID 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
    echo "[$(date +%H:%M:%S)] Server stopped."
}
trap cleanup EXIT INT TERM

# ========================
#  WAIT FOR SERVER
# ========================

echo "[$(date +%H:%M:%S)] Waiting for model to load ..."
MAX_WAIT=600
WAITED=0

while [ $WAITED -lt $MAX_WAIT ]; do
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM process died. Scroll up for the error."
        exit 1
    fi
    if curl -s -o /dev/null -w "%{http_code}" "$SERVER_URL/health" 2>/dev/null | grep -q "200"; then
        echo "[$(date +%H:%M:%S)] Server ready! (waited ${WAITED}s)"
        break
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    [ $((WAITED % 30)) -eq 0 ] && echo "[$(date +%H:%M:%S)] Still loading ... (${WAITED}s)"
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "ERROR: Server didn't start within ${MAX_WAIT}s"
    exit 1
fi

# ========================
#  RUN CLIENT (host Python, not container)
# ========================

echo "[$(date +%H:%M:%S)] Starting inference ..."

CLIENT_ARGS=(
    python run_inference.py
    --server "$SERVER_URL"
    --model-name "$MODEL"
    --id-to-index "$ID_TO_INDEX"
    --output "$OUTPUT_PREFIX"
    --concurrency "$CONCURRENCY"
    --max-tokens "$MAX_TOKENS"
    --timeout "$TIMEOUT"
    --retries "$RETRIES"
)

if [ "$USE_INPAINTED" = "true" ]; then
    if [ -z "${INPAINTED_DIR:-}" ]; then
        echo "ERROR: USE_INPAINTED=true but INPAINTED_DIR is empty"
        exit 1
    fi
    CLIENT_ARGS+=(--inpainted)
    CLIENT_ARGS+=(--inpainted-dir "$INPAINTED_DIR")
else
    if [ -z "${CLASS_CAT:-}" ]; then
        echo "ERROR: USE_INPAINTED=false but CLASS_CAT is not set"
        exit 1
    fi
    CLIENT_ARGS+=(--no-inpainted)
    CLIENT_ARGS+=(--class-cat "$CLASS_CAT")
fi

if [ -n "${ORIGINAL_IMAGES_DIR:-}" ]; then
    CLIENT_ARGS+=(--original-images-dir "$ORIGINAL_IMAGES_DIR")
fi

if [ -n "${LIMIT:-}" ]; then
    CLIENT_ARGS+=(--limit "$LIMIT")
fi

echo "Command: ${CLIENT_ARGS[*]}"
echo ""
"${CLIENT_ARGS[@]}"
CLIENT_EXIT=$?

echo ""
echo "[$(date +%H:%M:%S)] Finished (exit code $CLIENT_EXIT)"
echo "  Parquet:    ${OUTPUT_PREFIX}.parquet"
echo "  Checkpoint: ${OUTPUT_PREFIX}.checkpoint.jsonl"

exit $CLIENT_EXIT
