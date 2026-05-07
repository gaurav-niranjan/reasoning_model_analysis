#!/bin/bash
#SBATCH --job-name=judge-kitchen-bg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=h100-ferranti
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=00:14:00
#SBATCH --output=logs/judge/judge_%j.log
#SBATCH --error=logs/judge/judge_%j.err

# ============================================================
# LLM-as-judge: classify thinking traces for background signals
#
# Same architecture as run_job.sh:
#   - vLLM server in Singularity container (GPU)
#   - Client runs with host Python (just needs aiohttp/pandas)
#
# Resume after crash: just re-submit.
#
# Usage:
#   mkdir -p logs results
#   sbatch run_judge_job.sh
# ============================================================

cd /weka/eickhoff/esx139/thinking_models
source .venv/bin/activate
cd /weka/eickhoff/esx139/thinking_models/vllm_files

set -euo pipefail

# ========================
#  PATHS — CHANGE THESE
# ========================

SIF="/weka/eickhoff/esx139/thinking_models/vllm.sif"
HF_CACHE="/weka/eickhoff/esx139/.cache/huggingface"

# Input: the parquet output from run_inference.py
INPUT_PARQUET="results/cat_0/lab/qwen8B/sbbench_inpainted_8B.parquet"

# Output: where judge results go
OUTPUT_PREFIX="results/judge/cat_0/lab/qwen8B/judge_lab_bg"

# ========================
#  JUDGE CONFIG
# ========================

MODEL="google/gemma-3-27b-it"
MAX_MODEL_LEN=16384         # traces can be long, but judge response is short
GPU_MEM_UTIL=0.92

# What signal to look for (natural language description)
#SIGNAL="the image background being a kitchen or kitchen-like setting"
#SIGNAL="the image background being a cluttered, messy, or disorganized room"
SIGNAL="the image background being a laboratory or science lab setting"

CONCURRENCY=64              # text-only = lighter, can go higher than vision
MAX_TOKENS=256              # judge only needs ~50 tokens, buffer for safety
TIMEOUT=120
RETRIES=2
LIMIT=""                    # set to e.g. "50" for a test run

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
echo "Input:        $INPUT_PARQUET"
echo "Output:       $OUTPUT_PREFIX"
echo "Signal:       $SIGNAL"
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
        --dtype bfloat16 \
        --trust-remote-code \
        --no-enable-log-requests \
    &

# NOTE: no --reasoning-parser needed — Gemma-3 IT is not a thinking model

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
#  RUN JUDGE CLIENT
# ========================

echo "[$(date +%H:%M:%S)] Starting judge ..."

CLIENT_ARGS=(
    python run_judge.py
    --server "$SERVER_URL"
    --model-name "$MODEL"
    --input "$INPUT_PARQUET"
    --output "$OUTPUT_PREFIX"
    --signal "$SIGNAL"
    --concurrency "$CONCURRENCY"
    --max-tokens "$MAX_TOKENS"
    --timeout "$TIMEOUT"
    --retries "$RETRIES"
)

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