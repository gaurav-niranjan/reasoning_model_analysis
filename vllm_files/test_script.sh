#!/bin/bash
# test_script.sh — Run from inside a compute node (srun --pty bash)
#
# Usage:
#   bash test_script.sh

cd /weka/eickhoff/esx139/thinking_models
source .venv/bin/activate

cd /weka/eickhoff/esx139/thinking_models/vllm_files

set -euo pipefail

PORT=12345
SERVER_URL="http://127.0.0.1:${PORT}"

# ========================
#  START SERVER
# ========================

echo "[$(date +%H:%M:%S)] Starting vLLM server ..."

singularity exec --nv \
    --bind /weka:/weka \
    --env HF_HOME=/weka/eickhoff/esx139/.cache/huggingface \
    --env TRANSFORMERS_CACHE=/weka/eickhoff/esx139/.cache/huggingface \
    --env HUGGINGFACE_HUB_CACHE=/weka/eickhoff/esx139/.cache/huggingface \
    /weka/eickhoff/esx139/thinking_models/vllm.sif \
    python3 -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen3-VL-8B-Thinking \
        --port "$PORT" \
        --host 127.0.0.1 \
        --reasoning-parser deepseek_r1 \
        --dtype bfloat16 \
        --trust-remote-code \
    &

VLLM_PID=$!
echo "[$(date +%H:%M:%S)] vLLM PID: $VLLM_PID"

# Kill server on exit (Ctrl+C, error, or normal finish)
cleanup() {
    echo ""
    echo "[$(date +%H:%M:%S)] Killing vLLM ..."
    kill $VLLM_PID 2>/dev/null || true
    pkill -P $VLLM_PID 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
    echo "[$(date +%H:%M:%S)] Done."
}
trap cleanup EXIT INT TERM

# ========================
#  WAIT FOR SERVER
# ========================

echo "[$(date +%H:%M:%S)] Waiting for model to load (this takes 2-5 min) ..."

WAITED=0
while [ $WAITED -lt 600 ]; do
    # Check server process is still alive
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM process died. Scroll up for the error."
        exit 1
    fi

    # Check health endpoint
    if curl -s -o /dev/null -w "%{http_code}" "$SERVER_URL/health" 2>/dev/null | grep -q "200"; then
        echo "[$(date +%H:%M:%S)] Server ready! (took ${WAITED}s)"
        break
    fi

    sleep 5
    WAITED=$((WAITED + 5))
    [ $((WAITED % 30)) -eq 0 ] && echo "[$(date +%H:%M:%S)] Still loading ... (${WAITED}s)"
done

if [ $WAITED -ge 600 ]; then
    echo "ERROR: Server didn't start within 10 minutes"
    exit 1
fi

# ========================
#  SMOKE TEST
# ========================

echo ""
echo "[$(date +%H:%M:%S)] Running smoke test ..."
python check_server.py --server "$SERVER_URL"

# ========================
#  SMALL INFERENCE RUN
# ========================

echo ""
echo "[$(date +%H:%M:%S)] Running inference on 5 samples ..."
python run_inference.py \
    --server "$SERVER_URL" \
    --inpainted-dir /weka/eickhoff/esx139/flux_inpainting/flux_klein/inpainting_results/cat_2/F \
    --id-to-index /weka/eickhoff/esx139/inpainting/experiments/id_to_index.json \
    --inpainted \
    --output results/test \
    --limit 5

# ========================
#  INSPECT
# ========================

echo ""
echo "[$(date +%H:%M:%S)] Quick look at results:"
python -c "
import pandas as pd
df = pd.read_parquet('results/test.parquet')
print(f'Samples: {len(df)}')
print(f'Succeeded: {(df[\"status\"]==\"ok\").sum()}')
print(f'Failed: {(df[\"status\"]!=\"ok\").sum()}')
print()
for _, row in df.head(2).iterrows():
    print(f'--- ID: {row[\"id\"]} ---')
    print(f'Question: {row[\"question\"][:100]}')
    print(f'Thinking ({row[\"thinking_word_count\"]} words): {row[\"thinking\"][:200]}...')
    print(f'Answer: {row[\"answer\"]}')
    print(f'Ground truth: {row[\"ground_truth\"]}')
    print()
"
