#!/bin/bash
# Start vLLM server for Qwen3-4B-Instruct
# Run this in WSL2 or Linux (vLLM does not support native Windows)
#
# Prerequisites:
#   pip install vllm
#   GPU with ~8-10 GB VRAM (FP16)

MODEL_NAME="Qwen/Qwen3-4B-Instruct"
PORT=8000
MAX_MODEL_LEN=32768

echo "Starting vLLM server with ${MODEL_NAME} on port ${PORT}..."
echo "Maximum model length: ${MAX_MODEL_LEN}"
echo ""
echo "Press Ctrl+C to stop the server."
echo ""

python -m vllm.entrypoints.openai.api_server \
    --model /data/zyx/model/Qwen3-4B-Instruct \
    --served-model-name qwen3-4b \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --trust-remote-code \
    --dtype auto \
    --host 0.0.0.0 \

