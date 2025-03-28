#!/bin/bash 

SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")
RESULT_DIR="$SCRIPT_DIR/result"
VLLM_ROOT_DIR=$(realpath "$SCRIPT_DIR/../../")
export CUDA_VISIBLE_DEVICES=0

clean_up() {
    echo "Cleaning up..."
    pgrep -f "vllm serve" >/dev/null && pgrep -f "vllm serve" | xargs kill
}

trap clean_up EXIT

mkdir -p $RESULT_DIR

evaluate_vllm() {
    echo "Evaluating vllm"
    conda run -n vllm --no-capture-output \
    vllm serve llava-hf/llava-1.5-7b-hf \
    --port=8888 \
    --chat-template=$VLLM_ROOT_DIR/examples/template_llava.jinja \
    --enforce-eager \
    > $RESULT_DIR/vllm_api_server.log 2>&1 &
}

evaluate_vllm
sleep 20