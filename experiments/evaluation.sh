#!/bin/bash 

SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")
RESULT_DIR="$SCRIPT_DIR/result"
VLLM_ROOT_DIR=$(realpath "$SCRIPT_DIR/../")
export CUDA_VISIBLE_DEVICES=0
export TEST=1
export PRINT_LATENCY=1

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
    --enable-chunked-prefill\
    --max-num-batched-tokens=1024\
    --enforce-eager \
    > $RESULT_DIR/vllm_api_server.log 2>&1 &

    retry=0
    while ! nc -z 127.0.0.1 8888; do
        echo "Waiting for vllm to start..."
        retry=$((retry + 1))
        if [ $retry -gt 50 ]; then
            echo "vllm failed to start after 50 attempts. Exiting."
            exit 1
        fi
        sleep 5
    done

    sleep 3
    echo "vllm is running on port 8888"
    echo "Start benchmarking"

    conda run -n vllm --no-capture-output \
    python benchmark.py --num-requests=100 --model=llava-hf/llava-1.5-7b-hf --port=8888 \
    --test-correctness \
    --test-performance \
    --slo-analysis \
    --request-rate 2 \
    > $RESULT_DIR/result.log

    echo "Finished evaluating vllm"

    clean_up
}

evaluate_vllm