#!/bin/bash 

SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")
RESULT_DIR="$SCRIPT_DIR/result"
VLLM_ROOT_DIR=$(realpath "$SCRIPT_DIR/../")
MODEL="llava-hf/llava-1.5-7b-hf"
MODEL_PATH="/mnt/cfs/9n-das-admin/llm_models/llava-1.5-7b-hf"
REQUEST_RATES="4 5 6 7 8"
export CUDA_VISIBLE_DEVICES=1
export TEST=1
export PRINT_LATENCY=1
export DEBUG_SCHEDULE=1
export TPOT_SLO=0.16

clean_up() {
    echo "Cleaning up..."
    pgrep -f "vllm serve" >/dev/null && pgrep -f "vllm serve" | xargs kill
}

trap clean_up EXIT

mkdir -p $RESULT_DIR

evaluate_vllm() {
    export STAGE_LEVEL_SCHEDULE=0
    echo "Evaluating baseline"
    conda run -n vllm --no-capture-output \
    vllm serve $MODEL_PATH \
    --host=127.0.0.1 \
    --port=8888 \
    --chat-template=$VLLM_ROOT_DIR/examples/template_llava.jinja \
    --enable-chunked-prefill \
    --no-enable-prefix-caching \
    --max-num-batched-tokens=2048 \
    --enforce-eager \
    > $RESULT_DIR/baseline_api_server.log 2>&1 &

    retry=0
    while ! nc -z 127.0.0.1 8888; do
        echo "Waiting for apiserver to start..."
        retry=$((retry + 1))
        if [ $retry -gt 50 ]; then
            echo "apiserver failed to start after 50 attempts. Exiting."
            exit 1
        fi
        sleep 5
    done

    sleep 3
    echo "apiserver is running on port 8888"
    echo "Start benchmarking"

    conda run -n vllm --no-capture-output \
    python benchmark.py --num-requests=100 --model=$MODEL_PATH --port=8888 \
    --test-correctness \
    --test-performance \
    --slo-analysis \
    --request-rate ${REQUEST_RATES} \
    > $RESULT_DIR/baseline_result.log


    conda run -n vllm --no-capture-output \
    python get_latency.py --input-name baseline --output-name baseline

    echo "Finished evaluating baseline"

    clean_up
}

evaluate_stage_level_schedule() {
    export STAGE_LEVEL_SCHEDULE=1
    echo "Evaluating stage level schedule"
    conda run -n vllm --no-capture-output \
    vllm serve $MODEL_PATH \
    --host=127.0.0.1 \
    --port=8888 \
    --chat-template=$VLLM_ROOT_DIR/examples/template_llava.jinja \
    --enable-chunked-prefill \
    --no-enable-prefix-caching \
    --max-num-batched-tokens=2048 \
    --enforce-eager \
    > $RESULT_DIR/stage_level_schedule_api_server.log 2>&1 &

    retry=0
    while ! nc -z 127.0.0.1 8888; do
        echo "Waiting for apiserver to start..."
        retry=$((retry + 1))
        if [ $retry -gt 50 ]; then
            echo "apiserver failed to start after 50 attempts. Exiting."
            exit 1
        fi
        sleep 5
    done

    sleep 3
    echo "apiserver is running on port 8888"
    echo "Start benchmarking"

    conda run -n vllm --no-capture-output \
    python benchmark.py --num-requests=100 --model=$MODEL_PATH --port=8888 \
    --test-correctness \
    --test-performance \
    --slo-analysis \
    --request-rate ${REQUEST_RATES} \
    > $RESULT_DIR/stage_level_schedule_result.log


    conda run -n vllm --no-capture-output \
    python get_latency.py --input-name stage_level_schedule --output-name stage_level_schedule

    echo "Finished evaluating stage level schedule"

    clean_up
}

evaluate_vllm
sleep 5
evaluate_stage_level_schedule