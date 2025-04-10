#!/bin/bash 

SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")
VLLM_ROOT_DIR=$(realpath "$SCRIPT_DIR/../")
MODEL_PATH="/mnt/cfs/9n-das-admin/llm_models/llava-v1.6-vicuna-7b-hf"
REQUEST_RATES="2 3 4 5 6 7 8 9 10 11 12"
NUM_REQUESTS=200
SHARED_PARAMS="--enable-chunked-prefill --no-enable-prefix-caching --max-num-batched-tokens=1152 --tensor-parallel-size=4"
export CUDA_VISIBLE_DEVICES=1,2,3,4
export TEST=0
export PRINT_LATENCY=1
export DEBUG_SCHEDULE=1
export TTFT_SLO=1.0
export TPOT_SLO=0.08

RESULT_DIR=$(echo "$SCRIPT_DIR/$(date +%Y%m%d_%H%M%S)_TTFT_SLO${TTFT_SLO}_TPOT_SLO_${TPOT_SLO}_REQUEST_RATES_${REQUEST_RATES}_NUM_REQUESTS_${NUM_REQUESTS}_${SHARED_PARAMS}" | tr ' ' '_')

 scenarios=(
    "--textcaps=1 --pope=0 --mme=0 --text_vqa=0 --vizwiz_vqa=0"
    # "--textcaps=0 --pope=1 --mme=0 --text_vqa=0 --vizwiz_vqa=0"
    # "--textcaps=0 --pope=0 --mme=1 --text_vqa=0 --vizwiz_vqa=0"
    # "--textcaps=0 --pope=0 --mme=0 --text_vqa=1 --vizwiz_vqa=0"
    # "--textcaps=0 --pope=0 --mme=0 --text_vqa=0 --vizwiz_vqa=1"
)

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
    $SHARED_PARAMS \
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

    for scenario in "${scenarios[@]}"; do
        echo "Running scenario: $scenario"
        conda run -n vllm --no-capture-output \
        python benchmark.py --num-requests=$NUM_REQUESTS --model=$MODEL_PATH --port=8888 \
        --test-correctness \
        --test-performance \
        --slo-analysis \
        $scenario \
        --request-rate ${REQUEST_RATES} \
        > $RESULT_DIR/baseline_${scenario// /_}_result.log
    done


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
    ${SHARED_PARAMS} \
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

    for scenario in "${scenarios[@]}"; do
        echo "Running scenario: $scenario"
        conda run -n vllm --no-capture-output \
        python benchmark.py --num-requests=$NUM_REQUESTS --model=$MODEL_PATH --port=8888 \
        --test-correctness \
        --test-performance \
        --slo-analysis \
        $scenario \
        --request-rate ${REQUEST_RATES} \
        > $RESULT_DIR/stage_level_schedule_${scenario// /_}_result.log
    done

    echo "Finished evaluating stage level schedule"

    clean_up
}

evaluate_vllm
sleep 20
evaluate_stage_level_schedule

conda run -n vllm --no-capture-output \
    python get_latency.py --dir=${RESULT_DIR} --input-name baseline --output-name baseline

conda run -n vllm --no-capture-output \
    python get_latency.py --dir=${RESULT_DIR} --input-name stage_level_schedule --output-name stage_level_schedule