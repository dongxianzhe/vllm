#!/bin/bash 

# show command

export CUDA_VISIBLE_DEVICES=1
export TEST=0
export PRINT_LATENCY=1
export DEBUG_SCHEDULE=1

SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")
VLLM_ROOT_DIR=$(realpath "$SCRIPT_DIR/../")
MODEL_PATH="/mnt/cfs/9n-das-admin/llm_models/llava-v1.6-vicuna-7b-hf"
REQUEST_RATES="1 2 3 4 5 6 7 8 9 10"
NUM_REQUESTS=200
RESULT_DIR=$(echo "$SCRIPT_DIR/$(date +%Y%m%d_%H%M%S)_${MODEL_PATH##*/}_REQUEST_RATES_${REQUEST_RATES}_NUM_REQUESTS_${NUM_REQUESTS}" | tr ' ' '_')
SHARED_PARAMS="\
    --host=127.0.0.1 \
    --port=8888 \
    --enable-chunked-prefill \
    --no-enable-prefix-caching \
    --max-num-batched-tokens=2048 \
    --enforce-eager \
    --chat-template=$VLLM_ROOT_DIR/examples/template_llava.jinja"

scenarios=(
    "--textcaps=1 --pope=0 --mme=0 --text_vqa=0 --vizwiz_vqa=0"
    "--textcaps=0 --pope=1 --mme=0 --text_vqa=0 --vizwiz_vqa=0"
    "--textcaps=0 --pope=0 --mme=1 --text_vqa=0 --vizwiz_vqa=0"
    "--textcaps=0 --pope=0 --mme=0 --text_vqa=1 --vizwiz_vqa=0"
    "--textcaps=0 --pope=0 --mme=0 --text_vqa=0 --vizwiz_vqa=1"
)

methods=(
    "vllm"
    "ours"
)

clean_up() {
    echo "Cleaning up..."
    pgrep -f "vllm serve" >/dev/null && pgrep -f "vllm serve" | xargs kill
}

trap clean_up EXIT

mkdir -p $RESULT_DIR

for method in "${methods[@]}"; do
    echo "Evaluating ${method}"

    if [ "$method" == "ours" ]; then
        export STAGE_LEVEL_SCHEDULE=1
    else
        export STAGE_LEVEL_SCHEDULE=0
    fi

    conda run -n vllm --no-capture-output \
        vllm serve $MODEL_PATH $SHARED_PARAMS > $RESULT_DIR/${method}_api_server.log 2>&1 &

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
            --method-name=${method} \
            --log-request-data-path="$RESULT_DIR/${method}_${scenario// /_}_method_results.pkl" \
            $scenario \
            --request-rate ${REQUEST_RATES}
    done

    echo "Finished evaluating ${method}"
    clean_up
    sleep 20
done

for scenario in "${scenarios[@]}"; do
    echo "slo attainment analysis scenario: $scenario"
    method_results_paths=""
    for method in "${methods[@]}"; do
        method_results_paths="$method_results_paths $RESULT_DIR/${method}_${scenario// /_}_method_results.pkl"
    done
    echo "method_results_paths $method_results_paths"

    conda run -n vllm --no-capture-output \
        python find_best_slo.py --method-results-paths $method_results_paths \
        > $RESULT_DIR/${scenario// /_}_slo_attainment_analysis.log
done

for method in "${methods[@]}"; do
    conda run -n vllm --no-capture-output \
        python get_latency.py --log-path $RESULT_DIR/${method}_api_server.log \
        > $RESULT_DIR/${method}_latency_analysis.log
done