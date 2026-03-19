wait_api_server() {
    local ip=$1
    local port=$2
    local pid=$3
    local name="api server at $ip $port"
    local retry=0
    while ! nc -z $ip $port; do
        echo "Waiting for $name to start..."
        retry=$((retry + 1))
        if ! kill -0 $pid 2>/dev/null; then
            echo "Backend process with PID $pid has exited."
            return 1
        fi
        if [ $retry -gt 50 ]; then
            echo "$name failed to start after 50 attempts. Exiting."
            return 1
        fi
        sleep 5
    done
    echo "api server is running on $ip $port"
    return 0
}

clean_up() {
    echo "Cleaning up..."
    pgrep -f "vllm serve" >/dev/null && pgrep -f "vllm serve" | xargs kill
    pgrep -f "text-generation-launcher" >/dev/null && pgrep -f "text-generation-launcher" | xargs kill
    pgrep -f "sglang.launch_server" >/dev/null && pgrep -f "sglang.launch_server" | xargs kill
    pgrep -f "lmdeploy serve" >/dev/null && pgrep -f "lmdeploy serve" | xargs kill
}
trap clean_up EXIT
SCRIPT=$(readlink -f "$0")
SCRIPT_PATH=$(dirname "$SCRIPT")
RESULT_PATH=$(echo "$SCRIPT_PATH/result/${method}/$(date +%Y%m%d_%H%M%S)")
mkdir -p $RESULT_PATH
api_server_log_path="$RESULT_PATH/api_server.log"
MODEL_PATH="/mnt/cephfs/user_xianzhedong/models/Qwen3-Omni-30B-A3B-Captioner"
DATASET_PATH="/mnt/cephfs/user_xianzhedong/datasets/caption2.mp3"
host=127.0.0.1
port=8901
start_server=1
send_request=1

if [[ "$start_server" == "1" ]]; then
    vllm serve $MODEL_PATH \
        --port $port \
        --host $host \
        --dtype bfloat16 \
        --max-model-len 32768 \
        --allowed-local-media-path / \
        -tp 4 > $api_server_log_path 2>&1 &
    ln -sf $api_server_log_path latest_api_server.log

    apiserver_pid=$!

    if wait_api_server $host $port $apiserver_pid; then
        echo "server start success."
    else
        echo "server start failed. Exit 1."
        clean_up
        exit 1
    fi
fi

DATABASE64=$(base64 -w 0 "$DATASET_PATH")
cat > payload.json <<EOF
{
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "input_audio",
          "input_audio": {
            "data": "${DATABASE64}",
            "format": "mp3"
          }
        }
      ]
    }
  ]
}
EOF
if [[ "$send_request" == "1" ]]; then
    curl http://$host:$port/v1/chat/completions   -H "Content-Type: application/json"   -d @payload.json | tee "$RESULT_PATH/response.json"
fi

sleep 3
clean_up