#!/bin/bash
set -e
# =============================================================================
# HA Global Cache Pooling test script — REDIS backend (single redis + multi-master + failover)
# Mirror of run_ha.sh, but replaces the 3-node etcd cluster with a SINGLE redis
# instance. The 3 mooncake_master use redis (lease-based leader election) instead
# of etcd raft. Motivation: redis avoids introducing etcd as an extra component.
#
# Flow (identical to run_ha.sh):
#   1. start a single redis instance
#   2. start 3 HA masters (one is elected leader via a redis lease)
#   3. start 2 FastDeploy instances sharing the global cache pool
#   4. verify pooling (before failover): warmup on server_0, reuse on server_1
#   5. kill the leader master, wait for a standby to be re-elected
#   6. verify pooling (after failover) with a BRAND-NEW prompt
# =============================================================================

export PYTHONPATH="/workspace/mooncake-test/FastDeploy:$PYTHONPATH"
export MODEL_NAME="/workspace/models/Ernie-0.3B"
export MOONCAKE_CONFIG_PATH=./ha_redis_mooncake_config.json
export FD_DEBUG=1

unset http_proxy && unset https_proxy

echo "begin"
source ./utils.sh

# ---- topology ---------------------------------------------------------------
# redis:        client port = 6399 (single instance)
# master node i: rpc port = 808${i}, metrics port = 909${i}
REDIS_PORT=6399
REDIS_SERVER_BIN="$(command -v redis-server || echo /usr/local/redis/bin/redis-server)"
REDIS_CLI_BIN="$(command -v redis-cli || echo /usr/local/redis/bin/redis-cli)"
REDIS_CONN="redis://127.0.0.1:${REDIS_PORT}"          # for mooncake_master + client config

CLUSTER_ID="mooncake_cluster"
# redis master_view key uses a hash-tag {cluster_id} so all related keys land in
# the same Redis Cluster slot; it is a HASH with fields leader_address/view_version/owner_token.
MASTER_VIEW_KEY="mooncake-store/{${CLUSTER_ID}}/master_view"

S0_PORT=52700
S1_PORT=52800

# ---- helpers ----------------------------------------------------------------

# Query redis for the current leader's "rpc_address:rpc_port".
# The master_view is a redis HASH; the leader endpoint lives in field leader_address.
# redis-cli prints raw (unquoted) output when piped, so no extra unquoting needed.
get_leader_addr() {
    "${REDIS_CLI_BIN}" -p "${REDIS_PORT}" hget "${MASTER_VIEW_KEY}" leader_address 2>/dev/null \
        | tr -d '[:space:]'
}

# Wait until a leader is elected and published into redis.
wait_for_leader() {
    local timeout=${1:-60}
    local start_time=$(date +%s)
    while true; do
        local leader=$(get_leader_addr)
        if [ -n "${leader}" ]; then
            echo "${leader}"
            return 0
        fi
        if [ $(( $(date +%s) - start_time )) -ge ${timeout} ]; then
            echo ""
            return 1
        fi
        sleep 1
    done
}

# Kill the mooncake_master process(es) that own the given rpc_port (leader).
kill_master_by_rpc_port() {
    local rpc_port=$1
    # match "--rpc_port 8081" or "--rpc_port=8081" on the full command line
    local pids=$(pgrep -af mooncake_master | grep -E "rpc_port[ =]${rpc_port}([^0-9]|$)" | awk '{print $1}')
    if [ -z "${pids}" ]; then
        echo "⚠️  no mooncake_master process found for rpc_port=${rpc_port}"
        return
    fi
    # also collect direct children by ppid, in case a child's cmdline didn't match.
    local all_pids="${pids}"
    for p in ${pids}; do
        local kids=$(pgrep -P "${p}" 2>/dev/null)
        [ -n "${kids}" ] && all_pids="${all_pids} ${kids}"
    done
    echo "kill leader master pids=$(echo ${all_pids} | tr '\n' ' ')(rpc_port=${rpc_port})"
    kill -9 ${all_pids} 2>/dev/null || true

}

# Send a chat request to a FastDeploy server.
send_request() {
    local port=$1
    local content=$2
    curl -s -X POST "http://0.0.0.0:${port}/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d "{
        \"messages\": [
          {\"role\": \"user\", \"content\": \"${content}\"}
        ],
        \"max_tokens\": 50,
        \"stream\": false,
        \"top_p\": 0
      }"
    echo
}

# ---- 1. start a single redis instance ---------------------------------------
echo "=== [1/6] start redis ==="
pkill -9 -f "redis-server .*:${REDIS_PORT}" || true
sleep 1

check_ports "${REDIS_PORT}" || {
    echo "❌ redis port ${REDIS_PORT} is in use. Please release it."
    exit 1
}

# disable persistence; this is a throwaway coordination store.
"${REDIS_SERVER_BIN}" --port "${REDIS_PORT}" --save "" --appendonly no \
    --daemonize no > log_redis 2>&1 &
sleep 2
echo "=== redis health check ==="
"${REDIS_CLI_BIN}" -p "${REDIS_PORT}" ping

# ---- 2. start 3 HA masters (redis backend) ----------------------------------
echo "=== [2/6] start 3 HA mooncake_master (redis backend) ==="
pkill -9 -f mooncake_master || true
sleep 1

master_ports=(8081 8082 8083 9091 9092 9093)
check_ports "${master_ports[@]}" || {
    echo "❌ Some master ports are in use. Please release them."
    exit 1
}

for i in 1 2 3; do
    # --ha_backend_type redis + --ha_backend_connstring redis://...
    mooncake_master \
        --enable_ha \
        --ha_backend_type redis \
        --ha_backend_connstring "${REDIS_CONN}" \
        --cluster_id "${CLUSTER_ID}" \
        --rpc_address "127.0.0.1" \
        --rpc_port 808${i} \
        --metrics_port=909${i} > log_master_${i} 2>&1 &
done

echo "waiting for leader election..."
LEADER_ADDR=$(wait_for_leader 60) || {
    echo "❌ no leader elected within timeout"
    exit 1
}
echo "✅ current leader: ${LEADER_ADDR}"

# ---- 3. start 2 FastDeploy instances ----------------------------------------
echo "=== [3/6] start FastDeploy instances ==="

# clean up any lingering FastDeploy services so the ports are free.
# the api_server runs under gunicorn; killing the gunicorn masters takes the
# workers down with them.
pkill -f "gunicorn: master" || true
sleep 2

rm -rf log_0 log_1

fd_ports=("$S0_PORT" "$S1_PORT")
check_ports "${fd_ports[@]}" || {
    echo "❌ Some ports are in use. Please release them."
    exit 1
}

# Launch FD server 0
export CUDA_VISIBLE_DEVICES=6
export FD_LOG_DIR="log_0"
mkdir -p ${FD_LOG_DIR}
echo "server 0 port: ${S0_PORT}"

nohup python -m fastdeploy.entrypoints.openai.api_server \
       --model ${MODEL_NAME} \
       --port ${S0_PORT} \
       --max-model-len 32768 \
       --max-num-seqs 32 \
       --kvcache-storage-backend mooncake \
       2>&1 >${FD_LOG_DIR}/nohup &

# Launch FD server 1
export CUDA_VISIBLE_DEVICES=7
export FD_LOG_DIR="log_1"
mkdir -p ${FD_LOG_DIR}
echo "server 1 port: ${S1_PORT}"

nohup python -m fastdeploy.entrypoints.openai.api_server \
       --model ${MODEL_NAME} \
       --port ${S1_PORT} \
       --max-model-len 32768 \
       --max-num-seqs 32 \
       --kvcache-storage-backend mooncake \
       2>&1 >${FD_LOG_DIR}/nohup &

wait_for_health ${S0_PORT}
wait_for_health ${S1_PORT}
# ---- 4. verify pooling before failover (warmup on s0, reuse on s1) ----------
# msg_a: warmed on server_0, then reused on server_1.
msg_a="深圳是中国经济实力最强的城市之一。近年来，深圳GDP持续稳步增长，2023年突破3.4万亿元人民币，2024年接近3.7万亿元。长期位居全国城市前列。深圳经济以第二产业和第三产业为主，高端制造业、电子信息产业和现代服务业发达，形成了以科技创新为核心的产业结构。依托华为、腾讯、大疆等龙头企业，深圳在数字经济、人工智能、新能源等领域具有显著优势。同时，深圳进出口总额常年位居全国城市第一，是中国对外开放和高质量发展的重要引擎。深圳持续推进创新驱动发展战略，不断加大研发投入，全社会研发投入占GDP比重长期保持较高水平。深圳拥有完善的创业生态体系，吸引了大量科技企业和创新人才。近年来，深圳积极布局半导体、生物医药、低空经济和智能网联汽车等战略性新兴产业，进一步增强经济增长动能。请总结深圳经济发展的核心优势。"

echo "=== [4/6] verify pooling before failover ==="
echo ">>> warmup msg_a on server_0 (${S0_PORT})"
send_request ${S0_PORT} "${msg_a}"
sleep 5
echo ">>> reuse msg_a on server_1 (${S1_PORT}), expect cache hit"
send_request ${S1_PORT} "${msg_a}"

# ---- 5. kill the leader, wait for re-election -------------------------------
echo "=== [5/6] kill leader and wait for failover ==="
OLD_LEADER_ADDR=$(get_leader_addr)
OLD_LEADER_PORT="${OLD_LEADER_ADDR##*:}"
echo "old leader: ${OLD_LEADER_ADDR} (rpc_port=${OLD_LEADER_PORT})"
kill_master_by_rpc_port "${OLD_LEADER_PORT}"

echo "waiting for a new leader to be elected..."
NEW_LEADER_ADDR=""
start_time=$(date +%s)
while true; do
    cur=$(get_leader_addr)
    if [ -n "${cur}" ] && [ "${cur}" != "${OLD_LEADER_ADDR}" ]; then
        NEW_LEADER_ADDR="${cur}"
        break
    fi
    if [ $(( $(date +%s) - start_time )) -ge 60 ]; then
        echo "❌ no new leader elected within timeout"
        exit 1
    fi
    sleep 1
done
echo "✅ new leader: ${NEW_LEADER_ADDR} (was ${OLD_LEADER_ADDR})"

# wait for the new leader to finish recovery and reach serving state
# (and for clients to reconnect) before sending requests; 5s was too short.
sleep 10

# ---- 6. verify pooling after failover with a BRAND-NEW prompt ---------------
# Use a different prompt (msg_b) never sent before the failover, so a hit on
# server_1 proves the cache was written/read through the NEW leader's global
# pool (not stale local cache from step 4).
msg_b="人工智能已经成为全球科技竞争的重要方向。近年来，大模型技术快速发展，在自然语言处理、代码生成、多模态理解以及智能代理等领域取得显著突破。越来越多企业开始将人工智能技术应用于客服、办公自动化、内容生成、金融风控和软件开发等场景。与此同时，人工智能的发展也带来了新的挑战，包括算力成本快速上升、训练数据质量参差不齐、模型幻觉问题以及隐私保护需求增强。各国政府正在制定相应监管框架，以平衡技术创新和风险控制之间的关系。未来几年，人工智能有望进一步推动生产力提升，并深刻影响教育、医疗、科研和工业制造等行业的发展模式。请列出人工智能当前面临的主要挑战。"

echo "=== [6/6] verify pooling after failover (new prompt msg_b) ==="
echo ">>> warmup msg_b on server_0 (${S0_PORT})"
send_request ${S0_PORT} "${msg_b}"
sleep 5
echo ">>> reuse msg_b on server_1 (${S1_PORT}), expect cache hit via new leader"
send_request ${S1_PORT} "${msg_b}"

echo
echo "=== HA (redis) test completed ==="
echo "Check cache hit:  grep -E 'storage_cache_token_num' log_*/cache_storage.log* "
echo "Master logs:      log_master_1 / log_master_2 / log_master_3"
echo "Redis log:        log_redis"
echo "Current leader:   ${REDIS_CLI_BIN} -p ${REDIS_PORT} hget '${MASTER_VIEW_KEY}' leader_address"
