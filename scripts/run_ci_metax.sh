#!/bin/bash

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tests_path="$DIR/../tests/"
export PYTEST_INI="$DIR/../tests/cov_pytest.ini"
run_path=$( realpath "$DIR/../")


export COVERAGE_FILE=${COVERAGE_FILE:-$DIR/../coveragedata/.coverage}
export COVERAGE_RCFILE=${COVERAGE_RCFILE:-$DIR/../scripts/.coveragerc}


LOG_ROOT_PATH=${run_path}/metax_log
LOG_SUBDIR=${LOG_ROOT_PATH}/logs
LOG_RESULT_TMP=$(mktemp)
PASS_FILE_LIST=${LOG_ROOT_PATH}/passed_files.txt
FAIL_FILE_LIST=${LOG_ROOT_PATH}/failed_files.txt
SUMMARY_FILE_LIST=${LOG_ROOT_PATH}/summary.txt
trap 'rm -f "$LOG_RESULT_TMP"' EXIT

mkdir -p "$LOG_ROOT_PATH" "$LOG_SUBDIR"

OVERWRITE_OLD_RESULT="yes"
METAX_GPU_TARGET=C500
PARALLEL_NUM="4"
PYTEST_EXTRA_ARGS="${3:-}"


declare -a IGNORE_PATHS=()
while IFS= read -r line; do
    if [ -z "$line" ]; then
        continue
    fi
    path=$(echo "$line" | sed 's/^\s*--ignore=//')
    if [ -n "$path" ]; then
        IGNORE_PATHS+=("$path")
    fi
done < <(grep -E '^\s*--ignore=' "$PYTEST_INI")


declare -A SEEN=()
declare -a EXCLUDE_PATHS=()
for path in "${IGNORE_PATHS[@]}"; do
    if [[ -z "${SEEN[$path]}" ]]; then
        SEEN[$path]=1
        EXCLUDE_PATHS+=("$path")
    fi
done

# declare -a CUSTOM_EXCLUDE_PATHS=(
#     "tests/e2e"
#     "tests/model_loader"
#     "tests/pooling"
#     "tests/entrypoints"
# )

# for path in "${CUSTOM_EXCLUDE_PATHS[@]}"; do
#     if [[ -z "${SEEN[$path]}" ]]; then
#         SEEN[$path]=1
#         EXCLUDE_PATHS+=("$path")
#     fi
# done


is_excluded() {
    local target_path="$1"
    for exclude in "${EXCLUDE_PATHS[@]}"; do
        if [ -z "$exclude" ]; then
            continue
        fi
        if [[ "$target_path" == *"$exclude"* ]]; then
            return 0
        fi
    done
    return 1
}
# FIND_PATTERN="test_*.py"
# declare -a ALL_PATHS=()

# while IFS= read -r path; do
#     [[ -n "$path" ]] && ALL_PATHS+=("$path")
# done < <(find "${tests_path}" -type f -name "$FIND_PATTERN" | sort | uniq)


declare -a FILTERED_PATHS=()

METAX_CI_CASELIST=(
    "tests/metax_ci/test_fused_moe.py"
    "tests/metax_ci/test_cache_kv_with_rope.py"
    "tests/operators/test_limit_thinking_content_length.py"
    "tests/operators/test_update_inputs_v1.py"
    "tests/operators/test_set_value_by_flags_and_idx.py"
    "tests/operators/test_get_token_penalty_multi_scores.py"
    "tests/operators/test_speculate_get_token_penalty_multi_scores.py"
    "tests/operators/test_token_penalty.py"
    "tests/operators/test_stop_generation_multi_ends.py"
    "tests/operators/test_get_padding_offset.py"
    "tests/operators/test_rebuild_padding.py"
    "tests/operators/test_share_external_data.py"
    "tests/operators/test_rejection_top_p_sampling.py"
    "tests/layers/test_min_sampling.py"
)
for path in "${METAX_CI_CASELIST[@]}"; do
    TEST_FILE_PATH=$run_path/$path

    if [ -e "$TEST_FILE_PATH" ]; then
        FILTERED_PATHS+=("$TEST_FILE_PATH")
    else
        echo "Test file: [ $path ] does not exist, skip it."
    fi

    # if ! is_excluded "$path"; then
    #     FILTERED_PATHS+=("$run_path/$path")
    # fi
done


echo -e "\n================== Metax CI test total num ( ${#FILTERED_PATHS[@]} ) =================="

if [ "$OVERWRITE_OLD_RESULT" = "yes" ]; then
    > "${SUMMARY_FILE_LIST}"
    > "${PASS_FILE_LIST}"
    > "${FAIL_FILE_LIST}"

    rm -f "${LOG_SUBDIR}"/*.log
else
    echo -e "\n================== $(date +%Y-%m-%d_%H:%M:%S) =================" >> "${SUMMARY_FILE_LIST}"
fi


get_max_free_gpu() {
    local log=$(mx-smi)
    local gpu_lines=$(echo "$log" | grep -E "MetaX ${METAX_GPU_TARGET}|MiB" | grep -v "Process" | grep -v "Board Name")

    local GPU_INFO=()
    local current_gpu_idx=""

    while IFS= read -r line; do
        if echo "$line" | grep -q "MetaX ${METAX_GPU_TARGET}"; then
            current_gpu_idx=$(echo "$line" | awk '{print $2}' | grep -E '^[0-9]+$')
        elif echo "$line" | grep -q "MiB"; then
            if [ -n "$current_gpu_idx" ]; then
                mem_used=$(echo "$line" | awk '{for(i=1;i<=NF;i++){if($i ~ /MiB/){split($(i-1),mem,"/");print mem[1]}}}' )
                mem_total=$(echo "$line" | awk '{for(i=1;i<=NF;i++){if($i ~ /MiB/){split($(i-1),mem,"/");print mem[2]}}}' )
                mem_free=$((mem_total - mem_used))
                GPU_INFO+=("$current_gpu_idx:$mem_used:$mem_total:$mem_free")
                # echo "$current_gpu_idx - ${mem_used}"
                current_gpu_idx=""
            fi
        fi
    done <<< "$gpu_lines"

    gpu_mem_info=${GPU_INFO[@]}

    local sorted_gpus=$(echo "$gpu_mem_info" | tr ' ' '\n' | sort -t ':' -k4,4nr -k1,1n)
    echo "${sorted_gpus}"

    local count=0
    local top_n=1
    local gpu_list=""
    while IFS= read -r gpu && [ $count -lt $top_n ]; do
        if [ -n "$gpu" ]; then
            local gpu_idx=$(echo "$gpu" | cut -d ':' -f 1)
            local gpu_free=$(echo "$gpu" | cut -d ':' -f 4)
            gpu_list="$gpu_list,$gpu_idx"
            count=$((count + 1))
        fi
    done <<< "$sorted_gpus"

    gpu_list=${gpu_list:1:${#gpu_list}}
    echo "$gpu_list"
}

test_env_init() {
    export MACA_PATH=/opt/maca
    if [ ! -d ${HOME}/cu-bridge ]; then
            `${MACA_PATH}/tools/cu-bridge/tools/pre_make`
    fi

    export CUDA_PATH=${HOME}/cu-bridge/CUDA_DIR
    export LD_LIBRARY_PATH=${CUDA_PATH}/lib64:${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:$LD_LIBRARY_PATH
    export PADDLE_XCCL_BACKEND=metax_gpu
    export FLAGS_weight_only_linear_arch=80
    export FD_MOE_BACKEND=cutlass # or triton
    export FD_METAX_KVCACHE_MEM=8
    export FD_ENC_DEC_BLOCK_NUM=2

    export MACA_VISIBLE_DEVICES=$(get_max_free_gpu)
}


test_single_file() {
    test_env_init

    local file="$1"
    local extra_args="$2"
    local log_file="${LOG_SUBDIR}/$(basename "$file").log"
    local start_time=$(date +%s)

    pytest "$file" $extra_args > "$log_file" 2>&1
    local exit_code=$?
    local end_time=$(date +%s)
    local cost_time=$((end_time - start_time))

    echo "$file,$exit_code,$cost_time" >> "$LOG_RESULT_TMP"

    if [ $exit_code -eq 0 ]; then
        echo "✅ $(basename "$file") passed (cost: ${cost_time}s)"
    else
        echo "❌ $(basename "$file") failed (cost: ${cost_time}s, exit code: $exit_code)"
    fi
}

export -f get_max_free_gpu
export -f test_env_init
export -f test_single_file
export LOG_RESULT_TMP PYTEST_EXTRA_ARGS LOG_SUBDIR


# if [ "$PARALLEL_NUM" = "auto" ]; then
#     PARALLEL_NUM=$(nproc --all 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
# fi


printf "%s\n" "${FILTERED_PATHS[@]}" | xargs -I {} -P "$PARALLEL_NUM" bash -c 'test_single_file "{}" "$PYTEST_EXTRA_ARGS"'


PASS_COUNT=0
FAIL_COUNT=0
TOTAL_COST_TIME=0
declare -a FAIL_FILES=()

while IFS=, read -r file exit_code cost_time; do
    TOTAL_COST_TIME=$((TOTAL_COST_TIME + cost_time))
    if [ "$exit_code" -eq 0 ]; then
        PASS_COUNT=$((PASS_COUNT + 1))
        echo "$file" >> ${PASS_FILE_LIST}
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAIL_FILES+=$(basename "$file")
        echo "$file" >> ${FAIL_FILE_LIST}
        echo -e "\n\n+++++++++++++++++++++++++ [ $(basename "$file") ] Fail Info +++++++++++++++++++++++++\n\n"
        cat ${LOG_SUBDIR}/$(basename "$file").log
    fi
done < "$LOG_RESULT_TMP"

{
    echo "============================= TEST RESULT SUMMARY ================================"
    echo "Pytest date: $(date +%Y-%m-%d\ %H:%M:%S)"
    echo "Pytest parallel num: $PARALLEL_NUM"
    echo "Pytest extra args: $PYTEST_EXTRA_ARGS"
    echo "Pytest total num: $((PASS_COUNT + FAIL_COUNT))"
    echo "Pytest successful: $PASS_COUNT"
    echo "Pytest failed: $FAIL_COUNT"
    echo "Pytest all cost: $TOTAL_COST_TIME s"
    echo "=================================================================================="
} >> "${SUMMARY_FILE_LIST}"


cat ${SUMMARY_FILE_LIST}


if [ "$FAIL_COUNT" -ne 0 ]; then
    # echo "Failed test cases are listed in $failed_tests_file"
    # cat "$FAIL_FILE_LIST"
    echo ${FAIL_FILES[@]}
    exit 1
fi
