#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tests_path="$DIR/../tests/"
export PYTEST_INI="$DIR/../tests/cov_pytest.ini"
run_path=$( realpath "$DIR/../")

export COVERAGE_FILE=${COVERAGE_FILE:-$DIR/../coveragedata/.coverage}
export COVERAGE_RCFILE=${COVERAGE_RCFILE:-$DIR/../scripts/.coveragerc}

# ============================================================
#  Classify tests into one of the following categories
#  - multi_gpu: requires multiple GPUs / ports (run sequentially)
#  - single_gpu: independent tests (can run in parallel)
# ============================================================
classify_tests() {
    local test_file=$1
    # Rule 1: distributed tests (explicit multi-GPU launch)
    if [[ "$test_file" =~ tests/distributed/.*test_.*\.py ]]; then
        echo "multi_gpu"
        return
    fi

    # Rule 2: e2e tests (usually involve service / ports)
    if [[ "$test_file" =~ tests/e2e/.*test_.*\.py ]]; then
        echo "multi_gpu"
        return
    fi

    # Rule 3: model loader tests (allocate multiple GPUs)
    if [[ "$test_file" =~ tests/model_loader/.*test_.*\.py ]]; then
        echo "multi_gpu"
        return
    fi

    # Rule 4: check file content for tensor_parallel_size=[234] or --tensor-parallel-size [234]
    #    or CUDA_VISIBLE_DEVICES="0,1"
    #    or PORT environment variables
    if [ -f "$test_file" ]; then
        if grep -q '"tensor_parallel_size".*[1234]\|--tensor-parallel-size.*[1234]\|tensor_parallel_size.*=[1234]\|CUDA_VISIBLE_DEVICES.*0.*1\|paddle\.distributed\.launch.*--gpus.*0.*1\|FD_API_PORT\|FLASK_PORT\|FD_ENGINE_QUEUE_PORT\|FD_METRICS_PORT\|FD_CACHE_QUEUE_PORT\|FD_ROUTER_PORT\|FD_CONNECTOR_PORT\|FD_RDMA_PORT' "$test_file" 2>/dev/null; then
            echo "multi_gpu"
            return
        fi
    fi

    # Rule 5: high-risk OOM tests (treat as multi_gpu for sequential execution)
    if [[ "$test_file" =~ ^tests/entrypoints/cli/ ||
          "$test_file" == "tests/layers/test_append_attention_with_output.py" ||
          "$test_file" == "tests/operators/test_get_position_ids_and_mask_encoder_batch.py" ||
          "$test_file" == "tests/operators/test_group_swiglu_with_masked.py" ||
          "$test_file" == "tests/operators/test_hybrid_mtp_ngram.py" ||
          "$test_file" == "tests/operators/test_moe_top_k_select.py" ||
          "$test_file" == "tests/operators/test_noaux_tc.py" ||
          "$test_file" == "tests/operators/test_qk_rmsnorm_fused.py" ||
          "$test_file" == "tests/output/test_get_save_output_v1.py" ||
          "$test_file" == "tests/output/test_process_batch_draft_tokens.py" ||
          "$test_file" == "tests/output/test_process_batch_output.py" ]]; then
        echo "multi_gpu"
        return
    fi

    # ========== Single-GPU tests (no port required, can run in parallel) ==========
    echo "single_gpu"
}

# ============================================================
# Run Test With Logging
# ============================================================
run_test_with_logging() {
    local test_file=$1
    local log_prefix=$2
    local status

    echo "Running pytest file: $test_file"

    # Create isolated log directory for this test to avoid race conditions
    # Format: unittest_logs/<test_dir>/<test_file_base>/log
    local test_rel_path="${test_file#tests/}"
    local test_dir=$(dirname "$test_rel_path")
    local test_name=$(basename "$test_file" .py)
    local isolated_log_dir="${run_path}/unittest_logs/${test_dir}/${test_name}/log"
    mkdir -p "$isolated_log_dir"

    # Set FD_LOG_DIR to isolate logs for each test
    export FD_LOG_DIR="$isolated_log_dir"

    # Run test
    timeout 600 python -m coverage run -m pytest -c ${PYTEST_INI} "$test_file" -vv -s
    status=$?

    if [ "$status" -ne 0 ]; then
        echo "$test_file" >> "$log_prefix"
        echo ""
        echo "==================== Test Failed: $test_file ===================="

        # Use isolated log directory for this test
        if [ -d "$isolated_log_dir" ]; then
            echo
            echo ">>>> Processing log directory: ${isolated_log_dir}"

            # workerlog
            worker_logs=("${isolated_log_dir}"/workerlog.0)

            if [ -f "${worker_logs[0]}" ]; then
                for worker_log in "${worker_logs[@]}"; do
                    [ -f "${worker_log}" ] || continue
                    echo "---------------- ${worker_log} (last 100 lines) ----------------"
                    tail -n 100 "${worker_log}" || true
                    echo "---------------------------------------------------------------"
                done
            fi

            echo ">>> grep error in ${isolated_log_dir}"
            grep -Rni --color=auto "error" "${isolated_log_dir}" --exclude="pytest_*_error.log" || true
        fi

        # print all server logs
        server_logs=("${run_path}"/*.log)
        if [ "${#server_logs[@]}" -gt 0 ]; then
            for server_log in "${server_logs[@]}"; do
                # skip failed_tests_file
                [[ "$(basename "$server_log")" == "$failed_tests_file" ]] && continue
                if [ -f "${server_log}" ]; then
                    echo
                    echo "---------------- ${server_log} (last 100 lines) ----------------"
                    tail -n 100 "${server_log}" || true
                    echo "---------------------------------------------------------------"
                fi
            done
        else
            echo "No *.log files found"
        fi

        echo "======================================================="
    fi

    # Clean up port-related processes
    if [ -n "$FD_CACHE_QUEUE_PORT" ]; then
        ps -ef | grep "${FD_CACHE_QUEUE_PORT}" | grep -v grep | awk '{print $2}' | xargs -r kill -9 || true
    fi
    if [ -n "$FD_ENGINE_QUEUE_PORT" ]; then
        ps -ef | grep "${FD_ENGINE_QUEUE_PORT}" | grep -v grep | awk '{print $2}' | xargs -r kill -9 || true
    fi

    # if passed, remove the isolated log directory and server logs
    if [ "$status" -eq 0 ]; then
        rm -rf "${isolated_log_dir}" || true
        # Clean up server logs in run_path on pass
        for f in "${run_path}"/*.log; do
            [[ "$(basename "$f")" != "${failed_tests_file}" ]] && rm -f "$f" || true
        done
    fi

    # Unset FD_LOG_DIR to avoid affecting next test
    unset FD_LOG_DIR
    return $status
}

# ============================================================
# Run a shard of tests on a dedicated GPU
#   - one shard = one process = one GPU
# ============================================================
run_shard() {
    local shard_name=$1
    local gpu_id=$2
    shift 2
    local tests=("$@")

    echo "===================================="
    echo "Starting shard '${shard_name}' on GPU ${gpu_id}"
    echo "Tests count: ${#tests[@]}"
    echo "===================================="

    # Set GPU
    export CUDA_VISIBLE_DEVICES="$gpu_id"
    export COVERAGE_FILE="${DIR}/../coveragedata/.coverage.${shard_name}"

    # Failed log filename (no path, directly in project root)
    local failed_log="${shard_name}_failed.txt"
    rm -f "$failed_log"
    > "$failed_log"

    local success_count=0
    local failed_count=0

    for file in "${tests[@]}"; do
        echo "[${shard_name}] Running: $file"

        run_test_with_logging "$file" "$failed_log"
        local status=$?

        if [ "$status" -eq 0 ]; then
            success_count=$((success_count + 1))
        else
            failed_count=$((failed_count + 1))
        fi
    done

    unset COVERAGE_FILE

    echo "===================================="
    echo "Shard '${shard_name}' completed"
    echo "Successful: $success_count"
    echo "Failed: $failed_count"
    echo "===================================="

    unset CUDA_VISIBLE_DEVICES

    return $failed_count
}

# ============================================================
# Main Flow
# ============================================================

failed_tests_file="failed_tests.log"
> "$failed_tests_file"

echo "===================================="
echo "Coverage Test Execution with Parallel Single-GPU Tests"
echo "===================================="

# ============================================================
# Step 1: Collect & classify tests
# ============================================================
echo "Step 1: Collecting and classifying tests"

ALL_TEST_FILES=$(
    python -m pytest --collect-only -q -c "${PYTEST_INI}" "${tests_path}" --rootdir="${run_path}" --disable-warnings 2>&1 \
    | grep -E 'tests/.+\/test_.*\.py' \
    | sed -E 's@.*(tests/[^: ]*test_[^: ]*\.py).*@\1@' \
    | sort -u
)

if [ -z "$ALL_TEST_FILES" ]; then
    echo "ERROR: No test files found!"
    exit 1
fi

MULTI_GPU_TESTS=()
SINGLE_GPU_TESTS=()

TOTAL_TESTS=0
for file in $ALL_TEST_FILES; do
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    test_type=$(classify_tests "$file")

    case "$test_type" in
        "multi_gpu")
            MULTI_GPU_TESTS+=("$file")
            ;;
        "single_gpu")
            SINGLE_GPU_TESTS+=("$file")
            ;;
    esac
done

echo "Multi-GPU tests: ${#MULTI_GPU_TESTS[@]}"
echo "Single-GPU tests: ${#SINGLE_GPU_TESTS[@]}"
echo "Total tests: $TOTAL_TESTS"

# ============================================================
# Step 2: Run multi-GPU tests (sequential)
# ============================================================
echo "Step 2: Running multi-GPU tests"

if [ ${#MULTI_GPU_TESTS[@]} -gt 0 ]; then
    for file in "${MULTI_GPU_TESTS[@]}"; do
        run_test_with_logging "$file" "$failed_tests_file"
    done
else
    echo "No multi-GPU tests to run."
fi

# ============================================================
# Step 3: Run single-GPU tests (parallel shards)
# ============================================================
echo "Step 3: Running single-GPU tests in parallel"

if [ ${#SINGLE_GPU_TESTS[@]} -gt 0 ]; then
    # Split single-GPU tests into 2 shards (1 per GPU)
    TOTAL=${#SINGLE_GPU_TESTS[@]}
    HALF=$(( TOTAL / 2 ))

    SHARD_1=("${SINGLE_GPU_TESTS[@]:0:$HALF}")
    SHARD_2=("${SINGLE_GPU_TESTS[@]:$HALF}")

    echo "Shard 1: ${#SHARD_1[@]} tests on GPU 0"
    echo "Shard 2: ${#SHARD_2[@]} tests on GPU 1"

    # Run in parallel (1 process per GPU)
    run_shard "shard1" 0 "${SHARD_1[@]}" &
    PID1=$!
    run_shard "shard2" 1 "${SHARD_2[@]}" &
    PID2=$!

    # Wait for all shards to complete
    wait $PID1
    EXIT_CODE1=$?
    wait $PID2
    EXIT_CODE2=$?

    # Merge shard failed logs to main failed log
    for shard in shard1 shard2; do
        if [ -f "${shard}_failed.txt" ]; then
            cat "${shard}_failed.txt" >> "$failed_tests_file"
            rm -f "${shard}_failed.txt"
        fi
    done

    echo ""
    echo "===================================="
    echo "Parallel execution completed"
    echo "Shard 1 exit code: $EXIT_CODE1"
    echo "Shard 2 exit code: $EXIT_CODE2"
    echo "===================================="
else
    echo "No single-GPU tests to run."
fi

# ============================================================
# Step 4: Summary
# ============================================================
echo "Step 4: Summary"

# Count failed tests
if [ -f "$failed_tests_file" ]; then
    failed_count=$(wc -l < "$failed_tests_file" | tr -d ' ')
else
    failed_count=0
fi

success_count=$((TOTAL_TESTS - failed_count))

echo "Pytest total: $TOTAL_TESTS"
echo "Pytest successful: $success_count"
echo "Pytest failed: $failed_count"

echo "===================================="

# Exit with error and package logs if there were failures
if [ "$failed_count" -ne 0 ]; then
    echo "Failed test cases are listed in $failed_tests_file"
    cat "$failed_tests_file"

    # clean the empty directories
    if [ -d "${run_path}/unittest_logs" ]; then
        echo "Cleaning empty directories..."

        # remove *error.log* files (cleanup logs from stopped processes)
        find "${run_path}/unittest_logs" \( -name "console_error.log*" -o -name "error.log*" \) -delete || true

        # perform multi-round clean until no more empty directories are found
        while true; do
            before=$(find "${run_path}/unittest_logs" -type d | wc -l)
            find "${run_path}/unittest_logs" -mindepth 1 -type d -empty -delete || true
            after=$(find "${run_path}/unittest_logs" -type d | wc -l)
            [ "$before" -eq "$after" ] && break
        done
    fi

    # Only package logs when there are failures
    echo "===================================="
    echo "Step 5: Packaging logs (only on failure)"
    echo "===================================="

    if [ -d "${run_path}/unittest_logs" ]; then
        tar -czf "${run_path}/unittest_logs.tar.gz" -C "${run_path}" unittest_logs
        echo "Logs packaged to: ${run_path}/unittest_logs.tar.gz"
        ls -lh "${run_path}/unittest_logs.tar.gz"
    else
        echo "No unittest_logs directory found."
    fi

    echo "===================================="

    exit 8
fi

echo "All tests passed!"
exit 0
