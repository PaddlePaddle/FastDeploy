#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tests_path="$DIR/../tests/"
export PYTEST_INI="$DIR/../tests/cov_pytest.ini"
run_path=$( realpath "$DIR/../")

export COVERAGE_FILE=${COVERAGE_FILE:-$DIR/../coveragedata/.coverage}
export COVERAGE_RCFILE=${COVERAGE_RCFILE:-$DIR/../scripts/.coveragerc}


failed_tests_file="failed_tests.log"
> "$failed_tests_file"


##################################
# Run pytest, one file at a time
# Use pytest's --collect-only output to extract the actual test file paths (e.g., tests/.../test_*.py).
# Note: pytest may output lines like "ERROR tests/xxx/test_xxx.py::test_xxx ..." on collection failure,
# to avoid treating prefixes like "ERROR"/"FAILED"/"collecting" as filenames,
# we only keep the "tests/.../test_*.py" portion and discard everything else.
TEST_FILES=$(
  python -m pytest --collect-only -q -c "${PYTEST_INI}" "${tests_path}" --rootdir="${run_path}" --disable-warnings 2>&1 \
    | grep -E 'tests/.+\/test_.*\.py' \
    | sed -E 's@.*(tests/[^: ]*test_[^: ]*\.py).*@\1@' \
    | sort -u
)


failed_pytest=0
success_pytest=0

# nullglob: if no match, the pattern expands to nothing
shopt -s nullglob

for file in $TEST_FILES; do
    echo "Running pytest file: $file"
    # Clean up previous logs
    rm -rf "${run_path}"/log* || true
    for f in "${run_path}"/*.log; do
        [[ "$(basename "$f")" != "${failed_tests_file}" ]] && rm -f "$f"
    done

    # Run pytest with coverage for the current file
    # Set timeout to 600 seconds to avoid infinite loop
    timeout 600 python -m coverage run -m pytest -c ${PYTEST_INI} "$file" -vv -s
    status=$?
    if [ "$status" -ne 0 ]; then
        echo "$file" >> "$failed_tests_file"
        failed_pytest=$((failed_pytest+1))

        echo ""
        echo "==================== Dumping Logs ===================="

        for log_dir in "${run_path}"/log*; do
            if [ -d "${log_dir}" ]; then
                echo
                echo ">>>> Processing log directory: ${log_dir}"

                # print all workerlog.0
                worker_logs=("${log_dir}"/workerlog.0)
                if [ "${#worker_logs[@]}" -gt 0 ]; then
                    for worker_log in "${worker_logs[@]}"; do
                        if [ -f "${worker_log}" ]; then
                            echo "---------------- ${worker_log} (last 100 lines) ----------------"
                            tail -n 100 "${worker_log}" || true
                            echo "---------------------------------------------------------------"
                        fi
                    done
                else
                    echo "No workerlog.0 found in ${log_dir}"
                fi

                echo ">>> grep error in ${log_dir}"
                grep -Rni --color=auto "error" "${log_dir}" || true
            fi
        done

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

        echo "======================================================"
    else
        success_pytest=$((success_pytest+1))
    fi
    ps -ef | grep "${FD_CACHE_QUEUE_PORT}" | grep -v grep | awk '{print $2}' | xargs -r kill -9
    ps -ef | grep "${FD_ENGINE_QUEUE_PORT}" | grep -v grep | awk '{print $2}' | xargs -r kill -9
done
shopt -u nullglob

##################################
# Summary
##################################
echo "===================================="
echo "Pytest total: $((failed_pytest + success_pytest))"
echo "Pytest successful: $success_pytest"
echo "Pytest failed: $failed_pytest"


if [ "$failed_pytest" -ne 0 ]; then
    echo "Failed test cases are listed in $failed_tests_file"
    cat "$failed_tests_file"
    exit 8
fi

echo "All tests passed!"
