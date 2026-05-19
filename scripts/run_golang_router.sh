#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GOLANG_ROUTER_CASES_DIR="${REPO_ROOT}/tests/e2e/golang_router"
FAILED_CASE_FILE="${REPO_ROOT}/failed_cases.txt"

FAILED_COUNT=0

rm -f "${FAILED_CASE_FILE}"

shopt -s nullglob
test_files=("${GOLANG_ROUTER_CASES_DIR}"/test_*.py)

if [ "${#test_files[@]}" -eq 0 ]; then
    echo "ERROR: No test files found under: ${GOLANG_ROUTER_CASES_DIR}"
    exit 1
fi

for test_file in "${test_files[@]}"; do
    echo "------------------------------------------------------------"
    echo "Running pytest: ${test_file}"
    echo "------------------------------------------------------------"
    # Clean up previous logs
    rm -rf "${REPO_ROOT}"/log* || true
    rm -rf "${REPO_ROOT}"/*.log || true

    if ! python -m pytest -sv --tb=short "${test_file}"; then
        echo "Pytest failed for: ${test_file}"
        echo "${test_file}" >> "${FAILED_CASE_FILE}"
        FAILED_COUNT=$((FAILED_COUNT + 1))

        echo ""
        echo "==================== Dumping Logs ===================="

        for log_dir in "${REPO_ROOT}"/log*; do
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
                grep -Rni --color=auto "error" "${log_dir}" --exclude="pytest_*_error.log" || true
            fi
        done

        # print all server logs
        server_logs=("${REPO_ROOT}"/*.log)
        if [ "${#server_logs[@]}" -gt 0 ]; then
            for server_log in "${server_logs[@]}"; do
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
    fi
done

echo ""
echo "============================================================"

shopt -u nullglob

if [ "${FAILED_COUNT}" -ne 0 ]; then
    echo "${FAILED_COUNT} test file(s) failed:"
    cat "${FAILED_CASE_FILE}"
    exit 1
else
    echo "All golang_router end-to-end tests passed"
    exit 0
fi
