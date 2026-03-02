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

    if ! python -m pytest -sv --tb=short "${test_file}"; then
        echo "Pytest failed for: ${test_file}"
        echo "${test_file}" >> "${FAILED_CASE_FILE}"
        FAILED_COUNT=$((FAILED_COUNT + 1))

        # print all workerlog.0
        for log_dir in "${REPO_ROOT}"/log_*; do
            worker_log="${log_dir}/workerlog.0"
            if [ -f "${worker_log}" ]; then
                echo "---------------- ${worker_log} (last 200 lines) -------------"
                tail -n 200 "${worker_log}"
                echo "------------------------------------------------------------"
            fi
        done

        # print all server_*.log
        for server_log in "${REPO_ROOT}"/server_*.log; do
            if [ -f "${server_log}" ]; then
                echo "---------------- ${server_log} (last 200 lines) ---------------"
                tail -n 200 "${server_log}"
                echo "------------------------------------------------------------"
            fi
        done
    fi
done

shopt -u nullglob

if [ "${FAILED_COUNT}" -ne 0 ]; then
    echo "${FAILED_COUNT} test file(s) failed:"
    cat "${FAILED_CASE_FILE}"
    exit 1
else
    echo "All golang_router end-to-end tests passed"
    exit 0
fi
