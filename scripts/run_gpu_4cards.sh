#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

E2E_CASES_DIR="${REPO_ROOT}/tests/e2e/4cards_cases"
FAILED_CASE_FILE="${REPO_ROOT}/failed_cases.txt"

FAILED_COUNT=0

rm -f "${FAILED_CASE_FILE}"

shopt -s nullglob
test_files=("${E2E_CASES_DIR}"/test_*.py)

if [ "${#test_files[@]}" -eq 0 ]; then
    echo "ERROR: No test files found under: ${E2E_CASES_DIR}"
    exit 1
fi

for test_file in "${test_files[@]}"; do
    echo "------------------------------------------------------------"
    echo "Running pytest on ${test_file}"
    echo "------------------------------------------------------------"
    # Clean up previous logs
    rm -rf "${REPO_ROOT}"/log* || true
    rm -rf "${REPO_ROOT}"/*.log || true

    timeout 600 python -m pytest -sv --tb=short "${test_file}"
    exit_code=$?

    if [ $exit_code -ne 0 ]; then
        if [ $exit_code -eq 124 ]; then
            echo "Pytest timeout (10 min) for: ${test_file}"
        else
            echo "Pytest failed for: ${test_file}"
        fi

        echo "${test_file}" >> "${FAILED_CASE_FILE}"
        FAILED_COUNT=$((FAILED_COUNT + 1))

        echo ""
        echo "==================== Dumping Logs ===================="

        if [ -d "${REPO_ROOT}/log" ]; then
            echo ">>> grep error in ${REPO_ROOT}/log/"
            grep -Rni --color=auto "error" "${REPO_ROOT}/log/" --exclude="pytest_*_error.log" || true
        else
            echo "${REPO_ROOT}/log directory not found"
        fi

        if [ -f "${REPO_ROOT}/log/log_0/workerlog.0" ]; then
            echo "---------------- workerlog.0 (last 100 lines) -------------"
            tail -n 100 "${REPO_ROOT}/log/log_0/workerlog.0"
            echo "------------------------------------------------------------"
        fi

        if [ -f "${REPO_ROOT}/server.log" ]; then
            echo "---------------- server.log (last 100 lines) ---------------"
            tail -n 100 "${REPO_ROOT}/server.log"
            echo "------------------------------------------------------------"
        fi
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
    echo "All 4-GPU end-to-end tests passed"
    exit 0
fi
