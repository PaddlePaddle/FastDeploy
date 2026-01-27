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
    echo "Running pytest: ${test_file}"
    echo "------------------------------------------------------------"

    if ! python -m pytest -sv --tb=short "${test_file}"; then
        echo "Pytest failed for: ${test_file}"
        echo "${test_file}" >> "${FAILED_CASE_FILE}"
        FAILED_COUNT=$((FAILED_COUNT + 1))

        if [ -f "${REPO_ROOT}/log/log_0/workerlog.0" ]; then
            echo "---------------- workerlog.0 (last 200 lines) -------------"
            tail -n 200 "${REPO_ROOT}/log/log_0/workerlog.0"
            echo "------------------------------------------------------------"
        fi

        if [ -f "${REPO_ROOT}/server.log" ]; then
            echo "---------------- server.log (last 200 lines) ---------------"
            tail -n 200 "${REPO_ROOT}/server.log"
            echo "------------------------------------------------------------"
        fi
    fi
done

shopt -u nullglob

if [ "${FAILED_COUNT}" -ne 0 ]; then
    echo "${FAILED_COUNT} test file(s) failed:"
    cat "${FAILED_CASE_FILE}"
    exit 1
else
    echo "All 4-GPU end-to-end tests passed"
    exit 0
fi
