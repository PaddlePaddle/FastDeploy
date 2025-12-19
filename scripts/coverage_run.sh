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
# Run pytest, executing each file independently.
# Use pytest's --collect-only output and extract the actual test file paths (tests/.../test_*.py).
# Note: when collection fails, pytest prints lines like "ERROR tests/xxx/test_xxx.py::test_xxx ...".
# To avoid treating prefixes like "ERROR"/"FAILED"/"collecting" as file names, keep only the
# "tests/.../test_*.py" segment from each line and discard the rest.
TEST_FILES=$(
  python -m pytest --collect-only -q -c "${PYTEST_INI}" "${tests_path}" --rootdir="${run_path}" --disable-warnings 2>&1 \
    | grep -E 'tests/.+\/test_.*\.py' \
    | sed -E 's@.*(tests/[^: ]*test_[^: ]*\.py).*@\1@' \
    | sort -u
)


failed_pytest=0
success_pytest=0

for file in $TEST_FILES; do
    echo "Running pytest file: $file"
    python -m coverage run -m pytest -c ${PYTEST_INI} "$file" -vv -s
    status=$?
    if [ "$status" -ne 0 ]; then
        echo "$file" >> "$failed_tests_file"
        failed_pytest=$((failed_pytest+1))
    else
        success_pytest=$((success_pytest+1))
    fi
    ps -ef | grep "${FD_CACHE_QUEUE_PORT}" | grep -v grep | awk '{print $2}' | xargs -r kill -9
    ps -ef | grep "${FD_ENGINE_QUEUE_PORT}" | grep -v grep | awk '{print $2}' | xargs -r kill -9
done

##################################
# Summaries
##################################
echo "===================================="
echo "Pytest total: $((failed_pytest + success_pytest))"
echo "Pytest successful: $success_pytest"
echo "Pytest failed: $failed_pytest"

echo "Special tests total: ${#special_tests[@]}"
echo "Special tests successful: $success_special"

if [ "$failed_pytest" -ne 0 ]; then
    echo "Failed test cases are listed in $failed_tests_file"
    cat "$failed_tests_file"
    exit 8
fi

echo "All tests passed!"
