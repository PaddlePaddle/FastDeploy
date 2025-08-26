#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
run_path="$DIR/../tests/"
export PYTEST_INI="$DIR/../tests/cov_pytest.ini"

export COVERAGE_FILE=${COVERAGE_FILE:-$DIR/../coveragedata/.coverage}
export COVERAGE_RCFILE=${COVERAGE_RCFILE:-$DIR/../scripts/.coveragerc}
export COVERAGE_PROCESS_START=${COVERAGE_PROCESS_START:-$DIR/../scripts/.coveragerc}
cd "$run_path" || exit 1

failed_tests_file="failed_tests.log"
> "$failed_tests_file"


##################################
# 执行 pytest，每个文件单独跑
##################################
# 收集 pytest 文件
TEST_FILES=$(python -m pytest --collect-only -q -c ${PYTEST_INI} --disable-warnings | grep -Eo '^.*test_.*\.py' | sort | uniq)


failed_pytest=0
success_pytest=0

for file in $TEST_FILES; do
    echo "Running pytest file: $file"
    python -m pytest -c ${PYTEST_INI} --cov-config=${COVERAGE_RCFILE} "$file" -vv -s
    status=$?
    if [ "$status" -ne 0 ]; then
        echo "$file" >> "$failed_tests_file"
        failed_pytest=$((failed_pytest+1))
    else
        success_pytest=$((success_pytest+1))
    fi
done

##################################
# 汇总结果
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
