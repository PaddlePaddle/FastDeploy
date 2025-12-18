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
# 执行 pytest，每个文件单独跑
# 使用 pytest 的 --collect-only 输出，并从每行中提取真正的测试文件路径（形如 tests/.../test_*.py）。
# 注意：pytest 在收集失败时会输出形如 "ERROR tests/xxx/test_xxx.py::test_xxx ..." 的行，
# 为了避免把前缀 "ERROR"/"FAILED"/"collecting" 等误当成文件名，这里只保留行中出现的
# "tests/.../test_*.py" 这一段，其他前后内容直接丢弃。
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
