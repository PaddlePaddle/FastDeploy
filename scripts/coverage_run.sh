#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
run_path="$DIR/../tests/"
export PYTEST_INI="$DIR/../pytest.ini"

cd "$run_path" || exit 1

failed_tests_file="failed_tests.log"
> "$failed_tests_file"

# 定义不使用 pytest 的文件列表（可以根据需要加目录或文件）
special_tests=(
    "graph_optimization/test_cuda_graph_dynamic_subgraph.py"
    "graph_optimization/test_cuda_graph_spec_decode.py"
    "layers/test_quant_layer.py"
    "operators/test_token_penalty.py"
    "operators/test_split_fuse.py"
    "operators/test_flash_mask_attn.py"
    "operators/test_w4afp8_gemm.py"
)

# 执行特殊测试文件
for test_file in "${special_tests[@]}"; do
    if [ -f "$test_file" ]; then
        echo "Running special test: $test_file"
        python -m coverage run "$test_file"
        status=$?
        if [ "$status" -ne 0 ]; then
            echo "$test_file" >> "$failed_tests_file"
        fi
    else
        echo "Warning: $test_file not found"
    fi
done

# 运行所有测试（pytest.ini 会自动忽略不需要的）
# --maxfail=0 表示不提前停止
# --junitxml 可以生成 XML 报告，便于后续统计
python -m coverage run -m pytest -c $PYTEST_INI --maxfail=0 --disable-warnings -q --junitxml=report.xml
status=$?

# 提取失败的 case 列表（可选）
grep "<testcase" report.xml | grep failure | \
    sed -E 's/.*classname="([^"]*)" name="([^"]*)".*/\1::\2/' > "$failed_tests_file"

echo "===================================="
total=$(grep -c "<testcase" report.xml)
failed=$(grep -c "<failure" report.xml)
success=$((total - failed))

echo "Total test cases: $total"
echo "Successful tests: $success"
echo "Failed tests: $failed"


echo "Special tests total: ${#special_tests[@]}"
echo "Special tests failed: $failed_special"
echo "Special tests successful: $success_special"

if [ "$failed_pytest" -ne 0 ] || [ "$failed_special" -ne 0 ]; then
    echo "Failed test cases are listed in $failed_tests_file"
    cat "$failed_tests_file"
    exit 8
fi
