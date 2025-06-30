#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "$DIR"

python -m pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
python -m pip install --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu126/
python -m pip install -r requirements.txt
python -m pip install jsonschema aistudio_sdk==0.2.6
bash build.sh || exit 1

failed_files=()
run_path="$DIR/../test/ci_use"
pushd "$run_path" || exit 1  # 目录不存在时退出

for file in test_*; do
    if [ -f "$file" ]; then
        abs_path=$(realpath "$file")
        echo "Running pytest on $abs_path"
        if ! python -m pytest -sv "$abs_path"; then
            echo "Test failed: $file"
            failed_files+=("$file")
        fi
    fi
done
popd


if [ ${#failed_files[@]} -gt 0 ]; then
    echo "The following tests failed:"
    for f in "${failed_files[@]}"; do
        echo "$f"
    done
    exit 1
else
    echo "All tests passed!"
    exit 0
fi