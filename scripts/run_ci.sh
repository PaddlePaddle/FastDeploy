#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "$DIR"

python -m pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
python -m pip install --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu126/
python -m pip install -r requirements.txt
python -m pip install jsonschema aistudio_sdk==0.2.6
bash build.sh || exit 1

failed_files=()
run_path="$DIR/../test/ci_use/"

# load all test files
for subdir in "$run_path"*/; do
    if [ -d "$subdir" ]; then
        pushd "$subdir" > /dev/null || continue  # into test dir or continue

        # search for test_*.py files
        for file in test_*.py; do
            if [ -f "$file" ]; then
                echo "============================================================"
                echo "Running pytest on $(realpath "$file")"
                echo "------------------------------------------------------------"

                set +e
                timeout 360 python -m pytest --disable-warnings -sv "$file"
                exit_code=$?
                set -e

                if [ $exit_code -ne 0 ]; then
                    if [ -f "${subdir%/}/log/workerlog.0" ]; then
                        echo "---------------- log/workerlog.0 -------------------"
                        cat "${subdir%/}/log/workerlog.0"
                        echo "----------------------------------------------------"
                    fi

                    if [ -f "${subdir%/}/server.log" ]; then
                        echo "---------------- server.log ----------------"
                        cat "${subdir%/}/server.log"
                        echo "--------------------------------------------"
                    fi

                    if [ "$exit_code" -eq 1 ] || [ "$exit_code" -eq 124 ]; then
                        echo "[ERROR] $file 起服务或执行异常，exit_code=$exit_code"
                        if [ "$exit_code" -eq 124 ]; then
                            echo "[TIMEOUT] $file 脚本执行超过 6 分钟, 任务超时退出！"
                        fi
                    fi

                    failed_files+=("$subdir$file")
                    exit 1
                fi
                echo "------------------------------------------------------------"
            fi
        done
        popd > /dev/null  # back to test dir
    fi
done

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