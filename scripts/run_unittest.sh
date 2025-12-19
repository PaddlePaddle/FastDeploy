#!/bin/bash
set -x
set -e
cd FastDeploy

export no_proxy=agent.baidu.com:8118,localhost,127.0.0.1,localaddress,.localdomain.com,.cdn.bcebos.com,.baidu.com,bcebos.com
export https_proxy=
export http_proxy=
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LD_LIBRARY_PATH=/usr/local/cuda/compat/:/usr/lib64/:$LD_LIBRARY_PATH

ldconfig

python -V
pwd

git config --global --add safe.directory /workspace1/FastDeploy

python -m pip install --force-reinstall --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu126/
python -m pip install --upgrade --force-reinstall -r requirements/unittest/requirements.txt
python -m pip install xgrammar==0.1.19 torch==2.6.0
bash tools/build_wheel.sh


# Identify the GPU with the lowest memory usage
gpu_info=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null)

if [ -z "$gpu_info" ]; then
  echo "Error: No GPU found or nvidia-smi is unavailable."
  exit 1
fi

min_vram=999999
min_gpu=-1

while read -r line; do
  gpu_id=$(echo "$line" | awk -F', ' '{print $1}' | tr -d ' ')
  vram=$(echo "$line" | awk -F', ' '{print $2}' | tr -d ' ')

  if [ "$vram" -lt "$min_vram" ]; then
    min_vram=$vram
    min_gpu=$gpu_id
  fi
done <<< "$gpu_info"

export CUDA_VISIBLE_DEVICES=${min_gpu}

# Locate Python test files under the tests directory
test_files=$(find tests -type f -name "test*.py")

# Execute each discovered test file
for test_file in $test_files; do
    python $test_file

    # Verify the exit code of the previous command
    if [ $? -ne 0 ]; then
        echo $test_file
        exit 1
    fi
done

echo "All tests passed."
exit 0
