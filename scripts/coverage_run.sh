#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "$DIR"

run_path="$DIR/../test/"
cd ${run_path}
ls

exclude=("ci_use" "ce")
for d in */ ; do
  dir_name="${d%/}"
  if [[ -d "$dir_name" ]]; then
    skip=false
    for ex in "${exclude[@]}"; do
      if [[ "$dir_name" == "$ex" ]]; then
        skip=true
        break
      fi
    done
    if ! $skip; then
      dirs+=("$dir_name")
    fi
  fi
done

failed_tests_file="failed_tests.log"
> "$failed_tests_file"
disabled_tests=(
  layers/test_sampler.py
  layers/test_append_attention.py
  layers/test_attention.py
  operators/test_rejection_top_p_sampling.py
  operators/test_perchannel_gemm.py
  operators/test_scaled_gemm_f8_i4_f16.py
  operators/test_topp_sampling.py
  operators/test_stop_generation.py
  operators/test_air_topp_sampling.py
  operators/test_fused_moe.py
  layers/test_repetition_early_stopper.py
  operators/test_stop_generation_multi_ends.py
  graph_optimization/test_cuda_graph.py
)
is_disabled() {
  local test_file_rel="$1"
  for disabled in "${disabled_tests[@]}"; do
    if [[ "$test_file_rel" == "$disabled" ]]; then
      return 0
    fi
  done
  return 1
}

total=0
fail=0
success=0

for dir in "${dirs[@]}"; do
  if [ -d "$dir" ]; then
    echo "Running tests in directory: $dir"
    while IFS= read -r -d '' test_file; do
      total=$((total + 1))
      echo "Running $test_file"

      if is_disabled "$test_file"; then
        echo "Skipping disabled test: $test_file"
        continue
      fi
      # TODO: Add a framework to manage unit test execution time
      timeout 600 python -m coverage run "$test_file"
      if [ $? -ne 0 ]; then
        echo "$test_file" >> "$failed_tests_file"
        fail=$((fail + 1))

        PORTS=($FLASK_PORT $FD_API_PORT $FD_ENGINE_QUEUE_PORT $FD_METRICS_PORT)
        echo "==== PORT CLEAN AFTER UT FAILED ===="

        for port in "${PORTS[@]}"; do
            PIDS=$(lsof -t -i :$port)
            if [ -n "$PIDS" ]; then
                echo "Port $port is occupied by PID(s): $PIDS"
                echo "$PIDS" | xargs -r kill -9
                echo "Port $port cleared"
            else
                echo "Port $port is free"
            fi
        done
      else
        success=$((success + 1))
      fi
    done < <(find "$dir" -type f -name "test_*.py" -print0)
  else
    echo "Directory $dir not found, skipping."
  fi
done

echo "===================================="
echo "Total test files run: $total"
echo "Successful tests: $success"
echo "Failed tests: $fail"
echo "Failed test cases are listed in $failed_tests_file"

if [ "$fail" -ne 0 ]; then
  echo "Failed test cases:"
  cat "$failed_tests_file"
  exit 8
fi
