#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "$DIR"

run_path="$DIR/../test/"
cd ${run_path}
ls

dirs=("layers" "operators" "worker")
failed_tests_file="failed_tests.log"
> "$failed_tests_file"

total=0
fail=0
success=0

for dir in "${dirs[@]}"; do
  if [ -d "$dir" ]; then
    echo "Running tests in directory: $dir"
    while IFS= read -r -d '' test_file; do
      total=$((total + 1))
      echo "Running $test_file"
      python -m coverage run "$test_file"
      if [ $? -ne 0 ]; then
        echo "$test_file" >> "$failed_tests_file"
        fail=$((fail + 1))
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
