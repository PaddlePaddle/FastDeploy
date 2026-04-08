import argparse
import glob
import os
import re
from typing import List, Optional


def extract_entropy_values(log_path: str) -> List[float]:
    pattern = r"entropy:\s*([0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?)"

    entropy_values = []
    with open(log_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            match = re.search(pattern, line)
            if match:
                try:
                    entropy_value = float(match.group(1))
                    entropy_values.append(entropy_value)
                except ValueError:
                    continue

    return entropy_values


def calculate_average(entropy_values: List[float], drop_ratio: float = 0.1) -> Optional[float]:
    if not entropy_values:
        return None
    sorted_vals = sorted(entropy_values)
    n = len(sorted_vals)
    drop_count = int(n * drop_ratio)
    filtered_vals = sorted_vals[drop_count : n - drop_count] if drop_count > 0 else sorted_vals
    if not filtered_vals:
        return None, []
    avg = sum(filtered_vals) / len(filtered_vals)
    return avg, filtered_vals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=str, required=True)
    parser.add_argument("--drop-ratio", "-d", type=float, default=0.1)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--start-id", "-s", type=int)
    parser.add_argument("--end-id", "-e", type=int)
    args = parser.parse_args()
    log_files = glob.glob(os.path.join(args.log_dir, "data_processor.log.*"))
    if not log_files:
        print(f"No log files found in {args.log_dir}")
        return

    entropy_values = []
    for log_file in log_files:
        entropy_values.extend(extract_entropy_values(log_file))
    if args.start_id and args.end_id:
        entropy_values = entropy_values[args.start_id : args.end_id]
    average_entropy, filtered_vals = calculate_average(entropy_values, args.drop_ratio)

    print(f"{len(entropy_values)} entropy values were found")
    print(f"effective entropy values: {len(filtered_vals)} (drop ratio {args.drop_ratio})")
    print(f"Average entropy: {average_entropy:.10f}")
    if args.verbose:
        print("\nentropy details:")
        for i, value in enumerate(filtered_vals, 1):
            print(f"  {i}. {value}")


if __name__ == "__main__":
    main()
