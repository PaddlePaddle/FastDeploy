# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Worker Parameter Dump Module

This module provides utilities to dump and compare worker process parameters
between old and new engine architectures for verification.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class WorkerParamDumper:
    """Dump worker process parameters for comparison."""

    def __init__(self, dump_dir: Optional[str] = None):
        """
        Initialize parameter dumper.

        Args:
            dump_dir: Directory to dump parameter files. Defaults to
                      environment variable FD_PARAM_DUMP_DIR or ./param_dumps
        """
        if dump_dir is None:
            dump_dir = os.getenv("FD_PARAM_DUMP_DIR", "./param_dumps")

        self.dump_dir = Path(dump_dir)
        self.dump_dir.mkdir(parents=True, exist_ok=True)

        self.arch_type = os.getenv("FD_USE_NEW_ENGINE_ARCHITECTURE", "0")
        self.arch_label = "new" if self.arch_type == "1" else "old"

    def is_enabled(self) -> bool:
        """Check if parameter dumping is enabled."""
        return os.getenv("FD_DUMP_WORKER_PARAMS", "0") == "1"

    def _serialize_args(self, args: Any) -> Dict[str, Any]:
        """Serialize args object to dictionary."""
        if hasattr(args, "__dict__"):
            # Namespace or object with __dict__
            return {k: self._serialize_value(v) for k, v in vars(args).items()}
        elif isinstance(args, dict):
            return {k: self._serialize_value(v) for k, v in args.items()}
        return args

    def _serialize_value(self, value: Any) -> Any:
        """Serialize a value for JSON storage."""
        if value is None:
            return None
        elif isinstance(value, (str, int, float, bool)):
            return value
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        elif hasattr(value, "__dict__"):
            # Object or dataclass
            return {k: self._serialize_value(v) for k, v in vars(value).items()}
        else:
            return str(value)

    def dump_params(self, args: Any) -> Optional[str]:
        """
        Dump parsed arguments to file.

        Args:
            args: Parsed arguments object (usually from argparse)

        Returns:
            File path if dumped, None otherwise
        """
        if not self.is_enabled():
            return None

        # Convert args to dict
        params_dict = self._serialize_args(args)

        # Add metadata
        params_dict["_metadata"] = {
            "architecture": self.arch_label,
            "pid": os.getpid(),
            "timestamp": __import__("time").time(),
        }

        # Generate filename
        filename = f"worker_params_{self.arch_label}_{params_dict['_metadata']['pid']}.json"
        filepath = self.dump_dir / filename

        # Write to file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(params_dict, f, indent=2, default=str, ensure_ascii=False)

        print(f"[PARAM_DUMP] Dumped parameters to: {filepath}")
        return str(filepath)


def dump_worker_params(args: Any) -> Optional[str]:
    """
    Convenience function to dump worker parameters.

    Args:
        args: Parsed arguments object

    Returns:
        File path if dumped, None otherwise
    """
    dumper = WorkerParamDumper()
    return dumper.dump_params(args)


class ParamComparator:
    """Compare parameters between old and new architectures."""

    def __init__(self, dump_dir: str = "./param_dumps"):
        """
        Initialize parameter comparator.

        Args:
            dump_dir: Directory containing dumped parameter files
        """
        self.dump_dir = Path(dump_dir)

    def load_params(self, arch_type: str) -> Optional[Dict[str, Any]]:
        """
        Load parameters for specified architecture type.

        Args:
            arch_type: Either "old" or "new"

        Returns:
            Parameters dictionary, or None if not found
        """
        pattern = f"worker_params_{arch_type}_*.json"
        files = list(self.dump_dir.glob(pattern))

        if not files:
            return None

        # Use the most recent file
        latest_file = max(files, key=lambda p: p.stat().st_mtime)

        with open(latest_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def compare(self) -> Tuple[bool, List[str]]:
        """
        Compare old and new architecture parameters.

        Returns:
            Tuple of (is_identical, list_of_differences)
        """
        old_params = self.load_params("old")
        new_params = self.load_params("new")

        if old_params is None:
            return False, ["Old architecture parameters not found"]
        if new_params is None:
            return False, ["New architecture parameters not found"]

        # Remove metadata for comparison
        old_copy = dict(old_params)
        new_copy = dict(new_params)

        old_metadata = old_copy.pop("_metadata", {})
        new_metadata = new_copy.pop("_metadata", {})

        # Compare parameters
        differences = self._compare_dicts(old_copy, new_copy, "")

        is_identical = len(differences) == 0
        return is_identical, differences

    def _compare_dicts(
        self,
        old: Dict[str, Any],
        new: Dict[str, Any],
        prefix: str,
    ) -> List[str]:
        """Compare two dictionaries recursively."""
        differences = []

        # Check for missing keys
        old_keys = set(old.keys())
        new_keys = set(new.keys())

        missing_in_new = old_keys - new_keys
        missing_in_old = new_keys - old_keys

        if missing_in_new:
            differences.append(f"{prefix}Missing in new: {missing_in_new}")
        if missing_in_old:
            differences.append(f"{prefix}Missing in old: {missing_in_old}")

        # Compare common keys
        for key in sorted(old_keys & new_keys):
            old_value = old[key]
            new_value = new[key]
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(old_value, dict) and isinstance(new_value, dict):
                differences.extend(self._compare_dicts(old_value, new_value, full_key))
            elif isinstance(old_value, list) and isinstance(new_value, list):
                differences.extend(self._compare_lists(old_value, new_value, full_key))
            elif old_value != new_value:
                differences.append(
                    f"{full_key}: old={repr(old_value)}, new={repr(new_value)}"
                )

        return differences

    def _compare_lists(
        self,
        old: List[Any],
        new: List[Any],
        key: str,
    ) -> List[str]:
        """Compare two lists."""
        differences = []

        if len(old) != len(new):
            differences.append(f"{key}: length differs (old={len(old)}, new={len(new)})")
            # Still compare elements up to min length
            max_len = min(len(old), len(new))
        else:
            max_len = len(old)

        for i in range(max_len):
            if isinstance(old[i], dict) and isinstance(new[i], dict):
                differences.extend(self._compare_dicts(old[i], new[i], f"{key}[{i}]"))
            elif old[i] != new[i]:
                differences.append(
                    f"{key}[{i}]: old={repr(old[i])}, new={repr(new[i])}"
                )

        return differences

    def print_report(self) -> bool:
        """
        Print comparison report to stdout.

        Returns:
            True if parameters are identical, False otherwise
        """
        is_identical, differences = self.compare()

        print("=" * 80)
        print("Worker Parameter Comparison Report")
        print("=" * 80)

        # Load parameters for display
        old_params = self.load_params("old")
        new_params = self.load_params("new")

        if old_params:
            metadata = old_params.get("_metadata", {})
            print(f"\nOld Architecture:")
            print(f"  File: worker_params_old_{metadata.get('pid', 'N/A')}.json")
            print(f"  PID: {metadata.get('pid', 'N/A')}")
        if new_params:
            metadata = new_params.get("_metadata", {})
            print(f"\nNew Architecture:")
            print(f"  File: worker_params_new_{metadata.get('pid', 'N/A')}.json")
            print(f"  PID: {metadata.get('pid', 'N/A')}")

        print("\n" + "-" * 80)
        print(f"Comparison Result: {'IDENTICAL ✓' if is_identical else 'DIFFERENT ✗'}")
        print("-" * 80)

        if differences:
            print(f"\nFound {len(differences)} difference(s):\n")
            for diff in differences:
                print(f"  ✗ {diff}")
        else:
            print("\nNo differences found! Parameters are identical.")

        print("=" * 80)

        return is_identical


def compare_params(dump_dir: str = "./param_dumps") -> bool:
    """
    Compare parameters and return result.

    Args:
        dump_dir: Directory containing dumped parameter files

    Returns:
        True if identical, False otherwise
    """
    comparator = ParamComparator(dump_dir)
    return comparator.print_report()


if __name__ == "__main__":
    # Command line interface
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare worker parameters between old and new architectures"
    )
    parser.add_argument(
        "--dump-dir",
        "-d",
        default="./param_dumps",
        help="Directory containing dumped parameter files",
    )

    args = parser.parse_args()

    success = compare_params(args.dump_dir)
    sys.exit(0 if success else 1)
