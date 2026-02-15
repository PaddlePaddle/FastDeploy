#!/usr/bin/env python3
"""
Module to capture launch parameters from Engine before subprocess.Popen is called.

This module provides utilities to capture the exact command line parameters
that would be passed to worker processes in both old and new architectures.
"""

import json
import os
from pathlib import Path


def save_launch_params(pd_cmd: str, arch_type: str, dump_dir: str = None):
    """
    Save the launch command parameters to a file for verification.

    Args:
        pd_cmd: The full launch command string
        arch_type: Either "old" or "new"
        dump_dir: Directory to save the parameters. Defaults to env var or ./param_dumps
    """
    # Get dump directory
    if dump_dir is None:
        dump_dir = os.getenv("FD_PARAM_DUMP_DIR", "./param_dumps")

    dump_path = Path(dump_dir)
    dump_path.mkdir(parents=True, exist_ok=True)

    # Save the command for reference
    cmd_file = dump_path / f"launch_command_{arch_type}.sh"
    with open(cmd_file, 'w') as f:
        f.write(pd_cmd)

    # Extract and parse the worker arguments
    # Find the worker_process.py part
    if 'worker_process.py' in pd_cmd:
        worker_args_part = pd_cmd.split('worker_process.py')[1]

        # Parse arguments similar to argparse
        import shlex
        tokens = shlex.split(worker_args_part)

        params = {}
        i = 0
        while i < len(tokens):
            token = tokens[i]

            if token.startswith('--'):
                if '=' in token:
                    # --key=value format
                    key, value = token[2:].split('=', 1)
                    # Remove quotes if present
                    value = value.strip()
                    if value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    elif value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    params[key] = parse_value(value)
                elif i + 1 < len(tokens) and not tokens[i + 1].startswith('--'):
                    # --key value format
                    key = token[2:]
                    value = parse_value(tokens[i + 1])
                    params[key] = value
                    i += 1
                else:
                    # --key flag (boolean)
                    params[token[2:]] = True

            i += 1

        # Save parsed parameters
        params_file = dump_path / f"worker_params_{arch_type}.json"
        with open(params_file, 'w') as f:
            json.dump(params, f, indent=2)

        print(f"[LAUNCH_PARAMS] Saved {arch_type.upper()} architecture launch parameters:")
        print(f"  Command: {cmd_file}")
        print(f"  Parameters: {params_file}")
        print(f"  Total params: {len(params)}")

    return True


def parse_value(value: str):
    """Parse a value string, handling JSON strings and numbers."""
    value = value.strip()

    # Try to parse as JSON first
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try to parse as number
    try:
        if '.' in value:
            return float(value)
        return int(value)
    except ValueError:
        pass

    # Return as string
    return value


def should_capture_launch_params() -> bool:
    """Check if launch parameter capture is enabled."""
    return os.getenv("FD_CAPTURE_LAUNCH_PARAMS", "0") == "1"


# Convenience function for integration
def capture_if_enabled(pd_cmd: str, arch_type: str, dump_dir: str = None):
    """
    Capture launch parameters if enabled.

    Args:
        pd_cmd: The full launch command string
        arch_type: Either "old" or "new"
        dump_dir: Directory to save the parameters

    Returns:
        True if captured, False otherwise
    """
    if should_capture_launch_params():
        return save_launch_params(pd_cmd, arch_type, dump_dir)
    return False
