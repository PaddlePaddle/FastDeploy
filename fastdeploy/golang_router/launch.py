"""
FastDeploy Golang Router Launcher.

Launches the pre-compiled fd-router binary that is bundled inside the
FastDeploy wheel package.

Usage:
    python -m fastdeploy.golang_router.launch --port 9000
    python -m fastdeploy.golang_router.launch --port 9000 --splitwise
    python -m fastdeploy.golang_router.launch --config_path config.yaml
    python -m fastdeploy.golang_router.launch --version
"""

import argparse
import importlib.resources
import os
import subprocess
import sys


def _get_fd_router_path() -> str:
    """Locate the fd-router binary inside the installed package.

    Uses importlib.resources so it works both with regular installs
    (site-packages) and editable installs.
    """
    res = importlib.resources.files("fastdeploy.golang_router") / "fd-router"
    with importlib.resources.as_file(res) as path:
        binary_path = str(path)

    if not os.path.isfile(binary_path):
        raise FileNotFoundError(
            f"fd-router binary not found at {binary_path}. "
            "Please rebuild FastDeploy with the golang router enabled."
        )

    if not os.access(binary_path, os.X_OK):
        os.chmod(binary_path, 0o755)

    return binary_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser aligned with fd-router flags."""
    parser = argparse.ArgumentParser(
        description="FastDeploy Golang Router - high-performance load balancer",
    )
    parser.add_argument(
        "--port",
        type=str,
        default="",
        help="Listen port of the router (default: from config or 9000)",
    )
    parser.add_argument(
        "--splitwise",
        action="store_true",
        default=False,
        help="Enable splitwise (prefill/decode disaggregated) mode",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="",
        help="Path to the router config YAML file",
    )
    parser.add_argument(
        "--version",
        "-V",
        action="store_true",
        default=False,
        help="Print fd-router version info and exit",
    )
    return parser


def main() -> None:
    """Entry point: parse args and launch fd-router binary."""
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        binary_path = _get_fd_router_path()

        cmd = [binary_path]
        if args.port:
            cmd.extend(["--port", args.port])
        if args.splitwise:
            cmd.append("--splitwise")
        if args.config_path:
            cmd.extend(["--config_path", args.config_path])
        if args.version:
            cmd.append("--version")

        result = subprocess.run(cmd)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        sys.exit(130)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except PermissionError as exc:
        print(
            f"Error: unable to access or execute fd-router binary: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
