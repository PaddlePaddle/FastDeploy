"""Pytest configuration for model_executor tests.

This conftest handles special dependencies required only by specific tests,
avoiding pollution of the global test environment.
"""

import os
import shlex
import subprocess
import sys

import pytest


def get_package_version(package_name):
    """Get the version of an installed package.

    Args:
        package_name: Name of the package

    Returns:
        Version string or "not installed" if package is not found
    """
    try:
        import importlib.metadata

        version = importlib.metadata.version(package_name)
        return version
    except Exception:
        try:
            # Fallback for older Python versions
            import pkg_resources

            version = pkg_resources.get_distribution(package_name).version
            return version
        except Exception:
            return "not installed"


def print_package_versions():
    """Print versions of key packages (paddlepaddle, paddlefleet, paddleformers)."""
    print("\n" + "=" * 70)
    print("[conftest] Package Versions:")
    print("=" * 70)

    packages = ["paddlepaddle-gpu", "paddlefleet", "paddleformers", "transformers"]
    for pkg in packages:
        version = get_package_version(pkg)
        status = "✓" if version != "not installed" else "✗"
        print(f"[conftest] {status} {pkg:20s}: {version}")

    print("=" * 70 + "\n")


def pytest_configure(config):
    """Configure pytest before test collection."""
    # Register custom marker for paddlefleet tests
    config.addinivalue_line("markers", "paddlefleet: tests that require paddlefleet and paddleformers dependencies")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle paddlefleet dependencies.

    This hook runs after test collection but before test execution.
    It checks if any collected tests require paddlefleet dependencies
    and installs them in an isolated manner if needed.
    """
    # Check if any test in this session requires paddlefleet
    has_paddlefleet_tests = any("test_fallback_fleet_model.py" in item.nodeid for item in items)
    print("has_paddlefleet_tests:", has_paddlefleet_tests)
    if not has_paddlefleet_tests:
        return

    # Check if dependencies are already installed with correct versions
    try:
        import paddlefleet  # noqa: F401

        print("\n" + "=" * 70)
        print("[conftest] paddlefleet already installed, skipping installation")
        print("=" * 70)
        print_package_versions()
        return
    except ImportError:
        pass

    # Print versions before installation
    print("\n" + "=" * 70)
    print("[conftest] Package versions BEFORE installing paddlefleet dependencies:")
    print("=" * 70)
    print_package_versions()

    # Install dependencies only when running paddlefleet tests
    print("=" * 70)
    print("[conftest] Installing paddlefleet-specific dependencies...")
    print("=" * 70)

    try:
        # Install paddleformers
        paddleformers_url = os.getenv(
            "PADDLEFORMERS_WHEEL_URL",
            "paddleformers==1.1.0.dev20260507 --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/ --extra-index-url https://www.paddlepaddle.org.cn/packages/nightly/cu126/",  # fallback to PyPI name
        )
        install_args = [sys.executable, "-m", "pip", "install"] + shlex.split(paddleformers_url) + ["--quiet"]
        subprocess.check_call(install_args)
        print(f"[conftest] ✓ Installed paddleformers 1.1.0.dev20250507 from {paddleformers_url}")

        # Install paddlefleet (skip paddlepaddle dependency, use existing version)
        paddlefleet_url = os.getenv(
            "PADDLEFLEET_WHEEL_URL",
            "paddlefleet==0.3.0.dev20260527 --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/ --extra-index-url https://www.paddlepaddle.org.cn/packages/nightly/cu126/",  # fallback to PyPI name
        )
        # Use --no-deps to avoid reinstalling paddlepaddle
        install_args = (
            [sys.executable, "-m", "pip", "install"] + shlex.split(paddlefleet_url) + ["--no-deps", "--quiet"]
        )
        subprocess.check_call(install_args)
        print(f"[conftest] ✓ Installed paddlefleet 0.3.0.dev20260527 (--no-deps) from {paddlefleet_url}")
        print("[conftest] ℹ Using existing paddlepaddle from environment")

        # Print versions after installation
        print("\n" + "=" * 70)
        print("[conftest] Package versions AFTER installing paddlefleet dependencies:")
        print("=" * 70)
        print_package_versions()

    except subprocess.CalledProcessError as e:
        print(f"[conftest] ✗ Failed to install dependencies: {e}")
        print("[conftest] Tests requiring paddlefleet will be skipped")

        # Mark all paddlefleet tests to skip
        skip_marker = pytest.mark.skip(reason="Failed to install paddlefleet dependencies")
        for item in items:
            if "test_fallback_fleet_model.py" in item.nodeid:
                item.add_marker(skip_marker)

    print("=" * 70 + "\n")


def pytest_sessionfinish(session, exitstatus):
    """Optional: cleanup after test session if needed.

    You can uninstall the dependencies here to keep the environment clean,
    but this may slow down subsequent test runs.
    """
    # Uncomment the following to auto-cleanup after tests
    # if os.getenv("CLEANUP_PADDLEFLEET_DEPS", "false").lower() == "true":
    #     try:
    #         subprocess.check_call([
    #             sys.executable, "-m", "pip", "uninstall",
    #             "paddlefleet", "paddleformers", "-y", "--quiet"
    #         ])
    #         print("[conftest] Cleaned up paddlefleet dependencies")
    #     except Exception:
    #         pass
    pass
