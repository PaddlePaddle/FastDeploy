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


def check_package_version(package_name, required_version):
    """Check if a package is installed with the required version.

    Args:
        package_name: Name of the package
        required_version: Required version string (e.g., "1.1.0.dev20250507")

    Returns:
        bool: True if package is installed with required version, False otherwise
    """
    try:
        import importlib.metadata

        installed_version = importlib.metadata.version(package_name)

        # For dev versions, do exact match
        if installed_version == required_version:
            return True

        # Also accept if major.minor.patch matches (ignore post/dev suffixes)
        # e.g., "1.1.1.post20260401" matches "1.1.1"
        if required_version in installed_version:
            return True

        return False
    except Exception:
        return False


def print_package_versions():
    """Print versions of key packages (paddlepaddle, paddlefleet, paddleformers)."""
    print("\n" + "=" * 70)
    print("[conftest] Package Versions:")
    print("=" * 70)

    packages = ["paddlepaddle-gpu", "paddlefleet", "paddleformers"]
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

    if not has_paddlefleet_tests:
        return

    # Check if dependencies are already installed with correct versions
    # Define required versions
    REQUIRED_PADDLEFLEET_VERSION = "0.3.0.dev20260527"
    REQUIRED_PADDLEFORMERS_VERSION = "1.1.0.dev20250507"

    paddlefleet_ok = check_package_version("paddlefleet", REQUIRED_PADDLEFLEET_VERSION)
    paddleformers_ok = check_package_version("paddleformers", REQUIRED_PADDLEFORMERS_VERSION)

    if paddlefleet_ok and paddleformers_ok:
        print("\n" + "=" * 70)
        print("[conftest] paddlefleet and paddleformers already installed with required versions")
        print("=" * 70)
        print_package_versions()
        return

    # If versions don't match, show what needs to be installed
    if not paddlefleet_ok:
        print("\n[conftest] paddlefleet version mismatch or not installed")
        print(f"  Required: {REQUIRED_PADDLEFLEET_VERSION}")
        print(f"  Current: {get_package_version('paddlefleet')}")

    if not paddleformers_ok:
        print("[conftest] paddleformers version mismatch or not installed")
        print(f"  Required: {REQUIRED_PADDLEFORMERS_VERSION}")
        print(f"  Current: {get_package_version('paddleformers')}")

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
        # Install paddleformers if needed
        if not paddleformers_ok:
            paddleformers_url = os.getenv(
                "PADDLEFORMERS_WHEEL_URL",
                f"paddleformers=={REQUIRED_PADDLEFORMERS_VERSION} --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/ --extra-index-url https://www.paddlepaddle.org.cn/packages/nightly/cu126/",
            )

            # Split the string into separate arguments (handles --extra-index-url flags)
            install_args = (
                [sys.executable, "-m", "pip", "install"] + shlex.split(paddleformers_url) + ["--no-deps", "--quiet"]
            )
            subprocess.check_call(install_args)
            print(f"[conftest] ✓ Installed paddleformers (--no-deps) from {paddleformers_url}")
        else:
            print(f"[conftest] ℹ paddleformers {REQUIRED_PADDLEFORMERS_VERSION} already satisfied")

        # Install paddlefleet if needed
        if not paddlefleet_ok:
            paddlefleet_url = os.getenv("PADDLEFLEET_WHEEL_URL", f"paddlefleet=={REQUIRED_PADDLEFLEET_VERSION}")

            # Use --no-deps to avoid reinstalling paddlepaddle
            # Split in case the URL contains spaces or flags
            install_args = (
                [sys.executable, "-m", "pip", "install"] + shlex.split(paddlefleet_url) + ["--no-deps", "--quiet"]
            )
            subprocess.check_call(install_args)
            print(f"[conftest] ✓ Installed paddlefleet (--no-deps) from {paddlefleet_url}")
        else:
            print(f"[conftest] ℹ paddlefleet {REQUIRED_PADDLEFLEET_VERSION} already satisfied")
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
