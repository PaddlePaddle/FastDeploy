"""
Configuration for trace.py tests
"""

import os
import sys

import pytest

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@pytest.fixture(autouse=True)
def setup_tracing_environment():
    """Setup and teardown for tracing tests"""
    # Save original environment
    original_env = os.environ.copy()

    # Setup test environment variables
    os.environ.update(
        {
            "TRACES_ENABLE": "true",
            "FD_SERVICE_NAME": "test_service",
            "EXPORTER_OTLP_ENDPOINT": "http://localhost:4317",
            "FD_HOST_NAME": "test_host",
        }
    )

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)
