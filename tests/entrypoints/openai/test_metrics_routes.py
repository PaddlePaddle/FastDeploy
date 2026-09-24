"""
Unit tests for metrics routes on the main API port (no --metrics-port set).
Mimics the patching pattern used by other tests under tests/entrypoints/openai.
"""

import asyncio
import importlib
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch


def _build_mock_args():
    # Provide all attributes used at import time by api_server
    return SimpleNamespace(
        # basic
        workers=1,
        model="test-model",
        revision=None,
        chat_template=None,
        tool_parser_plugin=None,
        # server/network
        host="0.0.0.0",
        port=8000,
        metrics_port=None,  # key: not set -> metrics on main port
        controller_port=-1,
        # concurrency & limits
        max_concurrency=16,
        max_model_len=32768,
        max_waiting_time=-1,
        # distributed/engine args referenced during import
        tensor_parallel_size=1,
        data_parallel_size=1,
        enable_logprob=False,
        enable_prefix_caching=False,
        splitwise_role=None,
        max_processor_cache=0,
        # optional API key list
        api_key=None,
        # timeout args for gunicorn
        timeout_graceful_shutdown=0,
        timeout=0,
        # misc used later but safe defaults
        tokenizer=None,
        served_model_name=None,
        ips=None,
        enable_mm_output=False,
        tokenizer_base_url=None,
        dynamic_load_weight=False,
        reasoning_parser=None,
    )


def _build_mock_args_with_side_metrics():
    args = _build_mock_args()
    # Force metrics served on the side metrics_app (different port)
    args.metrics_port = args.port + 1
    return args


def _get_route(app, path: str):
    for r in getattr(app, "routes", []):
        if getattr(r, "path", "") == path and "GET" in getattr(r, "methods", {"GET"}):
            return r
    return None


def test_metrics_route():
    with (
        patch("fastdeploy.utils.FlexibleArgumentParser.parse_args") as mock_parse_args,
        patch("fastdeploy.utils.retrive_model_from_server") as mock_retrive_model,
        patch("fastdeploy.entrypoints.chat_utils.load_chat_template") as mock_load_template,
    ):
        mock_parse_args.return_value = _build_mock_args()
        mock_retrive_model.return_value = "test-model"
        mock_load_template.return_value = None

        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["PROMETHEUS_MULTIPROC_DIR"] = tmpdir

            from fastdeploy.entrypoints.openai import api_server as api_server_mod

            api_server = importlib.reload(api_server_mod)

            from fastdeploy.metrics import metrics as metrics_mod

            if not hasattr(metrics_mod.main_process_metrics, "cache_config_info"):
                metrics_mod.main_process_metrics.cache_config_info = None
            metrics_route = _get_route(api_server.app, "/metrics")
            assert metrics_route is not None
            metrics_resp = asyncio.run(metrics_route.endpoint())
            assert getattr(metrics_resp, "media_type", "").startswith("text/plain")
            metrics_text = (
                metrics_resp.body.decode("utf-8")
                if isinstance(metrics_resp.body, (bytes, bytearray))
                else str(metrics_resp.body)
            )
            assert "fastdeploy:" in metrics_text


def test_metrics_app_routes_when_metrics_port_diff():
    # Cover metrics_app '/metrics'
    with (
        patch("fastdeploy.utils.FlexibleArgumentParser.parse_args") as mock_parse_args,
        patch("fastdeploy.utils.retrive_model_from_server") as mock_retrive_model,
        patch("fastdeploy.entrypoints.chat_utils.load_chat_template") as mock_load_template,
    ):
        mock_parse_args.return_value = _build_mock_args_with_side_metrics()
        mock_retrive_model.return_value = "test-model"
        mock_load_template.return_value = None

        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["PROMETHEUS_MULTIPROC_DIR"] = tmpdir

            from fastdeploy.entrypoints.openai import api_server as api_server_mod

            api_server = importlib.reload(api_server_mod)

            metrics_route = _get_route(api_server.metrics_app, "/metrics")
            assert metrics_route is not None
            resp = asyncio.run(metrics_route.endpoint())
            assert getattr(resp, "media_type", "").startswith("text/plain")
            text = resp.body.decode("utf-8") if isinstance(resp.body, (bytes, bytearray)) else str(resp.body)
            assert "fastdeploy:" in text
