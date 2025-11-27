"""
Unit tests for metrics routes on the main API port (no --metrics-port set).
Mimics the patching pattern used by other tests under tests/entrypoints/openai.
"""

import asyncio
import importlib
import json
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


def test_metrics_and_config_routes():
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

            # 1) /metrics
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

            # 2) /config-info
            # Inject a fake engine so /config-info returns 200
            from types import SimpleNamespace as NS

            api_server.llm_engine = NS(cfg=NS(dummy="value"))

            cfg_route = _get_route(api_server.app, "/config-info")
            assert cfg_route is not None

            cfg_resp = cfg_route.endpoint()
            assert cfg_resp.status_code == 200
            assert getattr(cfg_resp, "media_type", "").startswith("application/json")
            cfg_text = (
                cfg_resp.body.decode("utf-8") if isinstance(cfg_resp.body, (bytes, bytearray)) else str(cfg_resp.body)
            )
            data = json.loads(cfg_text)
            assert isinstance(data, dict)
            assert "env_config" in data


def test_config_info_engine_not_loaded_returns_500():
    # Ensure we take the branch where llm_engine is None
    with (
        patch("fastdeploy.utils.FlexibleArgumentParser.parse_args") as mock_parse_args,
        patch("fastdeploy.utils.retrive_model_from_server") as mock_retrive_model,
        patch("fastdeploy.entrypoints.chat_utils.load_chat_template") as mock_load_template,
    ):
        mock_parse_args.return_value = _build_mock_args()
        mock_retrive_model.return_value = "test-model"
        mock_load_template.return_value = None

        from fastdeploy.entrypoints.openai import api_server as api_server_mod

        api_server = importlib.reload(api_server_mod)

        # Fresh import sets llm_engine to None
        cfg_route = _get_route(api_server.app, "/config-info")
        assert cfg_route is not None

        resp = cfg_route.endpoint()
        assert resp.status_code == 500
        # message body is simple text
        assert b"Engine not loaded" in getattr(resp, "body", b"")


def test_config_info_process_object_branches():
    # Cover forcing json default() to handle
    # both an object with __dict__ and one without.
    with (
        patch("fastdeploy.utils.FlexibleArgumentParser.parse_args") as mock_parse_args,
        patch("fastdeploy.utils.retrive_model_from_server") as mock_retrive_model,
        patch("fastdeploy.entrypoints.chat_utils.load_chat_template") as mock_load_template,
    ):
        mock_parse_args.return_value = _build_mock_args()
        mock_retrive_model.return_value = "test-model"
        mock_load_template.return_value = None

        from fastdeploy.entrypoints.openai import api_server as api_server_mod

        api_server = importlib.reload(api_server_mod)

        # Build a cfg with values that exercise both branches of process_object()
        class WithDict:
            pass

        has_dict = WithDict()
        has_dict.a = 1
        no_dict = object()

        from types import SimpleNamespace as NS

        api_server.llm_engine = NS(cfg=NS(with_dict=has_dict, without_dict=no_dict))

        cfg_route = _get_route(api_server.app, "/config-info")
        assert cfg_route is not None

        resp = cfg_route.endpoint()
        assert resp.status_code == 200
        data = json.loads(resp.body.decode("utf-8"))
        # The object with __dict__ becomes its dict; the one without becomes null
        assert data.get("with_dict") == {"a": 1}
        assert "without_dict" in data and data["without_dict"] is None


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


def test_metrics_app_config_info_branches():
    # Cover metrics_app '/config-info' 500 branch and success path
    # including process_object branches and response
    with (
        patch("fastdeploy.utils.FlexibleArgumentParser.parse_args") as mock_parse_args,
        patch("fastdeploy.utils.retrive_model_from_server") as mock_retrive_model,
        patch("fastdeploy.entrypoints.chat_utils.load_chat_template") as mock_load_template,
    ):
        mock_parse_args.return_value = _build_mock_args_with_side_metrics()
        mock_retrive_model.return_value = "test-model"
        mock_load_template.return_value = None

        from fastdeploy.entrypoints.openai import api_server as api_server_mod

        api_server = importlib.reload(api_server_mod)

        # First, llm_engine is None -> 500
        cfg_route = _get_route(api_server.metrics_app, "/config-info")
        assert cfg_route is not None
        resp = cfg_route.endpoint()
        assert resp.status_code == 500

        # Then set a fake engine with cfg carrying both serializable and non-serializable objects
        class WithDict:
            pass

        has_dict = WithDict()
        has_dict.x = 42
        no_dict = object()

        from types import SimpleNamespace as NS

        api_server.llm_engine = NS(cfg=NS(with_dict=has_dict, without_dict=no_dict))

        resp2 = cfg_route.endpoint()
        assert resp2.status_code == 200
        data = json.loads(resp2.body.decode("utf-8"))
        assert data.get("with_dict") == {"x": 42}
        assert "without_dict" in data and data["without_dict"] is None
        assert "env_config" in data
