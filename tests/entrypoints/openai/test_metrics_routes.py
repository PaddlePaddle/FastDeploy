"""
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
        task=None,
        model_config_name=None,
        tool_call_parser=None,
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
        assert "without_dict" in data and isinstance(data["without_dict"], str)


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
        assert "without_dict" in data and isinstance(data["without_dict"], str)
        assert "env_config" in data


def _reload_api_server():
    """Helper: reload api_server with standard mocks, return the module."""
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
    return api_server


def test_config_info_server_config_matches_args():
    """Verify server_config values are populated from args."""
    api_server = _reload_api_server()
    from types import SimpleNamespace as NS

    api_server.llm_engine = NS(cfg=NS())

    resp = _get_route(api_server.app, "/config-info").endpoint()
    assert resp.status_code == 200
    data = json.loads(resp.body.decode("utf-8"))

    sc = data["server_config"]
    assert sc["host"] == "0.0.0.0"
    assert sc["port"] == 8000
    assert sc["workers"] == 1
    assert sc["metrics_port"] is None
    assert sc["controller_port"] == -1
    assert sc["max_concurrency"] == 16
    assert sc["max_waiting_time"] == -1
    assert sc["timeout"] == 0
    assert sc["timeout_graceful_shutdown"] == 0
    assert sc["served_model_name"] is None
    assert sc["task"] is None
    assert sc["model_config_name"] is None
    assert sc["tokenizer_base_url"] is None
    assert sc["enable_mm_output"] is False
    assert sc["tool_call_parser"] is None
    assert sc["tool_parser_plugin"] is None


def test_config_info_top_level_fields():
    """Verify version_info, chat_template, device_info, env_config all present."""
    api_server = _reload_api_server()
    from types import SimpleNamespace as NS

    api_server.llm_engine = NS(cfg=NS(key="val"))

    resp = _get_route(api_server.app, "/config-info").endpoint()
    data = json.loads(resp.body.decode("utf-8"))

    assert "version_info" in data
    assert "chat_template" in data
    assert "device_info" in data
    assert "env_config" in data
    assert isinstance(data["env_config"], dict)
    # cfg field should propagate
    assert data["key"] == "val"


def test_config_info_process_object_set_and_frozenset():
    """Cover process_object branch for set/frozenset -> list."""
    api_server = _reload_api_server()
    from types import SimpleNamespace as NS

    api_server.llm_engine = NS(
        cfg=NS(
            my_set={3, 1, 2},
            my_frozenset=frozenset(["b", "a"]),
        )
    )

    resp = _get_route(api_server.app, "/config-info").endpoint()
    assert resp.status_code == 200
    data = json.loads(resp.body.decode("utf-8"))

    assert isinstance(data["my_set"], list)
    assert sorted(data["my_set"]) == [1, 2, 3]
    assert isinstance(data["my_frozenset"], list)
    assert sorted(data["my_frozenset"]) == ["a", "b"]


def test_config_info_non_ascii_content():
    """Cover ensure_ascii=False path with unicode in cfg."""
    api_server = _reload_api_server()
    from types import SimpleNamespace as NS

    api_server.llm_engine = NS(cfg=NS(desc="中文描述", emoji="🚀"))

    resp = _get_route(api_server.app, "/config-info").endpoint()
    assert resp.status_code == 200
    raw = resp.body.decode("utf-8")
    # Non-ASCII chars should appear directly, not as \uXXXX escapes
    assert "中文描述" in raw
    assert "🚀" in raw
    data = json.loads(raw)
    assert data["desc"] == "中文描述"
    assert data["emoji"] == "🚀"


def test_config_info_cfg_fields_propagated():
    """Verify that all cfg.__dict__ entries end up in the response."""
    api_server = _reload_api_server()
    from types import SimpleNamespace as NS

    api_server.llm_engine = NS(
        cfg=NS(
            model_name="Qwen-7B",
            max_seq_len=4096,
            use_fp16=True,
            parallel_config=None,
        )
    )

    resp = _get_route(api_server.app, "/config-info").endpoint()
    data = json.loads(resp.body.decode("utf-8"))

    assert data["model_name"] == "Qwen-7B"
    assert data["max_seq_len"] == 4096
    assert data["use_fp16"] is True
    assert data["parallel_config"] is None


def test_config_info_nested_objects():
    """Cover process_object with nested custom objects."""
    api_server = _reload_api_server()
    from types import SimpleNamespace as NS

    class Inner:
        pass

    inner = Inner()
    inner.lr = 0.01
    inner.steps = 100

    api_server.llm_engine = NS(cfg=NS(train_config=inner))

    resp = _get_route(api_server.app, "/config-info").endpoint()
    assert resp.status_code == 200
    data = json.loads(resp.body.decode("utf-8"))

    assert data["train_config"] == {"lr": 0.01, "steps": 100}
