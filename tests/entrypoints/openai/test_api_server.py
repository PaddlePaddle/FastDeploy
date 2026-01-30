import asyncio
import importlib
import sys
import types
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Shared test fixtures
from fastdeploy.entrypoints.openai.protocol import (
    ChatCompletionResponse,
    CompletionResponse,
    ErrorInfo,
    ErrorResponse,
    ModelInfo,
    ModelList,
    UsageInfo,
)


class DummyErrorInfo:
    def __init__(self, message: str, code=None, **_):
        self.message = message
        self.code = str(code) if code is not None else code


class DummyErrorResponse:
    def __init__(self, error):
        self.error = error

    def model_dump(self):
        return {"error": {"message": self.error.message, "code": self.error.code}}


def _build_args(**overrides):
    """Return a SimpleNamespace with all attributes accessed at import time."""
    base = dict(
        workers=1,
        model="test-model",
        revision=None,
        chat_template=None,
        tool_parser_plugin=None,
        host="0.0.0.0",
        port=9000,
        metrics_port=None,
        controller_port=-1,
        max_concurrency=4,
        max_model_len=1024,
        max_waiting_time=-1,
        max_logprobs=0,
        tensor_parallel_size=1,
        data_parallel_size=1,
        max_num_seqs=8,
        api_key=None,
        tokenizer=None,
        served_model_name=None,
        ips=None,
        enable_mm_output=False,
        tokenizer_base_url=None,
        dynamic_load_weight=False,
        timeout_graceful_shutdown=0,
        timeout=0,
        local_data_parallel_id=0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _reload_api_server(args):
    """Import/reload api_server with patched dependencies."""
    fake_envs_mod = types.ModuleType("fastdeploy.envs")

    class _FakeEnvVars:
        @staticmethod
        def get(key, default=None):
            return [] if key == "FD_API_KEY" else default

    fake_envs_mod.TRACES_ENABLE = "false"
    fake_envs_mod.FD_SERVICE_NAME = ""
    fake_envs_mod.FD_HOST_NAME = ""
    fake_envs_mod.TRACES_EXPORTER = "console"
    fake_envs_mod.EXPORTER_OTLP_ENDPOINT = ""
    fake_envs_mod.EXPORTER_OTLP_HEADERS = ""
    fake_envs_mod.FD_SUPPORT_MAX_CONNECTIONS = 1024
    fake_envs_mod.environment_variables = _FakeEnvVars()

    # Save original sys.argv and replace with minimal valid args to avoid parse errors
    original_argv = sys.argv[:]
    sys.argv = ["api_server.py", "--model", "test-model", "--port", "9000"]

    try:
        with (
            patch("fastdeploy.utils.FlexibleArgumentParser.parse_args", return_value=args),
            patch("fastdeploy.utils.retrive_model_from_server", return_value=args.model),
            patch("fastdeploy.entrypoints.chat_utils.load_chat_template", return_value=None),
            patch.dict("sys.modules", {"fastdeploy.envs": fake_envs_mod}),
            patch("fastdeploy.envs", fake_envs_mod),
        ):
            from fastdeploy.entrypoints.openai import api_server as api_server_mod

            return importlib.reload(api_server_mod)
    finally:
        sys.argv = original_argv


def _dummy_engine_args(config_parallel_id=0):
    cfg = SimpleNamespace(parallel_config=SimpleNamespace(local_data_parallel_id=config_parallel_id))

    class DummyArgs:
        def create_engine_config(self, port_availability_check=True):
            return cfg

    return DummyArgs()


def _dummy_engine_client():
    class DummyConnMgr:
        async def initialize(self):
            pass

        async def close(self):
            pass

    class DummyClient:
        def __init__(self, *_, **__):
            self.connection_manager = DummyConnMgr()
            self.zmq_client = SimpleNamespace(close=lambda: None)
            self.data_processor = "dp"
            self.pid = None

        def create_zmq_client(self, *_, **__):
            self.zmq_client = SimpleNamespace(close=lambda: None)

        def check_health(self):
            return True, "ok"

        def is_workers_alive(self):
            return True, "ok"

        async def rearrange_experts(self, request_dict):
            return {"data": request_dict}, 201

        async def get_per_expert_tokens_stats(self, request_dict):
            return {"stats": request_dict}, 202

        async def check_redundant(self, request_dict):
            return {"redundant": request_dict}, 203

    return DummyClient


def _fake_handlers():
    class Handler:
        def __init__(self, *_, **__):
            pass

        async def create_chat_completion(self, *args, **kwargs):
            return args[0] if args else None

        async def create_completion(self, *args, **kwargs):
            return args[0] if args else None

        async def create_embedding(self, req):
            return SimpleNamespace(model_dump=lambda: {"emb": True})

        async def create_reward(self, req):
            return SimpleNamespace(model_dump=lambda: {"reward": True})

        async def list_models(self):
            return SimpleNamespace(model_dump=lambda: {"list": True})

    return Handler


def _patch_common_imports(args, engine_client_cls=None, handler_cls=None):
    engine_client_cls = engine_client_cls or _dummy_engine_client()
    handler_cls = handler_cls or _fake_handlers()

    fake_paddle = types.ModuleType("paddle")
    fake_prom = types.ModuleType("prometheus_client")
    fake_prom.multiprocess = SimpleNamespace(mark_process_dead=lambda *_: None)
    fake_metrics = types.ModuleType("fastdeploy.metrics.metrics")
    fake_metrics.get_filtered_metrics = lambda: ""
    fake_metrics_pkg = types.ModuleType("fastdeploy.metrics")
    fake_metrics_pkg.metrics = fake_metrics
    fake_zmq = types.ModuleType("zmq")
    fake_zmq.PUSH = "PUSH"

    stack = ExitStack()
    stack.enter_context(patch.dict("sys.modules", {"paddle": fake_paddle}))
    stack.enter_context(patch.dict("sys.modules", {"prometheus_client": fake_prom}))
    stack.enter_context(patch.dict("sys.modules", {"fastdeploy.metrics": fake_metrics_pkg}))
    stack.enter_context(patch.dict("sys.modules", {"fastdeploy.metrics.metrics": fake_metrics}))
    stack.enter_context(patch.dict("sys.modules", {"zmq": fake_zmq}))
    stack.enter_context(
        patch("fastdeploy.entrypoints.openai.api_server.EngineArgs.from_cli_args", return_value=_dummy_engine_args())
    )
    stack.enter_context(patch("fastdeploy.entrypoints.openai.api_server.EngineClient", engine_client_cls))
    stack.enter_context(patch("fastdeploy.entrypoints.openai.api_server.OpenAIServingModels", handler_cls))
    stack.enter_context(patch("fastdeploy.entrypoints.openai.api_server.OpenAIServingChat", handler_cls))
    stack.enter_context(patch("fastdeploy.entrypoints.openai.api_server.OpenAIServingCompletion", handler_cls))
    stack.enter_context(patch("fastdeploy.entrypoints.openai.api_server.OpenAIServingEmbedding", handler_cls))
    stack.enter_context(patch("fastdeploy.entrypoints.openai.api_server.OpenAIServingReward", handler_cls))
    stack.enter_context(patch("fastdeploy.entrypoints.openai.api_server.ToolParserManager.import_tool_parser"))
    return stack


def test_tool_parser_and_load_engine_branches():
    args = _build_args(tool_parser_plugin="plugin")
    with (
        patch("fastdeploy.utils.FlexibleArgumentParser.parse_args", return_value=args),
        patch("fastdeploy.utils.retrive_model_from_server", return_value=args.model),
        patch("fastdeploy.entrypoints.chat_utils.load_chat_template", return_value=None),
        patch("fastdeploy.entrypoints.openai.api_server.ToolParserManager.import_tool_parser") as import_mock,
        patch("fastdeploy.entrypoints.openai.api_server.LLMEngine.from_engine_args") as llm_from_args,
        patch("fastdeploy.entrypoints.openai.api_server.EngineArgs.from_cli_args", return_value=_dummy_engine_args()),
    ):
        from fastdeploy.entrypoints.openai import api_server as api_server_mod

        api_server = importlib.reload(api_server_mod)
        import_mock.assert_called_once()

        api_server.llm_engine = "cached"
        assert api_server.load_engine() == "cached"

        api_server.llm_engine = None
        llm_from_args.return_value = SimpleNamespace(start=MagicMock(return_value=False))
        assert api_server.load_engine() is None

    with patch.object(api_server_mod.BaseApplication, "__init__", return_value=None):
        app_instance = api_server_mod.StandaloneApplication("app", {"bind": "0.0.0.0:1", "unused": None})
        app_instance.cfg = SimpleNamespace(settings={"bind": True})
        app_instance.cfg.set = MagicMock()
        app_instance.load_config()
        app_instance.cfg.set.assert_called_once()
        assert app_instance.load() == "app"


def test_load_data_service_branches():
    args = _build_args()
    api_server = _reload_api_server(args)
    cfg = SimpleNamespace(parallel_config=SimpleNamespace(local_data_parallel_id=1))
    engine_args = SimpleNamespace(create_engine_config=lambda: cfg)
    expert = MagicMock()
    expert.start.side_effect = [False, True]

    with (
        patch("fastdeploy.entrypoints.openai.api_server.EngineArgs.from_cli_args", return_value=engine_args),
        patch("fastdeploy.entrypoints.openai.api_server.ExpertService", return_value=expert),
    ):
        api_server.llm_engine = None
        assert api_server.load_data_service() is None
        api_server.llm_engine = None
        assert api_server.load_data_service() is expert
        assert api_server.load_data_service() is expert


@pytest.mark.asyncio
async def test_connection_manager_timeout_branch():
    args = _build_args()
    api_server = _reload_api_server(args)

    class SlowSemaphore:
        async def acquire(self):
            await asyncio.sleep(0.01)

        def status(self):
            return "busy"

    with patch("fastdeploy.entrypoints.openai.api_server.connection_semaphore", SlowSemaphore()):
        with pytest.raises(api_server.HTTPException) as exc:
            async with api_server.connection_manager():
                pass
        assert exc.value.status_code == 429


def test_health_and_routes():
    args = _build_args()
    api_server = _reload_api_server(args)
    engine_client = MagicMock()
    engine_client.check_health.return_value = (True, "ok")
    engine_client.is_workers_alive.return_value = (False, "dead")
    api_server.app.state.engine_client = engine_client

    assert api_server.health(MagicMock()).status_code == 304
    assert api_server.ping(MagicMock()).status_code == 304

    engine_client.is_workers_alive.return_value = (True, "ok")
    assert api_server.health(MagicMock()).status_code == 200

    routes = asyncio.run(api_server.list_all_routes())
    assert isinstance(routes, dict) and routes["routes"]


@pytest.mark.asyncio
async def test_wrap_streaming_generator():
    args = _build_args()
    api_server = _reload_api_server(args)
    sem = MagicMock()

    # Error path with span
    span = MagicMock()
    span.is_recording.return_value = True
    with (
        patch("opentelemetry.trace.get_current_span", return_value=span),
        patch("fastdeploy.entrypoints.openai.api_server.connection_semaphore", sem),
    ):

        async def gen():
            yield "first"
            raise RuntimeError("boom")

        wrapped = api_server.wrap_streaming_generator(gen())
        with pytest.raises(RuntimeError):
            async for _ in wrapped():
                pass
    span.record_exception.assert_called()
    sem.release.assert_called_once()

    # Success path without span
    api_server.connection_semaphore = SimpleNamespace(status=lambda: "ok", release=MagicMock())
    with patch("fastdeploy.entrypoints.openai.api_server.trace.get_current_span", return_value=None):

        async def gen2():
            yield "a"
            yield "b"

        wrapped = api_server.wrap_streaming_generator(gen2())
        out = []
        async for item in wrapped():
            out.append(item)
    assert out == ["a", "b"]
    api_server.connection_semaphore.release.assert_called_once()

    # Success path with span and last_chunk event (count > 0)
    span = MagicMock()
    span.is_recording.return_value = True
    api_server.connection_semaphore = SimpleNamespace(status=lambda: "ok", release=MagicMock())
    with patch("opentelemetry.trace.get_current_span", return_value=span):

        async def gen3():
            yield "chunk1"
            yield "chunk2"

        wrapped = api_server.wrap_streaming_generator(gen3())
        out = []
        async for item in wrapped():
            out.append(item)
        assert out == ["chunk1", "chunk2"]
        span.add_event.assert_called()
        api_server.connection_semaphore.release.assert_called_once()


@pytest.mark.asyncio
async def test_chat_and_completion_routes():
    args = _build_args(dynamic_load_weight=True)
    api_server = _reload_api_server(args)
    api_server.app.state.dynamic_load_weight = True
    api_server.app.state.engine_client = MagicMock()
    api_server.app.state.engine_client.is_workers_alive.return_value = (False, "down")
    fake_req = SimpleNamespace(headers={})
    body = SimpleNamespace(model_dump_json=lambda: "{}", stream=False)

    # Unhealthy path
    resp = await api_server.create_chat_completion(body, fake_req)
    assert resp.status_code == 304
    resp = await api_server.create_completion(body, fake_req)
    assert resp.status_code == 304

    # Healthy path with dynamic_load_weight=True (missing branch 383, 419)
    api_server.app.state.dynamic_load_weight = True
    api_server.app.state.engine_client.is_workers_alive.return_value = (True, "ok")
    api_server.connection_semaphore = SimpleNamespace(acquire=AsyncMock(), release=MagicMock(), status=lambda: "ok")
    success_resp = ChatCompletionResponse(id="1", model="m", choices=[], usage=UsageInfo())
    api_server.app.state.chat_handler = SimpleNamespace(create_chat_completion=AsyncMock(return_value=success_resp))
    api_server.app.state.completion_handler = SimpleNamespace(create_completion=AsyncMock(return_value=success_resp))

    class DummyCM:
        async def __aenter__(self):
            return None

        async def __aexit__(self, exc_type, exc, tb):
            return False

    with patch("fastdeploy.entrypoints.openai.api_server.connection_manager", return_value=DummyCM()):
        resp = await api_server.create_chat_completion(body, fake_req)
        assert resp.status_code == 200
        resp = await api_server.create_completion(body, fake_req)
        assert resp.status_code == 200

    # Healthy paths
    api_server.app.state.dynamic_load_weight = False
    api_server.connection_semaphore = SimpleNamespace(acquire=AsyncMock(), release=MagicMock(), status=lambda: "ok")

    error_resp = ErrorResponse(error=ErrorInfo(message="err"))
    chat_handler = MagicMock()
    chat_handler.create_chat_completion = AsyncMock(return_value=error_resp)
    api_server.app.state.chat_handler = chat_handler

    with patch("fastdeploy.entrypoints.openai.api_server.connection_manager", return_value=DummyCM()):
        assert (await api_server.create_chat_completion(body, fake_req)).status_code == 500

    success_resp = ChatCompletionResponse(id="1", model="m", choices=[], usage=UsageInfo())
    api_server.app.state.chat_handler.create_chat_completion = AsyncMock(return_value=success_resp)

    with patch("fastdeploy.entrypoints.openai.api_server.connection_manager", return_value=DummyCM()):
        assert (await api_server.create_chat_completion(body, fake_req)).status_code == 200

    async def stream_gen():
        yield "data"

    api_server.app.state.chat_handler.create_chat_completion = AsyncMock(return_value=stream_gen())

    with patch("fastdeploy.entrypoints.openai.api_server.connection_manager", return_value=DummyCM()):
        assert isinstance(await api_server.create_chat_completion(body, fake_req), api_server.StreamingResponse)

    # Completion handler
    completion_handler = MagicMock()
    completion_handler.create_completion = AsyncMock(return_value=error_resp)
    api_server.app.state.completion_handler = completion_handler

    with patch("fastdeploy.entrypoints.openai.api_server.connection_manager", return_value=DummyCM()):
        assert (await api_server.create_completion(body, fake_req)).status_code == 500

    api_server.app.state.completion_handler.create_completion = AsyncMock(return_value=success_resp)

    with patch("fastdeploy.entrypoints.openai.api_server.connection_manager", return_value=DummyCM()):
        assert (await api_server.create_completion(body, fake_req)).status_code == 200

    # HTTPException handling
    class RaiseHTTP:
        async def __aenter__(self):
            raise api_server.HTTPException(status_code=418, detail="teapot")

        async def __aexit__(self, exc_type, exc, tb):
            return False

    with patch("fastdeploy.entrypoints.openai.api_server.connection_manager", return_value=RaiseHTTP()):
        assert (await api_server.create_chat_completion(body, fake_req)).status_code == 418
        assert (await api_server.create_completion(body, fake_req)).status_code == 418


@pytest.mark.asyncio
async def test_chat_completion_tracing():
    args = _build_args(dynamic_load_weight=False)
    api_server = _reload_api_server(args)
    api_server.envs.TRACES_ENABLE = "true"
    api_server.app.state.dynamic_load_weight = False

    fake_req = SimpleNamespace(headers={"x-request-id": "1"})
    body = SimpleNamespace(model_dump_json=lambda: "{}", stream=False)

    chat_resp = ChatCompletionResponse(id="1", model="m", choices=[], usage=UsageInfo())
    completion_resp = CompletionResponse(id="2", model="m", choices=[], usage=UsageInfo())

    api_server.app.state.chat_handler = SimpleNamespace(create_chat_completion=AsyncMock(return_value=chat_resp))
    api_server.app.state.completion_handler = SimpleNamespace(
        create_completion=AsyncMock(return_value=completion_resp)
    )
    api_server.connection_semaphore = SimpleNamespace(acquire=AsyncMock(), release=MagicMock(), status=lambda: "ok")

    class DummyCM:
        async def __aenter__(self):
            return None

        async def __aexit__(self, exc_type, exc, tb):
            return False

    with (
        patch("fastdeploy.entrypoints.openai.api_server.extract", return_value="ctx"),
        patch("fastdeploy.entrypoints.openai.api_server.connection_manager", return_value=DummyCM()),
    ):
        resp_chat = await api_server.create_chat_completion(body, fake_req)
        resp_comp = await api_server.create_completion(body, fake_req)

    assert resp_chat.status_code == 200
    assert resp_comp.status_code == 200
    assert getattr(body, "trace_context", None) == "ctx"

    # TRACES_ENABLE=True but req.headers is None/empty (missing branch 379, 415)
    api_server.envs.TRACES_ENABLE = "true"
    fake_req_no_headers = SimpleNamespace(headers=None)
    body2 = SimpleNamespace(model_dump_json=lambda: "{}", stream=False)
    api_server.app.state.chat_handler = SimpleNamespace(create_chat_completion=AsyncMock(return_value=chat_resp))
    api_server.app.state.completion_handler = SimpleNamespace(
        create_completion=AsyncMock(return_value=completion_resp)
    )

    with patch("fastdeploy.entrypoints.openai.api_server.connection_manager", return_value=DummyCM()):
        resp_chat2 = await api_server.create_chat_completion(body2, fake_req_no_headers)
        resp_comp2 = await api_server.create_completion(body2, fake_req_no_headers)

    assert resp_chat2.status_code == 200
    assert resp_comp2.status_code == 200
    assert not hasattr(body2, "trace_context")


@pytest.mark.asyncio
async def test_reward_embedding_and_weights():
    args = _build_args(dynamic_load_weight=True)
    api_server = _reload_api_server(args)
    api_server.app.state.dynamic_load_weight = True
    api_server.app.state.engine_client = MagicMock()
    api_server.app.state.engine_client.is_workers_alive.return_value = (False, "down")

    assert (await api_server.create_reward(SimpleNamespace())).status_code == 304
    assert (await api_server.create_embedding(SimpleNamespace())).status_code == 304

    api_server.app.state.dynamic_load_weight = False
    api_server.app.state.reward_handler = MagicMock(
        create_reward=AsyncMock(return_value=SimpleNamespace(model_dump=lambda: {"ok": True}))
    )
    api_server.app.state.embedding_handler = MagicMock(
        create_embedding=AsyncMock(return_value=SimpleNamespace(model_dump=lambda: {"ok": True}))
    )
    assert (await api_server.create_reward(SimpleNamespace())).status_code == 200
    assert (await api_server.create_embedding(SimpleNamespace())).status_code == 200

    # Weight update/clear
    api_server.app.state.dynamic_load_weight = True
    api_server.app.state.engine_client.update_model_weight.return_value = (404, "fail")
    assert api_server.update_model_weight(MagicMock()).status_code == 404
    api_server.app.state.engine_client.update_model_weight.return_value = (200, "ok")
    assert api_server.update_model_weight(MagicMock()).status_code == 200

    api_server.app.state.engine_client.clear_load_weight.return_value = (404, "fail")
    assert api_server.clear_load_weight(MagicMock()).status_code == 404
    api_server.app.state.engine_client.clear_load_weight.return_value = (200, "ok")
    assert api_server.clear_load_weight(MagicMock()).status_code == 200

    # Disabled branch
    api_server.app.state.dynamic_load_weight = False
    assert api_server.update_model_weight(MagicMock()).status_code == 404
    assert api_server.clear_load_weight(MagicMock()).status_code == 404


@pytest.mark.asyncio
async def test_expert_and_stats_routes():
    args = _build_args()
    with _patch_common_imports(args, engine_client_cls=_dummy_engine_client()):
        api_server = _reload_api_server(args)

    api_server.app.state.engine_client = _dummy_engine_client()()
    req = MagicMock()
    req.json = AsyncMock(return_value={"a": 1})

    assert (await api_server.rearrange_experts(req)).status_code == 201
    assert (await api_server.get_per_expert_tokens_stats(req)).status_code == 202
    assert (await api_server.check_redundant(req)).status_code == 203


def test_launchers_and_controller():
    args = _build_args()
    api_server = _reload_api_server(args)

    with patch("fastdeploy.entrypoints.openai.api_server.is_port_available", return_value=False):
        with pytest.raises(Exception):
            api_server.launch_api_server()

    with (
        patch("fastdeploy.entrypoints.openai.api_server.is_port_available", return_value=True),
        patch("fastdeploy.entrypoints.openai.api_server.StandaloneApplication.run", side_effect=RuntimeError("fail")),
    ):
        api_server.launch_api_server()

    with patch("fastdeploy.entrypoints.openai.api_server.uvicorn.run") as uv_run:
        api_server.run_metrics_server()
        api_server.run_controller_server()
        assert uv_run.call_count == 2

    with (
        patch("fastdeploy.entrypoints.openai.api_server.is_port_available", return_value=True),
        patch("fastdeploy.entrypoints.openai.api_server.run_metrics_server"),
    ):
        api_server.args.metrics_port = api_server.args.port + 1
        api_server.launch_metrics_server()

    with patch("fastdeploy.entrypoints.openai.api_server.is_port_available", return_value=False):
        api_server.args.metrics_port = api_server.args.port + 2
        with pytest.raises(Exception):
            api_server.launch_metrics_server()

    api_server.args.controller_port = -1
    api_server.launch_controller_server()

    api_server.args.controller_port = api_server.args.port + 5
    with patch("fastdeploy.entrypoints.openai.api_server.is_port_available", return_value=False):
        with pytest.raises(Exception):
            api_server.launch_controller_server()

    with (
        patch("fastdeploy.entrypoints.openai.api_server.is_port_available", return_value=True),
        patch("fastdeploy.entrypoints.openai.api_server.run_controller_server"),
    ):
        api_server.launch_controller_server()


def test_worker_monitor_and_main():
    args = _build_args()
    api_server = _reload_api_server(args)

    api_server.llm_engine = SimpleNamespace(worker_proc=SimpleNamespace(poll=lambda: 1, returncode=9))
    with patch("os.kill") as kill_mock:
        api_server.launch_worker_monitor()
        kill_mock.assert_called()

    api_server.args.local_data_parallel_id = 0
    with patch("fastdeploy.entrypoints.openai.api_server.load_engine", return_value=False):
        api_server.main()

    api_server.args.local_data_parallel_id = 1
    with patch("fastdeploy.entrypoints.openai.api_server.load_data_service", return_value=False):
        api_server.main()

    api_server.args.local_data_parallel_id = 0
    with (
        patch("fastdeploy.entrypoints.openai.api_server.load_engine", return_value=True),
        patch("fastdeploy.entrypoints.openai.api_server.launch_metrics_server"),
        patch("fastdeploy.entrypoints.openai.api_server.launch_worker_monitor"),
        patch("fastdeploy.entrypoints.openai.api_server.launch_controller_server"),
        patch("fastdeploy.entrypoints.openai.api_server.launch_api_server"),
    ):
        api_server.main()


@pytest.mark.asyncio
async def test_lifespan_and_health():
    args = _build_args()
    with _patch_common_imports(args):
        api_server = _reload_api_server(args)
        engine_client = MagicMock()
        engine_client.check_health.return_value = (False, "bad")
        api_server.app.state.engine_client = engine_client

        assert api_server.health(MagicMock()).status_code == 404
        routes = await api_server.list_all_routes()
        assert isinstance(routes, dict)


@pytest.mark.asyncio
async def test_list_models():
    args = _build_args()
    with _patch_common_imports(args):
        api_server = _reload_api_server(args)
        api_server.app.state.dynamic_load_weight = False

        class FakeErrorResponse:
            def model_dump(self):
                return {"err": True}

        api_server.ErrorResponse = FakeErrorResponse
        api_server.app.state.model_handler = MagicMock(list_models=AsyncMock(return_value=FakeErrorResponse()))
    resp = await api_server.list_models()
    assert resp.status_code == 200
    assert resp.body

    # dynamic_load_weight=True but workers_alive returns True (missing branch 442)
    api_server.app.state.dynamic_load_weight = True
    api_server.app.state.engine_client = MagicMock()
    api_server.app.state.engine_client.is_workers_alive.return_value = (True, "ok")

    # Return ModelList instead of ErrorResponse (missing branch 449)
    model_list = ModelList(data=[ModelInfo(id="test-model", object="model")])
    api_server.app.state.model_handler.list_models = AsyncMock(return_value=model_list)
    resp2 = await api_server.list_models()
    assert resp2.status_code == 200
    assert "data" in resp2.body.decode() if hasattr(resp2.body, "decode") else True


def test_control_scheduler():
    args = _build_args()
    with _patch_common_imports(args):
        api_server = _reload_api_server(args)

        with (
            patch("fastdeploy.entrypoints.openai.api_server.ErrorInfo", DummyErrorInfo),
            patch("fastdeploy.entrypoints.openai.api_server.ErrorResponse", DummyErrorResponse),
        ):
            # Engine not loaded
            api_server.llm_engine = None
            req = SimpleNamespace(reset=False, load_shards_num=None, reallocate_shard=False)
            assert api_server.control_scheduler(req).status_code == 500

            # Without update_config
            sched = SimpleNamespace()
            api_server.llm_engine = SimpleNamespace(engine=SimpleNamespace(clear_data=MagicMock(), scheduler=sched))
            req = SimpleNamespace(reset=False, load_shards_num=1, reallocate_shard=True)
            assert api_server.control_scheduler(req).status_code == 400

            # Success path
            scheduler = SimpleNamespace(update_config=MagicMock(), reset=MagicMock())
            engine = SimpleNamespace(clear_data=MagicMock(), scheduler=scheduler)
            api_server.llm_engine = SimpleNamespace(engine=engine)
            req = SimpleNamespace(reset=True, load_shards_num=2, reallocate_shard=True)
            resp = api_server.control_scheduler(req)

            assert resp.status_code == 200
            engine.clear_data.assert_called_once()
            scheduler.reset.assert_called_once()
            scheduler.update_config.assert_called_once()

            # Only reset, no update_config (missing branch 681)
            scheduler2 = SimpleNamespace(update_config=MagicMock(), reset=MagicMock())
            engine2 = SimpleNamespace(clear_data=MagicMock(), scheduler=scheduler2)
            api_server.llm_engine = SimpleNamespace(engine=engine2)
            req2 = SimpleNamespace(reset=True, load_shards_num=None, reallocate_shard=False)
            resp2 = api_server.control_scheduler(req2)

            assert resp2.status_code == 200
            engine2.clear_data.assert_called_once()
            scheduler2.reset.assert_called_once()
            scheduler2.update_config.assert_not_called()


def test_config_info():
    args = _build_args()
    with _patch_common_imports(args):
        api_server = _reload_api_server(args)
        api_server.llm_engine = None
    assert api_server.config_info().status_code == 500
