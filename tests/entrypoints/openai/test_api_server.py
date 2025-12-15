"""
Extra coverage for `fastdeploy.entrypoints.openai.api_server`.
Tests are lightweight and mock heavy dependencies to exercise branches
that were previously uncovered.
"""

import asyncio
import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _build_args(**overrides):
    """Return a SimpleNamespace with all attributes accessed at import time."""
    base = dict(
        # basic
        workers=1,
        model="test-model",
        revision=None,
        chat_template=None,
        tool_parser_plugin=None,
        # network
        host="0.0.0.0",
        port=9000,
        metrics_port=None,
        controller_port=-1,
        # limits
        max_concurrency=4,
        max_model_len=1024,
        max_waiting_time=-1,
        max_logprobs=0,
        # engine/distributed knobs
        tensor_parallel_size=1,
        data_parallel_size=1,
        enable_expert_parallel=False,
        enable_logprob=False,
        enable_early_stop=False,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        max_num_seqs=8,
        max_num_partial_prefills=0,
        max_long_partial_prefills=0,
        long_prefill_token_threshold=0,
        cache_transfer_protocol=None,
        scheduler_name=None,
        scheduler_host=None,
        scheduler_port=None,
        scheduler_db=None,
        scheduler_password=None,
        scheduler_topic=None,
        splitwise_role=None,
        max_processor_cache=0,
        # misc
        api_key=None,
        tokenizer=None,
        served_model_name=None,
        ips=None,
        enable_mm_output=False,
        tokenizer_base_url=None,
        dynamic_load_weight=False,
        timeout_graceful_shutdown=0,
        timeout=0,
        controller_port_start=None,
        controller_port_end=None,
        local_data_parallel_id=0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _reload_api_server(args):
    """Import/reload api_server with patched parse_args/model loader/template."""
    with (
        patch("fastdeploy.utils.FlexibleArgumentParser.parse_args", return_value=args),
        patch("fastdeploy.utils.retrive_model_from_server", return_value=args.model),
        patch("fastdeploy.entrypoints.chat_utils.load_chat_template", return_value=None),
    ):
        from fastdeploy.entrypoints.openai import api_server as api_server_mod

        return importlib.reload(api_server_mod)


def _dummy_engine_args(config_parallel_id=0):
    cfg = SimpleNamespace(parallel_config=SimpleNamespace(local_data_parallel_id=config_parallel_id))

    class DummyArgs:
        def create_engine_config(self, port_availability_check=True):
            return cfg

    return DummyArgs()


def _dummy_engine_client():
    class DummyConnMgr:
        async def initialize(self):
            self.inited = True

        async def close(self):
            self.closed = True

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

        async def create_chat_completion(self, req):
            return req  # will be swapped in tests

        async def create_completion(self, req):
            return req

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
    return (
        patch("fastdeploy.entrypoints.openai.api_server.EngineArgs.from_cli_args", return_value=_dummy_engine_args()),
        patch("fastdeploy.entrypoints.openai.api_server.EngineClient", engine_client_cls),
        patch("fastdeploy.entrypoints.openai.api_server.OpenAIServingModels", handler_cls),
        patch("fastdeploy.entrypoints.openai.api_server.OpenAIServingChat", handler_cls),
        patch("fastdeploy.entrypoints.openai.api_server.OpenAIServingCompletion", handler_cls),
        patch("fastdeploy.entrypoints.openai.api_server.OpenAIServingEmbedding", handler_cls),
        patch("fastdeploy.entrypoints.openai.api_server.OpenAIServingReward", handler_cls),
        patch("fastdeploy.entrypoints.openai.api_server.ToolParserManager.import_tool_parser"),
    )


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
        import_mock.assert_called_once()  # line ~85

        api_server.llm_engine = "cached"
        assert api_server.load_engine() == "cached"  # line ~113

        api_server.llm_engine = None
        llm_from_args.return_value = SimpleNamespace(start=MagicMock(return_value=False))
        assert api_server.load_engine() is None  # lines ~119-120

    # StandaloneApplication load_config/load
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
        assert api_server.load_data_service() is None  # failure branch 138-140
        api_server.llm_engine = None
        assert api_server.load_data_service() is expert  # success branch 131-142
        # Subsequent call returns cached engine (line ~131)
        assert api_server.load_data_service() is expert


@pytest.mark.asyncio
async def test_lifespan_context_initializes_and_closes():
    args = _build_args()
    with _patch_common_imports(args):
        api_server = _reload_api_server(args)

    async with api_server.lifespan(api_server.app):
        assert hasattr(api_server.app.state, "chat_handler")
        assert hasattr(api_server.app.state, "engine_client")
    # ensure cleanup executed without raising (lines ~235-243)


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
        assert exc.value.status_code == 429  # lines 263-268


def test_health_ping_and_route_listing():
    args = _build_args()
    api_server = _reload_api_server(args)
    engine_client = MagicMock()
    engine_client.check_health.return_value = (True, "ok")
    engine_client.is_workers_alive.return_value = (False, "dead")
    api_server.app.state.engine_client = engine_client

    resp = api_server.health(MagicMock())
    assert resp.status_code == 304  # lines 278-284

    ping_resp = api_server.ping(MagicMock())
    assert ping_resp.status_code == 304  # line 323

    routes = api_server.list_all_routes()
    assert isinstance(routes, dict) and routes["routes"]  # lines 309-317


@pytest.mark.asyncio
async def test_wrap_streaming_generator_error_span_branch():
    args = _build_args()
    api_server = _reload_api_server(args)
    span = MagicMock()
    span.is_recording.return_value = True
    sem = MagicMock()
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


@pytest.mark.asyncio
async def test_chat_completion_branches_and_completion_branches():
    args = _build_args(dynamic_load_weight=True)
    api_server = _reload_api_server(args)
    api_server.app.state.dynamic_load_weight = True
    api_server.app.state.engine_client = MagicMock()
    api_server.app.state.engine_client.is_workers_alive.return_value = (False, "down")

    # dynamic_load_weight unhealthy path
    resp = await api_server.create_chat_completion(SimpleNamespace(model_dump_json=lambda: "{}"))
    assert resp.status_code == 304  # lines 374-398

    # Healthy path with ErrorResponse -> ChatCompletionResponse -> streaming
    api_server.app.state.dynamic_load_weight = False
    api_server.connection_semaphore = MagicMock()
    api_server.connection_semaphore.release = MagicMock()

    from fastdeploy.entrypoints.openai.protocol import (
        ChatCompletionResponse,
        ErrorInfo,
        ErrorResponse,
        UsageInfo,
    )

    error_resp = ErrorResponse(error=ErrorInfo(message="err"))
    api_server.app.state.chat_handler = MagicMock(create_chat_completion=AsyncMock(return_value=error_resp))
    resp2 = await api_server.create_chat_completion(SimpleNamespace(model_dump_json=lambda: "{}"))
    assert resp2.status_code == 500

    success_resp = ChatCompletionResponse(id="1", model="m", choices=[], usage=UsageInfo())
    api_server.app.state.chat_handler.create_chat_completion = AsyncMock(return_value=success_resp)
    resp3 = await api_server.create_chat_completion(SimpleNamespace(model_dump_json=lambda: "{}"))
    assert resp3.status_code == 200

    async def stream_gen():
        yield "data"

    api_server.app.state.chat_handler.create_chat_completion = AsyncMock(return_value=stream_gen())
    stream_resp = await api_server.create_chat_completion(SimpleNamespace(model_dump_json=lambda: "{}"))
    assert isinstance(stream_resp, api_server.StreamingResponse)

    # completion handler mirrors chat path
    api_server.app.state.completion_handler = MagicMock(create_completion=AsyncMock(return_value=error_resp))
    resp4 = await api_server.create_completion(SimpleNamespace(model_dump_json=lambda: "{}"))
    assert resp4.status_code == 500
    api_server.app.state.completion_handler.create_completion = AsyncMock(return_value=success_resp)
    resp5 = await api_server.create_completion(SimpleNamespace(model_dump_json=lambda: "{}"))
    assert resp5.status_code == 200

    # completion dynamic_load_weight unhealthy branch
    api_server.app.state.dynamic_load_weight = True
    api_server.app.state.engine_client.is_workers_alive.return_value = (False, "down")
    resp6 = await api_server.create_completion(SimpleNamespace(model_dump_json=lambda: "{}"))
    assert resp6.status_code == 304

    # HTTPException handling for chat/completion
    api_server.app.state.dynamic_load_weight = False

    class RaiseHTTP:
        async def __aenter__(self):
            raise api_server.HTTPException(status_code=418, detail="teapot")

        async def __aexit__(self, exc_type, exc, tb):
            return False

    with patch("fastdeploy.entrypoints.openai.api_server.connection_manager", return_value=RaiseHTTP()):
        resp_err = await api_server.create_chat_completion(SimpleNamespace(model_dump_json=lambda: "{}"))
        assert resp_err.status_code == 418
        resp_err2 = await api_server.create_completion(SimpleNamespace(model_dump_json=lambda: "{}"))
        assert resp_err2.status_code == 418


@pytest.mark.asyncio
async def test_other_routes_reward_embedding_and_weights():
    args = _build_args(dynamic_load_weight=True)
    api_server = _reload_api_server(args)
    api_server.app.state.engine_client = MagicMock()
    api_server.app.state.engine_client.is_workers_alive.return_value = (False, "down")

    # reward/embedding unhealthy path
    reward_resp = await api_server.create_reward(SimpleNamespace())
    embed_resp = await api_server.create_embedding(SimpleNamespace())
    assert reward_resp.status_code == 304 and embed_resp.status_code == 304  # lines 450-470

    api_server.app.state.dynamic_load_weight = False
    api_server.app.state.reward_handler = MagicMock(
        create_reward=AsyncMock(return_value=SimpleNamespace(model_dump=lambda: {"ok": True}))
    )
    api_server.app.state.embedding_handler = MagicMock(
        create_embedding=AsyncMock(return_value=SimpleNamespace(model_dump=lambda: {"ok": True}))
    )
    assert (await api_server.create_reward(SimpleNamespace())).status_code == 200
    assert (await api_server.create_embedding(SimpleNamespace())).status_code == 200

    # weight update/clear
    api_server.app.state.dynamic_load_weight = True
    api_server.app.state.engine_client.update_model_weight.return_value = (False, "fail")
    assert api_server.update_model_weight(MagicMock()).status_code == 404
    api_server.app.state.engine_client.update_model_weight.return_value = (True, "ok")
    assert api_server.update_model_weight(MagicMock()).status_code == 200

    api_server.app.state.engine_client.clear_load_weight.return_value = (False, "fail")
    assert api_server.clear_load_weight(MagicMock()).status_code == 404
    api_server.app.state.engine_client.clear_load_weight.return_value = (True, "ok")
    assert api_server.clear_load_weight(MagicMock()).status_code == 200


@pytest.mark.asyncio
async def test_expert_and_stats_routes():
    args = _build_args()
    with _patch_common_imports(args, engine_client_cls=_dummy_engine_client()):
        api_server = _reload_api_server(args)

    api_server.app.state.engine_client = _dummy_engine_client()()

    # rearrange_experts
    req = MagicMock()
    req.json = AsyncMock(return_value={"a": 1})
    rearrange_resp = await api_server.rearrange_experts(req)
    assert rearrange_resp.status_code == 201  # lines 506-508

    stats_resp = await api_server.get_per_expert_tokens_stats(req)
    assert stats_resp.status_code == 202  # lines 516-518

    redundant_resp = await api_server.check_redundant(req)
    assert redundant_resp.status_code == 203  # lines 526-528


def test_launchers_and_controller_paths():
    args = _build_args()
    api_server = _reload_api_server(args)

    # launch_api_server port in use path (line ~536)
    with patch("fastdeploy.entrypoints.openai.api_server.is_port_available", return_value=False):
        with pytest.raises(Exception):
            api_server.launch_api_server()

    # launch_api_server exception branch (line ~554)
    with (
        patch("fastdeploy.entrypoints.openai.api_server.is_port_available", return_value=True),
        patch("fastdeploy.entrypoints.openai.api_server.StandaloneApplication.run", side_effect=RuntimeError("fail")),
    ):
        api_server.launch_api_server()

    # metrics server and controller server
    with patch("fastdeploy.entrypoints.openai.api_server.uvicorn.run") as uv_run:
        api_server.run_metrics_server()
        api_server.run_controller_server()
        assert uv_run.call_count == 2  # lines ~604 and ~673

    with (
        patch("fastdeploy.entrypoints.openai.api_server.is_port_available", return_value=True),
        patch("fastdeploy.entrypoints.openai.api_server.run_metrics_server"),
    ):
        api_server.args.metrics_port = api_server.args.port + 1
        api_server.launch_metrics_server()  # lines ~610-614

    with patch("fastdeploy.entrypoints.openai.api_server.is_port_available", return_value=False):
        api_server.args.metrics_port = api_server.args.port + 2
        with pytest.raises(Exception):
            api_server.launch_metrics_server()

    api_server.args.controller_port = -1
    api_server.launch_controller_server()  # early return branch 684-686
    api_server.args.controller_port = api_server.args.port + 5
    with patch("fastdeploy.entrypoints.openai.api_server.is_port_available", return_value=False):
        with pytest.raises(Exception):
            api_server.launch_controller_server()
    with (
        patch("fastdeploy.entrypoints.openai.api_server.is_port_available", return_value=True),
        patch("fastdeploy.entrypoints.openai.api_server.run_controller_server"),
    ):
        api_server.launch_controller_server()  # lines 687-692


def test_controller_routes_and_models_listing():
    args = _build_args(dynamic_load_weight=True)
    api_server = _reload_api_server(args)

    # reset_scheduler branch when llm_engine is None
    api_server.llm_engine = None
    resp = api_server.reset_scheduler()
    assert resp.status_code == 500  # lines 627-632

    # reset_scheduler success branch
    mock_engine = SimpleNamespace(
        engine=SimpleNamespace(clear_data=MagicMock(), scheduler=MagicMock(reset=MagicMock()))
    )
    api_server.llm_engine = mock_engine
    resp2 = api_server.reset_scheduler()
    assert resp2.status_code == 200

    # control_scheduler: engine not loaded
    api_server.llm_engine = None
    from fastdeploy.entrypoints.openai.protocol import ControlSchedulerRequest

    ctrl_req = ControlSchedulerRequest()
    resp3 = api_server.control_scheduler(ctrl_req)
    assert resp3.status_code == 500

    # control_scheduler reset and update_config branches
    sched = MagicMock()
    sched.update_config = MagicMock()
    mock_engine2 = SimpleNamespace(engine=SimpleNamespace(clear_data=MagicMock(), scheduler=sched))
    api_server.llm_engine = mock_engine2
    ctrl_req = ControlSchedulerRequest(reset=True, load_shards_num=2, reallocate_shard=True)
    resp4 = api_server.control_scheduler(ctrl_req)
    assert resp4.status_code == 200
    sched.update_config.assert_called_once()

    # list_models with dynamic_load_weight True and unhealthy workers
    api_server.app.state.dynamic_load_weight = True
    api_server.app.state.engine_client = MagicMock(is_workers_alive=MagicMock(return_value=(False, "down")))
    models_resp_fail = asyncio.run(api_server.list_models())
    assert models_resp_fail.status_code == 304  # lines 433-437

    # list_models success ModelList branch
    api_server.app.state.dynamic_load_weight = False
    api_server.app.state.model_handler = MagicMock(
        list_models=AsyncMock(return_value=SimpleNamespace(model_dump=lambda: {"models": [1]}))
    )
    models_resp = asyncio.run(api_server.list_models())
    assert models_resp.status_code == 200  # lines 433-442


def test_worker_monitor_and_main_paths():
    args = _build_args()
    api_server = _reload_api_server(args)

    # launch_worker_monitor hitting poll branch without killing process
    api_server.llm_engine = SimpleNamespace(worker_proc=SimpleNamespace(poll=lambda: 1, returncode=9))
    with patch("os.kill") as kill_mock:
        api_server.launch_worker_monitor()
        kill_mock.assert_called()  # lines 702-709

    # main branches: local_data_parallel_id toggles load_engine/load_data_service
    api_server.args.local_data_parallel_id = 0
    with patch("fastdeploy.entrypoints.openai.api_server.load_engine", return_value=False):
        api_server.main()  # exits early lines 718-723
    api_server.args.local_data_parallel_id = 1
    with patch("fastdeploy.entrypoints.openai.api_server.load_data_service", return_value=False):
        api_server.main()  # exits early with data service branch

    # success path to hit logging and launcher calls
    api_server.args.local_data_parallel_id = 0
    with (
        patch("fastdeploy.entrypoints.openai.api_server.load_engine", return_value=True),
        patch("fastdeploy.entrypoints.openai.api_server.launch_metrics_server"),
        patch("fastdeploy.entrypoints.openai.api_server.launch_worker_monitor"),
        patch("fastdeploy.entrypoints.openai.api_server.launch_controller_server"),
        patch("fastdeploy.entrypoints.openai.api_server.launch_api_server"),
    ):
        api_server.main()  # lines ~729 etc.
