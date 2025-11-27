"""# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import asyncio
import json
import os
import signal
import threading
import time
import traceback
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
import zmq
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from gunicorn.app.base import BaseApplication
from opentelemetry import trace

from fastdeploy.engine.args_utils import EngineArgs
from fastdeploy.engine.engine import LLMEngine
from fastdeploy.engine.expert_service import ExpertService
from fastdeploy.entrypoints.chat_utils import load_chat_template
from fastdeploy.entrypoints.engine_client import EngineClient
from fastdeploy.entrypoints.openai.middleware import AuthenticationMiddleware
from fastdeploy.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatRewardRequest,
    CompletionRequest,
    CompletionResponse,
    ControlSchedulerRequest,
    EmbeddingRequest,
    ErrorInfo,
    ErrorResponse,
    ModelList,
)
from fastdeploy.entrypoints.openai.serving_chat import OpenAIServingChat
from fastdeploy.entrypoints.openai.serving_completion import OpenAIServingCompletion
from fastdeploy.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from fastdeploy.entrypoints.openai.serving_models import ModelPath, OpenAIServingModels
from fastdeploy.entrypoints.openai.serving_reward import OpenAIServingReward
from fastdeploy.entrypoints.openai.tool_parsers import ToolParserManager
from fastdeploy.entrypoints.openai.utils import UVICORN_CONFIG, make_arg_parser
from fastdeploy.envs import environment_variables
from fastdeploy.metrics.metrics import get_filtered_metrics
from fastdeploy.metrics.metrics_middleware import PrometheusMiddleware
from fastdeploy.metrics.trace_util import (
    fd_start_span,
    inject_to_metadata,
    instrument,
    lable_span,
)
from fastdeploy.utils import (
    ExceptionHandler,
    FlexibleArgumentParser,
    StatefulSemaphore,
    api_server_logger,
    console_logger,
    is_port_available,
    retrive_model_from_server,
)

parser = make_arg_parser(FlexibleArgumentParser())
args = parser.parse_args()

console_logger.info(f"Number of api-server workers: {args.workers}.")

args.model = retrive_model_from_server(args.model, args.revision)
chat_template = load_chat_template(args.chat_template, args.model)
if args.tool_parser_plugin:
    ToolParserManager.import_tool_parser(args.tool_parser_plugin)
llm_engine = None

MAX_CONCURRENT_CONNECTIONS = (args.max_concurrency + args.workers - 1) // args.workers
connection_semaphore = StatefulSemaphore(MAX_CONCURRENT_CONNECTIONS)


class StandaloneApplication(BaseApplication):
    def __init__(self, app, options=None):
        self.application = app
        self.options = options or {}
        super().__init__()

    def load_config(self):
        config = {key: value for key, value in self.options.items() if key in self.cfg.settings and value is not None}
        for key, value in config.items():
            self.cfg.set(key.lower(), value)

    def load(self):
        return self.application


def load_engine():
    """
    load engine
    """
    global llm_engine
    if llm_engine is not None:
        return llm_engine

    api_server_logger.info(f"FastDeploy LLM API server starting... {os.getpid()}, port: {args.port}")
    engine_args = EngineArgs.from_cli_args(args)
    engine = LLMEngine.from_engine_args(engine_args)
    if not engine.start(api_server_pid=args.port):
        api_server_logger.error("Failed to initialize FastDeploy LLM engine, service exit now!")
        return None

    llm_engine = engine
    return engine


def load_data_service():
    """
    load data service
    """
    global llm_engine
    if llm_engine is not None:
        return llm_engine
    api_server_logger.info(f"FastDeploy LLM API server starting... {os.getpid()}, port: {args.port}")
    engine_args = EngineArgs.from_cli_args(args)
    config = engine_args.create_engine_config()
    api_server_logger.info(f"local_data_parallel_id: {config.parallel_config}")
    expert_service = ExpertService(config, config.parallel_config.local_data_parallel_id)
    if not expert_service.start(args.port, config.parallel_config.local_data_parallel_id):
        api_server_logger.error("Failed to initialize FastDeploy LLM expert service, service exit now!")
        return None
    llm_engine = expert_service
    return expert_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    async context manager for FastAPI lifespan
    """
    import logging

    uvicorn_access = logging.getLogger("uvicorn.access")
    uvicorn_access.handlers.clear()

    # 使用 gunicorn 的格式
    formatter = logging.Formatter("[%(asctime)s] [%(process)d] [INFO] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    uvicorn_access.addHandler(handler)
    uvicorn_access.propagate = False

    if args.tokenizer is None:
        args.tokenizer = args.model
    pid = args.port
    api_server_logger.info(f"{pid}")

    if args.served_model_name is not None:
        served_model_names = args.served_model_name
        verification = True
    else:
        served_model_names = args.model
        verification = False
    model_paths = [ModelPath(name=served_model_names, model_path=args.model, verification=verification)]

    engine_args = EngineArgs.from_cli_args(args)
    config = engine_args.create_engine_config(port_availability_check=False)
    engine_client = EngineClient(
        model_name_or_path=args.model,
        tokenizer=args.tokenizer,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        pid=pid,
        port=int(os.environ.get("INFERENCE_MSG_QUEUE_ID", "0")),
        limit_mm_per_prompt=args.limit_mm_per_prompt,
        mm_processor_kwargs=args.mm_processor_kwargs,
        reasoning_parser=args.reasoning_parser,
        data_parallel_size=args.data_parallel_size,
        enable_logprob=args.enable_logprob,
        workers=args.workers,
        tool_parser=args.tool_call_parser,
        enable_prefix_caching=args.enable_prefix_caching,
        splitwise_role=args.splitwise_role,
        max_processor_cache=args.max_processor_cache,
        config=config,
    )
    await engine_client.connection_manager.initialize()
    app.state.dynamic_load_weight = args.dynamic_load_weight
    model_handler = OpenAIServingModels(
        model_paths,
        args.max_model_len,
        args.ips,
    )
    app.state.model_handler = model_handler
    chat_handler = OpenAIServingChat(
        engine_client,
        app.state.model_handler,
        pid,
        args.ips,
        args.max_waiting_time,
        chat_template,
        args.enable_mm_output,
        args.tokenizer_base_url,
    )
    completion_handler = OpenAIServingCompletion(
        engine_client,
        app.state.model_handler,
        pid,
        args.ips,
        args.max_waiting_time,
    )

    embedding_handler = OpenAIServingEmbedding(
        engine_client,
        app.state.model_handler,
        config,
        pid,
        args.ips,
        args.max_waiting_time,
        chat_template,
    )
    reward_handler = OpenAIServingReward(
        engine_client, app.state.model_handler, config, pid, args.ips, args.max_waiting_time, chat_template
    )
    engine_client.create_zmq_client(model=pid, mode=zmq.PUSH)
    engine_client.pid = pid
    app.state.engine_client = engine_client
    app.state.chat_handler = chat_handler
    app.state.completion_handler = completion_handler
    app.state.embedding_handler = embedding_handler
    app.state.reward_handler = reward_handler
    global llm_engine
    if llm_engine is not None:
        llm_engine.engine.data_processor = engine_client.data_processor
    yield
    # close zmq
    try:
        await engine_client.connection_manager.close()
        engine_client.zmq_client.close()
        from prometheus_client import multiprocess

        multiprocess.mark_process_dead(os.getpid())
        api_server_logger.info(f"Closing metrics client pid: {pid}")
    except Exception as e:
        api_server_logger.warning(f"exit error: {e}, {str(traceback.format_exc())}")


app = FastAPI(lifespan=lifespan)
app.add_exception_handler(RequestValidationError, ExceptionHandler.handle_request_validation_exception)
app.add_exception_handler(Exception, ExceptionHandler.handle_exception)
instrument(app)


env_api_key_func = environment_variables.get("FD_API_KEY")
env_tokens = env_api_key_func() if env_api_key_func else []
if tokens := [key for key in (args.api_key or env_tokens) if key]:
    app.add_middleware(AuthenticationMiddleware, tokens)

# add middleware for http metrics
app.add_middleware(PrometheusMiddleware)


@asynccontextmanager
async def connection_manager():
    """
    async context manager for connection manager
    """
    try:
        await asyncio.wait_for(connection_semaphore.acquire(), timeout=0.001)
        yield
    except asyncio.TimeoutError:
        api_server_logger.info(f"Reach max request concurrency, semaphore status: {connection_semaphore.status()}")
        raise HTTPException(
            status_code=429, detail=f"Too many requests,current max concurrency is {args.max_concurrency}"
        )


# TODO 传递真实引擎值 通过pid 获取状态
@app.get("/health")
def health(request: Request) -> Response:
    """Health check."""

    status, msg = app.state.engine_client.check_health()
    if not status:
        return Response(content=msg, status_code=404)
    status, msg = app.state.engine_client.is_workers_alive()
    if not status:
        return Response(content=msg, status_code=304)
    return Response(status_code=200)


@app.get("/load")
async def list_all_routes():
    """
    列出所有以/v1开头的路由信息

    Args:
        无参数

    Returns:
        dict: 包含所有符合条件的路由信息的字典，格式如下:
            {
                "routes": [
                    {
                        "path": str,  # 路由路径
                        "methods": list,  # 支持的HTTP方法列表，已排序
                        "tags": list  # 路由标签列表，默认为空列表
                    },
                    ...
                ]
            }

    """
    routes_info = []

    for route in app.routes:
        # 直接检查路径是否以/v1开头
        if route.path.startswith("/v1"):
            methods = sorted(route.methods)
            tags = getattr(route, "tags", []) or []
            routes_info.append({"path": route.path, "methods": methods, "tags": tags})
    return {"routes": routes_info}


@app.api_route("/ping", methods=["GET", "POST"])
def ping(raw_request: Request) -> Response:
    """Ping check. Endpoint required for SageMaker"""
    return health(raw_request)


def wrap_streaming_generator(original_generator: AsyncGenerator):
    """
    Wrap an async generator to release the connection semaphore when the generator is finished.
    """

    async def wrapped_generator():
        span = trace.get_current_span()
        if span is not None and span.is_recording():
            last_time = None
            count = 0
            try:
                async for chunk in original_generator:
                    last_time = time.time()
                    # 首包捕获
                    if count == 0 and span is not None and span.is_recording():
                        last_time = time.time()
                        span.add_event("first_chunk", {"time": last_time})
                    count += 1
                    yield chunk
            except Exception as e:
                # 错误捕获
                if span is not None and span.is_recording():
                    span.add_event("stream_error", {"time": time.time(), "error": str(e), "total_chunk": count})
                    span.record_exception(e)
                    span.set_status({"code": "ERROR", "description": str(e)})
                raise
            finally:
                # 尾包捕获
                if span is not None and span.is_recording() and count > 0:
                    span.add_event("last_chunk", {"time": last_time, "total_chunk": count})
                api_server_logger.debug(f"release: {connection_semaphore.status()}")
                connection_semaphore.release()
        else:
            try:
                async for chunk in original_generator:
                    yield chunk
            finally:
                api_server_logger.debug(f"release: {connection_semaphore.status()}")
                connection_semaphore.release()

    return wrapped_generator


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """
    Create a chat completion for the provided prompt and parameters.
    """
    api_server_logger.debug(f"Chat Received request: {request.model_dump_json()}")
    if app.state.dynamic_load_weight:
        status, msg = app.state.engine_client.is_workers_alive()
        if not status:
            return JSONResponse(content={"error": "Worker Service Not Healthy"}, status_code=304)
    try:
        async with connection_manager():
            inject_to_metadata(request)
            lable_span(request)
            generator = await app.state.chat_handler.create_chat_completion(request)
            if isinstance(generator, ErrorResponse):
                api_server_logger.debug(f"release: {connection_semaphore.status()}")
                connection_semaphore.release()
                return JSONResponse(content=generator.model_dump(), status_code=500)
            elif isinstance(generator, ChatCompletionResponse):
                api_server_logger.debug(f"release: {connection_semaphore.status()}")
                connection_semaphore.release()
                return JSONResponse(content=generator.model_dump())
            else:
                wrapped_generator = wrap_streaming_generator(generator)
                return StreamingResponse(content=wrapped_generator(), media_type="text/event-stream")

    except HTTPException as e:
        api_server_logger.error(f"Error in chat completion: {str(e)}")
        return JSONResponse(status_code=e.status_code, content={"detail": e.detail})


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """
    Create a completion for the provided prompt and parameters.
    """
    api_server_logger.info(f"Completion Received request: {request.model_dump_json()}")
    if app.state.dynamic_load_weight:
        status, msg = app.state.engine_client.is_workers_alive()
        if not status:
            return JSONResponse(content={"error": "Worker Service Not Healthy"}, status_code=304)
    try:
        async with connection_manager():
            lable_span(request)
            generator = await app.state.completion_handler.create_completion(request)
            if isinstance(generator, ErrorResponse):
                connection_semaphore.release()
                return JSONResponse(content=generator.model_dump(), status_code=500)
            elif isinstance(generator, CompletionResponse):
                connection_semaphore.release()
                return JSONResponse(content=generator.model_dump())
            else:
                wrapped_generator = wrap_streaming_generator(generator)
                return StreamingResponse(content=wrapped_generator(), media_type="text/event-stream")
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"detail": e.detail})


@app.get("/v1/models")
async def list_models() -> Response:
    """
    List all available models.
    """
    if app.state.dynamic_load_weight:
        status, msg = app.state.engine_client.is_workers_alive()
        if not status:
            return JSONResponse(content={"error": "Worker Service Not Healthy"}, status_code=304)

    models = await app.state.model_handler.list_models()
    if isinstance(models, ErrorResponse):
        return JSONResponse(content=models.model_dump())
    elif isinstance(models, ModelList):
        return JSONResponse(content=models.model_dump())


@app.post("/v1/reward")
async def create_reward(request: ChatRewardRequest):
    """
    Create reward for the input texts
    """
    if app.state.dynamic_load_weight:
        status, msg = app.state.engine_client.is_workers_alive()
        if not status:
            return JSONResponse(content={"error": "Worker Service Not Healthy"}, status_code=304)

    generator = await app.state.reward_handler.create_reward(request)
    return JSONResponse(content=generator.model_dump())


@app.post("/v1/embeddings")
async def create_embedding(request: EmbeddingRequest):
    """
    Create embeddings for the input texts
    """
    if app.state.dynamic_load_weight:
        status, msg = app.state.engine_client.is_workers_alive()
        if not status:
            return JSONResponse(content={"error": "Worker Service Not Healthy"}, status_code=304)

    generator = await app.state.embedding_handler.create_embedding(request)
    return JSONResponse(content=generator.model_dump())


@app.get("/update_model_weight")
def update_model_weight(request: Request) -> Response:
    """
    update model weight
    """
    if app.state.dynamic_load_weight:
        status, msg = app.state.engine_client.update_model_weight()
        if not status:
            return Response(content=msg, status_code=404)
        return Response(status_code=200)
    else:
        return Response(content="Dynamic Load Weight Disabled.", status_code=404)


@app.get("/clear_load_weight")
def clear_load_weight(request: Request) -> Response:
    """
    clear model weight
    """
    if app.state.dynamic_load_weight:
        status, msg = app.state.engine_client.clear_load_weight()
        if not status:
            return Response(content=msg, status_code=404)
        return Response(status_code=200)
    else:
        return Response(content="Dynamic Load Weight Disabled.", status_code=404)


@app.post("/rearrange_experts")
async def rearrange_experts(request: Request):
    """
    rearrange experts
    """
    request_dict = await request.json()
    content, status_code = await app.state.engine_client.rearrange_experts(request_dict=request_dict)
    return JSONResponse(content, status_code=status_code)


@app.post("/get_per_expert_tokens_stats")
async def get_per_expert_tokens_stats(request: Request):
    """
    get per expert tokens stats
    """
    request_dict = await request.json()
    content, status_code = await app.state.engine_client.get_per_expert_tokens_stats(request_dict=request_dict)
    return JSONResponse(content, status_code=status_code)


@app.post("/check_redundant")
async def check_redundant(request: Request):
    """
    check redundant
    """
    request_dict = await request.json()
    content, status_code = await app.state.engine_client.check_redundant(request_dict=request_dict)
    return JSONResponse(content, status_code=status_code)


def launch_api_server() -> None:
    """
    启动http服务
    """
    if not is_port_available(args.host, args.port):
        raise Exception(f"The parameter `port`:{args.port} is already in use.")

    api_server_logger.info(f"launch Fastdeploy api server... port: {args.port}")
    api_server_logger.info(f"args: {args.__dict__}")
    fd_start_span("FD_START")

    options = {
        "bind": f"{args.host}:{args.port}",
        "workers": args.workers,
        "worker_class": "uvicorn.workers.UvicornWorker",
        "loglevel": "info",
        "graceful_timeout": args.timeout_graceful_shutdown,
        "timeout": args.timeout,
    }

    try:
        StandaloneApplication(app, options).run()
    except Exception as e:
        api_server_logger.error(f"launch sync http server error, {e}, {str(traceback.format_exc())}")


metrics_app = FastAPI()

# Be tolerant to tests that monkeypatch/partially mock args.
_metrics_port = getattr(args, "metrics_port", None)
_main_port = getattr(args, "port", None)

if _metrics_port is None or (_main_port is not None and _metrics_port == _main_port):
    metrics_app = app


@metrics_app.get("/metrics")
async def metrics():
    """
    metrics
    """
    metrics_text = get_filtered_metrics()
    return Response(metrics_text, media_type="text/plain")


@metrics_app.get("/config-info")
def config_info() -> Response:
    """
    Get the current configuration of the API server.
    """
    global llm_engine
    if llm_engine is None:
        return Response("Engine not loaded", status_code=500)
    cfg = llm_engine.cfg

    def process_object(obj):
        if hasattr(obj, "__dict__"):
            # 处理有__dict__属性的对象
            return obj.__dict__
        return None  # 或其他默认处理

    cfg_dict = {k: v for k, v in cfg.__dict__.items()}
    env_dict = {k: v() for k, v in environment_variables.items()}
    cfg_dict["env_config"] = env_dict
    result_content = json.dumps(cfg_dict, default=process_object, ensure_ascii=False)
    return Response(result_content, media_type="application/json")


def run_metrics_server():
    """
    run metrics server
    """

    uvicorn.run(metrics_app, host="0.0.0.0", port=args.metrics_port, log_config=UVICORN_CONFIG, log_level="error")


def launch_metrics_server():
    """Metrics server running the sub thread"""
    if not is_port_available(args.host, args.metrics_port):
        raise Exception(f"The parameter `metrics_port`:{args.metrics_port} is already in use.")

    metrics_server_thread = threading.Thread(target=run_metrics_server, daemon=True)
    metrics_server_thread.start()
    time.sleep(1)


controller_app = FastAPI()


@controller_app.post("/controller/reset_scheduler")
def reset_scheduler():
    """
    reset scheduler
    """
    global llm_engine

    if llm_engine is None:
        return Response("Engine not loaded", status_code=500)

    llm_engine.engine.clear_data()
    llm_engine.engine.scheduler.reset()
    return Response("Scheduler Reset Successfully", status_code=200)


@controller_app.post("/controller/scheduler")
def control_scheduler(request: ControlSchedulerRequest):
    """
    Control the scheduler behavior with the given parameters.
    """

    content = ErrorResponse(error=ErrorInfo(message="Scheduler updated successfully", code=0))

    global llm_engine
    if llm_engine is None:
        content.message = "Engine is not loaded"
        content.code = 500
        return JSONResponse(content=content.model_dump(), status_code=500)

    if request.reset:
        llm_engine.engine.clear_data()
        llm_engine.engine.scheduler.reset()

    if request.load_shards_num or request.reallocate_shard:
        if hasattr(llm_engine.engine.scheduler, "update_config") and callable(
            llm_engine.engine.scheduler.update_config
        ):
            llm_engine.engine.scheduler.update_config(
                load_shards_num=request.load_shards_num,
                reallocate=request.reallocate_shard,
            )
        else:
            content.message = "This scheduler doesn't support the `update_config()` method."
            content.code = 400
            return JSONResponse(content=content.model_dump(), status_code=400)

    return JSONResponse(content=content.model_dump(), status_code=200)


def run_controller_server():
    """
    run controller server
    """
    uvicorn.run(
        controller_app,
        host="0.0.0.0",
        port=args.controller_port,
        log_config=UVICORN_CONFIG,
        log_level="error",
    )


def launch_controller_server():
    """Controller server running the sub thread"""
    if args.controller_port < 0:
        return

    if not is_port_available(args.host, args.controller_port):
        raise Exception(f"The parameter `controller_port`:{args.controller_port} is already in use.")

    controller_server_thread = threading.Thread(target=run_controller_server, daemon=True)
    controller_server_thread.start()
    time.sleep(1)


def launch_worker_monitor():
    """
    Detect whether worker process is alive. If not, stop the API serverby triggering llm_engine.
    """

    def _monitor():
        global llm_engine
        while True:
            if hasattr(llm_engine, "worker_proc") and llm_engine.worker_proc.poll() is not None:
                console_logger.error(
                    f"Worker process has died in the background (code={llm_engine.worker_proc.returncode}). API server is forced to stop."
                )
                os.kill(os.getpid(), signal.SIGINT)
                break
            time.sleep(5)

    worker_monitor_thread = threading.Thread(target=_monitor, daemon=True)
    worker_monitor_thread.start()
    time.sleep(1)


def main():
    """main函数"""
    if args.local_data_parallel_id == 0:
        if not load_engine():
            return
    else:
        if not load_data_service():
            return
    api_server_logger.info("FastDeploy LLM engine initialized!\n")
    if args.metrics_port is not None and args.metrics_port != args.port:
        launch_metrics_server()
        console_logger.info(f"Launching metrics service at http://{args.host}:{args.metrics_port}/metrics")
    else:
        console_logger.info(f"Launching metrics service at http://{args.host}:{args.port}/metrics")
    console_logger.info(f"Launching chat completion service at http://{args.host}:{args.port}/v1/chat/completions")
    console_logger.info(f"Launching completion service at http://{args.host}:{args.port}/v1/completions")

    launch_worker_monitor()
    launch_controller_server()
    launch_api_server()


if __name__ == "__main__":
    main()
