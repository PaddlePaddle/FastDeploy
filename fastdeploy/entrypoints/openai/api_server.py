"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
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
import os
import threading
import time
import traceback
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from multiprocessing import current_process

import uvicorn
import zmq
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST

from fastdeploy.engine.args_utils import EngineArgs
from fastdeploy.engine.engine import LLMEngine
from fastdeploy.engine.expert_service import ExpertService
from fastdeploy.entrypoints.chat_utils import load_chat_template
from fastdeploy.entrypoints.engine_client import EngineClient
from fastdeploy.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    ControlSchedulerRequest,
    ErrorInfo,
    ErrorResponse,
    ModelList,
)
from fastdeploy.entrypoints.openai.serving_chat import OpenAIServingChat
from fastdeploy.entrypoints.openai.serving_completion import OpenAIServingCompletion
from fastdeploy.entrypoints.openai.serving_models import ModelPath, OpenAIServingModels
from fastdeploy.entrypoints.openai.tool_parsers import ToolParserManager
from fastdeploy.entrypoints.openai.utils import UVICORN_CONFIG
from fastdeploy.metrics.metrics import (
    EXCLUDE_LABELS,
    cleanup_prometheus_files,
    get_filtered_metrics,
    main_process_metrics,
)
from fastdeploy.metrics.trace_util import fd_start_span, inject_to_metadata, instrument
from fastdeploy.utils import (
    ExceptionHandler,
    FlexibleArgumentParser,
    StatefulSemaphore,
    api_server_logger,
    console_logger,
    is_port_available,
    retrive_model_from_server,
)

parser = FlexibleArgumentParser()
parser.add_argument("--port", default=8000, type=int, help="port to the http server")
parser.add_argument("--host", default="0.0.0.0", type=str, help="host to the http server")
parser.add_argument("--workers", default=1, type=int, help="number of workers")
parser.add_argument("--metrics-port", default=8001, type=int, help="port for metrics server")
parser.add_argument("--controller-port", default=-1, type=int, help="port for controller server")
parser.add_argument(
    "--max-waiting-time",
    default=-1,
    type=int,
    help="max waiting time for connection, if set value -1 means no waiting time limit",
)
parser.add_argument("--max-concurrency", default=512, type=int, help="max concurrency")

parser.add_argument(
    "--enable-mm-output", action="store_true", help="Enable 'multimodal_content' field in response output. "
)
parser.add_argument(
    "--timeout-graceful-shutdown",
    default=0,
    type=int,
    help="timeout for graceful shutdown in seconds (used by uvicorn)",
)

parser = EngineArgs.add_cli_args(parser)
args = parser.parse_args()

console_logger.info(f"Number of api-server workers: {args.workers}.")

args.model = retrive_model_from_server(args.model, args.revision)
chat_template = load_chat_template(args.chat_template, args.model)
if args.tool_parser_plugin:
    ToolParserManager.import_tool_parser(args.tool_parser_plugin)
llm_engine = None


def load_engine():
    """
    load engine
    """
    global llm_engine
    if llm_engine is not None:
        return llm_engine

    api_server_logger.info(f"FastDeploy LLM API server starting... {os.getpid()}")
    engine_args = EngineArgs.from_cli_args(args)
    engine = LLMEngine.from_engine_args(engine_args)
    if not engine.start(api_server_pid=os.getpid()):
        api_server_logger.error("Failed to initialize FastDeploy LLM engine, service exit now!")
        return None

    llm_engine = engine
    return engine


app = FastAPI()

MAX_CONCURRENT_CONNECTIONS = (args.max_concurrency + args.workers - 1) // args.workers
connection_semaphore = StatefulSemaphore(MAX_CONCURRENT_CONNECTIONS)


def load_data_service():
    """
    load data service
    """
    global llm_engine
    if llm_engine is not None:
        return llm_engine
    api_server_logger.info(f"FastDeploy LLM API server starting... {os.getpid()}")
    engine_args = EngineArgs.from_cli_args(args)
    config = engine_args.create_engine_config()
    api_server_logger.info(f"local_data_parallel_id: {config.parallel_config}")
    expert_service = ExpertService(config, config.parallel_config.local_data_parallel_id)
    if not expert_service.start(os.getpid(), config.parallel_config.local_data_parallel_id):
        api_server_logger.error("Failed to initialize FastDeploy LLM expert service, service exit now!")
        return None
    llm_engine = expert_service
    return expert_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    async context manager for FastAPI lifespan
    """

    if args.tokenizer is None:
        args.tokenizer = args.model
    if current_process().name != "MainProcess":
        pid = os.getppid()
    else:
        pid = os.getpid()
    api_server_logger.info(f"{pid}")

    if args.served_model_name is not None:
        served_model_names = args.served_model_name
        verification = True
    else:
        served_model_names = args.model
        verification = False
    model_paths = [ModelPath(name=served_model_names, model_path=args.model, verification=verification)]

    engine_client = EngineClient(
        model_name_or_path=args.model,
        tokenizer=args.tokenizer,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        pid=pid,
        port=int(args.engine_worker_queue_port[args.local_data_parallel_id]),
        limit_mm_per_prompt=args.limit_mm_per_prompt,
        mm_processor_kwargs=args.mm_processor_kwargs,
        # args.enable_mm,
        reasoning_parser=args.reasoning_parser,
        data_parallel_size=args.data_parallel_size,
        enable_logprob=args.enable_logprob,
        workers=args.workers,
        tool_parser=args.tool_call_parser,
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
    engine_client.create_zmq_client(model=pid, mode=zmq.PUSH)
    engine_client.pid = pid
    app.state.engine_client = engine_client
    app.state.chat_handler = chat_handler
    app.state.completion_handler = completion_handler
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
    api_server_logger.info(f"Chat Received request: {request.model_dump_json()}")
    if app.state.dynamic_load_weight:
        status, msg = app.state.engine_client.is_workers_alive()
        if not status:
            return JSONResponse(content={"error": "Worker Service Not Healthy"}, status_code=304)
    try:
        async with connection_manager():
            inject_to_metadata(request)
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


def launch_api_server() -> None:
    """
    启动http服务
    """
    if not is_port_available(args.host, args.port):
        raise Exception(f"The parameter `port`:{args.port} is already in use.")

    api_server_logger.info(f"launch Fastdeploy api server... port: {args.port}")
    api_server_logger.info(f"args: {args.__dict__}")
    fd_start_span("FD_START")

    try:
        uvicorn.run(
            app="fastdeploy.entrypoints.openai.api_server:app",
            host=args.host,
            port=args.port,
            workers=args.workers,
            log_config=UVICORN_CONFIG,
            log_level="info",
            timeout_graceful_shutdown=args.timeout_graceful_shutdown,
        )  # set log level to error to avoid log
    except Exception as e:
        api_server_logger.error(f"launch sync http server error, {e}, {str(traceback.format_exc())}")


metrics_app = FastAPI()


@metrics_app.get("/metrics")
async def metrics():
    """
    metrics
    """
    metrics_text = get_filtered_metrics(
        EXCLUDE_LABELS,
        extra_register_func=lambda reg: main_process_metrics.register_all(reg, workers=args.workers),
    )
    return Response(metrics_text, media_type=CONTENT_TYPE_LATEST)


def run_metrics_server():
    """
    run metrics server
    """

    uvicorn.run(metrics_app, host="0.0.0.0", port=args.metrics_port, log_config=UVICORN_CONFIG, log_level="error")


def launch_metrics_server():
    """Metrics server running the sub thread"""
    if not is_port_available(args.host, args.metrics_port):
        raise Exception(f"The parameter `metrics_port`:{args.metrics_port} is already in use.")

    prom_dir = cleanup_prometheus_files(True)
    os.environ["PROMETHEUS_MULTIPROC_DIR"] = prom_dir
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


def main():
    """main函数"""
    if args.local_data_parallel_id == 0:
        if not load_engine():
            return
    else:
        if not load_data_service():
            return
    api_server_logger.info("FastDeploy LLM engine initialized!\n")
    console_logger.info(f"Launching metrics service at http://{args.host}:{args.metrics_port}/metrics")
    console_logger.info(f"Launching chat completion service at http://{args.host}:{args.port}/v1/chat/completions")
    console_logger.info(f"Launching completion service at http://{args.host}:{args.port}/v1/completions")

    launch_controller_server()
    launch_metrics_server()
    launch_api_server()


if __name__ == "__main__":
    main()
