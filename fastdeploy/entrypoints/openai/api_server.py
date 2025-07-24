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

import os
import threading
import time
from contextlib import asynccontextmanager
from multiprocessing import current_process

import uvicorn
import zmq
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST

from fastdeploy.engine.args_utils import EngineArgs
from fastdeploy.engine.engine import LLMEngine
from fastdeploy.entrypoints.engine_client import EngineClient
from fastdeploy.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    ControlSchedulerRequest,
    ErrorResponse,
)
from fastdeploy.entrypoints.openai.serving_chat import OpenAIServingChat
from fastdeploy.entrypoints.openai.serving_completion import OpenAIServingCompletion
from fastdeploy.metrics.metrics import (
    EXCLUDE_LABELS,
    cleanup_prometheus_files,
    get_filtered_metrics,
    main_process_metrics,
)
from fastdeploy.metrics.trace_util import inject_to_metadata, instrument
from fastdeploy.utils import (
    FlexibleArgumentParser,
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
parser = EngineArgs.add_cli_args(parser)
args = parser.parse_args()
args.model = retrive_model_from_server(args.model, args.revision)

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

    api_server_logger.info("FastDeploy LLM engine initialized!\n")
    console_logger.info(f"Launching metrics service at http://{args.host}:{args.metrics_port}/metrics")
    console_logger.info(f"Launching chat completion service at http://{args.host}:{args.port}/v1/chat/completions")
    console_logger.info(f"Launching completion service at http://{args.host}:{args.port}/v1/completions")
    llm_engine = engine
    return engine


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
    engine_client = EngineClient(
        args.tokenizer,
        args.max_model_len,
        args.tensor_parallel_size,
        pid,
        args.limit_mm_per_prompt,
        args.mm_processor_kwargs,
        args.enable_mm,
        args.reasoning_parser,
        args.data_parallel_size
    )
    app.state.dynamic_load_weight = args.dynamic_load_weight
    chat_handler = OpenAIServingChat(engine_client, pid, args.ips)
    completion_handler = OpenAIServingCompletion(engine_client, pid, args.ips)
    engine_client.create_zmq_client(model=pid, mode=zmq.PUSH)
    engine_client.pid = pid
    app.state.engine_client = engine_client
    app.state.chat_handler = chat_handler
    app.state.completion_handler = completion_handler
    yield
    # close zmq
    try:
        engine_client.zmq_client.close()
        from prometheus_client import multiprocess

        multiprocess.mark_process_dead(os.getpid())
        api_server_logger.info(f"Closing metrics client pid: {pid}")
    except Exception as e:
        api_server_logger.warning(e)


app = FastAPI(lifespan=lifespan)
instrument(app)


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


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """
    Create a chat completion for the provided prompt and parameters.
    """
    if app.state.dynamic_load_weight:
        status, msg = app.state.engine_client.is_workers_alive()
        if not status:
            return JSONResponse(content={"error": "Worker Service Not Healthy"}, status_code=304)
    inject_to_metadata(request)
    generator = await app.state.chat_handler.create_chat_completion(request)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.code)

    elif isinstance(generator, ChatCompletionResponse):
        return JSONResponse(content=generator.model_dump())

    return StreamingResponse(content=generator, media_type="text/event-stream")


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """
    Create a completion for the provided prompt and parameters.
    """
    if app.state.dynamic_load_weight:
        status, msg = app.state.engine_client.is_workers_alive()
        if not status:
            return JSONResponse(content={"error": "Worker Service Not Healthy"}, status_code=304)

    generator = await app.state.completion_handler.create_completion(request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.code)
    elif isinstance(generator, CompletionResponse):
        return JSONResponse(content=generator.model_dump())

    return StreamingResponse(content=generator, media_type="text/event-stream")


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

    try:
        uvicorn.run(
            app="fastdeploy.entrypoints.openai.api_server:app",
            host=args.host,
            port=args.port,
            workers=args.workers,
            log_level="info",
        )  # set log level to error to avoid log
    except Exception as e:
        api_server_logger.error(f"launch sync http server error, {e}")


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

    uvicorn.run(metrics_app, host="0.0.0.0", port=args.metrics_port, log_level="error")


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
    llm_engine.scheduler.reset()
    return Response("Scheduler Reset Successfully", status_code=200)


@controller_app.post("/controller/scheduler")
def control_scheduler(request: ControlSchedulerRequest):
    """
    Control the scheduler behavior with the given parameters.
    """
    content = ErrorResponse(object="", message="Scheduler updated successfully", code=0)

    global llm_engine
    if llm_engine is None:
        content.message = "Engine is not loaded"
        content.code = 500
        return JSONResponse(content=content.model_dump(), status_code=500)

    if request.reset:
        llm_engine.scheduler.reset()

    if request.load_shards_num or request.reallocate_shard:
        if hasattr(llm_engine.scheduler, "update_config") and callable(llm_engine.scheduler.update_config):
            llm_engine.scheduler.update_config(
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

    if load_engine() is None:
        return

    launch_controller_server()
    launch_metrics_server()
    launch_api_server()


if __name__ == "__main__":
    main()
