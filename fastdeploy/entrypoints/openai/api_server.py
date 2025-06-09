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
import shutil
import uvicorn
import zmq
import os
import sys
import ctypes
import signal
from fastapi import FastAPI, APIRouter, Request
import threading
from fastapi import FastAPI, Request
from multiprocessing import current_process
from fastapi.responses import JSONResponse, Response, StreamingResponse
from contextlib import asynccontextmanager
from prometheus_client import CONTENT_TYPE_LATEST
from fastdeploy.metrics.metrics import cleanup_prometheus_files, main_process_metrics, EXCLUDE_LABELS, \
    get_filtered_metrics
from fastdeploy.utils import FlexibleArgumentParser, api_server_logger, is_port_available
from fastdeploy.engine.args_utils import EngineArgs
from fastdeploy.engine.engine import LLMEngine
from fastdeploy.entrypoints.openai.protocol import (
    CompletionRequest,
    ChatCompletionRequest,
    ErrorResponse,
    ChatCompletionResponse,
    CompletionResponse
)

from fastdeploy.entrypoints.openai.serving_chat import OpenAIServingChat
from fastdeploy.entrypoints.openai.serving_completion import OpenAIServingCompletion
from fastdeploy.entrypoints.engine_client import EngineClient

parser = FlexibleArgumentParser()
parser.add_argument("--port", default=9904, type=int, help="port to the http server")
parser.add_argument("--host", default="0.0.0.0", type=str, help="host to the http server")
parser.add_argument("--workers", default=1, type=int, help="number of workers")
parser.add_argument("--metrics-port", default=8000, type=int, help="port for metrics server")
parser = EngineArgs.add_cli_args(parser)
args = parser.parse_args()


def load_engine():
    """
    Initialize and load the LLM engine.
    
    Raises:
        SystemExit: If engine initialization fails
    """
    api_server_logger.info(f"FastDeploy LLM API server starting... {os.getpid()}")
    engine_args = EngineArgs.from_cli_args(args)
    llm_engine = LLMEngine.from_engine_args(engine_args)

    if not llm_engine.start(api_server_pid=os.getpid()):
        api_server_logger.error("Failed to initialize FastDeploy LLM engine, service exit now!")
        exit(-1)
    else:
        api_server_logger.info(f"FastDeploy LLM engine initialized!\n")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Async context manager for FastAPI application lifespan events.
    
    Args:
        app (FastAPI): The FastAPI application instance
        
    Yields:
        None: After setting up engine client and handlers
    """

    if args.tokenizer is None:
        args.tokenizer = args.model
    if current_process().name != 'MainProcess':
        pid = os.getppid()
    else:
        pid = os.getpid()
    api_server_logger.info(f"{pid}")
    engine_client = EngineClient(args.tokenizer, args.max_model_len, args.tensor_parallel_size, pid, args.enable_mm)
    app.state.dynamic_load_weight = args.dynamic_load_weight
    chat_handler = OpenAIServingChat(engine_client, pid)
    completion_handler = OpenAIServingCompletion(engine_client, pid)
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


# TODO 传递真实引擎值 通过pid 获取状态
@app.get("/health")
def health(request: Request) -> Response:
    """
    Perform health check of the engine service.
    
    Args:
        request (Request): FastAPI request object
        
    Returns:
        Response: HTTP 200 if healthy, 404/304 if errors occur
    """

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
            routes_info.append({
                "path": route.path,
                "methods": methods,
                "tags": tags
            })
    return {"routes": routes_info}


@app.api_route("/ping", methods=["GET", "POST"])
def ping(raw_request: Request) -> Response:
    """
    Ping endpoint for service availability check.
    
    Args:
        raw_request (Request): FastAPI request object
        
    Returns:
        Response: Same as health check response
    """
    return health(raw_request)


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """
    Create chat completion based on the given request.
    
    Args:
        request (ChatCompletionRequest): Chat completion request parameters
        
    Returns:
        Union[JSONResponse, StreamingResponse]: Response containing either:
            - Error details if failed
            - Chat completion results
            - Stream of completion events
    """
    if app.state.dynamic_load_weight:
        status, msg = app.state.engine_client.is_workers_alive()
        if not status:
            return JSONResponse(content={"error": "Worker Service Not Healthy"}, status_code=304)
    generator = await app.state.chat_handler.create_chat_completion(request)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)

    elif isinstance(generator, ChatCompletionResponse):
        return JSONResponse(content=generator.model_dump())

    return StreamingResponse(content=generator, media_type="text/event-stream")


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """
    Create text completion based on the given request.
    
    Args:
        request (CompletionRequest): Completion request parameters
        
    Returns:
        Union[JSONResponse, StreamingResponse]: Response containing either:
            - Error details if failed
            - Completion results 
            - Stream of completion events
    """
    if app.state.dynamic_load_weight:
        status, msg = app.state.engine_client.is_workers_alive()
        if not status:
            return JSONResponse(content={"error": "Worker Service Not Healthy"}, status_code=304)

    generator = await app.state.completion_handler.create_completion(request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    elif isinstance(generator, CompletionResponse):
        return JSONResponse(content=generator.model_dump())

    return StreamingResponse(content=generator, media_type="text/event-stream")


@app.get("/update_model_weight")
def update_model_weight(request: Request) -> Response:
    """
    Update model weights dynamically if enabled.
    
    Args:
        request (Request): FastAPI request object
        
    Returns:
        Response: HTTP 200 if successful, 404 if failed or disabled
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
    Clear dynamically loaded model weights if enabled.
    
    Args:
        request (Request): FastAPI request object
        
    Returns:
        Response: HTTP 200 if successful, 404 if failed or disabled
    """
    if app.state.dynamic_load_weight:
        status, msg =  app.state.engine_client.clear_load_weight()
        if not status:
            return Response(content=msg, status_code=404)
        return Response(status_code=200)
    else:
        return Response(content="Dynamic Load Weight Disabled.", status_code=404)

def launch_api_server(args) -> None:
    """
    Launch the API server with given configuration.
    
    Args:
        args: Command line arguments containing server configuration
        
    Raises:
        Exception: If server launch fails
    """
    api_server_logger.info(f"launch Fastdeploy api server... port: {args.port}")
    api_server_logger.info(f"args: {args.__dict__}")

    try:
        prom_dir = cleanup_prometheus_files(True)
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = prom_dir
        metrics_server_thread = threading.Thread(target=run_main_metrics_server, daemon=True)
        metrics_server_thread.start()
        uvicorn.run(app="fastdeploy.entrypoints.openai.api_server:app",
                    host=args.host,
                    port=args.port,
                    workers=args.workers,
                    log_level="info")  # set log level to error to avoid log
    except Exception as e:
        api_server_logger.error(f"launch sync http server error, {e}")


main_app = FastAPI()


@main_app.get("/metrics")
async def metrics():
    """
    metrics
    """
    metrics_text = get_filtered_metrics(
        EXCLUDE_LABELS,
        extra_register_func=lambda reg: main_process_metrics.register_all(reg, workers=args.workers)
    )
    return Response(metrics_text, media_type=CONTENT_TYPE_LATEST)


def run_main_metrics_server():
    """
    Run metrics server in main process.
    
    Starts a Uvicorn server for Prometheus metrics endpoint.
    """

    uvicorn.run(
        main_app,
        host="0.0.0.0",
        port=args.metrics_port,
        log_level="error"
    )


def main():
    """
    Main entry point for the API server.
    
    Steps:
    1. Check port availability
    2. Load LLM engine
    3. Launch API server
    
    Raises:
        Exception: If ports are unavailable
    """
    if not is_port_available(args.host, args.port):
        raise Exception(f"The parameter `port`:{args.port} is already in use.")
    if not is_port_available(args.host, args.metrics_port):
        raise Exception(f"The parameter `metrics_port`:{args.metrics_port} is already in use.")
    load_engine()
    launch_api_server(args)


if __name__ == "__main__":
    main()
