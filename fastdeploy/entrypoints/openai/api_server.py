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

import argparse
import asyncio
import multiprocessing
import os
import signal
import socket
import sys
import threading
import time
import traceback
import weakref
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from multiprocessing import connection, current_process
from multiprocessing.process import BaseProcess
from typing import Any, Callable, Optional

import setproctitle
import uvicorn
import uvloop
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
    is_valid_ipv6_address,
    kill_process_tree,
    retrive_model_from_server,
)

llm_engine = None


def make_arg_parser(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
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
    return parser


def rewrite_args(args: argparse.Namespace) -> argparse.Namespace:
    console_logger.info(f"Number of api-server workers: {args.workers}.")

    args.model = retrive_model_from_server(args.model, args.revision)
    if args.tool_parser_plugin:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)
    return args


def load_engine(args: argparse.Namespace):
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


def load_data_service(args: argparse.Namespace) -> ExpertService:
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


# Adapted from: https://github.com/sgl-project/sglang/blob/v0.4.1/python/sglang/srt/utils.py#L630 # noqa: E501


def set_ulimit(target_soft_limit=65535):
    if sys.platform.startswith("win"):
        api_server_logger.info("Windows detected, skipping ulimit adjustment.")
        return

    import resource

    resource_type = resource.RLIMIT_NOFILE
    current_soft, current_hard = resource.getrlimit(resource_type)

    if current_soft < target_soft_limit:
        try:
            resource.setrlimit(resource_type, (target_soft_limit, current_hard))
        except ValueError as e:
            api_server_logger.warning(
                "Found ulimit of %s and failed to automatically increase "
                "with error %s. This can cause fd limit errors like "
                "`OSError: [Errno 24] Too many open files`. Consider "
                "increasing with ulimit -n",
                current_soft,
                e,
            )


class APIServerProcessManager:
    """Manages a group of API server processes.

    Handles creation, monitoring, and termination of API server worker
    processes. Also monitors extra processes to check if they are healthy.
    """

    def __init__(
        self,
        target_server_fn: Callable,
        listen_address: str,
        sock: Any,
        args: argparse.Namespace,
        num_servers: int,
    ):
        """Initialize and start API server worker processes.

        Args:
            target_server_fn: Function to call for each API server process
            listen_address: Address to listen for client connections
            sock: Socket for client connections
            args: Command line arguments
            num_servers: Number of API server processes to start
            stats_update_address: Optional stats update address
        """
        self.listen_address = listen_address
        self.sock = sock
        self.args = args

        # Start API servers
        spawn_context = multiprocessing.get_context("spawn")
        self.processes: list[BaseProcess] = []

        for i in range(num_servers):
            client_config = {"client_count": num_servers, "client_index": i}

            proc = spawn_context.Process(
                target=target_server_fn, name=f"ApiServer_{i}", args=(args, listen_address, sock, client_config)
            )
            self.processes.append(proc)
            proc.start()

        api_server_logger.info("Started %d API server processes", len(self.processes))

        # Shutdown only the API server processes on garbage collection
        # The extra processes are managed by their owners
        self._finalizer = weakref.finalize(self, self.shutdown, self.processes)

    def close(self) -> None:
        self._finalizer()

    # Note(rob): shutdown function cannot be a bound method,
    # else the gc cannot collect the object.
    def shutdown(self, procs: list[BaseProcess]):
        # Shutdown the process.
        for proc in procs:
            if proc.is_alive():
                proc.terminate()

        # Allow 5 seconds for remaining procs to terminate.
        deadline = time.monotonic() + 5
        for proc in procs:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if proc.is_alive():
                proc.join(remaining)

        for proc in procs:
            if proc.is_alive() and (pid := proc.pid) is not None:
                kill_process_tree(pid)


def create_server_socket(addr: tuple[str, int]) -> socket.socket:
    family = socket.AF_INET
    if is_valid_ipv6_address(addr[0]):
        family = socket.AF_INET6

    sock = socket.socket(family=family, type=socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind(addr)
    return sock


def setup_server(args):
    """Validate API server args, set up signal handler, create socket
    ready to serve."""
    # workaround to make sure that we bind the port before the engine is set up.
    # This avoids race conditions with ray.
    # see https://github.com/vllm-project/vllm/issues/8204
    sock_addr = (args.host or "", args.port)
    sock = create_server_socket(sock_addr)

    # workaround to avoid footguns where uvicorn drops requests with too
    # many concurrent requests active
    set_ulimit()

    def signal_handler(*_) -> None:
        # Interrupt server on sigterm while initializing
        raise KeyboardInterrupt("terminated")

    signal.signal(signal.SIGTERM, signal_handler)

    addr, port = sock_addr
    # is_ssl = args.ssl_keyfile and args.ssl_certfile
    is_ssl = False
    host_part = f"[{addr}]" if is_valid_ipv6_address(addr) else addr or "0.0.0.0"
    listen_address = f"http{'s' if is_ssl else ''}://{host_part}:{port}"
    return listen_address, sock


async def serve_http(app: FastAPI, sock: Optional[socket.socket], **uvicorn_kwargs: Any):
    config = uvicorn.Config(app, **uvicorn_kwargs)
    config.workers = 1
    config.load()
    server = uvicorn.Server(config=config)
    loop = asyncio.get_running_loop()
    server_task = loop.create_task(server.serve(sockets=[sock] if sock else None))

    async def dummy_shutdown() -> None:
        pass

    try:
        await server_task
        return dummy_shutdown()
    except asyncio.CancelledError:
        return server.shutdown()


def run_api_server_worker_proc(args, listen_address, sock, client_config=None, **uvicorn_kwargs) -> None:
    """Entrypoint for individual API server worker processes."""

    # 设置进程标题，并为标准输出和标准错误添加特定于进程的前缀
    # Set process title and add process-specific prefix to stdout and stderr.
    server_index = client_config.get("client_index", 0) if client_config else 0
    setproctitle.setproctitle(f"APIServer::{server_index}")

    uvloop.run(run_server_worker(listen_address, sock, args, client_config, **uvicorn_kwargs))


async def run_server_worker(listen_address, sock, args, client_config=None, **uvicorn_kwargs) -> None:
    api_server_app = ApiServerApp(args)
    # 异步启动HTTP服务
    shutdown_task = await serve_http(
        app=api_server_app.build_app(),
        sock=sock,
        host=args.host,
        port=args.port,
        log_config=UVICORN_CONFIG,
        log_level="info",
        **uvicorn_kwargs,
    )
    # NB: Await server shutdown only after the backend context is exited
    try:
        await shutdown_task
    finally:
        sock.close()


def run_multi_api_server(args: argparse.Namespace):
    listen_address, sock = setup_server(args)  # Construct common args for the APIServerProcessManager up-front.
    api_server_manager_kwargs = dict(
        target_server_fn=run_api_server_worker_proc,
        listen_address=listen_address,
        sock=sock,
        args=args,
        num_servers=args.workers,
    )

    api_server_manager = APIServerProcessManager(**api_server_manager_kwargs)
    wait_for_completion_or_failure(api_server_manager)


def wait_for_completion_or_failure(api_server_manager: APIServerProcessManager) -> None:
    """Wait for all processes to complete or detect if any fail.

    Raises an exception if any process exits with a non-zero status.

    Args:
        api_server_manager: The manager for API servers.
        engine_manager: The manager for engine processes.
            If CoreEngineProcManager, it manages local engines;
            if CoreEngineActorManager, it manages all engines.
        coordinator: The coordinator for data parallel.
    """

    try:
        api_server_logger.info("Waiting for API servers to complete ...")
        # Create a mapping of sentinels to their corresponding processes
        # for efficient lookup
        sentinel_to_proc: dict[Any, BaseProcess] = {proc.sentinel: proc for proc in api_server_manager.processes}

        # Check if any process terminates
        while sentinel_to_proc:
            # Wait for any process to terminate
            ready_sentinels: list[Any] = connection.wait(sentinel_to_proc, timeout=5)

            # Process any terminated processes
            for sentinel in ready_sentinels:
                proc = sentinel_to_proc.pop(sentinel)

                # Check if process exited with error
                if proc.exitcode != 0:
                    raise RuntimeError(
                        f"Process {proc.name} (PID: {proc.pid}) " f"died with exit code {proc.exitcode}"
                    )

    except KeyboardInterrupt:
        api_server_logger.info("Received KeyboardInterrupt, shutting down API servers...")
    except Exception as e:
        api_server_logger.exception("Exception occurred while running API servers: %s", str(e))
        raise
    finally:
        api_server_logger.info("Terminating remaining processes ...")
        api_server_manager.close()


class ApiServerApp(FastAPI):

    def __init__(self, args: argparse.Namespace):
        self.args = args

    def build_app(self) -> FastAPI:
        args = self.args
        MAX_CONCURRENT_CONNECTIONS = (args.max_concurrency + args.workers - 1) // args.workers
        connection_semaphore = StatefulSemaphore(MAX_CONCURRENT_CONNECTIONS)
        chat_template = load_chat_template(args.chat_template, args.model)

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
        instrument(app)
        app.add_exception_handler(RequestValidationError, ExceptionHandler.handle_request_validation_exception)
        app.add_exception_handler(Exception, ExceptionHandler.handle_exception)

        @asynccontextmanager
        async def connection_manager():
            """
            async context manager for connection manager
            """
            try:
                await asyncio.wait_for(connection_semaphore.acquire(), timeout=0.001)
                yield
            except asyncio.TimeoutError:
                api_server_logger.info(
                    f"Reach max request concurrency, semaphore status: {connection_semaphore.status()}"
                )
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

        return app

    def launch_api_server(self) -> None:
        """
        启动http服务
        """
        args = self.args
        if not is_port_available(args.host, args.port):
            raise Exception(f"The parameter `port`:{args.port} is already in use.")

        api_server_logger.info(f"launch Fastdeploy api server... port: {args.port}")
        api_server_logger.info(f"args: {args.__dict__}")
        fd_start_span("FD_START")

        try:
            if args.workers > 1:
                run_multi_api_server(args)
            else:
                app = self.build_app()
                uvicorn.run(
                    app=app,
                    host=args.host,
                    port=args.port,
                    workers=args.workers,
                    log_config=UVICORN_CONFIG,
                    log_level="info",
                    timeout_graceful_shutdown=args.timeout_graceful_shutdown,
                )  # set log level to error to avoid log
        except Exception as e:
            api_server_logger.error(f"launch sync http server error, {e}, {str(traceback.format_exc())}")
        print("fastdeploy api server stopped")


class MetricsServerApp(FastAPI):

    def __init__(self, args):
        self.args = args

    def build_app(self) -> FastAPI:
        metrics_app = FastAPI()

        @metrics_app.get("/metrics")
        async def metrics():
            """
            metrics
            """
            metrics_text = get_filtered_metrics(
                EXCLUDE_LABELS,
                extra_register_func=lambda reg: main_process_metrics.register_all(reg, workers=self.args.workers),
            )
            return Response(metrics_text, media_type=CONTENT_TYPE_LATEST)

        return metrics_app

    def run_metrics_server(self):
        """
        run metrics server
        """
        metrics_app = self.build_app()
        uvicorn.run(
            metrics_app, host="0.0.0.0", port=self.args.metrics_port, log_config=UVICORN_CONFIG, log_level="error"
        )

    def launch_metrics_server(self):
        """Metrics server running the sub thread"""
        args = self.args
        if not is_port_available(args.host, args.metrics_port):
            raise Exception(f"The parameter `metrics_port`:{args.metrics_port} is already in use.")

        prom_dir = cleanup_prometheus_files(True)
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = prom_dir
        metrics_server_thread = threading.Thread(target=self.run_metrics_server, daemon=True)
        metrics_server_thread.start()
        time.sleep(1)


class ControllerServerApp(FastAPI):

    def __init__(self, args):
        self.args = args

    def build_app(self) -> FastAPI:
        controller_app = FastAPI()

        @controller_app.post("/controller/reset_scheduler")
        def reset_scheduler():
            """
            reset scheduler
            """
            global llm_engine

            if llm_engine is None:
                return Response("Engine not loaded", status_code=500)
            llm_engine.engine.scheduler.reset()
            return Response("Scheduler Reset Successfully", status_code=200)

        @controller_app.post("/controller/scheduler")
        def control_scheduler(request: ControlSchedulerRequest):
            """
            Control the scheduler behavior with the given parameters.
            """
            content = ErrorResponse(error=ErrorInfo(message="Scheduler updated successfully", code="0"))

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

        return controller_app

    def run_controller_server(self):
        """
        run controller server
        """
        app = self.build_app()
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=self.args.controller_port,
            log_config=UVICORN_CONFIG,
            log_level="error",
        )

    def launch_controller_server(self):
        """Controller server running the sub thread"""
        args = self.args
        if args.controller_port < 0:
            return

        if not is_port_available(args.host, args.controller_port):
            raise Exception(f"The parameter `controller_port`:{args.controller_port} is already in use.")

        controller_server_thread = threading.Thread(target=self.run_controller_server, daemon=True)
        controller_server_thread.start()
        time.sleep(1)


def main(args: argparse.Namespace):
    """main函数"""
    args = rewrite_args(args)
    if args.local_data_parallel_id == 0:
        if not load_engine(args):
            return
    else:
        if not load_data_service(args):
            return
    api_server_logger.info("FastDeploy LLM engine initialized!\n")
    console_logger.info(f"Launching metrics service at http://{args.host}:{args.metrics_port}/metrics")
    console_logger.info(f"Launching chat completion service at http://{args.host}:{args.port}/v1/chat/completions")
    console_logger.info(f"Launching completion service at http://{args.host}:{args.port}/v1/completions")
    controller_server = ControllerServerApp(args)
    controller_server.launch_controller_server()
    metrics_server = MetricsServerApp(args)
    metrics_server.launch_metrics_server()
    api_server = ApiServerApp(args)
    api_server.launch_api_server()


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    main(args)
