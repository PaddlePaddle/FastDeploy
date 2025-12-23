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

import json
import os
import signal
import traceback

import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response, StreamingResponse

from fastdeploy.engine.args_utils import EngineArgs
from fastdeploy.engine.engine import LLMEngine
from fastdeploy.utils import (
    FlexibleArgumentParser,
    api_server_logger,
    is_port_available,
)

app = FastAPI()

llm_engine = None


def cleanup_engine():
    """清理引擎资源"""
    global llm_engine
    if llm_engine is not None:
        try:
            if hasattr(llm_engine, "worker_proc") and llm_engine.worker_proc is not None:
                try:
                    # 检查进程是否已经结束
                    if llm_engine.worker_proc.poll() is not None:
                        api_server_logger.info("Worker process already terminated")
                        return
                    
                    pgid = os.getpgid(llm_engine.worker_proc.pid)
                    api_server_logger.info(f"Terminating worker process group {pgid}")
                    os.killpg(pgid, signal.SIGTERM)
                    # 等待进程结束
                    llm_engine.worker_proc.wait(timeout=5)
                    api_server_logger.info("Worker process terminated successfully")
                except ProcessLookupError:
                    api_server_logger.info("Worker process already terminated")
                except Exception as e:
                    api_server_logger.error(f"Error terminating worker process: {e}")
        except Exception as e:
            api_server_logger.error(f"Error during cleanup: {e}")


def signal_handler(signum, frame):
    """处理SIGINT和SIGTERM信号"""
    sig_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
    api_server_logger.info(f"Received {sig_name}, initiating graceful shutdown...")
    cleanup_engine()
    # 让uvicorn处理实际的退出


def init_app(args):
    """
    init LLMEngine
    """

    global llm_engine
    engine_args = EngineArgs.from_cli_args(args)
    llm_engine = LLMEngine.from_engine_args(engine_args)
    if not llm_engine.start():
        api_server_logger.error("Failed to initialize FastDeploy LLM engine, service exit now!")
        return False

    api_server_logger.info("FastDeploy LLM engine initialized!")
    return True


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/generate")
async def generate(request: dict):
    """
    generate stream api
    """
    api_server_logger.info(f"Receive request: {request}")
    stream = request.get("stream", 0)

    if not stream:
        output = {}
        try:
            # 将生成过程包裹在try块中以捕获异常
            for result in llm_engine.generate(request, stream):
                output = result
        except Exception as e:
            # 记录完整的异常堆栈信息
            api_server_logger.error(f"Error during generation: {e!s}", exc_info=True)
            # 返回结构化的错误消息并终止流
            output = {"error": str(e), "error_type": e.__class__.__name__}
        return output

    async def event_generator():
        try:
            # 将生成过程包裹在try块中以捕获异常
            for result in llm_engine.generate(request, stream):
                yield f"data: {json.dumps(result)}\n\n"
        except Exception as e:
            # 记录完整的异常堆栈信息
            api_server_logger.error(f"Error during generation: {e!s}", exc_info=True)
            # 返回结构化的错误消息并终止流
            error_msg = {"error": str(e), "error_type": e.__class__.__name__}
            yield f"data: {json.dumps(error_msg)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def launch_api_server(args) -> None:
    """
    启动http服务
    """
    # 设置信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    if not is_port_available(args.host, args.port):
        raise Exception(f"The parameter `port`:{args.port} is already in use.")

    api_server_logger.info(f"launch Fastdeploy api server... port: {args.port}")
    api_server_logger.info(f"args: {args.__dict__}")

    if not init_app(args):
        api_server_logger.error("API Server launch failed.")
        return

    try:
        uvicorn.run(
            app=app,
            host=args.host,
            port=args.port,
            workers=args.workers,
            log_level="info",
        )  # set log level to error to avoid log
    except Exception as e:
        api_server_logger.error(f"launch sync http server error, {e}, {str(traceback.format_exc())}")
    finally:
        cleanup_engine()


def main():
    """main函数"""
    parser = FlexibleArgumentParser()
    parser.add_argument("--port", default=9904, type=int, help="port to the http server")
    parser.add_argument("--host", default="0.0.0.0", type=str, help="host to the http server")
    parser.add_argument("--workers", default=1, type=int, help="number of workers")
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    launch_api_server(args)


if __name__ == "__main__":
    main()
