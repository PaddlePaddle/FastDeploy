# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import os

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from server.http_server.api import (
    Req,
    chat_completion_generator,
    chat_completion_result,
)
from server.utils import http_server_logger

http_server_logger.info(f"create fastapi app...")
app = FastAPI()

@app.post("/v1/chat/completions")
def create_chat_completion(req: Req):
    """
    服务端路由函数
    返回：
        如果stream为True，流式返回
            如果正常，返回{'token': xxx, 'is_end': xxx, 'send_idx': xxx, ..., 'error_msg': '', 'error_code': 0}
            如果异常，返回{'error_msg': xxx, 'error_code': xxx}，error_msg字段不为空，error_code字段不为0
        如果stream为False，非流式返回
            如果正常，返回{'result': xxx, 'error_msg': '', 'error_code': 0}
            如果异常，返回{'result': '', 'error_msg': xxx, 'error_code': xxx}，error_msg字段不为空，error_code字段不为0
    """
    try:
        http_server_logger.info(f"receive request: {req.req_id}")
        grpc_port = int(os.getenv("GRPC_PORT", 0))
        if grpc_port == 0:
            return {"error_msg": f"GRPC_PORT ({grpc_port}) for infer service is invalid",
                    "error_code": 400}
        grpc_url = f"localhost:{grpc_port}"

        if req.stream:
            generator = chat_completion_generator(infer_grpc_url=grpc_url, req=req, yield_json=True)
            resp = StreamingResponse(generator, media_type="text/event-stream")
        else:
            resp = chat_completion_result(infer_grpc_url=grpc_url, req=req)
    except Exception as e:
        resp = {'error_msg': str(e), 'error_code': 501}
    finally:
        http_server_logger.info(f"finish request: {req.req_id}")
        return resp

def launch_http_server(port: int, workers: int) -> None:
    """
    启动http服务
    """
    http_server_logger.info(f"launch http server... port: {port}, workers: {workers}")
    try:
        uvicorn.run(app="server.http_server.app:app",
                    host='0.0.0.0',
                    port=port,
                    workers=workers,
                    log_level="error")
    except Exception as e:
        http_server_logger.error(f"launch http server error, {e}")

def main():
    """main函数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=9904, type=int, help="port to the http server")
    parser.add_argument("--workers", default=1, type=int, help="set the number of workers for the http service")
    args = parser.parse_args()
    launch_http_server(port=args.port, workers=args.workers)

if __name__ == "__main__":
    main()
