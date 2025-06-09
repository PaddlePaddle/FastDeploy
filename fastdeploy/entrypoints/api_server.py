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

import uvicorn
import json
from fastapi import FastAPI
from fastapi.responses import Response, StreamingResponse

from fastdeploy.utils import FlexibleArgumentParser, api_server_logger, is_port_available
from fastdeploy.engine.args_utils import EngineArgs
from fastdeploy.engine.engine import LLMEngine

app = FastAPI()

llm_engine = None

def init_app(args):
    """
    Initialize the LLMEngine instance.
    
    Args:
        args: Command line arguments containing engine configuration
        
    Returns:
        bool: True if initialization succeeded, False otherwise
    """

    global llm_engine
    engine_args = EngineArgs.from_cli_args(args)
    llm_engine = LLMEngine.from_engine_args(engine_args)
    if not llm_engine.start():
        api_server_logger.error("Failed to initialize FastDeploy LLM engine, service exit now!")
        return False

    api_server_logger.info(f"FastDeploy LLM engine initialized!")
    return True


@app.get("/health")
async def health() -> Response:
    """
    Health check endpoint for the API server.
    
    Returns:
        Response: HTTP 200 response if server is healthy
    """
    return Response(status_code=200)

@app.post("/generate")
async def generate(request: dict):
    """
    Generate text based on the given request.
    Supports both streaming and non-streaming modes.
    
    Args:
        request: Dictionary containing generation parameters and input text
        
    Returns:
        Response: Either a direct response (non-streaming) or streaming response
    """
    api_server_logger.info(f"Receive request: {request}")
    stream = request.get("stream", 0)

    if not stream:
        output = {}
        try:
            # Wrap generation in try block to handle exceptions
            for result in llm_engine.generate(request, stream):
                output = result
        except Exception as e:
            # Log full exception stack trace
            api_server_logger.error(f"Error during generation: {str(e)}", exc_info=True)
            # Return structured error message and terminate stream
            output = {"error": str(e), "error_type": e.__class__.__name__}
        return output

    async def event_generator():
        try:
            # Wrap generation in try block to handle exceptions
            for result in llm_engine.generate(request, stream):
                yield f"data: {json.dumps(result)}\n\n"
        except Exception as e:
            # Log full exception stack trace
            api_server_logger.error(f"Error during generation: {str(e)}", exc_info=True)
            # Return structured error message and terminate stream
            error_msg = {"error": str(e), "error_type": e.__class__.__name__}
            yield  f"data: {json.dumps(error_msg)}\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")

def launch_api_server(args) -> None:
    """
    Launch the FastDeploy API server.
    
    Args:
        args: Command line arguments containing server configuration
        
    Raises:
        Exception: If the specified port is already in use
    """
    if not is_port_available(args.host, args.port):
        raise Exception(f"The parameter `port`:{args.port} is already in use.")

    api_server_logger.info(f"launch Fastdeploy api server... port: {args.port}")
    api_server_logger.info(f"args: {args.__dict__}")

    if not init_app(args):
        api_server_logger.error("API Server launch failed.")
        return

    try:
        uvicorn.run(app=app,
                    host=args.host,
                    port=args.port,
                    workers=args.workers,
                    log_level="info")  # set log level to error to avoid log
    except Exception as e:
        api_server_logger.error(f"launch sync http server error, {e}")


def main():
    """
    Main entry point for the API server.
    Parses command line arguments and launches the server.
    """
    parser = FlexibleArgumentParser()
    parser.add_argument("--port", default=9904, type=int, help="port to the http server")
    parser.add_argument("--host", default="0.0.0.0", type=str, help="host to the http server")
    parser.add_argument("--workers", default=1, type=int, help="number of workers")
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    launch_api_server(args)
    

if __name__ == "__main__":
    main()
