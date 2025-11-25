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

# This file is modified from https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/run_batch.py

import argparse
import asyncio
import os
import tempfile
import uuid
from argparse import Namespace
from collections.abc import Awaitable
from http import HTTPStatus
from io import StringIO
from multiprocessing import current_process
from typing import Callable, List, Optional, Tuple

import aiohttp
import zmq
from tqdm import tqdm

from fastdeploy.engine.args_utils import EngineArgs
from fastdeploy.engine.engine import LLMEngine
from fastdeploy.entrypoints.chat_utils import load_chat_template
from fastdeploy.entrypoints.engine_client import EngineClient
from fastdeploy.entrypoints.openai.protocol import (
    BatchRequestInput,
    BatchRequestOutput,
    BatchResponseData,
    ChatCompletionResponse,
    ErrorResponse,
)
from fastdeploy.entrypoints.openai.serving_chat import OpenAIServingChat
from fastdeploy.entrypoints.openai.serving_models import ModelPath, OpenAIServingModels
from fastdeploy.entrypoints.openai.tool_parsers import ToolParserManager
from fastdeploy.utils import (
    FlexibleArgumentParser,
    api_server_logger,
    console_logger,
    retrive_model_from_server,
)

_BAR_FORMAT = "{desc}: {percentage:3.0f}% Completed | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]\n"
llm_engine = None
engine_client = None


def make_arg_parser(parser: FlexibleArgumentParser):
    parser.add_argument(
        "-i",
        "--input-file",
        required=True,
        type=str,
        help="The path or url to a single input file. Currently supports local file "
        "paths, or the http protocol (http or https). If a URL is specified, "
        "the file should be available via HTTP GET.",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        required=True,
        type=str,
        help="The path or url to a single output file. Currently supports "
        "local file paths, or web (http or https) urls. If a URL is specified,"
        " the file should be available via HTTP PUT.",
    )
    parser.add_argument(
        "--output-tmp-dir",
        type=str,
        default=None,
        help="The directory to store the output file before uploading it " "to the output URL.",
    )
    parser.add_argument(
        "--max-waiting-time",
        default=-1,
        type=int,
        help="max waiting time for connection, if set value -1 means no waiting time limit",
    )
    parser.add_argument("--port", default=8000, type=int, help="port to the http server")
    # parser.add_argument("--host", default="0.0.0.0", type=str, help="host to the http server")
    parser.add_argument("--workers", default=None, type=int, help="number of workers")
    parser.add_argument("--max-concurrency", default=512, type=int, help="max concurrency")
    parser.add_argument(
        "--enable-mm-output", action="store_true", help="Enable 'multimodal_content' field in response output. "
    )
    parser = EngineArgs.add_cli_args(parser)
    return parser


def parse_args():
    parser = FlexibleArgumentParser(description="FastDeploy OpenAI-Compatible batch runner.")
    args = make_arg_parser(parser).parse_args()
    return args


def init_engine(args: argparse.Namespace):
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


class BatchProgressTracker:

    def __init__(self):
        self._total = 0
        self._completed = 0
        self._pbar: Optional[tqdm] = None
        self._last_log_count = 0

    def submitted(self):
        self._total += 1

    def completed(self):
        self._completed += 1
        if self._pbar:
            self._pbar.update()

        if self._total > 0:
            log_interval = min(100, max(self._total // 10, 1))
            if self._completed - self._last_log_count >= log_interval:
                console_logger.info(f"Progress: {self._completed}/{self._total} requests completed")
                self._last_log_count = self._completed

    def pbar(self) -> tqdm:
        self._pbar = tqdm(
            total=self._total,
            unit="req",
            desc="Running batch",
            mininterval=10,
            bar_format=_BAR_FORMAT,
        )
        return self._pbar


async def read_file(path_or_url: str) -> str:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        async with aiohttp.ClientSession() as session, session.get(path_or_url) as resp:
            return await resp.text()
    else:
        with open(path_or_url, encoding="utf-8") as f:
            return f.read()


async def write_local_file(output_path: str, batch_outputs: list[BatchRequestOutput]) -> None:
    """
    Write the responses to a local file.
    output_path: The path to write the responses to.
    batch_outputs: The list of batch outputs to write.
    """
    # We should make this async, but as long as run_batch runs as a
    # standalone program, blocking the event loop won't effect performance.
    with open(output_path, "w", encoding="utf-8") as f:
        for o in batch_outputs:
            print(o.model_dump_json(), file=f)


async def upload_data(output_url: str, data_or_file: str, from_file: bool) -> None:
    """
    Upload a local file to a URL.
    output_url: The URL to upload the file to.
    data_or_file: Either the data to upload or the path to the file to upload.
    from_file: If True, data_or_file is the path to the file to upload.
    """
    # Timeout is a common issue when uploading large files.
    # We retry max_retries times before giving up.
    max_retries = 5
    # Number of seconds to wait before retrying.
    delay = 5

    for attempt in range(1, max_retries + 1):
        try:
            # We increase the timeout to 1000 seconds to allow
            # for large files (default is 300).
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1000)) as session:
                if from_file:
                    with open(data_or_file, "rb") as file:
                        async with session.put(output_url, data=file) as response:
                            if response.status != 200:
                                raise Exception(
                                    f"Failed to upload file.\n"
                                    f"Status: {response.status}\n"
                                    f"Response: {response.text()}"
                                )
                else:
                    async with session.put(output_url, data=data_or_file) as response:
                        if response.status != 200:
                            raise Exception(
                                f"Failed to upload data.\n"
                                f"Status: {response.status}\n"
                                f"Response: {response.text()}"
                            )

        except Exception as e:
            if attempt < max_retries:
                console_logger.error(
                    "Failed to upload data (attempt %d). Error message: %s.\nRetrying in %d seconds...",  # noqa: E501
                    attempt,
                    e,
                    delay,
                )
                await asyncio.sleep(delay)
            else:
                raise Exception(
                    f"Failed to upload data (attempt {attempt}). Error message: {str(e)}."  # noqa: E501
                ) from e


async def write_file(path_or_url: str, batch_outputs: list[BatchRequestOutput], output_tmp_dir: str) -> None:
    """
    Write batch_outputs to a file or upload to a URL.
    path_or_url: The path or URL to write batch_outputs to.
    batch_outputs: The list of batch outputs to write.
    output_tmp_dir: The directory to store the output file before uploading it
    to the output URL.
    """
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        if output_tmp_dir is None:
            console_logger.info("Writing outputs to memory buffer")
            output_buffer = StringIO()
            for o in batch_outputs:
                print(o.model_dump_json(), file=output_buffer)
            output_buffer.seek(0)
            console_logger.info("Uploading outputs to %s", path_or_url)
            await upload_data(
                path_or_url,
                output_buffer.read().strip().encode("utf-8"),
                from_file=False,
            )
        else:
            # Write responses to a temporary file and then upload it to the URL.
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=output_tmp_dir,
                prefix="tmp_batch_output_",
                suffix=".jsonl",
            ) as f:
                console_logger.info("Writing outputs to temporary local file %s", f.name)
                await write_local_file(f.name, batch_outputs)
                console_logger.info("Uploading outputs to %s", path_or_url)
                await upload_data(path_or_url, f.name, from_file=True)
    else:
        console_logger.info("Writing outputs to local file %s", path_or_url)
        await write_local_file(path_or_url, batch_outputs)


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


def make_error_request_output(request: BatchRequestInput, error_msg: str) -> BatchRequestOutput:
    batch_output = BatchRequestOutput(
        id=f"fastdeploy-{random_uuid()}",
        custom_id=request.custom_id,
        response=BatchResponseData(
            status_code=HTTPStatus.BAD_REQUEST,
            request_id=f"fastdeploy-batch-{random_uuid()}",
        ),
        error=error_msg,
    )
    return batch_output


async def make_async_error_request_output(request: BatchRequestInput, error_msg: str) -> BatchRequestOutput:
    return make_error_request_output(request, error_msg)


async def run_request(
    serving_engine_func: Callable,
    request: BatchRequestInput,
    tracker: BatchProgressTracker,
    semaphore: asyncio.Semaphore,
) -> BatchRequestOutput:
    async with semaphore:
        try:
            response = await serving_engine_func(request.body)

            if isinstance(response, ChatCompletionResponse):
                batch_output = BatchRequestOutput(
                    id=f"fastdeploy-{random_uuid()}",
                    custom_id=request.custom_id,
                    response=BatchResponseData(
                        status_code=200, body=response, request_id=f"fastdeploy-batch-{random_uuid()}"
                    ),
                    error=None,
                )
            elif isinstance(response, ErrorResponse):
                batch_output = BatchRequestOutput(
                    id=f"fastdeploy-{random_uuid()}",
                    custom_id=request.custom_id,
                    response=BatchResponseData(status_code=400, request_id=f"fastdeploy-batch-{random_uuid()}"),
                    error=response,
                )
            else:
                batch_output = make_error_request_output(request, error_msg="Request must not be sent in stream mode")

            tracker.completed()
            return batch_output

        except Exception as e:
            console_logger.error(f"Request {request.custom_id} processing failed: {str(e)}")
            tracker.completed()
            return make_error_request_output(request, error_msg=f"Request processing failed: {str(e)}")


def determine_process_id() -> int:
    """Determine the appropriate process ID."""
    if current_process().name != "MainProcess":
        return os.getppid()
    else:
        return os.getpid()


def create_model_paths(args: Namespace) -> List[ModelPath]:
    """Create model paths configuration."""
    if args.served_model_name is not None:
        served_model_names = args.served_model_name
        verification = True
    else:
        served_model_names = args.model
        verification = False

    return [ModelPath(name=served_model_names, model_path=args.model, verification=verification)]


async def initialize_engine_client(args: Namespace, pid: int) -> EngineClient:
    """Initialize and configure the engine client."""
    engine_args = EngineArgs.from_cli_args(args)
    config = engine_args.create_engine_config(port_availability_check=False)
    engine_client = EngineClient(
        model_name_or_path=args.model,
        tokenizer=args.tokenizer,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        pid=pid,
        port=int(args.engine_worker_queue_port[args.local_data_parallel_id]),
        limit_mm_per_prompt=args.limit_mm_per_prompt,
        mm_processor_kwargs=args.mm_processor_kwargs,
        reasoning_parser=args.reasoning_parser,
        data_parallel_size=args.data_parallel_size,
        enable_logprob=args.enable_logprob,
        workers=args.workers,
        tool_parser=args.tool_call_parser,
        config=config,
    )

    await engine_client.connection_manager.initialize()
    engine_client.create_zmq_client(model=pid, mode=zmq.PUSH)
    engine_client.pid = pid

    return engine_client


def create_serving_handlers(
    args: Namespace, engine_client: EngineClient, model_paths: List[ModelPath], chat_template: str, pid: int
) -> OpenAIServingChat:
    """Create model and chat serving handlers."""
    model_handler = OpenAIServingModels(
        model_paths,
        args.max_model_len,
        args.ips,
    )

    chat_handler = OpenAIServingChat(
        engine_client,
        model_handler,
        pid,
        args.ips,
        args.max_waiting_time,
        chat_template,
        args.enable_mm_output,
        args.tokenizer_base_url,
    )

    return chat_handler


async def setup_engine_and_handlers(args: Namespace) -> Tuple[EngineClient, OpenAIServingChat]:
    """Setup engine client and all necessary handlers."""

    if args.tokenizer is None:
        args.tokenizer = args.model

    pid = determine_process_id()
    console_logger.info(f"Process ID: {pid}")

    model_paths = create_model_paths(args)
    chat_template = load_chat_template(args.chat_template, args.model)

    # Initialize engine client
    engine_client = await initialize_engine_client(args, pid)
    engine_client = engine_client

    # Create handlers
    chat_handler = create_serving_handlers(args, engine_client, model_paths, chat_template, pid)

    # Update data processor if engine exists
    if llm_engine is not None:
        llm_engine.engine.data_processor = engine_client.data_processor

    return engine_client, chat_handler


async def run_batch(
    args: argparse.Namespace,
) -> None:

    # Setup engine and handlers
    engine_client, chat_handler = await setup_engine_and_handlers(args)

    concurrency = getattr(args, "max_concurrency", 512)
    workers = getattr(args, "workers", 1)
    max_concurrency = (concurrency + workers - 1) // workers
    semaphore = asyncio.Semaphore(max_concurrency)

    console_logger.info(f"concurrency: {concurrency}, workers: {workers}, max_concurrency: {max_concurrency}")

    tracker = BatchProgressTracker()
    console_logger.info("Reading batch from %s...", args.input_file)

    # Submit all requests in the file to the engine "concurrently".
    response_futures: list[Awaitable[BatchRequestOutput]] = []
    for request_json in (await read_file(args.input_file)).strip().split("\n"):
        # Skip empty lines.
        request_json = request_json.strip()
        if not request_json:
            continue

        request = BatchRequestInput.model_validate_json(request_json)

        # Determine the type of request and run it.
        if request.url == "/v1/chat/completions":
            chat_handler_fn = chat_handler.create_chat_completion if chat_handler is not None else None
            if chat_handler_fn is None:
                response_futures.append(
                    make_async_error_request_output(
                        request,
                        error_msg="The model does not support Chat Completions API",
                    )
                )
                continue

            response_futures.append(run_request(chat_handler_fn, request, tracker, semaphore))
            tracker.submitted()
        else:
            response_futures.append(
                make_async_error_request_output(
                    request,
                    error_msg=f"URL {request.url} was used. "
                    "Supported endpoints: /v1/chat/completions"
                    "See fastdeploy/entrypoints/openai/api_server.py for supported "
                    "/v1/chat/completions versions.",
                )
            )

    with tracker.pbar():
        responses = await asyncio.gather(*response_futures)

    success_count = sum(1 for r in responses if r.error is None)
    error_count = len(responses) - success_count
    console_logger.info(f"Batch processing completed: {success_count} success, {error_count} errors")

    await write_file(args.output_file, responses, args.output_tmp_dir)
    console_logger.info("Results written to output file")


async def main(args: argparse.Namespace):
    console_logger.info("Starting batch runner with args: %s", args)
    try:
        if args.workers is None:
            args.workers = max(min(int(args.max_num_seqs // 32), 8), 1)

        args.model = retrive_model_from_server(args.model, args.revision)

        if args.tool_parser_plugin:
            ToolParserManager.import_tool_parser(args.tool_parser_plugin)

        if not init_engine(args):
            return

        await run_batch(args)
    except Exception as e:
        print("Fatal error in main:")
        print(e)
        console_logger.error(f"Fatal error in main: {e}", exc_info=True)
        raise
    finally:
        await cleanup_resources()


async def cleanup_resources() -> None:
    """Clean up all resources during shutdown."""
    try:
        # stop engine
        if llm_engine is not None:
            try:
                llm_engine._exit_sub_services()
            except Exception as e:
                console_logger.error(f"Error stopping engine: {e}")

        # close client connections
        if engine_client is not None:
            try:
                if hasattr(engine_client, "zmq_client"):
                    engine_client.zmq_client.close()
                if hasattr(engine_client, "connection_manager"):
                    await engine_client.connection_manager.close()
            except Exception as e:
                console_logger.error(f"Error closing client connections: {e}")

        # garbage collect
        import gc

        gc.collect()
        print("run batch done")

    except Exception as e:
        console_logger.error(f"Error during cleanup: {e}")


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args=args))
