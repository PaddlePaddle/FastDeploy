"""
Async Router server for FastDeploy.
Handles client requests and manages prefill/decode/mixed instances.
This module references the router implementation of slglang and vllm.
"""

import asyncio
import copy
import json
import os
import random
import traceback
from dataclasses import dataclass
from itertools import chain
from typing import Dict, List, Optional
from uuid import uuid4

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import ORJSONResponse, Response, StreamingResponse

from fastdeploy.router.utils import (
    InstanceInfo,
    InstanceRole,
    check_service_health_async,
)
from fastdeploy.utils import FlexibleArgumentParser
from fastdeploy.utils import router_logger as logger

app = FastAPI()
_background_tasks = set()


@dataclass
class RouterArgs:
    host: str = "0.0.0.0"
    """
    Host address to bind the router server
    """
    port: int = 9000
    """
    Port to bind the router server.
    """
    splitwise: bool = False
    """
    Router uses splitwise deployment
    """
    request_timeout_secs: int = 1800
    """
    Request timeout in seconds
    """
    preempt_retry_count: int = 3
    """
    Max retry count when decode instance preempts a request in splitwise mode.
    """
    preempt_retry_exclude_decode: bool = False
    """
    Whether to exclude the previously used decode instance when retrying after preemption.
    """

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        parser.add_argument(
            "--host",
            type=str,
            default=RouterArgs.host,
            help="Host address to bind the router server.",
        )
        parser.add_argument(
            "--port",
            type=int,
            default=RouterArgs.port,
            help="Port number to bind the router server",
        )
        parser.add_argument(
            "--splitwise",
            action="store_true",
            default=RouterArgs.splitwise,
            help="Router uses splitwise deployment",
        )
        parser.add_argument(
            "--request-timeout-secs",
            type=int,
            default=RouterArgs.request_timeout_secs,
            help="Request timeout in seconds",
        )
        parser.add_argument(
            "--preempt-retry-count",
            type=int,
            default=RouterArgs.preempt_retry_count,
            help="Max retry count when decode instance preempts a request in splitwise mode.",
        )
        parser.add_argument(
            "--preempt-retry-exclude-decode",
            action="store_true",
            default=RouterArgs.preempt_retry_exclude_decode,
            help="Whether to exclude the previously used decode instance when retrying after preemption.",
        )
        return parser


class Router:
    """
    Router class that handles requests from client and
    collects prefill/decode instance information
    """

    def __init__(self, args):
        self.args = args
        self.host = args.host
        self.port = args.port
        self.splitwise = args.splitwise
        self.timeout = args.request_timeout_secs
        self.preempt_retry_count = args.preempt_retry_count
        self.preempt_retry_exclude_decode = args.preempt_retry_exclude_decode

        self.mixed_servers = []
        self.prefill_servers = []
        self.decode_servers = []
        self.lock = asyncio.Lock()  # async-safe lock
        logger.info("Router started at http://{}:{}".format(self.host, self.port))

    async def register_instance(self, instance_info_dict: dict):
        """Register an instance asynchronously"""
        try:
            inst_info = InstanceInfo.from_dict(instance_info_dict)
        except Exception as e:
            logger.error(f"register instance failed: {e}")
            raise

        if (self.splitwise and inst_info.role == InstanceRole.MIXED) or (
            not self.splitwise and inst_info.role != InstanceRole.MIXED
        ):
            raise ValueError(f"Invalid instance role: {inst_info.role}, splitwise: {self.splitwise}")

        if not await check_service_health_async(inst_info.url()):
            raise RuntimeError(f"Instance {inst_info} is not healthy")

        async with self.lock:
            instance_key = inst_info.get_key()

            if inst_info.role == InstanceRole.MIXED:
                self._update_or_add_instance(self.mixed_servers, inst_info, instance_key, "mixed")
            elif inst_info.role == InstanceRole.PREFILL:
                self._update_or_add_instance(self.prefill_servers, inst_info, instance_key, "prefill")
            elif inst_info.role == InstanceRole.DECODE:
                self._update_or_add_instance(self.decode_servers, inst_info, instance_key, "decode")

    def _update_or_add_instance(self, server_list: List, inst_info: InstanceInfo, key: str, role_name: str):
        """Update existing instance or add new one based on key (host_ip:port)."""
        for i, existing in enumerate(server_list):
            if existing.get_key() == key:
                if existing != inst_info:
                    server_list[i] = inst_info
                    logger.info(f"Updated {role_name} instance, key: {key}, inst_info: {inst_info}")
                return

        server_list.append(inst_info)
        logger.info(f"Register {role_name} instance success: {inst_info}, total {role_name}: {len(server_list)}")

    async def registered_number(self):
        """Get number of registered instances"""
        return {
            "mixed": len(self.mixed_servers),
            "prefill": len(self.prefill_servers),
            "decode": len(self.decode_servers),
        }

    async def get_decode_instances(self, version: Optional[str] = None) -> List[Dict]:
        """Get all registered decode instances, optionally filtered by version"""
        async with self.lock:
            instances = self.decode_servers
            if version is not None:
                instances = [inst for inst in instances if inst.version == version]
            return [inst.to_dict() for inst in instances]

    async def select_pd(self, exclude_decode=None):
        """Select one prefill and one decode server, optionally excluding a decode instance."""
        async with self.lock:
            if not self.prefill_servers:
                raise RuntimeError(f"No prefill servers available (decode={len(self.decode_servers)})")
            if not self.decode_servers:
                raise RuntimeError(f"No decode servers available (prefill={len(self.prefill_servers)})")
            pidx = random.randint(0, len(self.prefill_servers) - 1)
            available_decode = (
                [d for d in self.decode_servers if d is not exclude_decode] if exclude_decode else self.decode_servers
            )
            if not available_decode:
                available_decode = self.decode_servers
            didx = random.randint(0, len(available_decode) - 1)
            return self.prefill_servers[pidx], available_decode[didx]

    async def select_mixed(self):
        """Select one mixed server"""
        async with self.lock:
            if not self.mixed_servers:
                raise RuntimeError(f"No mixed servers available. Registered mixed servers: {len(self.mixed_servers)}")
            idx = random.randint(0, len(self.mixed_servers) - 1)
            return self.mixed_servers[idx]

    async def handle_request(self, request_data: dict, endpoint_name: str):
        if self.splitwise:
            return await self.handle_splitwise_request(request_data, endpoint_name)
        else:
            return await self.handle_mixed_request(request_data, endpoint_name)

    async def handle_mixed_request(self, request_data: dict, endpoint_name: str):
        logger.debug(f"Received request: {request_data}")
        mixed_server = await self.select_mixed()

        if request_data.get("stream", False):
            if request_data.get("divided_stream", int(os.environ.get("DIVIDED_STREAM", "0")) == 1):
                return await self._divided_generate_stream(request_data, [mixed_server.url()], endpoint=endpoint_name)
            else:
                return await self._generate_stream(request_data, [mixed_server.url()], endpoint=endpoint_name)
        else:
            return await self._generate(request_data, [mixed_server.url()], endpoint=endpoint_name)

    async def handle_splitwise_request(self, request_data: dict, endpoint_name: str):
        logger.debug(f"Received request: {request_data}")
        last_decode_server = None
        # Preserve client request_id on first attempt; append retry suffix on subsequent attempts
        base_request_id = request_data.get("request_id") or str(uuid4())
        max_attempts = self.preempt_retry_count + 1
        completion_token_ids = []

        for attempt in range(max_attempts):
            prefill_server, decode_server = await self.select_pd(
                exclude_decode=last_decode_server if self.preempt_retry_exclude_decode else None
            )
            logger.debug(f"Selected prefill server: {prefill_server}, decode server: {decode_server}")

            if prefill_server.tp_size != decode_server.tp_size and decode_server.tp_size != 1:
                raise HTTPException(
                    status_code=400,
                    detail="The tp_size of prefill and decode should be equal or the tp_size of decode is 1",
                )

            # TODO: unify the disaggregate_info in server and remove redundancy params
            is_same_node = prefill_server.host_ip == decode_server.host_ip
            is_support_ipc = "ipc" in prefill_server.transfer_protocol and "ipc" in decode_server.transfer_protocol
            is_same_tp_size = prefill_server.tp_size == decode_server.tp_size
            use_ipc = is_same_node and is_support_ipc and is_same_tp_size

            disaggregate_info = {
                "prefill_ip": prefill_server.host_ip,
                "decode_ip": decode_server.host_ip,
                "prefill_connector_port": prefill_server.connector_port,
                "decode_connector_port": decode_server.connector_port,
                "decode_device_ids": decode_server.device_ids,
                "decode_rdma_ports": decode_server.rdma_ports,
                "transfer_protocol": "ipc" if use_ipc else "rdma",
                "decode_tp_size": decode_server.tp_size,
            }

            modified_request = request_data.copy()
            modified_request["disaggregate_info"] = disaggregate_info
            if completion_token_ids:
                modified_request["completion_token_ids"] = completion_token_ids
            if attempt == 0:
                modified_request["request_id"] = base_request_id
            else:
                modified_request["request_id"] = f"{base_request_id}-retry{attempt}"

            logger.debug(f"Modified request: {modified_request}")

            if request_data.get("stream", False):
                return await self._generate_stream(
                    modified_request, [prefill_server.url(), decode_server.url()], endpoint=endpoint_name
                )
            else:
                ret_json, status_code = await self._do_generate(
                    modified_request, [prefill_server.url(), decode_server.url()], endpoint=endpoint_name
                )
                logger.debug(f"Get response of req {modified_request['request_id']}: {ret_json}")

                if self._is_need_reschedule(ret_json):
                    last_decode_server = decode_server
                    choices = ret_json.get("choices", [])
                    if choices:
                        completion_token_ids.extend(choices[0].get("message", {}).get("completion_token_ids", []))

                    logger.warning(
                        f"Preemption detected on attempt {attempt+1}/{max_attempts}, "
                        f"decode={decode_server.url()}, req_id {modified_request['request_id']},"
                        f"retrying with new PD instances..."
                    )
                else:
                    break

        logger.debug(f"Return response of req_id {base_request_id}: {ret_json}")
        return ORJSONResponse(content=ret_json, status_code=status_code)

    def _is_need_reschedule(self, ret_json: dict) -> bool:
        # ChatCompletionResponse format: choices[0].finish_reason == "pd_reschedule"
        choices = ret_json.get("choices", [])
        if choices:
            finish_reason = choices[0].get("finish_reason", "")
            if finish_reason == "pd_reschedule":
                logger.debug(f"PD reschedule request, ret_json: {ret_json}")
                return True
        # ErrorResponse format compatibility
        error = ret_json.get("error", {})
        if isinstance(error, dict) and "PD Error" in str(error.get("message", "")):
            return True
        return False

    async def _do_generate(
        self, modified_request, urls, return_result_url_index=-1, endpoint="v1/chat/completions"
    ) -> tuple:
        """Send requests and return (ret_json, status_code)."""
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            tasks = [session.post(f"{url}/{endpoint}", json=modified_request) for url in urls]
            results = await asyncio.gather(*tasks)
            ret_json = await results[return_result_url_index].json()
            return ret_json, results[return_result_url_index].status

    async def _generate(
        self, modified_request, urls, return_result_url_index=-1, endpoint="v1/chat/completions"
    ) -> ORJSONResponse:
        ret_json, status_code = await self._do_generate(modified_request, urls, return_result_url_index, endpoint)
        return ORJSONResponse(content=ret_json, status_code=status_code)

    async def _generate_stream(
        self, modified_request, urls, return_result_url_index=-1, endpoint="v1/chat/completions"
    ):
        async def stream_results():
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                tasks = [session.post(f"{url}/{endpoint}", json=modified_request) for url in urls]
                results = await asyncio.gather(*tasks)

                AIOHTTP_STREAM_READ_CHUNK_SIZE = 1024 * 64  # prevent aiohttp's "Chunk too big" error
                async for chunk in results[return_result_url_index].content.iter_chunked(
                    AIOHTTP_STREAM_READ_CHUNK_SIZE
                ):
                    logger.debug(f"receive response chunk: {chunk}")
                    yield chunk

        return StreamingResponse(stream_results(), media_type="text/event-stream")

    async def _divided_generate_stream(
        self,
        modified_request,
        urls,
        return_result_url_index=-1,
        endpoint="v1/chat/completions",
    ):
        """
        NOTE: Used for debugging, not used in production
        """

        async def stream_results():
            total_max_tokens = modified_request.get("max_tokens", 0)
            step_max_tokens = modified_request.get("step_max_tokens", 10)
            timeout = aiohttp.ClientTimeout(total=self.timeout)

            round_idx = -1
            generated_tokens = 0
            input_ids = []
            output_ids = []

            async with aiohttp.ClientSession(timeout=timeout) as session:
                while generated_tokens < total_max_tokens:
                    round_idx += 1
                    remain_tokens = total_max_tokens - generated_tokens
                    cur_max_tokens = min(step_max_tokens, remain_tokens)
                    is_last_round = remain_tokens <= step_max_tokens

                    cur_request = copy.deepcopy(modified_request)
                    cur_request["max_tokens"] = cur_max_tokens
                    cur_request["return_token_ids"] = True
                    cur_request["max_streaming_response_tokens"] = 1
                    if round_idx == 0:
                        cur_request["disable_chat_template"] = False
                    else:
                        cur_request["messages"] = []
                        cur_request["prompt_token_ids"] = input_ids + output_ids
                        cur_request["disable_chat_template"] = True

                    logger.debug(f"_divided_generate_stream, cur_request={cur_request}")

                    resp = await session.post(
                        f"{urls[return_result_url_index]}/{endpoint}",
                        json=cur_request,
                    )
                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(f"Request failed: {resp.status}, body={text}")

                    buffer = b""
                    chunk_idx = -1
                    is_real_finished = False
                    async for raw_chunk in resp.content.iter_chunked(64 * 1024):
                        try:
                            buffer += raw_chunk

                            while b"\n\n" in buffer:
                                event_bytes, buffer = buffer.split(b"\n\n", 1)
                                event_str = event_bytes.decode("utf-8")

                                for chunk in event_str.splitlines():
                                    logger.debug(f"receive response chunk: {chunk}")
                                    if not chunk:
                                        continue

                                    chunk_idx += 1
                                    if round_idx > 0 and chunk_idx == 0:
                                        continue

                                    assert chunk.startswith("data: "), f"Invalid response chunk: {chunk}"
                                    if chunk.startswith("data: [DONE]"):
                                        if is_real_finished:
                                            yield chunk + "\n\n"
                                    else:
                                        payload = json.loads(chunk[5:])
                                        choices = payload.get("choices", [])
                                        if not choices:
                                            continue
                                        delta = payload["choices"][0]["delta"]
                                        finish_reason = payload["choices"][0].get("finish_reason")

                                        if not input_ids and len(delta["prompt_token_ids"]) > 0:
                                            input_ids = delta["prompt_token_ids"]

                                        if finish_reason == "stop" or (is_last_round and finish_reason == "length"):
                                            is_real_finished = True

                                        token_ids = delta.get("completion_token_ids")
                                        if (
                                            token_ids
                                            and isinstance(token_ids, list)
                                            and (finish_reason is None or is_real_finished)
                                        ):
                                            output_ids.extend(token_ids)
                                            generated_tokens += len(token_ids)

                                        if finish_reason is None or is_real_finished:
                                            yield chunk + "\n\n"

                        except Exception as e:
                            logger.error(
                                f"Error decoding response chunk: {raw_chunk}, round_idx: {round_idx}, "
                                f"chunk_idx: {chunk_idx}, error: {e}, traceback:{traceback.format_exc()}"
                            )
                            pass

                    if not is_real_finished:
                        expected_tokens = (step_max_tokens - 1) * (round_idx + 1)
                        if generated_tokens != expected_tokens:
                            err_msg = (
                                f"Generated tokens mismatch: generated_tokens is {generated_tokens}, "
                                f"expected is {expected_tokens}"
                            )
                            logger.error(err_msg)
                            raise RuntimeError(err_msg)

                    if is_real_finished:
                        break

        return StreamingResponse(
            stream_results(),
            media_type="text/event-stream",
        )

    async def monitor_instance_health(self, interval_secs: float = 5.0):
        """
        Continuously check the health of prefill, decode, and mixed instances and remove unhealthy ones.
        """
        while True:
            try:
                prefill_to_remove = []
                decode_to_remove = []
                mixed_to_remove = []

                async with aiohttp.ClientSession() as session:
                    # check  servers
                    prefill_tasks = [(inst, session.get(f"{inst.url()}/health")) for inst in self.prefill_servers]
                    decode_tasks = [(inst, session.get(f"{inst.url()}/health")) for inst in self.decode_servers]
                    mixed_tasks = [(inst, session.get(f"{inst.url()}/health")) for inst in self.mixed_servers]

                    # gather all tasks concurrently
                    all_tasks = prefill_tasks + decode_tasks + mixed_tasks
                    for inst, coro in all_tasks:
                        try:
                            resp = await coro
                            if resp.status != 200:
                                logger.warning(f"Instance {inst.url()} unhealthy: {resp.status}")
                                if inst in self.prefill_servers:
                                    prefill_to_remove.append(inst)
                                elif inst in self.decode_servers:
                                    decode_to_remove.append(inst)
                                elif inst in self.mixed_servers:
                                    mixed_to_remove.append(inst)
                        except Exception as e:
                            logger.warning(f"Instance {inst.url()} check failed: {e}")
                            if inst in self.prefill_servers:
                                prefill_to_remove.append(inst)
                            elif inst in self.decode_servers:
                                decode_to_remove.append(inst)
                            elif inst in self.mixed_servers:
                                mixed_to_remove.append(inst)

                # remove unhealthy instances under lock
                async with self.lock:
                    if prefill_to_remove:
                        for inst in prefill_to_remove:
                            self.prefill_servers.remove(inst)
                            logger.info(f"Removed unhealthy prefill instance: {inst.url()}")
                    if decode_to_remove:
                        for inst in decode_to_remove:
                            self.decode_servers.remove(inst)
                            logger.info(f"Removed unhealthy decode instance: {inst.url()}")
                    if mixed_to_remove:
                        for inst in mixed_to_remove:
                            self.mixed_servers.remove(inst)
                            logger.info(f"Removed unhealthy mixed instance: {inst.url()}")

                await asyncio.sleep(interval_secs)

                prefill_instances = [inst.url() for inst in self.prefill_servers]
                decode_instances = [inst.url() for inst in self.decode_servers]
                mixed_instance = [inst.url() for inst in self.mixed_servers]
                logger.debug(
                    f"Healthy prefill instances: {prefill_instances}, "
                    f"Healthy decode instances: {decode_instances}, "
                    f"Healthy mixed instance: {mixed_instance}"
                )

            except Exception as e:
                logger.exception(f"Failed to monitor instance health: {e}")


@app.post("/register")
async def register(instance_info_dict: dict):
    """Register prefill/decode/mixed servers"""
    try:
        await app.state.router.register_instance(instance_info_dict)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "success"}


@app.get("/registered_number")
async def registered_number():
    """Get the number of registered prefill/decode/mixed servers"""
    return await app.state.router.registered_number()


@app.get("/decode_instances")
async def decode_instances(version: Optional[str] = None):
    """Get all registered decode instances, optionally filtered by version"""
    return await app.state.router.get_decode_instances(version)


@app.post("/v1/chat/completions")
async def create_chat_completion(request_data: dict):
    return await app.state.router.handle_request(request_data, "v1/chat/completions")


@app.post("/v1/completions")
async def create_completion(request_data: dict):
    return await app.state.router.handle_request(request_data, "v1/completions")


@app.get("/health")
async def health_check():
    """Basic health check"""
    return Response(status_code=200)


@app.get("/health_generate")
async def health_generate():
    """Check all prefill and decode servers are healthy"""
    router = app.state.router
    async with aiohttp.ClientSession() as session:
        tasks = [session.get(f"{s.url()}/health") for s in chain(router.prefill_servers, router.decode_servers)]
        for coro in asyncio.as_completed(tasks):
            resp = await coro
            if resp.status != 200:
                logger.warning(f"Server {resp.url} not healthy: {resp.status}")
    return Response(status_code=200)


@app.post("/v1/abort_requests")
async def abort_requests(request: Request):
    body = await request.json()
    prefill_servers = app.state.router.prefill_servers
    decode_servers = app.state.router.decode_servers
    all_servers = prefill_servers + decode_servers

    async def _forward_abort():
        async with aiohttp.ClientSession() as session:
            tasks = [session.post(f"{server.url()}/v1/abort_requests", json=body) for server in all_servers]
            await asyncio.gather(*tasks, return_exceptions=True)

    task = asyncio.create_task(_forward_abort())
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return Response(status_code=200)


def launch_router(router_args: RouterArgs):
    app.state.router_args = router_args
    print(f"Starting router with args: {router_args}")

    @app.on_event("startup")
    async def startup_event():
        app.state.router = Router(app.state.router_args)
        asyncio.create_task(app.state.router.monitor_instance_health(interval_secs=5))

    uvicorn.run(app, host=router_args.host, port=int(router_args.port))
