"""
Async Router server for FastDeploy.
Handles client requests and manages prefill/decode/mixed instances.
This module references the router implementation of slglang and vllm.
"""

import asyncio
import random
from dataclasses import dataclass
from itertools import chain
from uuid import uuid4

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse, Response, StreamingResponse

from fastdeploy.router.utils import (
    InstanceInfo,
    InstanceRole,
    check_service_health_async,
)
from fastdeploy.utils import FlexibleArgumentParser
from fastdeploy.utils import router_logger as logger

app = FastAPI()


@dataclass
class RouterArgs:
    host: str = "0.0.0.0"
    """
    Host address to bind the router server
    """
    port: str = "9000"
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
            type=str,
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

        self.mixed_servers = []
        self.prefill_servers = []
        self.decode_servers = []
        self.lock = asyncio.Lock()  # async-safe lock

    async def register_instance(self, instance_info_dict: dict):
        """Register an instance asynchronously"""
        try:
            inst_info = InstanceInfo(**instance_info_dict)
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
            if inst_info.role == InstanceRole.MIXED and inst_info not in self.mixed_servers:
                self.mixed_servers.append(inst_info)
                logger.info(
                    f"Register mixed instance success: {inst_info}, " f"total mixed: {len(self.mixed_servers)}"
                )
            elif inst_info.role == InstanceRole.PREFILL and inst_info not in self.prefill_servers:
                self.prefill_servers.append(inst_info)
                logger.info(
                    f"Register prefill instance success: {inst_info}, "
                    f"prefill: {len(self.prefill_servers)}, decode: {len(self.decode_servers)}"
                )
            elif inst_info.role == InstanceRole.DECODE and inst_info not in self.decode_servers:
                self.decode_servers.append(inst_info)
                logger.info(
                    f"Register decode instance success: {inst_info}, "
                    f"prefill: {len(self.prefill_servers)}, decode: {len(self.decode_servers)}"
                )

    async def registered_number(self):
        """Get number of registered instances"""
        return {
            "mixed": len(self.mixed_servers),
            "prefill": len(self.prefill_servers),
            "decode": len(self.decode_servers),
        }

    async def select_pd(self):
        """Select one prefill and one decode server"""
        async with self.lock:
            if not self.prefill_servers:
                raise RuntimeError("No prefill servers available")
            if not self.decode_servers:
                raise RuntimeError("No decode servers available")
            pidx = random.randint(0, len(self.prefill_servers) - 1)
            didx = random.randint(0, len(self.decode_servers) - 1)
            return self.prefill_servers[pidx], self.decode_servers[didx]

    async def select_mixed(self):
        """Select one mixed server"""
        async with self.lock:
            if not self.mixed_servers:
                raise RuntimeError("No mixed servers available")
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
            return await self._generate_stream(request_data, [mixed_server.url()], endpoint=endpoint_name)
        else:
            return await self._generate(request_data, [mixed_server.url()], endpoint=endpoint_name)

    async def handle_splitwise_request(self, request_data: dict, endpoint_name: str):
        logger.debug(f"Received request: {request_data}")
        prefill_server, decode_server = await self.select_pd()

        # TODO: unify the disaggregate_info in server and remove redundancy params
        is_same_node = prefill_server.host_ip == decode_server.host_ip
        use_ipc = (
            is_same_node and "ipc" in prefill_server.transfer_protocol and "ipc" in decode_server.transfer_protocol
        )

        cache_info = {}
        if use_ipc:
            cache_info["ipc"] = {
                "ip": decode_server.host_ip,
                "port": decode_server.engine_worker_queue_port,
                "device_ids": decode_server.device_ids,
            }
        else:
            cache_info["rdma"] = {
                "ip": decode_server.host_ip,
                "port": decode_server.connector_port,
                "rdma_port": decode_server.rdma_ports,
            }

        disaggregate_info = {
            "prefill": prefill_server.to_dict(),
            "decode": decode_server.to_dict(),
            "role": "decode",
            "cache_info": cache_info,
            "transfer_protocol": "ipc" if use_ipc else "rdma",
        }

        modified_request = request_data.copy()
        modified_request["disaggregate_info"] = disaggregate_info
        if "request_id" not in modified_request:
            modified_request["request_id"] = str(uuid4())

        logger.debug(f"Modified request: {modified_request}")

        if request_data.get("stream", False):
            return await self._generate_stream(
                modified_request, [prefill_server.url(), decode_server.url()], endpoint=endpoint_name
            )
        else:
            return await self._generate(
                modified_request, [prefill_server.url(), decode_server.url()], endpoint=endpoint_name
            )

    async def _generate(
        self, modified_request, urls, return_result_url_index=-1, endpoint="v1/chat/completions"
    ) -> ORJSONResponse:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            tasks = [session.post(f"{url}/{endpoint}", json=modified_request) for url in urls]
            results = await asyncio.gather(*tasks)
            ret_json = await results[return_result_url_index].json()
            return ORJSONResponse(content=ret_json, status_code=results[return_result_url_index].status)

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


def launch_router(router_args: RouterArgs):
    app.state.router_args = router_args
    print(f"Starting router with args: {router_args}")

    @app.on_event("startup")
    async def startup_event():
        app.state.router = Router(app.state.router_args)
        asyncio.create_task(app.state.router.monitor_instance_health(interval_secs=5))

    uvicorn.run(app, host=router_args.host, port=router_args.port)
