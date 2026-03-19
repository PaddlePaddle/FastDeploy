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

import threading
import time
import traceback
from typing import Callable, Dict, List
from uuid import uuid4

import requests

from fastdeploy import envs
from fastdeploy.router.utils import check_service_health
from fastdeploy.utils import register_manager_logger as logger


class RegisterManager:
    """
    Manages Prefill/Decode instance registration and RDMA connection for PD disaggregation.

    In PD (Prefill-Decode) disaggregated deployment:
    - All instances (Prefill/Decode) register to Router with heartbeat
    - Prefill instances fetch Decode instance list from Router
    - Prefill instances establish eager RDMA connections to Decode instances

    Thread Model:
    - _register_to_router: Periodic heartbeat registration thread
    - _eager_connect_loop: Periodic RDMA connection management thread
    - _get_connect_rdma_task_response_loop: Async RDMA connection result receiver thread
    """

    def __init__(
        self,
        cfg,
        engine_worker_queue,
        get_is_paused: Callable[[], bool],
    ):
        """
        Initialize RegisterManager.

        Args:
            cfg: FDConfig object containing router, scheduler, cache configurations
            engine_worker_queue: Queue for communicating RDMA connect tasks with worker
            get_is_paused: Callable that returns current engine pause state
        """
        self.cfg = cfg
        self.engine_worker_queue = engine_worker_queue
        self.get_is_paused = get_is_paused

        # Registration state
        self._is_registered = False

        # RDMA connection state (protected by _lock)
        self.connected_decodes: List[Dict] = []  # Successfully connected Decode instances
        self.connect_status: Dict[str, bool] = {}  # task_id -> connection result

        # Timing configuration (seconds)
        self._timeout = 5  # HTTP request and RDMA connect timeout
        self._sleep_seconds = 5  # Interval between iterations

        self._lock = threading.Lock()  # Protects connected_decodes and connect_status

    def start(self) -> None:
        """Start background threads for registration and RDMA connection management."""
        self._register_to_router()
        self._start_eager_connect_loop()

    def get_connected_decodes(self) -> List[Dict]:
        """
        Return a snapshot of successfully connected Decode instances.
        Thread-safe: returns a copy to avoid concurrent modification issues.
        """
        with self._lock:
            return list(self.connected_decodes)

    def is_registered(self) -> bool:
        """Return whether this instance has successfully registered to Router."""
        return self._is_registered

    def _register_to_router(self) -> None:
        """
        Start background thread for periodic Router registration (heartbeat).

        Registration info includes: host_ip, port, role, version, is_paused, etc.
        This serves as both initial registration and keep-alive heartbeat.
        """
        router_url = self.cfg.router_config.router
        if router_url is None:
            logger.info("Router is not enabled, skip registering to router")
            return

        def _register():
            while True:
                try:
                    api_server_host = self.cfg.router_config.api_server_host
                    api_server_port = self.cfg.router_config.api_server_port
                    api_server_url = f"http://{api_server_host}:{api_server_port}"
                    if not check_service_health(api_server_url):
                        logger.info("Wait for API service health and then register to router")
                        time.sleep(self._sleep_seconds)
                        continue

                    # Update registration info
                    self.cfg.register_info["is_paused"] = self.get_is_paused()
                    self.cfg.register_info["version"] = self.cfg.model_config.version
                    self.cfg.register_info["connected_decodes"] = self.get_connected_decodes()

                    resp = requests.post(
                        f"{router_url}/register",
                        json=self.cfg.register_info,
                        timeout=self._timeout,
                    )

                    if resp.ok:
                        if not self._is_registered:
                            self._is_registered = True
                            logger.info("Register to router successfully")
                    else:
                        logger.error(
                            f"Send server info to router failed: {resp.status_code}, "
                            f"{resp.text}, {self.cfg.register_info}"
                        )
                except Exception as e:
                    logger.exception(f"Unexpected error during router registration: {e}")

                time.sleep(self._sleep_seconds)

        register_thread = threading.Thread(target=_register, daemon=True)
        register_thread.start()

    def _start_eager_connect_loop(self) -> None:
        """
        Start background threads for eager RDMA connection management.

        Only enabled when all conditions are met:
        - Router is configured
        - This instance is Prefill role
        - FD_ENABLE_PD_RDMA_EAGER_CONNECT=1
        - RDMA transfer protocol is enabled

        Starts two threads:
        1. _eager_connect_loop: Periodically discovers and connects to Decode instances
        2. _get_connect_rdma_task_response_loop: Receives async RDMA connection results
        """
        if not self._should_enable_eager_connect():
            logger.info("Eager RDMA connect is not enabled, skip")
            return

        def _eager_connect_loop():
            while True:
                try:
                    self._eager_connect_iteration()
                except Exception as e:
                    logger.exception(f"Error in eager connect loop: {e}")
                time.sleep(self._sleep_seconds)

        connect_thread = threading.Thread(target=_eager_connect_loop, daemon=True)
        connect_thread.start()
        logger.info("Eager RDMA connect loop started")

        def _get_connect_rdma_task_response_loop():
            while True:
                try:
                    resp = self.engine_worker_queue.get_connect_rdma_task_response()
                    if resp:
                        task_id = resp["task_id"]
                        is_success = resp["success"]
                        with self._lock:
                            self.connect_status[task_id] = is_success
                        logger.debug(f"get_connect_rdma_task_response response: {resp}")
                except Exception as e:
                    logger.error(f"_keep_get_connect_rdma_task_response got error: {e}, " f"{traceback.format_exc()}")
                time.sleep(0.01)

        get_resp_thread = threading.Thread(target=_get_connect_rdma_task_response_loop, daemon=True)
        get_resp_thread.start()
        logger.info("Get connect rdma task response loop started")

    def _should_enable_eager_connect(self) -> bool:
        """
        Check if eager RDMA connect should be enabled.

        Returns True only when:
        - Router URL is configured
        - Instance role is 'prefill'
        - FD_ENABLE_PD_RDMA_EAGER_CONNECT env is set
        - RDMA protocol is in transfer_protocol list
        - Local RDMA ports are configured
        """
        if self.cfg.router_config.router is None:
            return False
        if self.cfg.scheduler_config.splitwise_role != "prefill":
            return False
        if not envs.FD_ENABLE_PD_RDMA_EAGER_CONNECT:
            return False

        transfer_protocol = self.cfg.register_info.get("transfer_protocol", [])
        rdma_ports = self.cfg.cache_config.local_rdma_comm_ports
        if not ("rdma" in transfer_protocol and rdma_ports):
            return False

        return True

    def _eager_connect_iteration(self) -> None:
        """
        Single iteration of the eager RDMA connect loop.

        Workflow:
        1. Fetch Decode instances from Router (filtered by model version)
        2. For new instances: check health -> check RDMA support -> establish RDMA connection
        3. For existing instances: verify health and RDMA connection status
        4. Remove unhealthy or disconnected instances from connected_decodes
        """
        if not self._is_registered:
            logger.info("This instance has not registered to router, skip eager connect in this step")
            return

        # Step 1: Fetch Decode instances from Router
        instances = self._fetch_decode_instances_internal()
        if not instances:
            return

        # Step 2: Process new instances - try to establish RDMA connection
        with self._lock:
            connected_decodes_snapshot = list(self.connected_decodes)
        existing_keys = {self._get_instance_key(inst) for inst in connected_decodes_snapshot}

        for instance in instances:
            try:
                instance_key = self._get_instance_key(instance)

                # Skip already connected instances
                if instance_key in existing_keys:
                    continue

                # Skip unhealthy instances
                if not self._check_instance_health(instance):
                    logger.debug(f"Instance {instance_key} is unhealthy, skip")
                    continue

                # Skip instances without RDMA support
                if not self._supports_rdma(instance):
                    continue

                # Try RDMA connection
                if self._try_rdma_connect(instance):
                    with self._lock:
                        if instance not in self.connected_decodes:
                            self.connected_decodes.append(instance)
                    logger.info(f"RDMA connect succeeded: {instance_key}")
                else:
                    logger.warning(f"RDMA connect failed: {instance_key}")
            except Exception as e:
                logger.exception(f"Error processing instance {instance}: {e}")

        # Step 3: Verify existing connections - check health and RDMA status
        to_remove = []
        for instance in connected_decodes_snapshot:
            instance_key = self._get_instance_key(instance)

            if not self._check_instance_health(instance):
                to_remove.append(instance)
                logger.warning(f"Instance {instance_key} is unhealthy, will remove")
                continue

            if not self._check_rdma_connection(instance):
                to_remove.append(instance)
                logger.warning(f"Instance {instance_key} RDMA connection lost, will remove")

        # Step 4: Remove failed instances from connected list
        for instance in to_remove:
            with self._lock:
                if instance in self.connected_decodes:
                    self.connected_decodes.remove(instance)

        logger.debug(
            f"Connected decodes num is {len(self.connected_decodes)}, "
            f"connected decodes are {[self._get_instance_key(inst) for inst in self.connected_decodes]}"
        )

    def _fetch_decode_instances_internal(self) -> List[Dict]:
        """
        Fetch Decode instance list from Router.

        Queries Router's /decode_instances endpoint with model version filter.
        Returns empty list on error to allow retry in next iteration.
        """
        router_url = self.cfg.router_config.router
        if router_url is None:
            return []

        try:
            version = self.cfg.model_config.version
            resp = requests.get(
                f"{router_url}/decode_instances",
                params={"version": version},
                timeout=self._timeout,
            )

            if resp.ok:
                instances = resp.json()
                logger.debug(
                    f"Fetched {len(instances)} decode instances from router, "
                    f"{[self._get_instance_key(instance) for instance in instances]}"
                )
                return instances
            else:
                logger.error(f"Fetch decode instances failed: {resp.status_code}")
                return []
        except Exception as e:
            logger.exception(f"Error fetching decode instances: {e}")
            return []

    def _get_instance_key(self, instance: Dict) -> str:
        """Generate unique identifier for an instance: 'host_ip:port'."""
        return f"{instance.get('host_ip')}:{instance.get('port')}"

    def _supports_rdma(self, instance: Dict) -> bool:
        """Check if instance supports RDMA transfer protocol and has RDMA ports configured."""
        transfer_protocol = instance.get("transfer_protocol", [])
        return "rdma" in transfer_protocol and instance.get("rdma_ports")

    def _check_instance_health(self, instance: Dict) -> bool:
        """Check if Decode instance is healthy via HTTP /health endpoint."""
        try:
            host_ip = instance.get("host_ip")
            port = instance.get("port")
            url = f"http://{host_ip}:{port}/health"
            response = requests.get(url, timeout=self._timeout)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"_is_decode_health error: {e}, host: {host_ip}, port: {port}")
            return False

    def _try_rdma_connect(self, instance: Dict) -> bool:
        """
        Attempt to establish RDMA connection to a Decode instance.

        Workflow:
        1. Generate unique task_id and submit connect task to engine_worker_queue
        2. Wait for connection result in connect_status dict (set by response loop)
        3. Return True if connected successfully within timeout, False otherwise

        Note: If already connected, the underlying RDMA layer will reuse existing connection.

        Args:
            instance: Decode instance info dict with 'host_ip' and 'rdma_ports'

        Returns:
            True if connection succeeded, False if failed or timeout
        """
        try:
            key = self._get_instance_key(instance)
            task_id = f"{key}-{uuid4().hex}"
            task = {"task_id": task_id, "ip": instance.get("host_ip"), "rdma_ports": instance.get("rdma_ports")}
            self.engine_worker_queue.put_connect_rdma_task(task)

            start_time = time.time()
            while time.time() - start_time <= self._timeout:
                with self._lock:
                    if task_id in self.connect_status:
                        result = self.connect_status[task_id]
                        del self.connect_status[task_id]
                        return result
                time.sleep(0.01)

            # Timeout: clean up any late-arriving result to prevent memory leak
            with self._lock:
                self.connect_status.pop(task_id, None)

        except Exception as e:
            logger.error(f"_try_rdma_connect error: {e}")
        return False

    def _check_rdma_connection(self, instance: Dict) -> bool:
        """
        Verify RDMA connection to instance is still alive.

        Reuses _try_rdma_connect() since the underlying RDMA layer:
        - Returns success immediately if connection already exists
        - Attempts reconnection if connection was lost
        """
        return self._try_rdma_connect(instance)
