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

from __future__ import annotations

import os
import signal
import threading
import time
import traceback
import weakref

import numpy as np

from fastdeploy.engine.resource_manager import ResourceManager
from fastdeploy.inter_communicator import EngineWorkerQueue
from fastdeploy.metrics.metrics import main_process_metrics
from fastdeploy.output.token_processor import TokenProcessor
from fastdeploy.splitwise.splitwise_connector import SplitwiseConnector
from fastdeploy.utils import EngineError, console_logger, llm_logger


class ExpertService:
    """
    Engine class responsible for managing the Large Language Model (LLM) operations.

    Attributes:
        cfg (Config): Configuration object containing all the parameters.
        local_data_parallel_id (int): Local data parallel ID.
    """

    def __init__(self, cfg, local_data_parallel_id):
        """
        Initializes the LLMEngine with the provided configuration.

        Args:
            cfg (Config): Config object containing all the configuration parameters.
        """
        self.cfg = cfg
        start_pos = (local_data_parallel_id * self.cfg.tensor_parallel_size) % self.cfg.worker_num_per_node
        end_pos = ((local_data_parallel_id + 1) * self.cfg.tensor_parallel_size) % self.cfg.worker_num_per_node
        self.cfg.cache_config.rdma_comm_ports = self.cfg.cache_config.rdma_comm_ports[start_pos:end_pos]
        self.cfg.local_device_ids = self.cfg.device_ids.split(",")[start_pos:end_pos]
        self.cfg.parallel_config.local_data_parallel_id = local_data_parallel_id
        self.cfg.disaggregate_info = None

        self.scheduler = cfg.scheduler_config.scheduler()

        self.scheduler.reset_nodeid(f"{self.scheduler.infer.nodeid}_{local_data_parallel_id!s}")

        self.cfg.parallel_config.local_data_parallel_id = local_data_parallel_id

        address = (cfg.master_ip, cfg.engine_worker_queue_port)
        self.engine_worker_queue = EngineWorkerQueue(
            address=address,
            is_server=False,
            client_id=0,
            num_client=cfg.tensor_parallel_size,
            local_data_parallel_id=local_data_parallel_id,
        )
        self.resource_manager = ResourceManager(
            cfg.max_num_seqs,
            cfg,
            cfg.tensor_parallel_size,
            cfg.splitwise_role,
            local_data_parallel_id,
        )

        if len(self.cfg.cache_config.pd_comm_port) == 1:
            self.cfg.cache_config.pd_comm_port[0] = int(self.cfg.cache_config.pd_comm_port[0]) + local_data_parallel_id
        else:
            self.cfg.cache_config.pd_comm_port = [self.cfg.cache_config.pd_comm_port[local_data_parallel_id]]

        self.split_connector = SplitwiseConnector(
            self.cfg,
            self.scheduler,
            self.engine_worker_queue,
            self.resource_manager,
        )

        self.token_processor = TokenProcessor(
            cfg=cfg,
            cached_generated_tokens=self.scheduler,
            engine_worker_queue=self.engine_worker_queue,
            split_connector=self.split_connector,
        )
        self.token_processor.set_resource_manager(self.resource_manager)

        self.partial_chunked_tokens = [0] * (self.cfg.max_num_partial_prefills + 1)
        for idx in range(1, self.cfg.max_num_partial_prefills + 1):
            self.partial_chunked_tokens[idx] = (
                (self.cfg.max_num_batched_tokens // idx)
                // self.cfg.cache_config.block_size
                * self.cfg.cache_config.block_size
            )

        self._finalizer = weakref.finalize(self, self._exit_sub_services)

    def start(self, ipc_signal_suffix, local_data_parallel_id):
        """
        Initializes the engine and starts its sub-services.
        If `api_server_pid` is defined, will launch a thread
        to keep getting request from zmq_server.
        """
        # assert not self.is_started, "The engine is already started."
        start_time = time.time()

        llm_logger.info(f"start expert service {local_data_parallel_id}")

        self.cache_manager_processes = self.resource_manager.cache_manager.launch_cache_manager(
            cache_config=self.cfg.cache_config,
            tensor_parallel_size=self.cfg.tensor_parallel_size,
            device_ids=self.cfg.local_device_ids,
            pod_ip=self.cfg.master_ip,
            engine_worker_queue_port=self.cfg.engine_worker_queue_port,
            pid_suffix=f"{local_data_parallel_id}_{ipc_signal_suffix}",
        )

        self.insert_task_to_worker_thread = threading.Thread(target=self._insert_task_to_worker, args=())
        self.insert_task_to_worker_thread.daemon = True
        self.insert_task_to_worker_thread.start()

        # Start TokenProcessor thread
        os.environ["INFERENCE_MSG_QUEUE_ID"] = str(local_data_parallel_id + int(self.cfg.engine_worker_queue_port))

        self.token_processor.run()

        self.split_mode_get_tasks()

        self.cfg.init_cache_info()

        role = self.cfg.splitwise_role
        host_ip = self.cfg.host_ip
        disaggregate = self.cfg.disaggregate_info
        self.scheduler.start(role, host_ip, disaggregate)
        self.cfg.print()

        console_logger.info(f"Worker processes are launched with {time.time() - start_time} seconds.")
        return True

    def _insert_task_to_worker(self):
        """
        Insert task to engine thread, monitor scheduler request queue.
        if the engine has resource, insert task to engine
        """
        current_id = -1
        while True:
            try:
                if self.resource_manager.available_batch() == 0:
                    time.sleep(0.001)
                    continue
                if self.engine_worker_queue.num_tasks() > 0:
                    time.sleep(0.001)
                    continue
                if len(self.split_connector.current_request_ids) > 0:
                    time.sleep(0.001)
                    continue

                num_prefill_batch = min(
                    int(self.resource_manager.available_batch()),
                    self.cfg.max_prefill_batch,
                )

                self.resource_manager.check_and_free_block_tables()
                tasks = self.scheduler.get_requests(
                    available_blocks=self.resource_manager.available_block_num(),
                    block_size=self.cfg.cache_config.block_size,
                    reserved_output_blocks=self.cfg.cache_config.enc_dec_block_num,
                    max_num_batched_tokens=self.cfg.max_num_batched_tokens,
                    batch=num_prefill_batch,
                )

                if len(tasks) == 0:
                    time.sleep(0.001)
                    continue

                if self.cfg.splitwise_role != "mixed":
                    llm_logger.info("Inserting splitwise tasks")
                    self.split_connector.send_splitwise_tasks(tasks, current_id)

                current_id = (current_id + 1) % 100003

                self.insert_tasks(tasks, current_id)

                main_process_metrics.num_requests_waiting.dec(len(tasks))
                main_process_metrics.num_requests_running.inc(len(tasks))
            except Exception as e:
                err_msg = f"Error happend while insert task to engine: {e}, {traceback.format_exc()!s}."
                llm_logger.error(err_msg)

    def split_mode_get_tasks(self):
        """
        Split mode get tasks
        """
        waiting_requests = []

        def receiver_loop():
            while True:
                try:
                    if len(waiting_requests) > 0:
                        for task in waiting_requests:
                            if self.resource_manager.is_resource_sufficient(task.prompt_token_ids_len):
                                self.insert_tasks([task])
                                waiting_requests.remove(task)
                            else:
                                break
                    if not self.engine_worker_queue.disaggregate_queue_empty():
                        items = self.engine_worker_queue.get_disaggregated_tasks()
                        for item in items:
                            role = item[0]
                            tasks = item[1]
                            if role == "prefill":
                                llm_logger.info("get prefill tasks")
                                for task in tasks:
                                    task.max_tokens = task.min_tokens = 2
                                self.insert_tasks(tasks)
                            elif role == "decode":
                                llm_logger.info(f"get decode tasks {tasks}")
                                if hasattr(tasks[0], "finished"):
                                    if not isinstance(tasks, list):
                                        tasks = [tasks]
                                    for task in tasks:
                                        task.finished = False
                                    # self.scheduler.put_results(tasks)

                                    self.insert_tasks(tasks, allocated=True)
                                else:
                                    if len(waiting_requests):
                                        for task in tasks:
                                            waiting_requests.append(task)
                                    else:
                                        for task in tasks:
                                            if not self.resource_manager.is_resource_sufficient(
                                                task.prompt_token_ids_len
                                            ):
                                                waiting_requests.append(task)
                                            else:
                                                self.insert_tasks([task])

                    else:
                        time.sleep(0.001)
                        continue
                except Exception as e:
                    llm_logger.error(f"get decode tasks error: {e}")

        threading.Thread(target=receiver_loop, daemon=True).start()

    def insert_tasks(self, tasks, current_id=-1, allocated=False):
        """
        Insert tasks to engine.
        """
        if allocated:
            current_tasks = []
            for task in tasks:
                cur_task_idx = self.resource_manager.req_dict[task.request_id]
                del self.resource_manager.req_dict[task.request_id]
                cur_task = self.resource_manager.tasks_list[cur_task_idx]
                if task.error_code != 200:
                    self.resource_manager.stop_flags[cur_task_idx] = True
                    self.resource_manager.tasks_list[cur_task_idx] = None
                    self.resource_manager._recycle_block_tables(cur_task)
                    if task.request_id in self.token_processor.tokens_counter:
                        del self.token_processor.tokens_counter[task.request_id]
                    self.scheduler.put_results([task])
                    llm_logger.warning(
                        f"{task.request_id} prefill failed with msg:{task.error_msg}, recycle resource."
                    )
                    continue
                llm_logger.info(f"{cur_task_idx} {task.request_id}")
                cur_task.prompt_token_ids[0] = task.outputs.token_ids[0]
                self.token_processor.tokens_counter[task.request_id] = 1
                current_tasks.append(cur_task)
            self.engine_worker_queue.put_tasks((current_tasks, self.resource_manager.real_bsz))
            return True

        self.resource_manager.check_and_free_block_tables()

        if not isinstance(tasks, list):
            tasks = [tasks]

        for item in tasks:
            item.schedule_start_time = time.time()

        available_batch = np.sum(self.resource_manager.stop_flags)
        if len(tasks) > available_batch:
            llm_logger.error(f"Inserting batch:{len(tasks)} exceeds the available batch:{available_batch}.")
            llm_logger.error("The exceeded part will be ignored!")
            tasks = tasks[:available_batch]

        req_ids = [t.request_id for t in tasks]

        tasks = self.resource_manager.allocate_resources_for_new_tasks(tasks)

        if not tasks:
            error_msg = f"The request required resources is exceed the limit, request id={req_ids}."
            llm_logger.error(error_msg)
            raise EngineError(error_msg, error_code=500)
            return False

        self.token_processor.number_of_tasks += len(tasks)

        is_decode = False
        is_prefill = False
        for i in range(len(tasks)):
            if tasks[i].disaggregate_info is not None:
                if tasks[i].disaggregate_info["role"] == "decode":
                    is_decode = True
                else:
                    is_prefill = True
            self.token_processor.number_of_input_tokens += tasks[i].prompt_token_ids_len

        self.split_connector.send_cache_infos(tasks, current_id)
        for task in tasks:
            task.infer_start_time = time.time()
        if not is_decode:
            llm_logger.info(f"Tasks are sent to engine, req_ids={req_ids}")
            if not is_prefill:
                if not self.cfg.enable_mm:
                    self.update_requests_chunk_size(tasks)
                else:
                    self.update_mm_requests_chunk_size(tasks)
            self.engine_worker_queue.put_tasks((tasks, self.resource_manager.real_bsz))
        return True

    def _exit_sub_services(self):
        """
        exit sub services
        """

        if hasattr(self, "cache_manager_processes"):
            self.resource_manager.cache_manager.shm_cache_task_flag_broadcast.clear()
            self.resource_manager.cache_manager.cache_ready_signal.clear()
            for p in self.cache_manager_processes:
                llm_logger.info(f"Killing cache manager process {p.pid}")
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except:
                    pass

        if hasattr(self, "zmq_server") and self.zmq_server is not None:
            self.zmq_server.close()


def start_expert_service(cfg, local_data_parallel_id, ipc_signal_suffix):
    """
    Start expert service
    """
    expert_service = ExpertService(cfg, local_data_parallel_id)
    try:
        expert_service.start(ipc_signal_suffix, local_data_parallel_id)
        expert_service.split_connector.start_receiver()
    except Exception as e:
        llm_logger.exception(f"Expert service failed to start: {e}")
