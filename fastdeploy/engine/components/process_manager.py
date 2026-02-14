# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""
Process Manager - Manages worker processes lifecycle.

This component is part of the new modular architecture and handles:
- Worker process startup
- Cache process startup
- Process health monitoring
- Process cleanup and restart
"""

import json
import os
import re
import signal
import sys
import threading
import time
import traceback
from multiprocessing import Process
from typing import List, Optional

from fastdeploy.utils import llm_logger


class ProcessManager:
    """
    Manages worker processes for the engine.

    This component is used in the new modular architecture.
    The old architecture (_use_new_architecture=False) uses the original EngineService code.
    """

    def __init__(self, cfg, ipc_manager):
        """
        Initialize process manager.

        Args:
            cfg: Configuration object
            ipc_manager: IPCManager instance for communication setup
        """
        self.cfg = cfg
        self.ipc = ipc_manager

        self._worker_procs: List[Process] = []
        self._cache_procs: List[Process] = []
        self._running = False

        # Process configuration
        self.ipc_signal_suffix = None
        self.cache_manager_processes = None

        # Worker management attributes
        self.do_profile = 1 if self.cfg.cache_config.num_gpu_blocks_override is None else 0
        self.worker_init_status = {}
        self.checking_worker_status_thread = None

        self.llm_logger = llm_logger

    def start_workers(self):
        """
        Start all worker processes.
        Migrated from EngineService._start_worker_service().

        Returns:
            subprocess.Popen instance of the started worker process
        """
        import subprocess

        log_dir = os.getenv("FD_LOG_DIR", default="log")
        command_prefix = self._setting_environ_variables()
        current_file_path = os.path.abspath(__file__)
        current_dir_path = os.path.split(current_file_path)[0]
        # TODO
        uncache_worker_stdout = "" if os.getenv("UNCACHE_WORKER_STDOUT", "0") == "1" else "-u"
        pd_cmd = f"{command_prefix} {sys.executable} {uncache_worker_stdout} -m paddle.distributed.launch"
        pd_cmd = pd_cmd + f" --log_dir {log_dir}"

        worker_path = "../worker/worker_process.py"
        py_script = os.path.join(current_dir_path, worker_path)

        # Access data_processor through scheduler coordinator
        data_processor = self.ipc._scheduler_coord._data_processor if hasattr(self.ipc, "_scheduler_coord") else None
        if data_processor is None:
            self.llm_logger.warning("Data processor not available, using default tokenizer values")
            ori_vocab_size = 0
            think_end_id = -1
            image_patch_id = -1
            line_break_id = -1
            eos_token_id_len = 1
            pad_token_id = 0
        else:
            ori_vocab_size = (
                len(data_processor.tokenizer.sp_model)
                if hasattr(data_processor.tokenizer, "sp_model")
                else len(data_processor.tokenizer.vocab)
            )

            think_end_id = data_processor.tokenizer.get_vocab().get("", -1)
            if think_end_id > 0:
                self.llm_logger.info(f"Get think_end_id {think_end_id} from vocab.")
            else:
                self.llm_logger.info("No  token found in vocabulary, model can not do reasoning.")
            image_patch_id = data_processor.tokenizer.get_vocab().get("<|IMAGE_PLACEHOLDER|>", -1)
            line_break_id = data_processor.tokenizer.get_vocab().get("\n", -1)
            eos_token_id_len = data_processor.eos_token_id_len
            pad_token_id = data_processor.pad_token_id

        ports = ",".join(map(str, self.cfg.parallel_config.engine_worker_queue_port))
        ips = None
        if self.cfg.ips is not None:
            ips = ",".join(self.cfg.ips)
        arguments = (
            f" --devices {self.cfg.parallel_config.device_ids} {py_script}"
            f" --max_num_seqs {self.cfg.scheduler_config.max_num_seqs} --max_model_len {self.cfg.model_config.max_model_len}"
            f" --gpu_memory_utilization {self.cfg.cache_config.gpu_memory_utilization}"
            f" --model {self.cfg.model_config.model!s}"
            f" --device_ids {self.cfg.parallel_config.device_ids}"
            f" --tensor_parallel_size {self.cfg.parallel_config.tensor_parallel_size}"
            f" --engine_worker_queue_port {ports}"
            f" --pod_ip {self.cfg.master_ip}"
            f" --block_size {self.cfg.cache_config.block_size}"
            f" --enc_dec_block_num {self.cfg.cache_config.enc_dec_block_num}"
            f" --eos_tokens_lens {eos_token_id_len}"
            f" --pad_token_id {pad_token_id}"
            f" --engine_pid {self.cfg.parallel_config.engine_worker_queue_port[0]}"
            f" --max_num_batched_tokens {self.cfg.scheduler_config.max_num_batched_tokens}"
            f" --splitwise_role {self.cfg.scheduler_config.splitwise_role}"
            f" --kv_cache_ratio {self.cfg.cache_config.kv_cache_ratio}"
            f" --expert_parallel_size {self.cfg.parallel_config.expert_parallel_size}"
            f" --chunked_moe_size {self.cfg.parallel_config.chunked_moe_size}"
            f" --data_parallel_size {self.cfg.parallel_config.data_parallel_size}"
            f" --quantization '{json.dumps(self.cfg.model_config.quantization)}'"
            f" --ori_vocab_size {ori_vocab_size}"
            f" --think_end_id {think_end_id}"
            f" --image_patch_id {image_patch_id}"
            f" --line_break_id {line_break_id}"
            f" --speculative_config '{self.cfg.speculative_config.to_json_string()}'"
            f" --graph_optimization_config '{self.cfg.graph_opt_config.to_json_string()}'"
            f" --guided_decoding_backend {self.cfg.structured_outputs_config.guided_decoding_backend}"
            f" --load_strategy {self.cfg.load_config.load_strategy}"
            f" --rsync_config '{json.dumps(self.cfg.load_config.rsync_config)}'"
            f" --early_stop_config '{self.cfg.early_stop_config.to_json_string()}'"
            f" --reasoning_parser {self.cfg.structured_outputs_config.reasoning_parser}"
            f" --load_choices {self.cfg.load_config.load_choices}"
            f" --plas_attention_config '{self.cfg.plas_attention_config.to_json_string()}'"
            f" --ips {ips}"
            f" --cache-transfer-protocol {self.cfg.cache_config.cache_transfer_protocol}"
            f" --runner {self.cfg.model_config.runner}"
            f" --convert {self.cfg.model_config.convert}"
            f" --override_pooler_config {self.cfg.model_config.override_pooler_config}"
            f" --logprobs_mode {self.cfg.model_config.logprobs_mode}"
            f" --max_logprobs {self.cfg.model_config.max_logprobs}"
            f" --eplb_config '{self.cfg.eplb_config.to_json_string()}'"
            f" --num_cpu_blocks {self.cfg.cache_config.num_cpu_blocks}"
        )
        if self.cfg.structured_outputs_config.logits_processors is not None:
            arguments += f" --logits-processors {' '.join(self.cfg.structured_outputs_config.logits_processors)}"
        mm_max_tokens = None
        if (
            hasattr(self.ipc._scheduler_coord, "mm_max_tokens_per_item")
            and self.ipc._scheduler_coord.mm_max_tokens_per_item is not None
        ):
            mm_max_tokens = self.ipc._scheduler_coord.mm_max_tokens_per_item
        if mm_max_tokens is not None:
            arguments += f" --mm_max_tokens_per_item '{json.dumps(mm_max_tokens)}'"

        worker_store_true_flag = {
            "enable_expert_parallel": self.cfg.parallel_config.enable_expert_parallel,
            "enable_prefix_caching": self.cfg.cache_config.enable_prefix_caching,
            "enable_chunked_prefill": self.cfg.cache_config.enable_chunked_prefill,
            "do_profile": self.do_profile,
            "dynamic_load_weight": self.cfg.load_config.dynamic_load_weight,
            "disable_any_whitespace": self.cfg.structured_outputs_config.disable_any_whitespace,
            "disable_custom_all_reduce": self.cfg.parallel_config.disable_custom_all_reduce,
            "use_internode_ll_two_stage": self.cfg.parallel_config.use_internode_ll_two_stage,
            "disable_sequence_parallel_moe": self.cfg.parallel_config.disable_sequence_parallel_moe,
            "enable_logprob": self.cfg.model_config.enable_logprob,
            "lm_head_fp32": self.cfg.model_config.lm_head_fp32,
            "enable_entropy": self.cfg.model_config.enable_entropy,
            "enable_overlap_schedule": self.cfg.scheduler_config.enable_overlap_schedule,
        }
        for worker_flag, value in worker_store_true_flag.items():
            if value:
                arguments = arguments + f" --{worker_flag}"

        worker_default_none_flag = {
            "num_gpu_blocks_override": self.cfg.cache_config.num_gpu_blocks_override,
            "kvcache_storage_backend": self.cfg.cache_config.kvcache_storage_backend,
        }
        for worker_flag, value in worker_default_none_flag.items():
            if value:
                arguments = arguments + f" --{worker_flag} {value}"

        if self.cfg.nnode > 1:
            pd_cmd = pd_cmd + f" --ips {ips} --nnodes {len(self.cfg.ips)}"
        pd_cmd = pd_cmd + arguments + f" 2>{log_dir}/launch_worker.log"
        self.llm_logger.info(f"Launch worker service command: {pd_cmd}")
        p = subprocess.Popen(
            pd_cmd,
            stdout=subprocess.PIPE,
            shell=True,
            preexec_fn=os.setsid,
        )
        self._worker_procs.append(p)
        self._running = True
        return p

    def start_cache_service(self, device_ids: List[str], suffix: int) -> List[Process]:
        """
        Start cache service processes.
        This is implemented at resource_manager.cache_manager level,
        ProcessManager just keeps track of the processes.

        Args:
            device_ids: List of device IDs for cache processes
            suffix: Suffix for identifying cache processes

        Returns:
            List of started cache processes
        """
        # The actual cache manager launch is handled by resource_manager.cache_manager
        # This method exists for tracking processes if needed
        return self._cache_procs

    def stop_workers(self):
        """
        Stop all worker and cache processes.
        Migrated from EngineService cleanup code.
        """
        self._running = False

        # Clean up worker processes
        self.llm_logger.info("Cleaning up worker processes...")
        for proc in self._worker_procs:
            if proc is not None:
                try:
                    pgid = os.getpgid(proc.pid)
                    os.killpg(pgid, signal.SIGTERM)
                except Exception as e:
                    self.llm_logger.error(
                        f"Error cleaning up worker process {proc.pid}: {e}, {str(traceback.format_exc())}"
                    )

        # Clean up cache manager processes
        if self._cache_procs:
            self.llm_logger.info("Cleaning up cache manager processes...")
            for p in self._cache_procs:
                self.llm_logger.info(f"Killing cache manager process {p.pid}")
                try:
                    pgid = os.getpgid(p.pid)
                    os.killpg(pgid, signal.SIGTERM)
                except Exception as e:
                    self.llm_logger.error(
                        f"Error killing cache manager process {p.pid}: {e}, {str(traceback.format_exc())}"
                    )

        self._worker_procs.clear()
        self._cache_procs.clear()

    def check_worker_health(self) -> bool:
        """
        Check health status of worker processes.
        Migrated from EngineService worker health check code.

        Returns:
            True if all workers are healthy, False otherwise
        """
        if not hasattr(self.ipc, "worker_healthy_live_signal") or self.ipc.worker_healthy_live_signal is None:
            return True
        current_time = int(time.time())
        for i, live_time in enumerate(self.ipc.worker_healthy_live_signal.value):
            elapsed = current_time - live_time
            timeout = 30  # Default 30 seconds timeout
            if elapsed > timeout:
                self.llm_logger.warning(f"Worker {i} not healthy for {elapsed} seconds")
                return False
        return True

    def _setting_environ_variables(self) -> str:
        """
        Set environment variables.
        Migrated from EngineService._setting_environ_variables().

        Returns:
            Command prefix string with environment variables
        """
        result = []
        result.append("ENABLE_FASTDEPLOY_LOAD_MODEL_CONCURRENCY=0")
        result.append("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python")
        result.append("FLAGS_use_append_attn=1")
        result.append("NCCL_ALGO=Ring")
        return " ".join(result)

    def check_worker_initialize_status(self) -> bool:
        """
        Check if worker has initialized successfully.
        Migrated from EngineService.check_worker_initialize_status().

        Returns:
            True if worker initialized successfully, False otherwise
        """
        from tqdm import tqdm

        def detect_thread():
            if len(self._worker_procs) == 0:
                self.worker_init_status["finished"] = True
                return
            proc = self._worker_procs[0]
            for line in proc.stdout:
                line = line.decode("utf-8", errors="ignore")
                if self.worker_init_status.get("finished", False):
                    break
                if match := re.search(
                    r"Loading (?:fastsafetensors |safetensors )?checkpoint shards:\s*(\d+)",
                    line,
                ):
                    self.worker_init_status["weight_loading"] = eval(match.group(1)) * 1.0 / 100
                elif (match := re.search(r"Start load layer (\d+)", line)) or (
                    match := re.search(r"set state for layer (\d+)", line)
                ):
                    progress = eval(match.group(1)) * 1.0 / self.cfg.model_config.num_hidden_layers
                    self.worker_init_status["layer_loading"] = progress
                    if self.worker_init_status["layer_loading"] == self.cfg.model_config.num_hidden_layers - 1:
                        self.worker_init_status["finished"] = True

        self.checking_worker_status_thread = threading.Thread(target=detect_thread, daemon=True)
        self.checking_worker_status_thread.start()

        # display weight loading progress
        with tqdm(total=100, desc="Loading Weights") as pbar:
            progress = 0
            while progress < 100:
                progress = int(self.worker_init_status.get("weight_loading", 0) * 100)
                if self.worker_init_status.get("layer_loading", 0) > 0 or self._worker_processes_ready():
                    progress = 100
                pbar.update(progress - pbar.n)
                pbar.refresh()
                time.sleep(0.5)
                if len(self._worker_procs) > 0 and self._worker_procs[0].poll() is not None:
                    return False

        # display layer loading progress
        with tqdm(total=100, desc="Loading Layers") as pbar:
            progress = 0
            while progress < 100:
                progress = int(self.worker_init_status.get("layer_loading", 0) * 100)
                if self._worker_processes_ready():
                    progress = 100
                pbar.update(progress - pbar.n)
                pbar.refresh()
                time.sleep(0.5)
                if len(self._worker_procs) > 0 and self._worker_procs[0].poll() is not None:
                    return False

        self.worker_init_status["finished"] = True
        try:
            self.checking_worker_status_thread.join(timeout=1)
        except Exception:
            pass
        return True

    def _worker_processes_ready(self) -> bool:
        """
        Judge if all worker processes are ready.
        Migrated from EngineService._worker_processes_ready().

        Returns:
            True if all workers are ready, False otherwise
        """
        import numpy as np

        if not hasattr(self.ipc, "worker_ready_signal") or self.ipc.worker_ready_signal is None:
            return False
        if np.sum(self.ipc.worker_ready_signal.value) == self.cfg.worker_num_per_node:
            return True
        return False

    @property
    def worker_proc(self) -> Optional[Process]:
        """
        Get first worker process (for compatibility).

        Returns:
            First worker process or None if no workers running
        """
        return self._worker_procs[0] if self._worker_procs else None

    @worker_proc.setter
    def worker_proc(self, value: Optional[Process]):
        """Set worker_proc for compatibility."""
        if self._worker_procs:
            self._worker_procs[0] = value
        else:
            self._worker_procs = [value] if value else []

    @property
    def cache_procs(self) -> List[Process]:
        """Get all cache processes."""
        return self._cache_procs

    @property
    def cache_manager_processes(self) -> Optional[List[Process]]:
        """Get cache manager processes (alias for compatibility)."""
        return self._cache_procs

    @cache_manager_processes.setter
    def cache_manager_processes(self, value: Optional[List[Process]]):
        """Set cache manager processes (alias for compatibility)."""
        self._cache_procs = value or []
