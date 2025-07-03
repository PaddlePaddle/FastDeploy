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
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.distributed.fleet as fleet

from fastdeploy.engine.config import ModelConfig
from fastdeploy.inter_communicator import EngineWorkerQueue, IPCSignal
from fastdeploy.utils import get_logger, none_or_str
from fastdeploy.worker.worker_process import initialize_fd_config, parse_args

logger = get_logger("worker", "worker.log")


class PrefillTracker:
    """
    Record the prefill time of the request
    """

    def __init__(
        self,
        engine_pid: int,
    ) -> None:
        """
        Initialize the PrefillTracker.
        """
        super().__init__()
        self.start_times = defaultdict(float)
        prefill_time_data = np.zeros([100], dtype=np.float32)
        self.prefill_time_signal = IPCSignal(name="prefill_time_signal",
                                             array=prefill_time_data,
                                             dtype=np.float32,
                                             suffix=engine_pid,
                                             create=False)
        self.current_index = 0
        self.executor = ThreadPoolExecutor(max_workers=1)

    def start_prefill(self, task_idx: int):
        """
        Record the start time of the prefill process for a given task index.

        Args:
            task_idx (int): The index of the task being prefetched.
        """
        self.start_times[task_idx] = time.time()

    def end_prefill(self, task_idx: int):
        """
        Record the end time of the prefill process for a given task index and
        asynchronously submit the duration for metric recording.

        Args:
            task_idx (int): The index of the task being prefetched.
        """
        if task_idx in self.start_times:
            duration = time.time() - self.start_times[task_idx]
            # Submit metric recording to the executor for asynchronous execution
            self.executor.submit(self._record_metrics, duration)
            del self.start_times[task_idx]

    def _record_metrics(self, duration: float):
        """
        Internal method to record the prefill duration into the signal buffer.
        Logs the duration and updates a circular buffer of timing metrics.

        Args:
            duration (float): Time taken for the prefill process in seconds.
        """

        self.prefill_time_signal.value[self.current_index] = duration
        self.current_index = (self.current_index + 1) % len(
            self.prefill_time_signal.value)

    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


class Worker:
    """
        Engine -> (WIP)Executor -> Worker -> ModelRunner -> Model
        Worker interface that allows inference framwork to cleanly separate implementations for different harware.
    """

    def __init__(
        self,
        args,
    ) -> None:
        """
        Initialize the Worker.
        """
        super().__init__()
        self.args = args
        self.MAX_INFER_SEED = 9223372036854775806
        paddle.set_default_dtype(args.dtype)
        self.device_ids = self.args.device_ids.split(",")
        self.model_cfg = ModelConfig(args.model_name_or_path)

        from fastdeploy.worker.vl_gpu_model_runner import GPUVLModelRunner

        self.init_dist_env()
        self.format_print_configuration()
        self.helper_tensors = {}

        local_rank = self.rank % self.args.tensor_parallel_size
        self.local_data_parallel_id = self.rank // self.args.tensor_parallel_size

        self.infer_engine = GPUVLModelRunner(config=self.model_cfg,
                                             args=self.args,
                                             nranks=self.nranks,
                                             rank=self.rank)
        self.prefill_tracker = PrefillTracker(args.engine_pid)

        # Only applicable for standalone (single-machine) inference
        address = ('0.0.0.0', self.args.engine_worker_queue_port)
        self.engine_worker_queue = EngineWorkerQueue(
            address=address,
            is_server=False,
            num_client=self.nranks,
            client_id=local_rank,
            local_data_parallel_id=self.local_data_parallel_id)
        self.init_health()

    def init_dist_env(self, seed=20):
        """
        init distributed env
        """

        self.nranks = dist.get_world_size()
        strategy = fleet.DistributedStrategy()

        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": self.nranks,
            "pp_degree": 1,
            "sharding_degree": 1,
        }

        # Set control in tensor parallel
        strategy.tensor_parallel_configs = {"tensor_init_seed": seed}
        fleet.init(is_collective=True, strategy=strategy)
        self.rank = fleet.worker_index()

    def init_health(self):
        """
        init health signals
        """
        # To perceive whether each worker process is ready
        worker_ready_signal_data = np.zeros(shape=[self.nranks],
                                            dtype=np.int32)
        self.worker_ready_signal = IPCSignal(name="worker_ready_signal",
                                             array=worker_ready_signal_data,
                                             dtype=np.int32,
                                             suffix=self.args.engine_pid,
                                             create=False)
        self.worker_ready_signal.value[self.rank] = 1

        # To monitor the liveness of worker processes and record each step's timestamp
        worker_healthy_live_recorded_time_array = np.zeros(shape=[self.nranks],
                                                           dtype=np.int32)
        self.worker_healthy_live_signal = IPCSignal(
            name="worker_healthy_live_signal",
            array=worker_healthy_live_recorded_time_array,
            dtype=np.int32,
            suffix=self.args.engine_pid,
            create=False)
        self.worker_healthy_live_signal.value[self.rank] = int(time.time())

        # To perceive whether there is a new task to be processed
        exist_task_signal_data = np.zeros([1], dtype=np.int32)
        self.exist_task_signal = IPCSignal(name="exist_task_signal",
                                           array=exist_task_signal_data,
                                           dtype=np.int32,
                                           suffix=self.args.engine_pid,
                                           create=False)

        # To detect whether there are swapped tasks in the worker
        exist_swapped_task_signal_data = np.zeros([1], dtype=np.int32)
        self.exist_swapped_task_signal = IPCSignal(
            name="exist_swapped_task_signal",
            array=exist_swapped_task_signal_data,
            dtype=np.int32,
            suffix=self.args.engine_pid,
            create=False)

        model_weights_status = np.zeros([1], dtype=np.int32)
        self.model_weights_status_signal = IPCSignal(
            name="model_weights_status",
            array=model_weights_status,
            dtype=np.int32,
            suffix=self.args.engine_pid,
            create=False)

    def format_print_configuration(self):
        """
        print model config
        """
        logger.info("===============   Model Information   ==============")
        for k, v in self.model_cfg.__dict__.items():
            logger.info("{:<20}:{:<6}{}".format(k, "", v))
        logger.info("=============== Service Configuration ===============")
        for k, v in vars(self.args).items():
            logger.info("{:<20}:{:<6}{}".format(k, "", v))
        logger.info("=====================================================\n")

    def step_cuda(self):
        """
        step cuda
        """
        from fastdeploy.model_executor.ops.gpu import (step_reschedule,
                                                       step_system_cache)

        if self.args.enable_prefix_caching:
            step_system_cache(
                self.infer_engine.share_inputs["stop_flags"],
                self.infer_engine.share_inputs["seq_lens_this_time"],
                self.infer_engine.share_inputs["step_seq_lens_encoder"],
                self.infer_engine.share_inputs["step_seq_lens_decoder"],
                self.infer_engine.share_inputs["seq_lens_encoder"],
                self.infer_engine.share_inputs["seq_lens_decoder"],
                self.infer_engine.share_inputs["block_tables"],
                self.infer_engine.share_inputs["encoder_block_lens"],
                self.infer_engine.share_inputs["is_block_step"],
                self.infer_engine.share_inputs["step_block_list"],
                self.infer_engine.share_inputs["step_lens"],
                self.infer_engine.share_inputs["recover_block_list"],
                self.infer_engine.share_inputs["recover_lens"],
                self.infer_engine.share_inputs["need_block_list"],
                self.infer_engine.share_inputs["need_block_len"],
                self.infer_engine.share_inputs["used_list_len"],
                self.infer_engine.share_inputs["free_list"],
                self.infer_engine.share_inputs["free_list_len"],
                self.infer_engine.share_inputs["input_ids"],
                self.infer_engine.share_inputs["pre_ids"],
                self.infer_engine.share_inputs["step_idx"],
                self.infer_engine.share_inputs["next_tokens"],
                self.infer_engine.share_inputs["first_token_ids"],
                self.args.block_size, self.args.enc_dec_block_num)

        else:
            step_reschedule(
                self.infer_engine.share_inputs["stop_flags"],
                self.infer_engine.share_inputs["seq_lens_this_time"],
                self.infer_engine.share_inputs["step_seq_lens_encoder"],
                self.infer_engine.share_inputs["seq_lens_encoder"],
                self.infer_engine.share_inputs["seq_lens_decoder"],
                self.infer_engine.share_inputs["block_tables"],
                self.infer_engine.share_inputs["encoder_block_lens"],
                self.infer_engine.share_inputs["is_block_step"],
                self.infer_engine.share_inputs["step_block_list"],
                self.infer_engine.share_inputs["step_lens"],
                self.infer_engine.share_inputs["recover_block_list"],
                self.infer_engine.share_inputs["recover_lens"],
                self.infer_engine.share_inputs["need_block_list"],
                self.infer_engine.share_inputs["need_block_len"],
                self.infer_engine.share_inputs["used_list_len"],
                self.infer_engine.share_inputs["free_list"],
                self.infer_engine.share_inputs["free_list_len"],
                self.infer_engine.share_inputs["input_ids"],
                self.infer_engine.share_inputs["pre_ids"],
                self.infer_engine.share_inputs["step_idx"],
                self.infer_engine.share_inputs["next_tokens"],
                self.infer_engine.share_inputs["first_token_ids"],
                self.args.block_size,
                self.args.enc_dec_block_num,
            )

    def check_model_weights_status(self):
        """
        check model weights status
        """
        is_stop = 0
        while self.model_weights_status_signal.value[0] != 0:
            if self.model_weights_status_signal.value[0] == 1:
                logger.info(
                    f"infer engine stopped! start to load new checkpoint... {self.rank}"
                )
                self.infer_engine.update_parameters(self.args.engine_pid)
            elif self.model_weights_status_signal.value[0] == -1:
                logger.info(
                    f"infer engine stopped! start to clear checkpoint... {self.rank}"
                )
                self.infer_engine.clear_parameters(self.args.engine_pid)

            while True:
                if self.model_weights_status_signal.value[0] == 0:
                    logger.info(f"finished loading new checkpoint {self.rank}")
                    break
                elif is_stop == 1 or (self.model_weights_status_signal.value[0]
                                      == -2 and is_stop == 0):
                    if is_stop == 0:
                        logger.info(
                            f"finished clearing checkpoint {self.rank}")
                        is_stop = 1
                    time.sleep(0.001)
                    break
                else:
                    time.sleep(0.001)

    def run(self):
        """
        run function, continuously get tasks and do inference.
        """
        infer_seed_increment = paddle.full(shape=[self.args.max_num_seqs, 1],
                                           fill_value=4,
                                           dtype="int64")
        self.nnode = 1

        while True:
            if self.rank == 0:
                if self.model_weights_status_signal.value[0] != 0:
                    self.exist_task_signal.value[0] = 2
                else:
                    self.exist_task_signal.value[0] = 0

            if self.nranks > 1:
                paddle.distributed.barrier()

            if self.exist_task_signal.value[0] == 2:
                self.check_model_weights_status()

            self.insert_step = False

            self.worker_healthy_live_signal.value[self.rank] = int(time.time())
            mp_num_per_node = self.nranks

            if self.rank % mp_num_per_node == 0:
                if self.engine_worker_queue.num_tasks(
                ) > 0 and self.infer_engine.prefill_finished():
                    if self.nnode > 1:
                        self.engine_worker_queue.read_finish_flag.set(1)
                    else:
                        self.exist_task_signal.value[0] = 1

            if self.nranks > 1:
                paddle.distributed.barrier()

            if self.exist_task_signal.value[
                    0] == 1 or self.engine_worker_queue.read_finish_flag.get(
                    ) == 1:
                logger.info(f"Rank: {self.rank} Detected new requests.")
                self.insert_step = True

                tasks, read_finish = self.engine_worker_queue.get_tasks()
                if read_finish:
                    self.exist_task_signal.value[0] = 0
                    self.engine_worker_queue.read_finish_flag.set(0)

                req_dicts = []
                for req_dict, bsz in tasks:
                    num_running_requests = int(bsz)

                    req_dicts.extend(req_dict)
                req_ids = [req.request_id for req in req_dicts]
                logger.info(f"Rank: {self.rank}, num_running_requests: {num_running_requests}, " \
                            f"num_insert_requests: {len(req_dicts)}. {req_ids}")

                self.infer_engine.dy_input_preprocess(req_dicts)
                for req_dict in req_dicts:
                    if self.infer_engine.share_inputs["seq_lens_this_time"][
                            req_dict.idx] > 1:
                        self.prefill_tracker.start_prefill(req_dict.idx)
                self.infer_engine.share_inputs["not_need_stop"][0] = True

            if not self.infer_engine.share_inputs["not_need_stop"]:
                time.sleep(0.001)
                continue

            self.infer_engine.generate()
            self.infer_engine.share_inputs["infer_seed"].add_(
                infer_seed_increment)
            self.infer_engine.share_inputs[
                "infer_seed"][:] %= self.MAX_INFER_SEED
            for req_dict in req_dicts:
                if (self.infer_engine.share_inputs["seq_lens_this_time"][
                        req_dict.idx] == 1
                        and req_dict.idx in self.prefill_tracker.start_times):
                    self.prefill_tracker.end_prefill(req_dict.idx)
            self.infer_engine.update_chunked_prefill(req_dicts)
            self.step_cuda()

    def determine_num_available_blocks(self):
        """Profiles the peak memory usage of the model to determine how many
        KV blocks may be allocated without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculate the maximum possible number of GPU and CPU blocks
        that can be allocated with the remaining free memory.

        .. tip::
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        start_time = time.time()

        GiB = 1024**3
        paddle.device.cuda.empty_cache()

        paddle.device.cuda.reset_max_memory_allocated()
        before_activation_gpu_memory = paddle.device.cuda.max_memory_allocated(
        ) / GiB
        logger.info(
            f"before activate gpu memory: {before_activation_gpu_memory} GiB.")

        import gc

        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(
            int(self.device_ids[self.rank]))
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_gpu_memory = meminfo.total / GiB
        used_gpu_memory = meminfo.used / GiB
        pynvml.nvmlShutdown()
        logger.info(f"used gpu memory: {used_gpu_memory} GiB.")

        self.run_profile()
        current_max_peak_gpu_memory = paddle.device.cuda.max_memory_reserved(
        ) / GiB
        logger.info(
            f"current max peak gpu memory: {current_max_peak_gpu_memory} GiB.")
        per_block_memory_used = self.infer_engine._cal_theortical_kvcache(
        ) / GiB
        logger.info(f"each kv cache block takes {per_block_memory_used} GiB.")
        used_cache_gpu_memory = self.args.total_block_num * per_block_memory_used
        logger.info(f"used cache gpu memory: {used_cache_gpu_memory} GiB.")
        model_weights_memory = used_gpu_memory - used_cache_gpu_memory
        paddle_peak_increase = current_max_peak_gpu_memory - before_activation_gpu_memory
        memory_for_current_instance = total_gpu_memory * self.args.gpu_memory_utilization
        available_kv_cache_memory = memory_for_current_instance - used_gpu_memory - \
                                    paddle_peak_increase + used_cache_gpu_memory

        num_gpu_blocks = max(
            int(available_kv_cache_memory // per_block_memory_used),
            self.args.total_block_num)
        profile_time = time.time() - start_time

        msg = (f"Memory profiling takes {profile_time:.2f} seconds\n"
               "the current instance can use "
               "total_gpu_memory "
               f"({(total_gpu_memory):.2f}GiB)"
               " x gpu_memory_utilization "
               f"({self.args.gpu_memory_utilization})"
               f" = {(memory_for_current_instance):.2f}GiB\n"
               "model weights take "
               f"{(model_weights_memory ):.2f}GiB;"
               " Paddle activation peak memory takes "
               f"{(paddle_peak_increase):.2f}GiB;"
               " the rest of the memory reserved for KV Cache is "
               f"{(available_kv_cache_memory):.2f}GiB.")

        self.infer_engine.record_profile_msg = {
            "per_block_memory_used": per_block_memory_used,
            "paddle_peak_increase": paddle_peak_increase,
        }

        logger.info(msg)
        # Final cleanup

        get_profile_block_num = np.zeros(shape=[self.nranks], dtype=np.int32)
        self.get_profile_block_num_signal = IPCSignal(
            name="get_profile_block_num",
            array=get_profile_block_num,
            dtype=np.int32,
            suffix=self.args.engine_pid,
            create=False)
        self.get_profile_block_num_signal.value[self.rank] = int(
            num_gpu_blocks)
        while np.any(self.get_profile_block_num_signal.value <= 0):
            time.sleep(0.01)
        num_gpu_blocks = self.get_profile_block_num_signal.value.min().item()
        self.get_profile_block_num_signal.value[self.rank] = int(
            num_gpu_blocks)
        logger.info(
            f"{self.get_profile_block_num_signal.value[self.rank]} GPU KV blocks can be allocated."
        )
        self.infer_engine.num_gpu_blocks = num_gpu_blocks
        self.infer_engine._update_share_input_block_num()

        paddle.device.cuda.empty_cache()
        gc.collect()

    def run_profile(self):
        """
        run profile
        """
        infer_seed_increment = paddle.full(shape=[self.args.max_num_seqs, 1],
                                           fill_value=4,
                                           dtype="int64")

        self.infer_engine.dummy_input(self.args.max_num_batched_tokens,
                                      self.args.max_num_seqs)
        while True:
            if self.nranks > 1:
                paddle.distributed.barrier()
            self.infer_engine.generate()
            self.infer_engine.share_inputs["infer_seed"].add_(
                infer_seed_increment)
            self.infer_engine.share_inputs[
                "infer_seed"][:] %= self.MAX_INFER_SEED
            self.step_cuda()
            if int((self.infer_engine.share_inputs['seq_lens_this_time']
                    > 0).sum()) == 0:
                break


def main():
    """
    start worker
    """
    args = parse_args()
    worker = Worker(args)
    if args.do_profile:
        worker.determine_num_available_blocks()
    worker.run()


if __name__ == "__main__":
    main()
