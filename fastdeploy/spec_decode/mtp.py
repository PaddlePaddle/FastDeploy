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

import os
import time
from abc import abstractmethod
from typing import TYPE_CHECKING, List

import numpy as np
import paddle
from paddleformers.utils.log import logger

from fastdeploy.engine.request import Request, RequestType
from fastdeploy.inter_communicator import IPCSignal
from fastdeploy.model_executor.forward_meta import ForwardMeta
from fastdeploy.model_executor.layers.attention import get_attention_backend
from fastdeploy.model_executor.layers.attention.base_attention_backend import (
    AttentionBackend,
)
from fastdeploy.model_executor.layers.sample.sampler import MTPSampler
from fastdeploy.model_executor.model_loader import get_model_loader
from fastdeploy.model_executor.models import ModelForCasualLM
from fastdeploy.platforms import current_platform

if current_platform.is_xpu():
    from fastdeploy.model_executor.ops.xpu import set_data_ipc, share_external_data
    from fastdeploy.model_executor.xpu_pre_and_post_process import async_set_value
else:
    from fastdeploy.model_executor.ops.gpu import set_data_ipc, share_external_data
    from fastdeploy.model_executor.pre_and_post_process import async_set_value

from fastdeploy.worker.input_batch import (
    ProposerInputBatch,
    reorder_split_prefill_and_decode_form_index_to_batch_id,
)

from .base import Proposer

if TYPE_CHECKING:
    from fastdeploy.config import FDConfig


class MTPProposer(Proposer):
    """
    Proposer for Multi-Token-Prediction(MTP)

    Base class containing common logic. Platform-specific behavior is
    implemented in MTPProposerCUDA and MTPProposerXPU subclasses.
    """

    def __init__(
        self,
        fd_config: "FDConfig",
        main_model: ModelForCasualLM,
        local_rank: int,
        device_id: int,  # physical device id
        target_model_inputs,  # main model share inputs
    ):
        super().__init__(fd_config)
        self.num_main_model_layers = self.model_config.num_hidden_layers
        self.local_rank = local_rank
        self.device_id = device_id
        self.use_attn_mask_offset = self.enable_mm

        self._update_mtp_config(main_model)
        self._load_model()
        self.target_model_inputs = target_model_inputs
        self.mtp_strategy = self.speculative_config.mtp_strategy
        self.hybrid_mode = self.mtp_strategy == "with_ngram" and self.max_draft_token_num > self.num_model_steps
        self.enable_logprob = self.model_config.enable_logprob
        self.enable_draft_logprob = self.speculative_config.enable_draft_logprob
        self.cache_kvs_map = {}

        # [mixed, prefill, decoder]
        self.role = self.scheduler_config.splitwise_role
        self.pd_disaggregation_mode = fd_config.parallel_config.pd_disaggregation_mode

        self.sampler = MTPSampler(fd_config)
        self.model_inputs = ProposerInputBatch(self.fd_config, self.target_model_inputs)
        self.model_inputs.init_share_inputs()

        # CUDA Graph
        self.draft_model_use_cudagraph = self.graph_opt_config.draft_model_use_cudagraph
        self.cudagraph_capture_sizes = list(reversed(self.graph_opt_config.cudagraph_capture_sizes))
        self.sot_warmup_sizes = self.graph_opt_config.sot_warmup_sizes

        self.attn_backends: list[AttentionBackend] = []
        self._initialize_attn_backend()

        # Forward meta store the global meta information of the forward
        self.forward_meta = None
        self.exist_prefill_flag = False

    # ======================== Abstract methods ========================
    # Subclasses (MTPProposerCUDA / MTPProposerXPU) must implement these.

    @abstractmethod
    def _initialize_forward_meta(
        self, step_use_cudagraph: bool = False, is_dummy_run: bool = False, substep: int = 0
    ) -> None:
        """Initialize forward meta and attention metadata for a substep."""
        ...

    @abstractmethod
    def _prepare_inputs(self, full_hidden_states: paddle.Tensor) -> None:
        """Prepare MTP inputs from target model hidden states (whole-proposer preprocessing)."""
        ...

    @abstractmethod
    def _propose(self, step_use_cudagraph: bool = False, is_dummy_run: bool = False, real_bsz: int = 0) -> None:
        """Execute the multi-step MTP inference loop (per-substep preprocessing / forward / sampling)."""
        ...

    @abstractmethod
    def _post_process(self, sampled_token_ids) -> None:
        """Per-substep post-processing after sampling."""
        ...

    @abstractmethod
    def _update_status(self) -> None:
        """Whole-proposer post-processing: update main-model forward info and manage MTP block allocation."""
        ...

    # ======================== Overridable hooks ========================
    def _extend_draft_token_with_ngram_match(self):
        """Extend draft tokens with ngram matching. CUDA-only feature; no-op by default."""
        pass

    # ======================== Common methods ========================

    def _update_mtp_config(self, main_model):
        """
        Update config for MTP from global config
        """
        self.forward_meta: ForwardMeta = None
        self.model_config.architectures[0] = self.model_config.architectures[0].replace("Moe", "MTP")
        self.speculative_config.sharing_model = main_model
        # TODO (wangyanpeng): The number of MTP layers should be read from model config
        self.model_config.num_hidden_layers = 1
        self.model_config.model = self.speculative_config.model
        if "Ernie" in self.model_config.architectures[0]:
            self.model_config.pretrained_config.prefix_name = "ernie.mtp_block"
            self.model_config.prefix_layer_name = "mtp_block"
        if self.speculative_config.quantization != "":
            self.model_config.quantization = self.speculative_config.quantization
        self.model_config.start_layer_index = self.num_main_model_layers
        self.speculative_config.model_type = "mtp"
        if not self.use_attn_mask_offset:
            self.model_config.causal = True

    def _load_model(self):
        """
        Load MTP Layer
        """
        model_loader = get_model_loader(load_config=self.fd_config.load_config)
        self.model = model_loader.load_model(fd_config=self.fd_config)

    def dummy_prefill_inputs(self, num_tokens: int, batch_size: int, expected_decode_len: int):
        """Set dummy prefill inputs to model_inputs"""
        max_dec_len = expected_decode_len + 1

        input_length = min(
            num_tokens // batch_size,
            self.model_config.max_model_len - max_dec_len,
        )

        # TODO(wanglongzhi): Figure out the accurate buffer size of DeepEP.
        if self.fd_config.parallel_config.enable_expert_parallel:
            input_length = min(input_length, 32)

        block_num = (
            input_length + self.cache_config.block_size - 1
        ) // self.cache_config.block_size + self.cache_config.enc_dec_block_num

        for i in range(batch_size):
            idx = i
            self.model_inputs["input_ids"][idx : idx + 1, :input_length] = np.array([5] * input_length)
            self.model_inputs["eos_token_id"][:] = np.array([2], dtype="int64").reshape(-1, 1)
            self.model_inputs["seq_lens_this_time_buffer"][idx : idx + 1] = input_length
            self.model_inputs["seq_lens_encoder"][idx : idx + 1] = input_length
            self.model_inputs["seq_lens_decoder"][idx : idx + 1] = 0
            self.model_inputs["step_idx"][idx : idx + 1] = 0
            self.model_inputs["max_dec_len"][idx : idx + 1] = max_dec_len
            self.model_inputs["stop_flags"][idx : idx + 1] = False

            self.model_inputs["encoder_block_lens"][idx : idx + 1] = block_num
            self.model_inputs["block_tables"][idx : idx + 1, :block_num] = np.arange(
                idx * block_num, (idx + 1) * block_num, 1
            )
        self.model_inputs.seq_lens_this_time = self.model_inputs["seq_lens_this_time_buffer"]

    def initialize_kv_cache(self, main_model_num_blocks, profile: bool = False):
        """
        Initialize kv cache
        """
        self.num_gpu_blocks = int(main_model_num_blocks * self.speculative_config.num_gpu_block_expand_ratio)
        self.cache_kvs = {}

        # Get kv cache dtype
        cache_type = self.model_config.dtype
        kv_cache_quant_type = None
        if (
            self.quant_config
            and hasattr(self.quant_config, "kv_cache_quant_type")
            and self.quant_config.kv_cache_quant_type is not None
        ):
            cache_type = self._get_cache_type()
            kv_cache_quant_type = self.quant_config.kv_cache_quant_type

        # Get kv cache shape
        key_cache_shape, value_cache_shape = self.attn_backends[0].get_kv_cache_shape(
            max_num_blocks=self.num_gpu_blocks, kv_cache_quant_type=kv_cache_quant_type
        )
        if kv_cache_quant_type == "block_wise_fp8":
            kv_cache_scale_shape = [key_cache_shape[0], key_cache_shape[1], key_cache_shape[2]]
        local_rank = self.local_rank % self.parallel_config.tensor_parallel_size

        cache_ready_signal_data = np.zeros(shape=[self.parallel_config.tensor_parallel_size], dtype=np.int32)
        cache_ready_signal = IPCSignal(
            name="cache_ready_signal",
            array=cache_ready_signal_data,
            dtype=np.int32,
            suffix=self.parallel_config.local_engine_worker_queue_port,
            create=False,
        )

        # Check if gpu runner needs to create kv cache
        # 1. During profiling, it creates its own kv cache.
        # 2. If no need to profile, create kv cache unless kvcache_storage_backend or
        #    p/d disaggregation is enabled. Note: CPU cache (num_cpu_blocks > 0) does NOT
        #    prevent GPU runner from creating GPU cache tensors; cache transfer manager
        #    handles CPU<->GPU swap on top of the GPU tensors created here.
        create_cache_tensor = profile or not (
            self.fd_config.cache_config.kvcache_storage_backend
            or self.fd_config.scheduler_config.splitwise_role != "mixed"
        )

        if not create_cache_tensor:
            logger.info(f"Waiting for cache managers to create kv cache.. {cache_ready_signal.value}")
            while cache_ready_signal.value[local_rank] != 1:
                time.sleep(1)
            logger.info(f"OK! Stop waiting. {cache_ready_signal.value}")

        logger.info(f"Initializing kv cache for all layers. {cache_ready_signal.value}")

        if not create_cache_tensor:
            cache_kvs_list = []
            for i in range(
                self.num_main_model_layers,
                self.num_main_model_layers + self.model_config.num_hidden_layers,
            ):
                logger.info(
                    f"..attaching kv cache for mtp layer {i}: key:{key_cache_shape}, value:{value_cache_shape}"
                )
                key_cache = paddle.empty(shape=[], dtype=cache_type)
                key_cache_name = f"key_caches_{i}_rank{local_rank}.device{self.device_id}"
                val_cache_name = f"value_caches_{i}_rank{local_rank}.device{self.device_id}"
                key_cache = self._share_external_data(key_cache, key_cache_name, key_cache_shape)
                self.cache_kvs_map[key_cache_name] = key_cache
                cache_kvs_list.append(key_cache)
                value_cache = paddle.empty(shape=[], dtype=cache_type)
                value_cache = self._share_external_data(value_cache, val_cache_name, value_cache_shape)
                self.cache_kvs_map[val_cache_name] = value_cache
                cache_kvs_list.append(value_cache)

                if kv_cache_quant_type == "block_wise_fp8":
                    scale_key_cache_name = f"key_cache_scales_{i}_rank{local_rank}.device{self.device_id}"
                    scale_val_cache_name = f"value_cache_scales_{i}_rank{local_rank}.device{self.device_id}"
                    key_scale_cache = paddle.empty(shape=[], dtype=paddle.get_default_dtype())
                    key_scale_cache = self._share_external_data(
                        key_scale_cache, scale_key_cache_name, kv_cache_scale_shape
                    )
                    self.cache_kvs_map[scale_key_cache_name] = key_scale_cache
                    cache_kvs_list.append(key_scale_cache)
                    value_scale_cache = paddle.empty(shape=[], dtype=paddle.get_default_dtype())
                    value_scale_cache = self._share_external_data(
                        value_scale_cache, scale_val_cache_name, kv_cache_scale_shape
                    )
                    self.cache_kvs_map[scale_val_cache_name] = value_scale_cache
                    cache_kvs_list.append(value_scale_cache)

            self.model_inputs["caches"] = cache_kvs_list
        else:
            cache_kvs_list = []
            for i in range(
                self.num_main_model_layers,
                self.num_main_model_layers + self.model_config.num_hidden_layers,
            ):
                logger.info(f"..creating kv cache for mtp layer {i}: key:{key_cache_shape}, value:{value_cache_shape}")
                key_cache = paddle.full(
                    shape=key_cache_shape,
                    fill_value=0,
                    dtype=cache_type,
                )
                key_cache_name = f"key_caches_{i}_rank{local_rank}.device{self.device_id}"
                set_data_ipc(key_cache, key_cache_name)
                self.cache_kvs_map[key_cache_name] = key_cache
                cache_kvs_list.append(key_cache)

                val_cache = paddle.full(
                    shape=value_cache_shape,
                    fill_value=0,
                    dtype=cache_type,
                )
                val_cache_name = f"value_caches_{i}_rank{local_rank}.device{self.device_id}"
                set_data_ipc(val_cache, val_cache_name)
                self.cache_kvs_map[val_cache_name] = val_cache
                cache_kvs_list.append(val_cache)

                if kv_cache_quant_type == "block_wise_fp8":
                    key_cache_scales = paddle.full(
                        shape=kv_cache_scale_shape,
                        fill_value=0,
                        dtype=paddle.get_default_dtype(),
                    )
                    key_cache_scales_name = f"key_cache_scales_{i}_rank{local_rank}.device{self.device_id}"
                    set_data_ipc(key_cache_scales, key_cache_scales_name)
                    self.cache_kvs_map[key_cache_scales_name] = key_cache_scales
                    cache_kvs_list.append(key_cache_scales)

                    val_cache_scales = paddle.full(
                        shape=kv_cache_scale_shape,
                        fill_value=0,
                        dtype=paddle.get_default_dtype(),
                    )
                    val_cache_scales_name = f"value_cache_scales_{i}_rank{local_rank}.device{self.device_id}"
                    set_data_ipc(val_cache_scales, val_cache_scales_name)
                    self.cache_kvs_map[val_cache_scales_name] = val_cache_scales
                    cache_kvs_list.append(val_cache_scales)

            self.model_inputs["caches"] = cache_kvs_list

        self._empty_cache()

    def _initialize_attn_backend(
        self,
    ) -> None:
        """
        Initialize attention backends and forward metadata
        """
        assert len(self.attn_backends) == 0

        num_heads = self.model_config.num_attention_heads // self.parallel_config.tensor_parallel_size
        self.model_config.kv_num_heads = max(
            1,
            int(self.model_config.num_key_value_heads) // self.parallel_config.tensor_parallel_size,
        )
        head_dim = self.model_config.head_dim

        # Initialize AttentionBackend buffers
        encoder_block_shape_q = 64
        decoder_block_shape_q = 16

        self.model_inputs["decoder_batch_ids"] = paddle.zeros_like(self.target_model_inputs["decoder_batch_ids"])
        self.model_inputs["decoder_tile_ids_per_batch"] = paddle.zeros_like(
            self.target_model_inputs["decoder_tile_ids_per_batch"]
        )
        if current_platform.is_xpu() or current_platform.is_maca():
            self.model_inputs["decoder_num_blocks_cpu"] = paddle.zeros_like(
                self.target_model_inputs["decoder_num_blocks_cpu"]
            ).cpu()
        else:
            self.model_inputs["decoder_num_blocks_cpu"] = paddle.zeros_like(
                self.target_model_inputs["decoder_num_blocks_cpu"]
            ).pin_memory()
        self.model_inputs["decoder_num_blocks_device"] = paddle.zeros_like(
            self.target_model_inputs["decoder_num_blocks_device"]
        )
        self.model_inputs["decoder_chunk_size_device"] = paddle.zeros_like(
            self.target_model_inputs["decoder_chunk_size_device"]
        )
        self.model_inputs["max_len_tensor_cpu"] = paddle.zeros_like(
            self.target_model_inputs["max_len_tensor_cpu"]
        ).cpu()

        self.model_inputs["encoder_batch_ids"] = paddle.zeros_like(self.target_model_inputs["encoder_batch_ids"])
        self.model_inputs["encoder_tile_ids_per_batch"] = paddle.zeros_like(
            self.target_model_inputs["encoder_tile_ids_per_batch"]
        )
        self.model_inputs["encoder_num_blocks_x_cpu"] = paddle.zeros_like(
            self.target_model_inputs["encoder_num_blocks_x_cpu"]
        ).cpu()
        self.model_inputs["kv_batch_ids"] = paddle.zeros_like(self.target_model_inputs["kv_batch_ids"])
        self.model_inputs["kv_tile_ids_per_batch"] = paddle.zeros_like(
            self.target_model_inputs["kv_tile_ids_per_batch"]
        )
        self.model_inputs["kv_num_blocks_x_cpu"] = paddle.zeros_like(
            self.target_model_inputs["kv_num_blocks_x_cpu"]
        ).cpu()

        # Get the attention backend
        attn_cls = get_attention_backend()
        attn_backend = attn_cls(
            self.fd_config,
            kv_num_heads=self.model_config.kv_num_heads,
            num_heads=num_heads,
            head_dim=head_dim,
            encoder_block_shape_q=encoder_block_shape_q,
            decoder_block_shape_q=decoder_block_shape_q,
        )
        if attn_backend is None:
            raise NotImplementedError(
                "Attention backend which you specified is not supported, please set FD_ATTENTION_BACKEND correctly."
            )
        self.attn_backends.append(attn_backend)

    def update_mtp_block_num(self, num_gpu_blocks, skip_cache_init: bool = False) -> None:
        """
        Update MTP block num by theoretical calculation

        Args:
            num_gpu_blocks: Main model GPU block count.
            skip_cache_init: When True, skip internal initialize_kv_cache call.
                Set this when the caller (e.g. gpu_model_runner with enable_cache_manager_v1)
                has already re-created MTP cache via cache_controller.
        """
        # Reset block table and kv cache with global block num
        self.main_model_num_gpu_blocks = num_gpu_blocks
        if not skip_cache_init:
            self.initialize_kv_cache(main_model_num_blocks=self.main_model_num_gpu_blocks)

        # Reset free list
        free_list = list(
            range(
                self.num_gpu_blocks - 1,
                int(self.main_model_num_gpu_blocks * self.cache_config.kv_cache_ratio) - 1,
                -1,
            )
        )
        self.free_list_len = len(free_list)
        self.model_inputs.update(
            {
                "free_list": paddle.to_tensor(free_list, dtype="int32"),
                "free_list_len": paddle.full([1], self.free_list_len, dtype="int32"),
            }
        )

    def insert_tasks_v1(
        self, req_dicts: List[Request], num_running_requests: int, target_model_index_to_batch_id: dict = {}
    ):

        if "caches" not in self.model_inputs:
            self.initialize_kv_cache()
        req_len = len(req_dicts)
        self.model_inputs["num_running_requests"] = num_running_requests
        self.model_inputs["running_requests_ids"] = range(num_running_requests)
        if target_model_index_to_batch_id:
            self.model_inputs.index_to_batch_id = dict(target_model_index_to_batch_id)
        for i in range(req_len):
            request = req_dicts[i]
            logger.debug(f"{i}th request-{request.request_id}: {request}")
            idx = self.model_inputs.get_index_by_batch_id(request.idx)
            if request.task_type.value == RequestType.PREFILL.value:  # prefill task
                prefill_start_index = request.prefill_start_index
                prefill_end_index = request.prefill_end_index
                length = prefill_end_index - prefill_start_index

                input_ids = request.prompt_token_ids + request.output_token_ids

                self.model_inputs["input_ids_len"][idx] = length - 1
                async_set_value(self.model_inputs["pre_ids"][idx : idx + 1], -1)
                self.model_inputs["input_ids"][idx : idx + 1, : length - 1] = self.target_model_inputs["input_ids"][
                    idx : idx + 1, 1:length
                ]
                # TODO: use token_all_ids replace with input_ids_cpu
                if getattr(self, "hybrid_mode", False) and "input_ids_cpu" in self.model_inputs:
                    self.model_inputs["input_ids_cpu"][idx : idx + 1, : length - 1] = self.target_model_inputs[
                        "input_ids"
                    ][idx : idx + 1, 1:length].cpu()
                encoder_block_num = len(request.block_tables)
                async_set_value(self.model_inputs["encoder_block_lens"][idx : idx + 1], encoder_block_num)
                async_set_value(self.model_inputs["block_tables"][idx : idx + 1, :], -1)
                async_set_value(
                    self.model_inputs["block_tables"][idx : idx + 1, :encoder_block_num], request.block_tables
                )

                async_set_value(self.model_inputs["stop_flags"][idx : idx + 1], False)
                async_set_value(self.model_inputs["batch_drop"][idx : idx + 1], False)

                async_set_value(self.model_inputs["seq_lens_encoder"][idx : idx + 1], length)
                self.exist_prefill_flag = True
                async_set_value(self.model_inputs["seq_lens_decoder"][idx : idx + 1], prefill_start_index)
                async_set_value(self.model_inputs["seq_lens_this_time_buffer"][idx : idx + 1], length)
                async_set_value(
                    self.model_inputs["step_idx"][idx : idx + 1],
                    len(request.output_token_ids) if prefill_end_index >= len(input_ids) else 0,
                )
                if self.use_attn_mask_offset:
                    inputs = request.multimodal_inputs
                    self.model_inputs["attn_mask_offsets_full"][idx][0 : prefill_end_index - prefill_start_index] = (
                        paddle.to_tensor(
                            inputs["attention_mask_offset"][prefill_start_index:prefill_end_index], dtype="int32"
                        )
                    )
                    # GPU don't need it anymore
                    # NOTE: XPU backend needs decoder attention mask offset; GPU backend does not use it
                    if current_platform.is_xpu():
                        self.model_inputs["attn_mask_offsets_decoder"][idx : idx + 1] = (
                            inputs["attention_mask_offset"][prefill_end_index - 1] + 1
                        )
                if (
                    self.fd_config.scheduler_config.splitwise_role == "decode"
                ):  # In PD, we continue to decode after P generates first token
                    async_set_value(self.model_inputs["seq_lens_encoder"][idx : idx + 1], 0)
                    self.exist_prefill_flag = False
                    async_set_value(self.model_inputs["seq_lens_this_time_buffer"][idx : idx + 1], length + 1)
                    # NOTE(liuzichang):
                    # extra 1 : P-D split need rollback one step

                    async_set_value(self.model_inputs["recompute_token_num"][idx : idx + 1], 0)
                    async_set_value(self.model_inputs["mask_rollback"][idx : idx + 1], 1)
                # has_prefill_task = True
            elif request.task_type.value == RequestType.DECODE.value:  # decode task
                encoder_block_num = len(request.block_tables)
                async_set_value(self.model_inputs["encoder_block_lens"][idx : idx + 1], encoder_block_num)
                async_set_value(self.model_inputs["block_tables"][idx : idx + 1, :], -1)
                if current_platform.is_cuda():
                    async_set_value(
                        self.model_inputs["block_tables"][idx : idx + 1, :encoder_block_num], request.block_tables
                    )
                else:
                    self.model_inputs["block_tables"][idx : idx + 1, :encoder_block_num] = np.array(
                        request.block_tables, dtype="int32"
                    )
            else:
                async_set_value(self.model_inputs["block_tables"][idx : idx + 1, :], -1)
                async_set_value(self.model_inputs["stop_flags"][idx : idx + 1], True)
                async_set_value(self.model_inputs["seq_lens_this_time_buffer"][idx : idx + 1], 0)
                async_set_value(self.model_inputs["seq_lens_decoder"][idx : idx + 1], 0)
                async_set_value(self.model_inputs["seq_lens_encoder"][idx : idx + 1], 0)
                async_set_value(self.model_inputs["is_block_step"][idx : idx + 1], False)
                continue

        # TODO(liuzichang): Solve splitewise-p bug to restore
        # self.model_inputs["seq_lens_this_time"] = self.model_inputs["seq_lens_this_time_buffer"][:num_running_requests]
        self.model_inputs.seq_lens_this_time = self.model_inputs["seq_lens_this_time_buffer"]

    def insert_prefill_inputs(self, req_dicts: List[Request], num_running_requests: int):
        """
        Process inputs for prefill tasks and insert it to model_inputs buffer
        """
        # TODO:Init role in initialize process
        if req_dicts[-1].disaggregate_info is not None:
            if req_dicts[-1].disaggregate_info["role"] == "prefill":
                self.role = "prefill"
                os.environ["PREFILL_NODE_ONE_STEP_STOP"] = "1"
            elif req_dicts[-1].disaggregate_info["role"] == "decode":
                self.role = "decode"
        else:
            self.role = "mixed"

        req_len = len(req_dicts)
        for i in range(req_len):
            request = req_dicts[i]
            idx = request.idx
            length = len(request.prompt_token_ids)
            self.model_inputs.input_ids_len[idx] = length - 1

            if req_dicts[i].disaggregate_info is not None and req_dicts[i].disaggregate_info["role"] == "decode":
                length = len(request.prompt_token_ids)
                if length > 1:
                    self.model_inputs["input_ids"][idx : idx + 1, : length - 1] = self.target_model_inputs[
                        "input_ids"
                    ][idx : idx + 1, 1:length]
                    self.model_inputs["input_ids_cpu"][idx : idx + 1, : length - 1] = np.array(
                        request.prompt_token_ids
                    )[1:]
                self.model_inputs["pre_ids"][idx : idx + 1] = request.prompt_token_ids[-1]
                prefill_token_num = self.max_draft_token_num + 1
                self.model_inputs["draft_tokens"][idx : idx + 1, 0:1] = paddle.to_tensor(
                    request.draft_token_ids[1:2], dtype="int64"
                )

                self.model_inputs["seq_lens_encoder"][idx : idx + 1] = 0
                self.model_inputs["seq_lens_decoder"][idx : idx + 1] = length
                self.model_inputs["seq_lens_this_time_buffer"][idx : idx + 1] = prefill_token_num

                self.model_inputs["stop_flags"][idx : idx + 1] = False
                self.model_inputs["batch_drop"][idx : idx + 1] = False
                self.model_inputs["step_idx"][idx : idx + 1] = 1
                encoder_block_num = len(request.block_tables)

                self.model_inputs["encoder_block_lens"][idx : idx + 1] = encoder_block_num
                self.model_inputs["block_tables"][idx : idx + 1, :] = -1
                self.model_inputs["block_tables"][idx : idx + 1, :encoder_block_num] = np.array(
                    request.block_tables, dtype="int32"
                )

            else:
                length = len(request.prompt_token_ids)

                if length > 1:
                    self.model_inputs["input_ids"][idx : idx + 1, : length - 1] = self.target_model_inputs[
                        "input_ids"
                    ][idx : idx + 1, 1:length]
                    self.model_inputs["input_ids_cpu"][idx : idx + 1, : length - 1] = np.array(
                        request.prompt_token_ids
                    )[1:]
                self.model_inputs["pre_ids"][idx : idx + 1] = -1
                self.model_inputs["step_idx"][idx : idx + 1] = 0
                if self.cache_config.enable_chunked_prefill:
                    token_chunk_size = request.prefill_chunk_info[0]
                    self.model_inputs["seq_lens_encoder"][idx : idx + 1] = token_chunk_size
                    self.model_inputs["seq_lens_this_time_buffer"][idx : idx + 1] = token_chunk_size
                else:
                    self.model_inputs["seq_lens_encoder"][idx : idx + 1] = length
                    self.model_inputs["seq_lens_this_time_buffer"][idx : idx + 1] = length

                self.model_inputs["seq_lens_decoder"][idx : idx + 1] = request.get("seq_lens_decoder", 0)
                self.model_inputs["stop_flags"][idx : idx + 1] = False
                self.model_inputs["batch_drop"][idx : idx + 1] = False

                encoder_block_num = len(request.get("block_tables"))
                self.model_inputs["encoder_block_lens"][idx : idx + 1] = encoder_block_num
                self.model_inputs["block_tables"][idx : idx + 1, :] = -1
                self.model_inputs["block_tables"][idx : idx + 1, :encoder_block_num] = np.array(
                    request.get("block_tables"), dtype="int32"
                )
        self.model_inputs.seq_lens_this_time = self.model_inputs["seq_lens_this_time_buffer"]

    def exist_prefill(self):
        """
        check whether prefill stage exist
        """
        return self.exist_prefill_flag

    def update_task_chunk_prefill(self, task):
        """
        Update single task's chunk_prefill info
        """
        idx = self.model_inputs.get_index_by_batch_id(task.idx)
        start_idx = sum(task.prefill_chunk_info[: task.chunk_idx])

        if task.chunk_idx == len(task.prefill_chunk_info):
            self.model_inputs["seq_lens_encoder"][idx : idx + 1] = 0
            self.model_inputs["step_idx"][idx : idx + 1] = 1
            self.model_inputs["seq_lens_decoder"][idx : idx + 1] = start_idx + task.get("seq_lens_decoder", 0)
        else:
            token_chunk_size = task.prefill_chunk_info[task.chunk_idx]

            if task.chunk_idx < len(task.prefill_chunk_info) - 1:
                self.model_inputs["input_ids"][idx, :token_chunk_size] = np.array(
                    task.prompt_token_ids[start_idx + 1 : start_idx + token_chunk_size + 1]
                )
            # Last prefill
            else:
                self.model_inputs["input_ids"][idx, : token_chunk_size - 1] = np.array(
                    task.prompt_token_ids[start_idx + 1 : start_idx + token_chunk_size]
                )

            self.model_inputs["seq_lens_this_time"][idx : idx + 1] = token_chunk_size
            self.model_inputs["seq_lens_encoder"][idx : idx + 1] = token_chunk_size
            self.model_inputs["step_idx"][idx : idx + 1] = 0
            self.model_inputs["seq_lens_decoder"][idx : idx + 1] = start_idx + task.get("seq_lens_decoder", 0)

    def _run_impl(
        self,
        full_hidden_states: paddle.Tensor,
        step_use_cudagraph: bool = False,
        is_dummy_run: bool = False,
        real_bsz: int = 0,
    ):
        """Execute Draft Model"""
        self._prepare_inputs(full_hidden_states)
        self._propose(step_use_cudagraph=step_use_cudagraph, is_dummy_run=is_dummy_run, real_bsz=real_bsz)
        self._update_status()
        if self.hybrid_mode:
            self._extend_draft_token_with_ngram_match()

    def is_chunk_prefill_enabled(self):
        """"""
        return True

    def _empty_cache(self):
        if current_platform.is_cuda():
            paddle.device.cuda.empty_cache()
        elif current_platform.is_xpu():
            paddle.device.xpu.empty_cache()
        else:
            paddle.device.empty_cache()

    def _get_cache_type(self):
        cache_type = None
        if current_platform.is_cuda():
            cache_type = "uint8"
        elif current_platform.is_xpu():
            cache_type = "int8"
        else:
            raise NotImplementedError
        return cache_type

    def reorder_inputs(self, target_model_input_batch):
        """
        Reorder inputs to split prefill and decode.
        """
        reorder_split_prefill_and_decode_form_index_to_batch_id(self.model_inputs, target_model_input_batch)

    def _share_external_data(self, cache, cache_name, cache_shape):
        if current_platform.is_xpu():
            return share_external_data(cache, cache_name, cache_shape, False)
        else:
            return share_external_data(cache, cache_name, cache_shape)


def create_mtp_proposer(fd_config, main_model, local_rank, device_id, share_inputs):
    """Factory function that returns the platform-specific MTPProposer subclass."""
    if current_platform.is_xpu():
        from fastdeploy.spec_decode.mtp_xpu import MTPProposerXPU

        return MTPProposerXPU(fd_config, main_model, local_rank, device_id, share_inputs)
    elif current_platform.is_cuda() or current_platform.is_maca():
        from fastdeploy.spec_decode.mtp_cuda import MTPProposerCUDA

        return MTPProposerCUDA(fd_config, main_model, local_rank, device_id, share_inputs)
    else:
        raise RuntimeError(
            f"Unsupported platform for MTP: {current_platform}. " f"Supported platforms: CUDA, MACA, XPU"
        )
