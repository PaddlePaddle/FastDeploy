"""
# Copyright (c) 2026  PaddlePaddle Authors. All Rights Reserved.
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

import time
from typing import List, Optional

import numpy as np
import paddle
from paddle import nn
from paddle.nn import functional as F
from paddleformers.utils.log import logger

from fastdeploy import envs
from fastdeploy.config import FDConfig
from fastdeploy.engine.request import Request, RequestType
from fastdeploy.inter_communicator import IPCSignal
from fastdeploy.model_executor.forward_meta import ForwardMeta, ForwardMode
from fastdeploy.model_executor.graph_optimization.utils import sot_warmup_guard
from fastdeploy.model_executor.layers.attention.flashinfer_backend import (
    FlashInferAttentionBackend,
)
from fastdeploy.model_executor.layers.sample.ops.top_k_top_p_sampling import (
    top_k_top_p_sampling,
)
from fastdeploy.model_executor.model_loader import get_model_loader
from fastdeploy.model_executor.pre_and_post_process import async_set_value
from fastdeploy.output.stream_transfer_data import DecoderState, StreamTransferData
from fastdeploy.platforms import current_platform
from fastdeploy.spec_decode import SpecMethod
from fastdeploy.utils import print_gpu_memory_use
from fastdeploy.worker.gpu.async_output import AsyncOutput, AsyncUploader
from fastdeploy.worker.gpu.block_table import BlockTable
from fastdeploy.worker.gpu.buffer_utils import async_to_tensor
from fastdeploy.worker.gpu.forward_meta import ForwardMetaV1
from fastdeploy.worker.gpu.input_batch import (
    InputBatch,
    InputBuffers,
    prepare_input_ids,
    prepare_query_start_loc_and_seq_lens,
)
from fastdeploy.worker.gpu.post_update import post_update
from fastdeploy.worker.gpu.request_state import RequestState
from fastdeploy.worker.gpu.rotary_embedding.mrope import compute_cos_sin_cache
from fastdeploy.worker.gpu.sampler import Sampler, SamplingState
from fastdeploy.worker.gpu.sampler.sampler import SamplerInputs
from fastdeploy.worker.model_runner_base import ModelRunnerBase
from fastdeploy.worker.output import LogprobsTensors

# Platform-specific imports for IPC functions
if current_platform.is_cuda():
    from fastdeploy.model_executor.ops.gpu import (
        set_data_ipc,
        share_external_data,
        unset_data_ipc,
    )
else:
    share_external_data = None
    set_data_ipc = None
    unset_data_ipc = None


class GPUModelRunnerV1(ModelRunnerBase):
    def __init__(
        self,
        fd_config: FDConfig,
        device: str,  # logic device
        device_id: int,  # physical device id
        rank: int,
        local_rank: int,
    ):
        super().__init__(fd_config=fd_config, device=device)
        self.MAX_INFER_SEED = 9223372036854775806
        self.enable_mm = self.fd_config.enable_mm_runtime
        self.rank = rank
        self.local_rank = local_rank
        self.device_id = device_id
        self.enable_early_stop = self.fd_config.early_stop_config.enable_early_stop
        self.is_pooling_model = self.fd_config.model_config.runner_type == "pooling"
        self.ori_vocab_size = self.fd_config.model_config.ori_vocab_size

        self.forward_batch_reqs_list: list[Request] = [None for _ in range(self.scheduler_config.max_num_seqs)]
        self.cache_kvs_map: dict = {}

        self.spec_method = self.fd_config.speculative_config.method
        self.speculative_decoding = self.spec_method is not None

        self.num_spec_tokens = self.fd_config.speculative_config.num_model_steps if self.speculative_decoding else 0

        # logprob
        self.enable_logprob = fd_config.model_config.enable_logprob
        self.max_logprobs = None
        if self.enable_logprob:
            self.max_logprobs = (
                self.ori_vocab_size
                if fd_config.model_config.max_logprobs == -1
                else fd_config.model_config.max_logprobs
            )
        self.temp_scaled_logprobs = True
        self.top_p_normalized_logprobs = True
        self.prompt_logprobs_reqs: dict[str, Request] = {}
        self.in_progress_prompt_logprobs: dict[str, LogprobsTensors] = {}

        # spec
        self.spec_method = self.fd_config.speculative_config.method
        self.increment_value = (
            4 if not self.speculative_decoding else (self.speculative_config.num_speculative_tokens + 1) * 4
        )
        self.infer_seed_increment = paddle.full(
            shape=[self.scheduler_config.max_num_seqs, 1], fill_value=self.increment_value, dtype="int64"
        )

        # CUDA Graph
        self.use_cudagraph = self.graph_opt_config.use_cudagraph
        self.cudagraph_capture_sizes = list(reversed(self.graph_opt_config.cudagraph_capture_sizes))
        self.cudagraph_capture_sizes_prefill = list(reversed(self.graph_opt_config.cudagraph_capture_sizes_prefill))
        self.cudagraph_only_prefill = self.graph_opt_config.cudagraph_only_prefill

        # Forward meta store the global meta information of the forward
        self.forward_meta: ForwardMeta = None

        # Rollout routing replay config
        self.routing_replay_manager = None

        self.async_uploader = AsyncUploader(
            local_rank=self.local_rank, port=self.fd_config.parallel_config.local_engine_worker_queue_port
        )

        self.enable_entropy = self.model_config.enable_entropy

        # init signal
        cache_ready_signal_data = np.zeros(shape=[self.parallel_config.tensor_parallel_size], dtype=np.int32)
        self.cache_ready_signal = IPCSignal(
            name="cache_ready_signal",
            array=cache_ready_signal_data,
            dtype=np.int32,
            suffix=self.parallel_config.local_engine_worker_queue_port,
            create=False,
        )

        # state flag
        self.is_kvcache_sleeping = False
        self.is_weight_sleeping = False

        # MRV1
        self.request_state = RequestState(
            max_num_seqs=self.scheduler_config.max_num_seqs,
            max_model_len=self.model_config.max_model_len,
            max_num_batched_tokens=self.scheduler_config.max_num_batched_tokens,
            num_speculative_steps=self.speculative_config.num_speculative_tokens,
            vocab_size=self.model_config.vocab_size,
        )

        self.sampling_state = SamplingState(
            max_num_seqs=self.scheduler_config.max_num_seqs,
            eos_tokens_len=self.model_config.eos_tokens_lens,
            max_stop_seqs_num=self.model_config.max_stop_seqs_num,
            stop_seqs_max_len=self.model_config.stop_seqs_max_len,
            bad_words_max_len=self.model_config.bad_words_max_len,
            max_bad_words_num=self.model_config.max_bad_words_num,
        )
        self.sampler = Sampler(self.sampling_state)
        self.async_output = None

        self.input_buffers = InputBuffers(
            max_num_seqs=self.scheduler_config.max_num_seqs,
            max_num_tokens=self.scheduler_config.max_num_batched_tokens,
            max_position_embeddings=self.model_config.max_model_len,
            rotary_dim=self.model_config.head_dim,
        )

        self.caches = None

        async_set_value(
            self.input_buffers.cos_sin_buffer,
            compute_cos_sin_cache(
                rotary_dim=self.model_config.head_dim,
                max_position_embeddings=self.model_config.max_model_len,
                base=self.model_config.rope_theta,
            ),
        )

    def exist_prefill(self):
        return self.request_state.exist_prefill()

    @property
    def is_sleeping(self):
        return self.is_weight_sleeping or self.is_kvcache_sleeping

    def exist_decode(self):
        return self.request_state.exist_decode()

    def insert_tasks_v1(self, req_dicts: List[Request], num_running_requests: int = None):
        for req in req_dicts:
            if req.task_type.value == RequestType.PREFILL.value:
                all_token_ids = req.prompt_token_ids + req.output_token_ids
                prompt_len = len(req.prompt_token_ids)
                prefill_len = len(req.prompt_token_ids)
                start_idx = req.prefill_start_index
                end_idx = req.prefill_end_index
                batched_input_ids = all_token_ids[start_idx:end_idx]
                num_tokens = end_idx - start_idx
                self.sampling_state.add_request(req.idx, req)

                self.request_state.add_request(
                    req.idx,
                    num_tokens,
                    prompt_len,
                    prefill_len,
                    all_token_ids,
                    batched_input_ids,
                    start_idx,
                )
                # Initialize total_len for this request (prompt + output so far)
                total_len_init = len(all_token_ids)
                self.request_state.total_len.stage_write_elem(req.idx, total_len_init)

                # Initialize num_output_tokens (decode tokens emitted so far)
                self.request_state.num_output_tokens[req.idx] = len(req.output_token_ids)

                # Set EOS token IDs from the first request (model-level, same for all)
                if not self.sampling_state.eos_inited and req.eos_token_ids is not None:
                    self.sampling_state.init_eos(req.eos_token_ids)

            elif req.task_type.value == RequestType.DECODE.value:
                # ScheduledDecodeTask: request state already exists from prefill.
                # Just mark it as active (1 token to decode) for this step.
                self.request_state.num_tokens_per_seq[req.idx] = 1
            elif req.task_type.value == RequestType.ABORT.value:
                # Abort task: mark request as inactive.
                self.request_state.num_tokens_per_seq[req.idx] = 0
                continue  # No block tables to append for abort tasks
            else:  # preempted task
                self.request_state.num_tokens_per_seq[req.idx] = 0

            if hasattr(req, "block_tables") and req.block_tables is not None:
                self.block_table.append_block_ids(
                    req.idx,
                    req.block_tables,
                )

        self.request_state.apply_staged_writes()
        self.sampling_state.apply_staged_writes()
        self.block_table.apply_staged_writes()

    def insert_prefill_inputs(self, req_dicts: List[Request], num_running_requests: int):
        raise NotImplementedError("GPUs only support KVCACHE SCHEDULER V1 in versions 2.6 and above.")

    def prepare_inputs(self) -> InputBatch:
        running_idx = self.request_state.num_tokens_per_seq > 0
        num_seqs = np.sum(running_idx)

        decoding_batch = self.request_state.num_tokens_per_seq == 1
        prefill_batch = self.request_state.num_tokens_per_seq > 1
        num_decodes = np.sum(decoding_batch)
        num_prefills = np.sum(prefill_batch)
        num_decode_tokens = np.sum(self.request_state.num_tokens_per_seq[decoding_batch])
        num_prefill_tokens = np.sum(self.request_state.num_tokens_per_seq[prefill_batch])
        num_draft_tokens = num_decode_tokens * self.num_spec_tokens
        num_tokens = num_decode_tokens + num_prefill_tokens + num_draft_tokens

        # 排序P在前，D在后
        valid_indices = np.where(running_idx)[0]
        sorted_num_tokens_per_seq = np.sort(self.request_state.num_tokens_per_seq[running_idx])[::-1]
        sorted_idx = np.argsort(self.request_state.num_tokens_per_seq[running_idx])[::-1]
        idx_mapping_np = valid_indices[sorted_idx]
        idx_mapping = async_to_tensor(idx_mapping_np)

        # Compute query_len per sequence (sorted: prefills first, decodes after).
        # prefill: query_len = num_tokens_per_seq (original token count)
        # decode:  query_len = 1 + num_spec_tokens (sampled + draft tokens)
        query_lens = sorted_num_tokens_per_seq.copy()
        query_lens[num_prefills:] = query_lens[num_prefills:] + self.num_spec_tokens
        query_start_loc_np = np.zeros(num_seqs + 1, dtype=np.int32)
        np.cumsum(query_lens, out=query_start_loc_np[1:])
        seq_lens_np = self.request_state.num_computed_tokens_np[idx_mapping_np] + query_lens

        sorted_num_tokens_per_seq_gpu = async_to_tensor(sorted_num_tokens_per_seq)

        prepare_query_start_loc_and_seq_lens(
            sorted_num_tokens_per_seq_gpu,
            idx_mapping,
            self.request_state.num_computed_tokens.gpu,
            self.input_buffers.query_start_loc,
            self.input_buffers.seq_lens,
            self.num_spec_tokens,
        )

        # cu_num_logits: prefill seqs -> 1 logit, decode seqs -> 1 + num_spec_tokens
        # (sorted P-first, D-after, so decode lanes start at index num_prefills).
        num_logits_per_seq = np.ones(num_seqs, dtype=np.int32)
        num_logits_per_seq[num_prefills:] += self.num_spec_tokens
        cu_num_logits_np = np.cumsum(np.insert(num_logits_per_seq, 0, 0)).astype(np.int32)
        total_num_logits = int(cu_num_logits_np[-1])
        cu_num_logits = async_to_tensor(cu_num_logits_np)

        # Outputs of prepare_input_ids that are not part of InputBuffers.
        logits_indices = paddle.empty([total_num_logits], dtype=paddle.int64)
        expanded_idx_mapping = paddle.empty([total_num_logits], dtype=paddle.int32)
        expanded_local_pos = paddle.empty([total_num_logits], dtype=paddle.int32)

        # Inner-loop sizing constants for the fused kernel.
        # sorted_num_tokens_per_seq is descending, so [0] is the longest prefill;
        # decode lanes contribute at most 1 + num_spec_tokens tokens.
        max_query_len = max(
            int(sorted_num_tokens_per_seq[0]) if num_seqs > 0 else 1,
            1 + self.num_spec_tokens,
        )
        max_num_logits = 1 + self.num_spec_tokens

        prepare_input_ids(
            self.input_buffers.input_ids,
            self.input_buffers.positions,
            logits_indices,
            expanded_idx_mapping,
            expanded_local_pos,
            idx_mapping,
            self.input_buffers.query_start_loc,
            cu_num_logits,
            self.request_state.num_computed_tokens.gpu,
            self.request_state.prefill_len.gpu,
            self.request_state.last_sampled_tokens,
            self.request_state.all_token_ids.gpu,
            self.request_state.draft_tokens,
            max_query_len=max_query_len,
            max_num_logits=max_num_logits,
        )

        return InputBatch(
            num_seqs=num_seqs,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            expanded_idx_mapping=expanded_idx_mapping,
            expanded_local_pos=expanded_local_pos,
            num_tokens=num_tokens,
            num_decode_tokens=num_decode_tokens,
            num_prefill_tokens=num_prefill_tokens,
            num_draft_tokens=num_draft_tokens,
            num_decodes=num_decodes,
            num_prefills=num_prefills,
            query_start_loc=self.input_buffers.query_start_loc[: num_seqs + 1],
            query_start_loc_np=query_start_loc_np,
            seq_lens=self.input_buffers.seq_lens[:num_seqs],
            seq_lens_np=seq_lens_np,
            input_ids=self.input_buffers.input_ids[:num_tokens],
            positions=self.input_buffers.positions[:num_tokens],
            logits_indices=logits_indices,
            cu_num_logits=cu_num_logits,
            cu_num_logits_np=cu_num_logits_np,
        )

    def prepare_attn(self, input_batch: InputBatch) -> tuple[paddle.Tensor, paddle.Tensor]:

        input_block_tables = self.block_table.gather_block_tables(input_batch.idx_mapping)
        slot_mappings = self.block_table.compute_slot_mappings(
            input_batch.idx_mapping,
            input_batch.query_start_loc,
            input_batch.positions,
        )

        return input_block_tables, slot_mappings

    def load_model(self) -> None:
        """load or download model"""
        logger.info(f"Starting to load model {self.model_config.architectures[0]}")
        # 1. Load original model
        model_loader = get_model_loader(load_config=self.fd_config.load_config)
        self.model = model_loader.load_model(fd_config=self.fd_config)

        # 2. Load lora model

        # 3. Load drafter model(for speculative decoding)

        # 4. Init proposer for speculative method
        # self._init_speculative_proposer()

        # Load RL dynamic model
        if self.fd_config.load_config.dynamic_load_weight:
            from fastdeploy.rl.dynamic_weight_manager import DynamicWeightManager

            if self.spec_method == SpecMethod.MTP:
                self.dynamic_weight_manager = DynamicWeightManager(
                    self.fd_config, [self.model, self.proposer.model], self.local_rank
                )
            else:
                self.dynamic_weight_manager = DynamicWeightManager(self.fd_config, self.model, self.local_rank)

    def get_model(self) -> nn.Layer:
        """Get current model"""
        return self.model

    def initialize_kv_cache(self, profile: bool = False) -> None:
        """
        Initialize kv cache
        """
        # Initialize kv_num_heads attribute (similar to old model_runner)
        self.model_config.kv_num_heads = max(
            1,
            int(self.model_config.num_key_value_heads) // self.parallel_config.tensor_parallel_size,
        )

        # cache_kvs = {}
        max_block_num = self.num_gpu_blocks

        # Get kv cache dtype
        cache_type = self.model_config.dtype
        kv_cache_quant_type = None

        self.block_table = BlockTable(
            block_size=self.cache_config.block_size,
            max_num_seqs=self.scheduler_config.max_num_seqs,
            max_num_batched_tokens=self.scheduler_config.max_num_batched_tokens,
            max_model_len=self.model_config.max_model_len,
        )

        self.attn_backends = [
            FlashInferAttentionBackend(
                self.fd_config,
                kv_num_heads=self.model_config.kv_num_heads,
                num_heads=self.model_config.num_attention_heads // self.parallel_config.tensor_parallel_size,
                head_dim=self.model_config.head_dim,
            )
        ]

        # NOTE:(changwenbin) Determine whether it is Multi-Head Latent Attention,
        # To rationalize the allocation of kvcache.
        self.mla_cache = envs.FD_ATTENTION_BACKEND == "MLA_ATTN"
        self.dsa_cache = envs.FD_ATTENTION_BACKEND == "DSA_ATTN"

        if (
            self.quant_config
            and hasattr(self.quant_config, "kv_cache_quant_type")
            and self.quant_config.kv_cache_quant_type is not None
        ):
            cache_type = "uint8"
            kv_cache_quant_type = self.quant_config.kv_cache_quant_type
        # Get kv cache shape
        if self.dsa_cache:
            # Determine dsa cache quant type
            kv_cache_quant_type = "uint8"
            cache_type = "uint8"

            # NOTE(changwenbin) Get dsa cache shape.
            key_cache_shape, value_cache_shape, indexer_cache_shape = self.attn_backends[0].get_kv_cache_shape(
                max_num_blocks=max_block_num, kv_cache_quant_type=kv_cache_quant_type
            )
        else:
            key_cache_shape, value_cache_shape = self.attn_backends[0].get_kv_cache_shape(
                max_num_blocks=max_block_num, kv_cache_quant_type=kv_cache_quant_type
            )
            indexer_cache_shape = []
        if kv_cache_quant_type == "block_wise_fp8":
            kv_cache_scale_shape = [key_cache_shape[0], key_cache_shape[1], key_cache_shape[2]]
        local_rank = self.local_rank % self.parallel_config.tensor_parallel_size

        # Check if gpu runner needs to create kv cache
        # 1. During profiling, it creates its own kv cache.
        # 2. If no need to profile, create kv cache if cache managers do not exist.
        create_cache_tensor = profile or not (
            self.fd_config.cache_config.num_cpu_blocks > 0
            or self.fd_config.cache_config.kvcache_storage_backend
            or self.fd_config.scheduler_config.splitwise_role != "mixed"
        )

        cache_ready_signal = self.cache_ready_signal
        if not create_cache_tensor:
            logger.info(f"Waiting for cache managers to create kv cache.. {cache_ready_signal.value}")
            while cache_ready_signal.value[local_rank] != 1:
                time.sleep(1)
            logger.info(f"OK! Stop waiting. {cache_ready_signal.value}")

        logger.info(f"Initializing kv cache for all layers. {cache_ready_signal.value}")
        cache_kvs_list = []

        for i in range(self.model_config.num_hidden_layers):
            # init key cache
            key_cache_name = f"key_caches_{i}_rank{local_rank}.device{self.device_id}"
            key_cache_scales_name = f"key_cache_scales_{i}_rank{local_rank}.device{self.device_id}"
            if value_cache_shape:
                val_cache_name = f"value_caches_{i}_rank{local_rank}.device{self.device_id}"
                value_cache_scales_name = f"value_cache_scales_{i}_rank{local_rank}.device{self.device_id}"
            elif indexer_cache_shape:
                indexer_cache_name = f"indexer_caches_{i}_rank{local_rank}.device{self.device_id}"
            if create_cache_tensor:
                logger.info(
                    f"..creating kv cache for layer {i}: key:{key_cache_shape}, value:{value_cache_shape}, indexer:{indexer_cache_shape}"
                )
                key_cache = paddle.full(shape=key_cache_shape, fill_value=0, dtype=cache_type)
                set_data_ipc(key_cache, key_cache_name)
                self.cache_kvs_map[key_cache_name] = key_cache
                if value_cache_shape:
                    val_cache = paddle.full(shape=value_cache_shape, fill_value=0, dtype=cache_type)
                    set_data_ipc(val_cache, val_cache_name)
                    self.cache_kvs_map[val_cache_name] = val_cache
                    cache_kvs_list.extend([key_cache, val_cache])
                elif indexer_cache_shape:
                    indexer_cache = paddle.full(shape=indexer_cache_shape, fill_value=0, dtype=cache_type)
                    set_data_ipc(indexer_cache, indexer_cache_name)
                    self.cache_kvs_map[indexer_cache_name] = indexer_cache
                    cache_kvs_list.extend([key_cache, indexer_cache])
                else:
                    cache_kvs_list.extend([key_cache])
                if kv_cache_quant_type == "block_wise_fp8":
                    key_cache_scales = paddle.full(
                        shape=kv_cache_scale_shape, fill_value=0, dtype=paddle.get_default_dtype()
                    )
                    set_data_ipc(key_cache_scales, key_cache_scales_name)
                    self.cache_kvs_map[key_cache_scales_name] = key_cache_scales
                    if value_cache_shape:
                        val_cache_scales = paddle.full(
                            shape=kv_cache_scale_shape, fill_value=0, dtype=paddle.get_default_dtype()
                        )
                        set_data_ipc(val_cache_scales, value_cache_scales_name)
                        self.cache_kvs_map[value_cache_scales_name] = val_cache_scales
                        cache_kvs_list.extend([key_cache_scales, val_cache_scales])
                    else:
                        cache_kvs_list.extend([key_cache_scales])
            else:
                logger.info(
                    f"..attaching kv cache for layer {i}: key:{key_cache_shape}, value:{value_cache_shape}, indexer:{indexer_cache_shape}"
                )
                key_cache = paddle.empty(shape=[], dtype=cache_type)
                key_cache = share_external_data(key_cache, key_cache_name, key_cache_shape)
                self.cache_kvs_map[key_cache_name] = key_cache
                if kv_cache_quant_type == "block_wise_fp8":
                    key_cache_scales = paddle.empty(shape=[], dtype=paddle.get_default_dtype())
                    key_cache_scales = share_external_data(
                        key_cache_scales, key_cache_scales_name, kv_cache_scale_shape
                    )
                    self.cache_kvs_map[key_cache_scales_name] = key_cache_scales
                if value_cache_shape:
                    val_cache = paddle.empty(shape=[], dtype=cache_type)
                    val_cache = share_external_data(val_cache, val_cache_name, value_cache_shape)
                    self.cache_kvs_map[val_cache_name] = val_cache
                    cache_kvs_list.extend([key_cache, val_cache])
                    if kv_cache_quant_type == "block_wise_fp8":
                        val_cache_scales = paddle.empty(shape=[], dtype=paddle.get_default_dtype())
                        val_cache_scales = share_external_data(
                            val_cache_scales, value_cache_scales_name, kv_cache_scale_shape
                        )
                        self.cache_kvs_map[value_cache_scales_name] = val_cache_scales
                        cache_kvs_list.extend([key_cache_scales, val_cache_scales])
                elif indexer_cache_shape:
                    indexer_cache = paddle.empty(shape=[], dtype=cache_type)
                    indexer_cache = share_external_data(indexer_cache, indexer_cache_name, indexer_cache_shape)
                    self.cache_kvs_map[indexer_cache_name] = indexer_cache
                    cache_kvs_list.extend([key_cache, indexer_cache])
                else:
                    cache_kvs_list.extend([key_cache])
                    if kv_cache_quant_type == "block_wise_fp8":
                        cache_kvs_list.extend([key_cache_scales])

        self.caches = cache_kvs_list

        if not profile and create_cache_tensor:
            cache_ready_signal.value[local_rank] = 1
            logger.info(f"✅ kv cache is ready! {cache_ready_signal.value}")

        paddle.device.cuda.empty_cache()
        logger.info("kv cache is initialized!")

    @sot_warmup_guard(True)
    def capture_model(self) -> None:
        """
        Trigger CUDA Graph capture for all shapes in cuda graph capture list
        """
        if not self.use_cudagraph:
            logger.info("Skipping CUDA graph capture. Please check GraphOptimizationConfig")
            return
        time_before_capture = time.perf_counter()
        expected_decode_len = 1
        capture_sizes = self.cudagraph_capture_sizes.copy()
        try:
            if self.fd_config.graph_opt_config.cudagraph_only_prefill:
                for num_tokens in sorted(capture_sizes, reverse=True):
                    self._dummy_run(
                        num_tokens=num_tokens,
                        batch_size=self.scheduler_config.max_num_seqs,
                        in_capturing=True,
                        expected_decode_len=expected_decode_len,
                        capture_prefill=True,
                    )
                    logger.info(
                        f"Warm up the model with the num_tokens:{num_tokens}, expected_decode_len:{expected_decode_len}"
                    )
            elif self.speculative_decoding and self.spec_method in [SpecMethod.MTP, SpecMethod.SUFFIX]:
                for capture_size in sorted(capture_sizes, reverse=True):
                    expected_decode_len = (self.speculative_config.num_speculative_tokens + 1) * 2
                    self._dummy_run(
                        num_tokens=self.fd_config.get_max_chunk_tokens(),
                        batch_size=int(capture_size / (self.speculative_config.num_speculative_tokens + 1)),
                        in_capturing=True,
                        expected_decode_len=expected_decode_len,
                        accept_all_drafts=True,
                    )
                    logger.info(
                        f"Warm up the model with the num_tokens:{capture_size}, expected_decode_len:{expected_decode_len}"
                    )
            else:
                for batch_size in sorted(capture_sizes, reverse=True):
                    self._dummy_run(
                        num_tokens=self.fd_config.get_max_chunk_tokens(),
                        batch_size=batch_size,
                        in_capturing=True,
                        expected_decode_len=expected_decode_len,
                    )
                    logger.info(
                        f"Warm up the model with the batch size:{batch_size}, num tokens:{expected_decode_len}"
                    )
        except RuntimeError as e:
            if "out of memory" in str(e):
                raise RuntimeError(
                    "CUDA out of memory occurred when warming up CUDAGraph "
                    f"with the capture sizes {capture_sizes}. Please try "
                    "lowering `max_num_seqs` or `gpu_memory_utilization` when "
                    "initializing the engine."
                ) from e
            if "CUDA error(700)" in str(e):
                raise RuntimeError(
                    "CUDA error(700), an illegal memory access was encountered, "
                    "when warming up CUDAGraph. Please try to set the startup parameter: "
                    "--graph-optimization-config '{\"use_cudagraph\": false}' to close CUDAGraph"
                ) from e
            else:
                raise e

        time_after_capture = time.perf_counter()
        logger.info(f"Cuda Graph capturing took {time_after_capture - time_before_capture} seconds")

    def _execute_empty_mtp_input(self, forward_meta) -> None:
        """
        run ep inference forward with empty input.
        """
        for _ in range(self.fd_config.speculative_config.num_model_steps):
            self.proposer.model.empty_input_forward(forward_meta)

    def initialize_forward_meta(
        self,
        input_batch: InputBatch,
        block_table_tensor: paddle.Tensor,
        slot_mappings: paddle.Tensor,
        is_dummy_or_profile_run: bool = False,
    ) -> None:
        # ── 1. Determine forward mode ─────────────────────────────────────
        exist_prefill = self.request_state.exist_prefill()

        if exist_prefill:
            forward_mode = ForwardMode.MIXED
        else:
            forward_mode = ForwardMode.DECODE

        # ── 2. Decide CUDA Graph usage ────────────────────────────────────
        # CUDA Graph is only valid for pure-decode steps with a batch size
        # that was previously captured.  Prefill invalidates graph replay.
        only_decode = not exist_prefill
        step_use_cudagraph = self.use_cudagraph and only_decode and not is_dummy_or_profile_run

        # ── 3. Construct ForwardMetaV1 ────────────────────────────────────
        self.forward_meta = ForwardMetaV1(
            input_batch=input_batch,
            # Trim slot_mapping to the actual number of query tokens so
            # FlashInfer writes KV into exactly the right cache slots.
            slot_mapping=slot_mappings[: input_batch.num_tokens],
            block_table_tensor=block_table_tensor,
            caches=self.caches,
            step_use_cudagraph=step_use_cudagraph,
            attn_backend=self.attn_backends[0],
            forward_mode=forward_mode,
            is_dummy_or_profile_run=is_dummy_or_profile_run,
            is_zero_size=(input_batch.num_tokens == 0),
            positions=input_batch.positions,
            cos_sin_cache=self.input_buffers.cos_sin_buffer,
        )

        # ── 4. Attention Backend Init ────────────────────────────────────────
        if not self.forward_meta.is_zero_size:
            self.attn_backends[0].init_attention_metadata(self.forward_meta)

    def execute_model(
        self,
        model_forward_batch: Optional[List[Request]] = None,
        num_running_requests: int = None,
    ) -> None:
        """
        One inference step: prepare inputs → build attn metadata → forward → sample → update state.
        """
        if np.sum(self.request_state.num_tokens_per_seq > 0) == 0:
            return

        # ── 1. Build input batch ─────────────────────────────────────────
        input_batch = self.prepare_inputs()

        # ── 2. Prepare paged-KV attention tensors ────────────────────────
        # gather_block_tables reorders the per-request block table into
        # batch order; compute_slot_mappings maps each token position to
        # its physical cache slot.
        input_block_tables, slot_mappings = self.prepare_attn(input_batch)

        # ── 3. Build ForwardMetaV1 + run FlashInfer planning ─────────────
        # After this call self.forward_meta is fully populated and the
        # FlashInfer prefill/decode wrappers are planned.
        self.initialize_forward_meta(input_batch, input_block_tables, slot_mappings)

        # ── 4. Model forward ─────────────────────────────────────────────
        model_inputs = {
            "ids_remove_padding": input_batch.input_ids[: input_batch.num_tokens],
            "generated_modality": paddle.zeros([input_batch.num_seqs], dtype=paddle.int32),
        }
        hidden_states = self.model(model_inputs, self.forward_meta)

        # Trim CUDA Graph padding from hidden states.
        if self.use_cudagraph and self.forward_meta.step_use_cudagraph:
            hidden_states = hidden_states[: input_batch.num_tokens]

        # ── 5. upload_last_batch ─────────────────────────────────────────
        self.upload_last_batch(input_batch)

        # ── 6. Compute logits (lm_head projection) ───────────────────────
        # Only the positions in logits_indices need a logit vector; these
        # correspond to the last token of each sequence (or draft tokens for
        # speculative decoding).
        hidden_states = hidden_states[input_batch.logits_indices]
        logits = self.model.compute_logits(hidden_states, self.forward_meta)

        # ── 7. Sample ────────────────────────────────────────────────────
        sampled_token_ids, sampler_inputs = self.sample(logits, input_batch)

        # ── 8. Check stop conditions ─────────────────────────────────────
        num_sampled = paddle.ones([input_batch.num_seqs], dtype=paddle.int32)
        num_rejected = paddle.zeros([input_batch.num_seqs], dtype=paddle.int32)

        # Reshape view (no copy) so the check_stop kernel can write the
        # forced-EOS token back into sampled_token_ids in place.
        sampled_flat = sampled_token_ids.reshape([-1])

        stop_flags = self.sampler.check_stop(
            sampled_tokens=sampled_flat,
            num_sampled=num_sampled,
            inputs=sampler_inputs,
        )

        # ── 9. Post-update: advance position counters and store samples ───
        post_update(
            idx_mapping=input_batch.idx_mapping,
            num_computed_tokens=self.request_state.num_computed_tokens.gpu,
            last_sampled_tokens=self.request_state.last_sampled_tokens,
            output_bin_counts=None,
            sampled_tokens=sampled_token_ids,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
            query_start_loc=input_batch.query_start_loc,
            all_token_ids=self.request_state.all_token_ids.gpu,
            total_len=self.request_state.total_len.gpu,
        )

        # Advance CPU-side per-seq counters for the next step.
        running = np.flatnonzero(self.request_state.num_tokens_per_seq)
        computed = self.request_state.num_computed_tokens_np[running]
        chunk = self.request_state.num_tokens_per_seq[running]
        prefill_len = self.request_state.prefill_len.np[running]
        last_prefill = running[computed + chunk == prefill_len]
        in_decode = running[computed >= prefill_len]
        # After the final prefill chunk, the seq switches to decode (1 token / step).
        self.request_state.num_tokens_per_seq[last_prefill] = 1
        self.request_state.num_computed_tokens_np[last_prefill] += 1
        self.request_state.num_computed_tokens_np[in_decode] += 1 + self.num_spec_tokens

        self.async_output = AsyncOutput(
            num_computed_tokens=self.request_state.num_computed_tokens.gpu,
            sampled_token_ids=sampled_token_ids,
            stop_flags=stop_flags,
        )

    def sample(
        self,
        logits: paddle.Tensor,
        input_batch: InputBatch,
        output_bin_counts: paddle.Tensor = None,
    ):
        """
        Sample next tokens using top-k/top-p sampling.
        """
        num_output_tokens_gpu = self.request_state.total_len.gpu - self.request_state.prompt_len.gpu

        sampler_inputs = SamplerInputs(
            idx_mapping=input_batch.idx_mapping,
            output_bin_counts=None,  # TODO: maintain token frequency histogram
            step_idx=num_output_tokens_gpu,
            min_dec_len=self.sampling_state.min_lens.gpu,
            max_dec_len=self.sampling_state.max_lens.gpu,
            eos_token_ids=self.sampling_state.eos_token_ids,
        )

        # Apply penalties and temperature
        logits = self.sampler.preprocess_logits(logits, sampler_inputs)

        # Gather per-request sampling params via idx_mapping
        top_p = paddle.index_select(self.sampling_state.top_p.gpu, input_batch.idx_mapping)
        top_k = paddle.index_select(self.sampling_state.top_k.gpu, input_batch.idx_mapping).cast(paddle.int64)
        seeds = paddle.index_select(self.sampling_state.seeds.gpu, input_batch.idx_mapping)

        # softmax to get probabilities
        probs = F.softmax(logits.cast(paddle.float32))

        # top-k / top-p sampling (reuses the old runner's proven implementation)
        _, sampled_ids = top_k_top_p_sampling(
            probs,
            top_p=top_p,
            top_k=top_k,
            topp_seed=seeds,
        )

        # sampled_token_ids expected shape: [num_seqs, 1] for non-speculative
        sampled_token_ids = sampled_ids.cast(paddle.int32)
        return sampled_token_ids, sampler_inputs

    def upload_last_batch(self, input_batch):
        if self.async_output is None:
            return
        self.async_output.get_output()

        stream_datas = []
        stop_flags_np = self.async_output.stop_flags_cpu.numpy()
        sampled_token_ids_np = self.async_output.sampled_token_ids_cpu.numpy()

        for batch_idx in range(input_batch.num_seqs):
            req_state_idx = int(input_batch.idx_mapping_np[batch_idx])
            tokens = sampled_token_ids_np[batch_idx]  # shape [1] for non-speculative

            if stop_flags_np[batch_idx]:
                self.request_state.num_tokens_per_seq[req_state_idx] = 0

            stream_datas.append(
                StreamTransferData(
                    decoder_state=DecoderState.TEXT,
                    batch_id=req_state_idx,
                    tokens=tokens,
                )
            )
        self.async_uploader.enqueue(stream_datas)
        self.async_output = None

    def cal_theortical_kvcache(self):
        """
        Calculate the total block memory required at the model level
        TODO(gongshaotian): Move to Attention Backend
        """
        """
        Byte of dtype:
        - default(bf16): 2
        - cache_int8: 1
        - cache_int4:
        """
        cache_quant_dtype = None
        if (
            self.quant_config
            and hasattr(self.quant_config, "kv_cache_quant_type")
            and self.quant_config.kv_cache_quant_type is not None
        ):
            cache_quant_dtype = self.quant_config.kv_cache_quant_type

        if cache_quant_dtype is not None:  # int8, int8_zp, fp8, fp8_zp
            byte_of_dtype = 1
        else:  # default
            byte_of_dtype = 2

        hidden_dim = self.model_config.head_dim * self.model_config.kv_num_heads
        # NOTE(liuzichang): Implement multi-layer MTP architecture in the future
        num_layers = (
            self.model_config.num_hidden_layers + self.speculative_config.num_gpu_block_expand_ratio
            if self.spec_method == SpecMethod.MTP
            else self.model_config.num_hidden_layers
        )

        # NOTE:(changwenbin) Determie whether it is Multi-Head Latent Attention,
        # To rationalize the allocation of kvcache.
        if self.fd_config.cache_config.use_mla_cache:
            required_memory = (
                byte_of_dtype
                * (self.fd_config.model_config.kv_lora_rank + self.fd_config.model_config.qk_rope_head_dim)
                * (self.cache_config.block_size)
                * num_layers
            )  # compress_kv + k_pe
        elif self.dsa_cache:
            required_memory = (
                1
                * (
                    self.fd_config.model_config.kv_lora_rank
                    + self.fd_config.model_config.kv_lora_rank // 128 * 4
                    + 2 * self.fd_config.model_config.qk_rope_head_dim
                    # indexer
                    + self.fd_config.model_config.index_head_dim
                    + self.fd_config.model_config.index_head_dim // 128 * 4
                )
                * (self.cache_config.block_size)
                * num_layers
            )
        else:
            required_memory = byte_of_dtype * 2 * (self.cache_config.block_size * hidden_dim) * num_layers  # k + v
        return required_memory

    def clear_parameters(self, pid):
        """Dynamic model loader use to clear parameters use for RL"""
        # Clear CUDAGraph
        if self.use_cudagraph:
            self.model.clear_grpah_opt_backend()
        # Clear parameters and Send single
        self.dynamic_weight_manager.clear_parameters(
            pid, self.fd_config.parallel_config.shutdown_comm_group_if_worker_idle
        )
        if self.spec_method == SpecMethod.MTP:
            self.proposer.model.clear_grpah_opt_backend()
            self.proposer.clear_mtp_cache()
        # self.clear_cache()
        paddle.device.cuda.empty_cache()

        self.dynamic_weight_manager._log_memory("dynamic weight manager clear all memory")

    def clear_requests(self):
        """Dynamic model loader use to clear requests use for RL"""
        # self.share_inputs["stop_flags"][:] = True
        # prompt_logprobs
        self.prompt_logprobs_reqs.clear()
        self.in_progress_prompt_logprobs.clear()
        self.forward_batch_reqs_list = [None for _ in range(self.scheduler_config.max_num_seqs)]

        # Routing Replay
        if self.routing_replay_manager:
            self.routing_replay_manager.clear_all_request()

    def update_parameters(self, pid):
        """Dynamic model loader use to update parameters use for RL"""
        # Update parameters
        self.dynamic_weight_manager.update_parameters(
            pid, self.fd_config.parallel_config.shutdown_comm_group_if_worker_idle
        )

        if self.spec_method == SpecMethod.MTP:
            self.proposer.model_inputs.reset_model_inputs()
            self.proposer.initialize_kv_cache(main_model_num_blocks=self.num_gpu_blocks)
        self.initialize_kv_cache()
        # Recapture CUDAGraph
        if self.use_cudagraph:
            self.capture_model()
        # Send single
        self.dynamic_weight_manager.finalize_update(pid)

        self.dynamic_weight_manager._log_memory("dynamic weight manager update all memory")

    def update_weights(self, version: str = None, verify_checksum: bool = False):
        return self.dynamic_weight_manager.update_weights_by_rdma(version, verify_checksum)

    def sleep(self, tags):

        logger.info(f">>> start offloading memory, tags: {tags}")
        start_time = time.perf_counter()

        # Clear weights, deepep_buffer, cudagraph, etc.
        if "weight" in tags.split(","):
            if self.is_weight_sleeping:
                logger.info("GPU model runner's weight is already sleeping, no need to sleep again!")
                return
            if self.use_cudagraph:
                self.model.clear_grpah_opt_backend()
            if self.fd_config.parallel_config.enable_expert_parallel:
                self.dynamic_weight_manager.clear_deepep_buffer()
            self.dynamic_weight_manager.clear_model_weight()
            if self.fd_config.parallel_config.shutdown_comm_group_if_worker_idle:
                self.dynamic_weight_manager.clear_communication_group()
            self.is_weight_sleeping = True

        # Clear KV cache
        if "kv_cache" in tags.split(","):
            if self.is_kvcache_sleeping:
                logger.info("GPU model runner's kv cache is already sleeping, no need to sleep again!")
                return
            if self.spec_method == SpecMethod.MTP:
                self.proposer.clear_mtp_cache()
            self.clear_cache()
            self.is_kvcache_sleeping = True

        paddle.device.cuda.empty_cache()
        logger.info(f"<<< finish offloading memory! time cost: {time.perf_counter()-start_time:.3f}s")
        print_gpu_memory_use(f"After offloading memory [{tags}]", self.local_rank, self.device_id)

    def wakeup(self, tags):

        if tags == "weight" and self.use_cudagraph and self.is_kvcache_sleeping:
            raise RuntimeError(
                "Waking up [weight] alone is not supported when CUDA Graph is enabled, "
                "as recapturing the graph requires the KV cache to be rebuilt first. "
                "Please wake up [kv_cache] first."
            )

        logger.info(f">>> start reloading memory, tags: {tags}")
        start_time = time.perf_counter()

        # Reinitialize KV cache
        if "kv_cache" in tags.split(","):
            if not self.is_kvcache_sleeping:
                logger.info("GPU model runner's kv cache is not sleeping, no need to wakeup!")
                return
            if self.spec_method == SpecMethod.MTP:
                self.proposer.initialize_kv_cache(main_model_num_blocks=self.num_gpu_blocks)
            self.initialize_kv_cache()
            self.is_kvcache_sleeping = False

        # Reload weights, deepep_buffer, cudagraph, etc.
        if "weight" in tags.split(","):
            if not self.is_weight_sleeping:
                logger.info("GPU model runner's weight is not sleeping, no need to wakeup!")
                return
            if self.fd_config.parallel_config.shutdown_comm_group_if_worker_idle:
                self.dynamic_weight_manager.restart_communication_group()
            if self.fd_config.parallel_config.enable_expert_parallel:
                self.dynamic_weight_manager.recreate_deepep_buffer()
            self.dynamic_weight_manager.reload_model_weights()
            if self.use_cudagraph:
                self.capture_model()
            self.is_weight_sleeping = False

        logger.info(f"<<< finish reloading memory! time cost: {time.perf_counter()-start_time:.3f}s")
        print_gpu_memory_use(f"After reloading memory [{tags}]", self.local_rank, self.device_id)

    def profile_run(self) -> None:
        """Execute a forward pass with dummy inputs to profile the memory usage of the model"""
        # Initialize kv cache for profile run. After profile run kv cache will be reset.
        self.num_gpu_blocks = self.cache_config.total_block_num
        self.initialize_kv_cache(profile=True)

        # TODO: Add proper dummy run for profiling
        # For now, just log and empty cache
        logger.info(f"Profile run with max_model_len={self.model_config.max_model_len}")

        # Clear cache after profile run
        self.clear_cache(profile=True)

    def update_share_input_block_num(self, num_gpu_blocks: int) -> None:
        """
        Set a globally unified block number and update the model's shared input.
        Args:
            num_gpu_blocks: Number of GPU blocks for KV cache
        """
        self.num_gpu_blocks = num_gpu_blocks

        # Reset block table and kv cache with global block num
        self.initialize_kv_cache()

        if self.spec_method == SpecMethod.MTP:
            self.proposer.update_mtp_block_num(num_gpu_blocks)

    def clear_cache(self, profile: bool = False) -> None:
        """Clear cached data from kv cache"""
        create_cache_tensor = profile or not (
            self.fd_config.cache_config.num_cpu_blocks > 0
            or self.fd_config.cache_config.kvcache_storage_backend
            or self.fd_config.scheduler_config.splitwise_role != "mixed"
        )
        local_rank = self.local_rank % self.parallel_config.tensor_parallel_size

        if not create_cache_tensor:
            for name, tensor in self.cache_kvs_map.items():
                if unset_data_ipc is not None:
                    unset_data_ipc(tensor, name, True, False)
            self.cache_ready_signal.value[local_rank] = 0
        self.cache_kvs_map.clear()

        if self.forward_meta is not None:
            self.forward_meta.clear_caches()
        paddle.device.cuda.empty_cache()

    def capture_model_prefill_and_mixed(self) -> None:
        """
        Trigger CUDA Graph capture for prefill/mixed phase in static split graph mode.
        """
        if not self.use_cudagraph:
            logger.info("Skipping CUDA graph capture. Please check GraphOptimizationConfig")
            return
        # TODO: Implement proper prefill/mixed capture
        logger.info("CUDA graph capture for prefill/mixed phase is not yet implemented in MRV1")

    def sot_warmup(self) -> None:
        """SOT warmup for the model"""
        # TODO: Implement proper SOT warmup
        logger.info("SOT warmup is not yet implemented in MRV1")

    def vision_encoder_compile(self):
        """Compile the vision encoder if applicable"""
        # TODO: Implement vision encoder compile
        logger.info("Vision encoder compile is not yet implemented in MRV1")
