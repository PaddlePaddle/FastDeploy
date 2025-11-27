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
import queue
import time
from concurrent.futures import Future
from threading import Thread
from typing import List, Optional, cast

import numpy as np
import paddle
from paddle import nn
from paddleformers.utils.log import logger

from fastdeploy.config import FDConfig
from fastdeploy.engine.pooling_params import PoolingParams
from fastdeploy.engine.request import Request, RequestType
from fastdeploy.model_executor.graph_optimization.utils import (
    GPUMemoryChecker,
    profile_run_guard,
    sot_warmup_guard,
)
from fastdeploy.model_executor.guided_decoding import (
    LogitsProcessorBase,
    get_guided_backend,
)
from fastdeploy.model_executor.layers.attention import get_attention_backend
from fastdeploy.model_executor.layers.attention.append_attn_backend import (
    allocate_launch_related_buffer,
)
from fastdeploy.model_executor.layers.attention.base_attention_backend import (
    AttentionBackend,
)
from fastdeploy.model_executor.layers.rotary_embedding import get_rope, get_rope_3d
from fastdeploy.model_executor.layers.sample.meta_data import SamplingMetadata
from fastdeploy.model_executor.layers.sample.sampler import Sampler, SpeculativeSampler
from fastdeploy.model_executor.model_loader import get_model_loader
from fastdeploy.platforms import current_platform

if current_platform.is_iluvatar():
    from fastdeploy.model_executor.ops.iluvatar import (
        set_data_ipc,
        set_value_by_flags_and_idx,
    )

    recover_decode_task = None
    share_external_data = None
elif current_platform.is_dcu():
    from fastdeploy.model_executor.ops.gpu import set_value_by_flags_and_idx

    recover_decode_task = None
    share_external_data = None
else:
    from fastdeploy.model_executor.ops.gpu import (
        recover_decode_task,
        set_value_by_flags_and_idx,
        share_external_data,
        speculate_schedule_cache,
        set_data_ipc,
    )

from fastdeploy.model_executor.pre_and_post_process import (
    post_process,
    pre_process,
    rebuild_padding,
    step_cuda,
)

if not (current_platform.is_dcu() or current_platform.is_iluvatar()):
    from fastdeploy.spec_decode import MTPProposer, NgramProposer

import zmq

from fastdeploy import envs
from fastdeploy.engine.tasks import PoolingTask
from fastdeploy.input.ernie4_5_vl_processor import DataProcessor
from fastdeploy.inter_communicator import IPCSignal, ZmqIpcClient
from fastdeploy.model_executor.forward_meta import ForwardMeta
from fastdeploy.model_executor.layers.pool.metadata import PoolingMetadata
from fastdeploy.model_executor.logits_processor import build_logits_processors
from fastdeploy.model_executor.models.ernie4_5_vl.modeling_resampler import ScatterOp
from fastdeploy.model_executor.models.interfaces_base import FdModelForPooling
from fastdeploy.output.pooler import PoolerOutput
from fastdeploy.worker.model_runner_base import ModelRunnerBase
from fastdeploy.worker.output import LogprobsTensors, ModelOutputData, ModelRunnerOutput


class GPUModelRunner(ModelRunnerBase):
    def __init__(
        self,
        fd_config: FDConfig,
        device: str,  # logic device
        device_id: int,  # physical device id
        rank: int,
        local_rank: int,
    ):
        super().__init__(fd_config=fd_config, device=device)
        self.enable_mm = self.model_config.enable_mm
        self.rank = rank
        self.local_rank = local_rank
        self.device_id = device_id
        self.speculative_method = self.fd_config.speculative_config.method
        self.speculative_decoding = self.speculative_method is not None
        self.enable_logprob = fd_config.model_config.enable_logprob
        self.enable_early_stop = self.fd_config.early_stop_config.enable_early_stop
        self.is_pooling_model = self.fd_config.model_config.runner_type == "pooling"
        self.ori_vocab_size = self.fd_config.model_config.ori_vocab_size
        self.max_logprobs = (
            self.ori_vocab_size if fd_config.model_config.max_logprobs == -1 else fd_config.model_config.max_logprobs
        )
        self.prompt_logprobs_reqs: dict[str, Request] = {}
        self.in_progress_prompt_logprobs: dict[str, LogprobsTensors] = {}

        # VL model config:
        if self.enable_mm:
            if "ernie" in self.fd_config.model_config.model_type:
                self._init_image_preprocess()

            self.amp_black = [
                "reduce_sum",
                "c_softmax_with_cross_entropy",
                "elementwise_div",
                "sin",
                "cos",
                "sort",
                "multinomial",
            ]
            self.amp_white = [
                "lookup_table",
                "lookup_table_v2",
                "flash_attn",
                "matmul",
                "matmul_v2",
                "fused_gemm_epilogue",
            ]

            if self.cache_config.max_encoder_cache > 0:
                self.encoder_cache: dict[str, paddle.Tensor] = {}
            else:
                self.encoder_cache = None

        #  Sampler
        if not self.speculative_decoding:
            self.sampler = Sampler(fd_config)
        else:
            self.sampler = SpeculativeSampler(fd_config)

        self.guided_backend = None
        if self.fd_config.structured_outputs_config.guided_decoding_backend != "off":
            self.guided_backend = get_guided_backend(fd_config=self.fd_config)
            self.sampler.set_reasoning_parser(self.guided_backend.get_reasoning_parser())

        # Lazy initialize kv cache after model loading
        # self.kv_caches: list[paddle.Tensor] = []

        # CUDA Graph
        self.use_cudagraph = self.graph_opt_config.use_cudagraph
        self.cudagraph_capture_sizes = list(reversed(self.graph_opt_config.cudagraph_capture_sizes))
        self.sot_warmup_sizes = self.graph_opt_config.sot_warmup_sizes
        self.cudagraph_only_prefill = self.graph_opt_config.cudagraph_only_prefill
        self.mem_checker = GPUMemoryChecker(device_id=self.device_id, print_debug_info=False)

        # Initialize share inputs
        self._init_share_inputs(self.scheduler_config.max_num_seqs)
        self.infer_seed_increment = paddle.full(
            shape=[self.scheduler_config.max_num_seqs, 1],
            fill_value=4,
            dtype="int64",
        ).cpu()

        self.restore_chunked_prefill_request = dict()

        # Initialize attention Backend
        # NOTE(gonshaotian): Currently, all attention layers share one attention backend instance.
        # In the future, we will expand it as a list.
        self.attn_backends: list[AttentionBackend] = []
        # self.attn_metadatas: list[AttentionMetadata] = []
        self._initialize_attn_backend()

        # Forward meta store the global meta information of the forward
        self.forward_meta: ForwardMeta = None

        # Postprocess Env params
        os.environ["INFERENCE_MSG_QUEUE_ID"] = str(self.parallel_config.engine_worker_queue_port)
        logger.info(f"queue id is {str(self.parallel_config.engine_worker_queue_port)}")

        self.zmq_client = None
        self.async_output_queue = None
        if envs.FD_USE_GET_SAVE_OUTPUT_V1:
            logger.info(f"zmq client get_save_output_rank{local_rank}")
            self.zmq_client = ZmqIpcClient(name=f"get_save_output_rank{local_rank}", mode=zmq.PUSH)
            self.zmq_client.connect()
            self.zmq_client.socket.SNDTIMEO = 3000
            self.async_output_queue: queue.Queue = queue.Queue()
            self.async_output_copy_thread = Thread(
                target=self._async_output_busy_loop,
                daemon=True,
                name="WorkerAsyncOutputCopy",
            )
            self.async_output_copy_thread.start()

    def _async_output_busy_loop(self):
        """Entrypoint for the thread which handles outputs asynchronously."""
        while True:
            try:
                output = self.async_output_queue.get()
                self.zmq_client.send_pyobj(output)
            except Exception as e:
                logger.exception("Exception in async output loop: %s", e)

    def exist_prefill(self):
        """
        check whether prefill stage exist
        """
        return np.any(self.share_inputs["seq_lens_encoder"].numpy() > 0)

    def exist_decode(self):
        """
        check whether decode stage exist
        """
        return np.any(self.share_inputs["seq_lens_decoder"].numpy() > 0)

    def only_prefill(self):
        """
        check whether prefill only
        """
        if_only_prefill = True
        decode_exists = None
        if self.fd_config.parallel_config.use_ep and self.fd_config.scheduler_config.splitwise_role == "mixed":
            only_prefill_batch_list = []
            decode_exists = self.exist_decode()
            paddle.distributed.all_gather_object(only_prefill_batch_list, not decode_exists)
            if_only_prefill = all(only_prefill_batch_list)

        if_only_prefill = if_only_prefill and not (decode_exists if decode_exists is not None else self.exist_decode())

        return if_only_prefill

    def only_decode(self):
        """
        check whether decode only
        """
        # Update Batch type for cuda graph for if_only_decode
        if_only_decode = True
        prefill_exists = None
        # mix ep in single node
        if self.fd_config.parallel_config.use_ep and self.fd_config.scheduler_config.splitwise_role == "mixed":
            only_decode_batch_list = []
            prefill_exists = self.exist_prefill()
            paddle.distributed.all_gather_object(only_decode_batch_list, not prefill_exists)
            if_only_decode = all(only_decode_batch_list)

        if_only_decode = if_only_decode and not (
            prefill_exists if prefill_exists is not None else self.exist_prefill()
        )

        return if_only_decode

    def _init_speculative_proposer(self):
        """
        Init speculative proposer
        """
        if self.speculative_method == "ngram":
            self.proposer = NgramProposer(self.fd_config)
        elif self.speculative_method == "mtp":
            self.share_inputs["seq_lens_this_time"] = self.seq_lens_this_time_buffer
            self.proposer = MTPProposer(
                self.fd_config,
                self.get_model(),
                self.local_rank,
                self.device_id,
                self.share_inputs,
            )
        else:
            self.proposer = None

    def _init_logits_processor(self, request) -> tuple[Future[LogitsProcessorBase],]:
        """
        init logits processor for guided decoding
        """
        assert self.guided_backend is not None, (
            "guided_backend is None, use " "--guided-decoding-backend to specify the backend at server startup."
        )

        if request.guided_json is not None:
            schemata_key = ("json", request.guided_json)
        elif request.guided_regex is not None:
            schemata_key = ("regex", request.guided_regex)
        elif request.guided_grammar is not None:
            schemata_key = ("grammar", request.guided_grammar)
        elif request.structural_tag is not None:
            schemata_key = ("structural_tag", request.structural_tag)

        return (
            self.guided_backend.get_logits_processor(
                schemata_key=schemata_key,
                enable_thinking=False,  # TODO cfg
            ),
            schemata_key,
        )

    def get_chunked_inputs(self, req: Request):
        """
        Get inputs in current chunk
        """
        prefill_start_index = req.prefill_start_index
        prefill_end_index = req.prefill_end_index
        inputs = req.multimodal_inputs
        input_ids = inputs["input_ids"][prefill_start_index:prefill_end_index]
        token_type_ids = inputs["token_type_ids"][prefill_start_index:prefill_end_index]
        image_type_ids = inputs["image_type_ids"][req.image_type_ids_start : req.image_type_ids_end]
        images = inputs["images"][req.image_start : req.image_end]
        grid_thw = inputs["grid_thw"][req.num_image_start : req.num_image_end]
        mm_hashes = inputs["mm_hashes"][req.num_image_start : req.num_image_end]

        return (
            input_ids,
            token_type_ids,
            image_type_ids,
            images,
            grid_thw,
            mm_hashes,
        )

    def batch_uncached_inputs(self, req: Request):
        """
        Batch uncached multimodal inputs
        """
        (input_ids, token_type_ids, image_type_ids, images, grid_thw, mm_hashes) = self.get_chunked_inputs(req)

        image_type_ids_size = grid_thw[:, 0]
        image_type_ids_split = np.cumsum(image_type_ids_size)[:-1]
        image_type_ids_lst = np.array_split(image_type_ids, image_type_ids_split, axis=0)

        images_size = np.prod(grid_thw, axis=1)
        images_split = np.cumsum(images_size)[:-1]
        images_lst = np.array_split(images, images_split, axis=0)

        assert len(image_type_ids_lst) == len(
            mm_hashes
        ), f"image_type_ids_lst length {len(image_type_ids_lst)} != mm_hashes length {len(mm_hashes)}"
        assert len(images_lst) == len(
            mm_hashes
        ), f"images_lst length {len(images_lst)} != mm_hashes length {len(mm_hashes)}"

        uncached_image_type_ids = []
        uncached_images = []
        uncached_grid_thw = []
        uncached_mm_hashes = []
        for i, mm_hash in enumerate(mm_hashes):
            if mm_hash in self.encoder_cache:
                continue
            uncached_image_type_ids.append(image_type_ids_lst[i])
            uncached_images.append(images_lst[i])
            uncached_grid_thw.append(grid_thw[i])
            uncached_mm_hashes.append(mm_hash)

        uncached_input_ids = paddle.to_tensor(input_ids, dtype=paddle.int64)
        uncached_token_type_ids = paddle.to_tensor(token_type_ids, dtype=paddle.int64)
        if len(uncached_mm_hashes) > 0:
            uncached_image_type_ids = paddle.to_tensor(np.hstack(uncached_image_type_ids), dtype=paddle.int64)
            uncached_images = paddle.to_tensor(
                np.vstack(uncached_images), dtype="uint8" if "ernie" in self.model_config.model_type else "bfloat16"
            )
            uncached_grid_thw = paddle.to_tensor(uncached_grid_thw, dtype=paddle.int64)

        return (
            uncached_input_ids,
            uncached_token_type_ids,
            uncached_image_type_ids,
            uncached_images,
            uncached_grid_thw,
            uncached_mm_hashes,
        )

    def scatter_and_cache_features(self, image_features, inputs):
        """
        Split batched image features and cache them
        """
        merge_size = 2
        grid_thw = inputs["grid_thw"]
        mm_hashes = inputs["mm_hashes"]
        image_features_size = (paddle.prod(grid_thw[:, 1:], axis=1) // (merge_size**2)).tolist()
        image_features_lst = paddle.split(image_features, image_features_size, axis=0)

        assert len(image_features_lst) == len(
            mm_hashes
        ), f"image_features_lst length {len(image_features_lst)} != mm_hashes length {len(mm_hashes)}"
        for i, mm_hash in enumerate(mm_hashes):
            self.encoder_cache[mm_hash] = image_features_lst[i].cpu()

    def _apply_mm_inputs(self, request: Request, multi_vision_inputs: dict, rope_3d_position_ids: dict):
        """
        Apply multimodal inputs to share_inputs
            - add image_features, extract and cache vision features from model
            - add rope_emb, rotate position embeddings
        """
        if self.encoder_cache:
            evict_mm_hashes = request.get("evict_mm_hashes", None)
            if evict_mm_hashes:
                for mm_hash in evict_mm_hashes:
                    self.encoder_cache.pop(mm_hash, None)

        inputs = request.multimodal_inputs
        if request.with_image:
            if envs.FD_ENABLE_MAX_PREFILL:
                multi_vision_inputs["images_lst"].append(
                    inputs["images"][request.image_start : request.image_end].cuda()
                )
                multi_vision_inputs["grid_thw_lst"].extend(
                    inputs["grid_thw"][request.num_image_start : request.num_image_end]
                )
                multi_vision_inputs["cu_seqlens"].extend(
                    inputs["vit_seqlen"][request.num_image_start : request.num_image_end]
                )
                multi_vision_inputs["vit_position_ids_lst"].extend(
                    inputs["vit_position_ids"][request.num_image_start : request.num_image_end]
                )
            else:
                vision_inputs = inputs
                if self.encoder_cache:
                    (
                        vision_inputs["input_ids"],
                        vision_inputs["token_type_ids"],
                        vision_inputs["image_type_ids"],
                        vision_inputs["images"],
                        vision_inputs["grid_thw"],
                        vision_inputs["mm_hashes"],
                    ) = self.batch_uncached_inputs(request)
                    if len(vision_inputs["mm_hashes"]) > 0:
                        # uncached multimodal inputs exist
                        image_features = self.extract_vision_features(vision_inputs)
                        self.scatter_and_cache_features(image_features, vision_inputs)

                    full_image_features_lst = []
                    for mm_hash in inputs["mm_hashes"][request.num_image_start : request.num_image_end]:
                        feature = self.encoder_cache[mm_hash].cuda()
                        full_image_features_lst.append(feature)
                    image_features = paddle.concat(full_image_features_lst, axis=0)
                else:
                    (
                        input_ids,
                        token_type_ids,
                        image_type_ids,
                        images,
                        grid_thw,
                        mm_hashes,
                    ) = self.get_chunked_inputs(request)
                    vision_inputs["input_ids"] = paddle.to_tensor(input_ids, dtype=paddle.int64)
                    vision_inputs["token_type_ids"] = paddle.to_tensor(token_type_ids, dtype=paddle.int64)
                    vision_inputs["image_type_ids"] = paddle.to_tensor(image_type_ids, dtype=paddle.int64)
                    vision_inputs["images"] = paddle.to_tensor(
                        images, dtype="uint8" if "ernie" in self.model_config.model_type else "bfloat16"
                    )
                    vision_inputs["grid_thw"] = paddle.to_tensor(grid_thw, dtype=paddle.int64)
                    vision_inputs["mm_hashes"] = mm_hashes

                    image_features = self.extract_vision_features(vision_inputs)

                # part of the first image may be already cached
                if "ernie" in self.model_config.model_type:
                    actual_image_token_num = paddle.sum(vision_inputs["input_ids"] == self.model_config.im_patch_id)
                elif "qwen" in self.model_config.model_type:
                    actual_image_token_num = paddle.sum(
                        vision_inputs["input_ids"] == vision_inputs["image_patch_id"]
                    ) + paddle.sum(vision_inputs["input_ids"] == vision_inputs["video_patch_id"])
                else:
                    raise ValueError(f"multiple modalities model {self.model_config.model_type} is not supported")
                self.share_inputs["image_features"] = image_features[-actual_image_token_num:]

        position_ids = request.multimodal_inputs["position_ids"]
        rope_3d_position_ids["position_ids_idx"].append(request.idx)
        rope_3d_position_ids["position_ids_lst"].append(position_ids)
        rope_3d_position_ids["position_ids_offset"].append(
            position_ids.shape[0] + rope_3d_position_ids["position_ids_offset"][-1]
        )
        rope_3d_position_ids["max_tokens_lst"].append(request.get("max_tokens", 2048))

    def insert_tasks_v1(self, req_dicts: List[Request], num_running_requests: int = None):
        """
        Process scheduler output tasks, used when ENABLE_V1_KVCACHE_SCHEDULER=1
        req_dict: A list of Request dict
        num_running_requests: batch_size
        """
        # NOTE(luotingdan): Lazy initialize kv cache
        if "caches" not in self.share_inputs:
            self.initialize_kv_cache()

        req_len = len(req_dicts)
        has_prefill_task = False
        has_decode_task = False

        batch_pooling_params = []
        self.share_inputs["image_features"] = None
        multi_vision_inputs = {"images_lst": [], "grid_thw_lst": [], "vit_position_ids_lst": [], "cu_seqlens": [0]}
        rope_3d_position_ids = {
            "position_ids_idx": [],
            "position_ids_lst": [],
            "position_ids_offset": [0],
            "max_tokens_lst": [],
        }

        for i in range(req_len):
            request = req_dicts[i]
            idx = request.idx

            if hasattr(request, "pooling_params") and request.pooling_params is not None:
                batch_pooling_params.append(request.pooling_params)

            logits_info = None
            prefill_tokens = []
            if request.task_type.value == RequestType.PREFILL.value:  # prefill task
                # guided decoding
                if (
                    request.guided_json is not None
                    or request.guided_regex is not None
                    or request.structural_tag is not None
                    or request.guided_grammar is not None
                ):
                    logits_info, schemata_key = self._init_logits_processor(request)
                    request.schemata_key = schemata_key

                    if self.scheduler_config.splitwise_role == "decode":
                        if (
                            hasattr(request, "prefill_end_index")
                            and hasattr(request, "prompt_token_ids")
                            and request.prefill_end_index > len(request.prompt_token_ids)
                        ):
                            if hasattr(request, "output_token_ids"):
                                prefill_tokens.extend(request.output_token_ids)

                prefill_start_index = request.prefill_start_index
                prefill_end_index = request.prefill_end_index
                length = prefill_end_index - prefill_start_index
                if self.enable_mm:
                    self._apply_mm_inputs(request, multi_vision_inputs, rope_3d_position_ids)

                if not self.is_pooling_model:
                    if request.get("enable_thinking", False) and request.get("reasoning_max_tokens", None) is not None:
                        # Enable thinking
                        self.share_inputs["max_think_lens"][idx : idx + 1, :] = request.get("reasoning_max_tokens")
                        self.share_inputs["limit_think_status"][idx : idx + 1, :] = 0
                    else:
                        # Disable thinking
                        self.share_inputs["max_think_lens"][idx : idx + 1, :] = -1
                        self.share_inputs["limit_think_status"][idx : idx + 1, :] = 0

                if isinstance(request.prompt_token_ids, np.ndarray):
                    prompt_token_ids = request.prompt_token_ids.tolist()
                else:
                    prompt_token_ids = request.prompt_token_ids
                input_ids = prompt_token_ids + request.output_token_ids
                prompt_len = len(prompt_token_ids)
                self.share_inputs["prompt_ids"][idx : idx + 1, :prompt_len] = np.array(prompt_token_ids, dtype="int64")
                logger.debug(
                    f"Handle prefill request {request} at idx {idx}, "
                    f"{prefill_start_index=}, {prefill_end_index=}, "
                    f"need_prefilled_token_num={len(input_ids)}"
                    f"prompt_len={prompt_len}"
                )
                self.share_inputs["input_ids"][idx : idx + 1, :length] = np.array(
                    input_ids[prefill_start_index:prefill_end_index]
                )
                encoder_block_num = len(request.block_tables)
                self.share_inputs["encoder_block_lens"][idx : idx + 1] = encoder_block_num
                self.share_inputs["block_tables"][idx : idx + 1, :] = -1
                self.share_inputs["block_tables"][idx : idx + 1, :encoder_block_num] = np.array(
                    request.block_tables, dtype="int32"
                )
                self.share_inputs["stop_flags"][idx : idx + 1] = False
                self.share_inputs["seq_lens_decoder"][idx : idx + 1] = prefill_start_index
                self.seq_lens_this_time_buffer[idx : idx + 1] = length
                self.share_inputs["seq_lens_encoder"][idx : idx + 1] = length
                self.share_inputs["step_seq_lens_decoder"][idx : idx + 1] = 0
                self.share_inputs["prompt_lens"][idx : idx + 1] = len(input_ids)
                self.share_inputs["is_block_step"][idx : idx + 1] = False
                self.share_inputs["step_idx"][idx : idx + 1] = (
                    len(request.output_token_ids) if prefill_end_index >= len(input_ids) else 0
                )
                self.share_inputs["pre_ids"][idx : idx + 1] = -1
                # pooling model request.sampling_params is None
                if request.sampling_params is not None and request.sampling_params.prompt_logprobs is not None:
                    self.prompt_logprobs_reqs[request.request_id] = request
                has_prefill_task = True
                if (
                    self.fd_config.scheduler_config.splitwise_role == "decode"
                ):  # In PD, we continue to decode after P generate first token
                    self.share_inputs["seq_lens_encoder"][idx : idx + 1] = 0
            elif request.task_type.value == RequestType.DECODE.value:  # decode task
                logger.debug(f"Handle decode request {request} at idx {idx}")
                encoder_block_num = len(request.block_tables)
                self.share_inputs["encoder_block_lens"][idx : idx + 1] = encoder_block_num
                self.share_inputs["block_tables"][idx : idx + 1, :] = -1
                self.share_inputs["block_tables"][idx : idx + 1, :encoder_block_num] = np.array(
                    request.block_tables, dtype="int32"
                )
                if self.share_inputs["is_block_step"][idx]:  # has tasks to continue to decode
                    has_decode_task = True
                continue
            else:  # preempted task
                logger.info(f"Handle preempted request {request} at idx {idx}")
                self.share_inputs["block_tables"][idx : idx + 1, :] = -1
                self.share_inputs["stop_flags"][idx : idx + 1] = True
                self.seq_lens_this_time_buffer[idx : idx + 1] = 0
                self.share_inputs["seq_lens_decoder"][idx : idx + 1] = 0
                self.share_inputs["seq_lens_encoder"][idx : idx + 1] = 0
                self.share_inputs["is_block_step"][idx : idx + 1] = False
                self.prompt_logprobs_reqs.pop(request.request_id, None)
                self.in_progress_prompt_logprobs.pop(request.request_id, None)
                continue

            assert len(request.eos_token_ids) == self.model_config.eos_tokens_lens
            self.share_inputs["eos_token_id"][:] = np.array(request.eos_token_ids, dtype="int64").reshape(-1, 1)

            self.share_inputs["top_p"][idx : idx + 1] = request.get("top_p", 0.7)
            self.share_inputs["top_k"][idx : idx + 1] = request.get("top_k", 0)
            self.share_inputs["top_k_list"][idx] = request.get("top_k", 0)
            self.share_inputs["min_p"][idx : idx + 1] = request.get("min_p", 0.0)
            self.share_inputs["min_p_list"][idx] = request.get("min_p", 0.0)
            self.share_inputs["temperature"][idx : idx + 1] = request.get("temperature", 0.95)
            self.share_inputs["penalty_score"][idx : idx + 1] = request.get("repetition_penalty", 1.0)
            self.share_inputs["frequency_score"][idx : idx + 1] = request.get("frequency_penalty", 0.0)
            self.share_inputs["presence_score"][idx : idx + 1] = request.get("presence_penalty", 0.0)
            self.share_inputs["temp_scaled_logprobs"][idx : idx + 1] = request.get("temp_scaled_logprobs", False)
            self.share_inputs["top_p_normalized_logprobs"][idx : idx + 1] = request.get(
                "top_p_normalized_logprobs", False
            )

            self.share_inputs["min_dec_len"][idx : idx + 1] = request.get("min_tokens", 1)
            self.share_inputs["max_dec_len"][idx : idx + 1] = request.get(
                "max_tokens", self.model_config.max_model_len
            )

            self.share_inputs["first_token_ids"][idx : idx + 1] = self.share_inputs["input_ids"][idx : idx + 1, :1]
            self.share_inputs["ori_seq_lens_encoder"][idx : idx + 1] = length

            if request.get("seed") is not None:
                self.share_inputs["infer_seed"][idx : idx + 1] = request.get("seed")

            if request.get("bad_words_token_ids") is not None and len(request.get("bad_words_token_ids")) > 0:
                bad_words_len = len(request.get("bad_words_token_ids"))
                self.share_inputs["bad_tokens_len"][idx : idx + 1] = bad_words_len
                self.share_inputs["bad_tokens"][idx : idx + 1, :bad_words_len] = np.array(
                    request.get("bad_words_token_ids"), dtype="int64"
                )
            else:
                self.share_inputs["bad_tokens_len"][idx : idx + 1] = 1
                self.share_inputs["bad_tokens"][idx : idx + 1, :] = np.array([-1], dtype="int64")

            if request.get("stop_token_ids") is not None and request.get("stop_seqs_len") is not None:
                stop_seqs_num = len(request.get("stop_seqs_len"))
                for i in range(stop_seqs_num, self.model_config.max_stop_seqs_num):
                    request.sampling_params.stop_seqs_len.append(0)
                self.share_inputs["stop_seqs_len"][idx : idx + 1, :] = np.array(
                    request.sampling_params.stop_seqs_len, dtype="int32"
                )
                self.share_inputs["stop_seqs"][
                    idx : idx + 1, :stop_seqs_num, : len(request.get("stop_token_ids")[0])
                ] = np.array(request.get("stop_token_ids"), dtype="int64")
            else:
                self.share_inputs["stop_seqs_len"][idx : idx + 1, :] = 0

            self.pooling_params = batch_pooling_params
            # For logits processors
            self.share_inputs["logits_processors_args"][idx] = request.get("logits_processors_args") or {}

            self.sampler.apply_logits_processor(idx, logits_info, prefill_tokens)

        if len(multi_vision_inputs["images_lst"]) > 0:
            self.share_inputs["image_features"] = self.extract_vision_features(multi_vision_inputs)

        if len(rope_3d_position_ids["position_ids_idx"]) > 0:
            packed_position_ids = paddle.to_tensor(
                np.concatenate(rope_3d_position_ids["position_ids_lst"]), dtype="int64"
            )
            rope_3d_lst = self.prepare_rope3d(
                packed_position_ids,
                rope_3d_position_ids["max_tokens_lst"],
                rope_3d_position_ids["position_ids_offset"],
            )
            for i, idx in enumerate(rope_3d_position_ids["position_ids_idx"]):
                self.share_inputs["rope_emb"][idx : idx + 1, :] = rope_3d_lst[i]

        if has_prefill_task or has_decode_task:
            self.share_inputs["not_need_stop"][0] = True
        self.share_inputs["seq_lens_this_time"] = self.seq_lens_this_time_buffer[:num_running_requests]
        if self.speculative_method in ["mtp"]:
            self.proposer.insert_tasks_v1(req_dicts, num_running_requests)

    def insert_prefill_inputs(self, req_dicts: List[Request], num_running_requests: int = None):
        """
        Process inputs for prefill tasks and insert it to share_inputs buffer
        req_dict: A list of Request dict
        num_running_requests: batch_size
        TODO(gongshaotian): Refactor this func
        """

        # NOTE(luotingdan): Set environment variable of prefill node
        if req_dicts[-1].disaggregate_info is not None and req_dicts[-1].disaggregate_info["role"] == "prefill":
            os.environ["PREFILL_NODE_ONE_STEP_STOP"] = "1"

        req_len = len(req_dicts)
        for i in range(req_len):
            request = req_dicts[i]
            idx = request.idx
            length = len(request.prompt_token_ids)
            assert length > 0, "The prompt requested must not be empty."

            logits_info = None
            prefill_tokens = []
            if (
                request.guided_json is not None
                or request.guided_regex is not None
                or request.structural_tag is not None
                or request.guided_grammar is not None
            ):
                logits_info, schemata_key = self._init_logits_processor(request)
                request.schemata_key = schemata_key

            # Is Decode Node
            if req_dicts[i].disaggregate_info is not None and req_dicts[i].disaggregate_info["role"] == "decode":
                prefill_tokens.append(request.prompt_token_ids[0])
                self.share_inputs["pre_ids"][idx : idx + 1] = request.prompt_token_ids[-1]
                self.share_inputs["input_ids"][idx : idx + 1, 0] = request.prompt_token_ids[0]
                self.share_inputs["prompt_ids"][idx : idx + 1, :length] = np.array(request.prompt_token_ids)
                self.share_inputs["seq_lens_encoder"][idx : idx + 1] = 0
                self.share_inputs["seq_lens_decoder"][idx : idx + 1] = length
                self.seq_lens_this_time_buffer[idx : idx + 1] = 1
                self.share_inputs["step_seq_lens_encoder"][idx : idx + 1] = 0
                self.share_inputs["step_seq_lens_decoder"][idx : idx + 1] = length
                self.share_inputs["prompt_lens"][idx : idx + 1] = length
                self.share_inputs["step_idx"][idx : idx + 1] = 1

                if self.speculative_decoding:
                    num_prefill_send_token = self.speculative_config.num_speculative_tokens + 1
                    self.share_inputs["draft_tokens"][idx : idx + 1, 0:num_prefill_send_token] = paddle.to_tensor(
                        request.draft_token_ids[0:num_prefill_send_token],
                        dtype="int64",
                    )
                    self.seq_lens_this_time_buffer[idx : idx + 1] = num_prefill_send_token
            else:
                self.share_inputs["pre_ids"][idx : idx + 1] = -1
                self.share_inputs["step_idx"][idx : idx + 1] = 0
                self.share_inputs["input_ids"][idx : idx + 1, :length] = np.array(request.prompt_token_ids)
                self.share_inputs["prompt_ids"][idx : idx + 1, :length] = np.array(request.prompt_token_ids)

                # Use chunked prefill
                if self.cache_config.enable_chunked_prefill:
                    request.set("chunk_idx", 1)
                    logger.info(f"prefill_chunk_info: {request.prefill_chunk_info}")
                    token_chunk_size = request.prefill_chunk_info[0]
                    if self.enable_mm:
                        inputs = self._preprocess_mm_task(token_chunk_size)
                        if inputs.get("images") is not None:
                            self.share_inputs["image_features"] = self.extract_vision_features(inputs)
                        else:
                            # Compatible with the situation that lacks images and videos
                            self.share_inputs["image_features"] = None
                        if request.multimodal_inputs["position_ids"] is not None:
                            position_ids = paddle.to_tensor(
                                request.multimodal_inputs["position_ids"],
                                dtype="int64",
                            )
                        else:
                            position_ids = None
                        token_chunk_size = inputs["input_ids"].shape[1]
                        request.set("start_idx", token_chunk_size)
                        self.share_inputs["input_ids"][idx : idx + 1, :token_chunk_size] = inputs["input_ids"]
                    else:
                        self.share_inputs["input_ids"][idx, :token_chunk_size] = np.array(
                            request.prompt_token_ids[:token_chunk_size]
                        )
                        self.share_inputs["seq_lens_decoder"][idx : idx + 1] = request.get("seq_lens_decoder", 0)
                        self.share_inputs["step_seq_lens_decoder"][idx : idx + 1] = request.get("seq_lens_decoder", 0)
                    self.seq_lens_this_time_buffer[idx : idx + 1] = token_chunk_size
                    self.share_inputs["step_seq_lens_encoder"][idx : idx + 1] = token_chunk_size
                    self.share_inputs["seq_lens_encoder"][idx : idx + 1] = token_chunk_size
                    self.share_inputs["prompt_lens"][idx : idx + 1] = token_chunk_size
                else:
                    if self.enable_mm:
                        inputs = self._preprocess_mm_task(request.multimodal_inputs)
                        if inputs.get("images") is not None:
                            self.share_inputs["image_features"] = self.extract_vision_features(inputs)
                        else:
                            # Compatible with the situation that lacks images and videos
                            self.share_inputs["image_features"] = None
                        position_ids = inputs["position_ids"]
                        length = inputs["input_ids"].shape[1]
                        self.share_inputs["input_ids"][idx : idx + 1, :length] = inputs["input_ids"]
                    else:
                        self.share_inputs["seq_lens_decoder"][idx : idx + 1] = request.get("seq_lens_decoder", 0)
                        self.share_inputs["step_seq_lens_decoder"][idx : idx + 1] = request.get("seq_lens_decoder", 0)
                    self.seq_lens_this_time_buffer[idx : idx + 1] = length
                    self.share_inputs["step_seq_lens_encoder"][idx : idx + 1] = length
                    self.share_inputs["seq_lens_encoder"][idx : idx + 1] = length
                    self.share_inputs["prompt_lens"][idx : idx + 1] = length

                if self.enable_mm:
                    self.share_inputs["rope_emb"][idx : idx + 1, :] = self.prepare_rope3d(
                        position_ids, [request.get("max_tokens", 2048)], [0, position_ids.shape[0]]
                    )[0]
                    self.share_inputs["seq_lens_decoder"][idx : idx + 1] = 0

                if not self.is_pooling_model:
                    if request.get("enable_thinking", False) and request.get("reasoning_max_tokens", None) is not None:
                        # Enable thinking
                        self.share_inputs["max_think_lens"][idx : idx + 1, :] = request.get("reasoning_max_tokens")
                        self.share_inputs["limit_think_status"][idx : idx + 1, :] = 0
                    else:
                        # Disable thinking
                        self.share_inputs["max_think_lens"][idx : idx + 1, :] = -1
                        self.share_inputs["limit_think_status"][idx : idx + 1, :] = 0

            def get_attr_from_request(request, attr, default_value=None):
                res = request.get(attr, default_value)
                if res is not None:
                    return res
                else:
                    return default_value

            assert len(request.eos_token_ids) == self.model_config.eos_tokens_lens
            self.share_inputs["eos_token_id"][:] = np.array(request.eos_token_ids, dtype="int64").reshape(-1, 1)
            self.share_inputs["top_p"][idx : idx + 1] = get_attr_from_request(request, "top_p", 0.7)
            self.share_inputs["top_k"][idx : idx + 1] = request.get("top_k", 0)
            self.share_inputs["top_k_list"][idx] = request.get("top_k", 0)
            self.share_inputs["min_p"][idx : idx + 1] = request.get("min_p", 0.0)
            self.share_inputs["min_p_list"][idx] = request.get("min_p", 0.0)

            self.share_inputs["temperature"][idx : idx + 1] = get_attr_from_request(request, "temperature", 0.95)
            self.share_inputs["penalty_score"][idx : idx + 1] = get_attr_from_request(
                request, "repetition_penalty", 1.0
            )
            self.share_inputs["frequency_score"][idx : idx + 1] = get_attr_from_request(
                request, "frequency_penalty", 0.0
            )
            self.share_inputs["presence_score"][idx : idx + 1] = get_attr_from_request(
                request, "presence_penalty", 0.0
            )
            self.share_inputs["temp_scaled_logprobs"][idx : idx + 1] = get_attr_from_request(
                request, "temp_scaled_logprobs", False
            )
            self.share_inputs["top_p_normalized_logprobs"][idx : idx + 1] = get_attr_from_request(
                request, "top_p_normalized_logprobs", False
            )

            self.share_inputs["min_dec_len"][idx : idx + 1] = request.get("min_tokens", 1)
            self.share_inputs["max_dec_len"][idx : idx + 1] = request.get(
                "max_tokens", self.model_config.max_model_len
            )
            self.share_inputs["stop_flags"][idx : idx + 1] = False

            self.share_inputs["first_token_ids"][idx : idx + 1] = self.share_inputs["input_ids"][idx : idx + 1, :1]
            self.share_inputs["ori_seq_lens_encoder"][idx : idx + 1] = length

            if request.get("seed") is not None:
                self.share_inputs["infer_seed"][idx : idx + 1] = request.get("seed")
            encoder_block_num = len(request.get("block_tables"))
            self.share_inputs["encoder_block_lens"][idx : idx + 1] = encoder_block_num
            self.share_inputs["block_tables"][idx : idx + 1, :] = -1
            self.share_inputs["block_tables"][idx : idx + 1, :encoder_block_num] = np.array(
                request.block_tables, dtype="int32"
            )

            if request.get("bad_words_token_ids") is not None and len(request.get("bad_words_token_ids")) > 0:
                bad_words_len = len(request.get("bad_words_token_ids"))
                self.share_inputs["bad_tokens_len"][idx : idx + 1] = bad_words_len
                self.share_inputs["bad_tokens"][idx : idx + 1, :bad_words_len] = np.array(
                    request.get("bad_words_token_ids"), dtype="int64"
                )
            else:
                self.share_inputs["bad_tokens_len"][idx : idx + 1] = 1
                self.share_inputs["bad_tokens"][idx : idx + 1, :] = np.array([-1], dtype="int64")

            if request.get("stop_token_ids") is not None and request.get("stop_seqs_len") is not None:
                stop_seqs_num = len(request.get("stop_seqs_len"))
                for i in range(stop_seqs_num, self.model_config.max_stop_seqs_num):
                    request.sampling_params.stop_seqs_len.append(0)
                self.share_inputs["stop_seqs_len"][idx : idx + 1, :] = np.array(
                    request.sampling_params.stop_seqs_len, dtype="int32"
                )
                self.share_inputs["stop_seqs"][
                    idx : idx + 1, :stop_seqs_num, : len(request.get("stop_token_ids")[0])
                ] = np.array(request.get("stop_token_ids"), dtype="int64")
            else:
                self.share_inputs["stop_seqs_len"][idx : idx + 1, :] = 0

            self.sampler.apply_logits_processor(idx, logits_info, prefill_tokens)

        self.share_inputs["not_need_stop"][0] = True

        self.share_inputs["seq_lens_this_time"] = self.seq_lens_this_time_buffer[:num_running_requests]

        if self.speculative_method in ["mtp"]:
            self.proposer.insert_prefill_inputs(req_dicts, num_running_requests)

    def get_input_length_list(
        self, num_tokens: int, batch_size: int, expected_decode_len: int, capture_prefill: bool = False
    ):
        """
        Generates some list for _dummy_prefill_inputs, when capture pure prefill or mtp,
        the list should be carefully constructed.

        This function addresses a specific problem: in the pure prefill stage, variable
        input lengths (e.g., `prompt[160, 0]` vs. `prompt[80, 80]`) can lead to different
        CUDA Grid dimensions for kernels like `split_q_block`. This prevents CUDA Graph
        reuse.

        The `split_q_block` kernel calculates the total number of blocks, which directly
        determines the `griddim.x` launch parameter for the `multi_query_append_attention_kernel`.
        The blocks for a single sequence are determined by the formula:
        `num_blocks = ceil((sequence_length * group_size) / block_shape_q)`

        Due to the `ceil` (ceiling) function, distributing a total number of tokens across
        a batch of shorter sequences will result in a larger total block count. For example,
        with a `group_size` of 5 and `block_shape_q` of 64:
        - A single sequence of 160 tokens requires `ceil((160 * 5) / 64) = 13` blocks.
        - Two sequences of 80 tokens each require `ceil((80 * 5) / 64) * 2 = 7 * 2 = 14` blocks.

        To ensure graph replayability, this function creates a "dummy" list of sequence
        lengths that's designed to produce the theoretical maximum `encoder_num_blocks_x_cpu`
        for the given `num_tokens` and `batch_size`. This strategy ensures the captured
        CUDA Graph has the largest possible grid dimensions. At runtime, if the actual number
        of blocks is less than or equal to this maximum, the kernel can safely execute by
        using an early-exit mechanism.

        Args:
            num_tokens (int): The total number of tokens across all sequences.
            batch_size (int): The number of sequences (requests) in the batch.

        Returns:
            List[int]: A list of integers representing the sequence length for each request.
                    This list is crafted to maximize the total number of blocks.
        """
        # NOTE(gongshaotian): The maximum decoding length is equal to the expected decoded tokens plus the eos token
        max_dec_len = expected_decode_len + 1
        input_length = min(
            num_tokens // (1 if capture_prefill else batch_size),
            self.model_config.max_model_len - max_dec_len,
        )

        # NOTE(wanglongzhi): When the full length is too large, DeepEP's buffer size will not be enough to cause the result to appear nan.
        # TODO(wanglongzhi): Figure out the accurate buffer size of DeepEP.
        if self.fd_config.parallel_config.enable_expert_parallel:
            input_length = min(input_length, 32)

        block_num = (
            input_length + self.cache_config.block_size - 1
        ) // self.cache_config.block_size + self.cache_config.enc_dec_block_num

        input_length_list = [input_length] * batch_size

        if capture_prefill:
            if num_tokens < batch_size:
                input_length_list = [1] * num_tokens
            else:
                input_length_list = [1] * (batch_size - 1)
                input_length_list.append(num_tokens - batch_size + 1)

        len_of_input_length_list = len(input_length_list)
        max_dec_len_list = [max_dec_len] * len_of_input_length_list

        return input_length_list, max_dec_len_list, block_num

    def get_supported_pooling_tasks(self) -> list[PoolingTask]:
        model = self.get_model()
        if not self.is_pooling_model:
            return []

        supported_tasks = list(model.pooler.get_supported_tasks())

        if self.cache_config.enable_chunked_prefill and "encode" in supported_tasks:
            supported_tasks.remove("encode")

            logger.debug(
                "Chunked prefill is not supported with "
                "encode task which using ALL pooling. "
                "Please turn off chunked prefill by export=FD_DISABLE_CHUNKED_PREFILL=1 before using it."
            )

        # score not support
        return supported_tasks

    def _dummy_prefill_inputs(self, input_length_list: List[int], max_dec_len_list: List[int], block_num: int):
        """Set dummy prefill inputs to share_inputs"""
        batch_size = len(input_length_list)
        for i in range(batch_size):
            idx = i
            input_length = input_length_list[i]
            max_dec_len = max_dec_len_list[i]
            self.share_inputs["input_ids"][idx : idx + 1, :input_length] = np.array([5] * input_length)
            self.share_inputs["prompt_ids"][idx : idx + 1, :input_length] = np.array([5] * input_length)
            self.share_inputs["eos_token_id"][:] = np.array(
                [2] * self.model_config.eos_tokens_lens, dtype="int64"
            ).reshape(-1, 1)
            self.seq_lens_this_time_buffer[idx : idx + 1] = input_length
            self.share_inputs["step_seq_lens_encoder"][idx : idx + 1] = input_length
            self.share_inputs["seq_lens_encoder"][idx : idx + 1] = input_length
            self.share_inputs["seq_lens_decoder"][idx : idx + 1] = 0
            self.share_inputs["prompt_lens"][idx : idx + 1] = 0
            self.share_inputs["step_idx"][idx : idx + 1] = 0
            self.share_inputs["max_dec_len"][idx : idx + 1] = max_dec_len
            self.share_inputs["min_dec_len"][idx : idx + 1] = max_dec_len
            self.share_inputs["stop_flags"][idx : idx + 1] = False
            self.share_inputs["temperature"][idx : idx + 1] = 1
            self.share_inputs["first_token_ids"][idx : idx + 1] = self.share_inputs["input_ids"][idx : idx + 1, :1]
            self.share_inputs["ori_seq_lens_encoder"][idx : idx + 1] = input_length

            self.share_inputs["encoder_block_lens"][idx : idx + 1] = block_num
            self.share_inputs["block_tables"][idx : idx + 1, :block_num] = np.arange(
                idx * block_num, (idx + 1) * block_num, 1
            )
        self.share_inputs["seq_lens_this_time"] = self.seq_lens_this_time_buffer

    def _init_share_inputs(self, max_num_seqs: int):
        """
        Initialize all share buffers for model inputs.
        """
        self.MAX_INFER_SEED = 9223372036854775806
        self.share_inputs = {}

        self.share_inputs["pre_ids"] = paddle.full(
            [max_num_seqs, self.model_config.max_model_len],
            -1,
            dtype="int64",
        )
        self.share_inputs["input_ids"] = paddle.full(
            [max_num_seqs, self.model_config.max_model_len],
            self.model_config.pad_token_id,
            dtype="int64",
        )
        self.share_inputs["prompt_ids"] = paddle.full(
            [max_num_seqs, self.model_config.max_model_len],
            self.model_config.pad_token_id,
            dtype="int64",
        )
        self.share_inputs["eos_token_id"] = paddle.full([self.model_config.eos_tokens_lens, 1], 0, dtype="int64")
        self.share_inputs["top_p"] = paddle.full([max_num_seqs, 1], self.model_config.top_p, dtype="float32")
        self.share_inputs["top_k"] = paddle.full([max_num_seqs, 1], 0, dtype="int64")
        self.share_inputs["top_k_list"] = [0] * max_num_seqs
        self.share_inputs["min_p"] = paddle.full([max_num_seqs, 1], 0.0, dtype="float32")
        self.share_inputs["min_p_list"] = [0.0] * max_num_seqs
        self.share_inputs["temperature"] = paddle.full(
            [max_num_seqs, 1], self.model_config.temperature, dtype="float32"
        )
        self.share_inputs["penalty_score"] = paddle.full(
            [max_num_seqs, 1], self.model_config.penalty_score, dtype="float32"
        )
        self.share_inputs["frequency_score"] = paddle.full(
            [max_num_seqs, 1],
            self.model_config.frequency_score,
            dtype="float32",
        )
        self.share_inputs["presence_score"] = paddle.full(
            [max_num_seqs, 1], self.model_config.presence_score, dtype="float32"
        )
        self.share_inputs["temp_scaled_logprobs"] = paddle.full([max_num_seqs, 1], False, dtype="bool")
        self.share_inputs["top_p_normalized_logprobs"] = paddle.full([max_num_seqs, 1], False, dtype="bool")

        self.share_inputs["min_dec_len"] = paddle.full([max_num_seqs, 1], self.model_config.min_length, dtype="int64")
        self.share_inputs["max_dec_len"] = paddle.full(
            [max_num_seqs, 1], self.model_config.max_model_len, dtype="int64"
        )
        self.seq_lens_this_time_buffer = paddle.full([max_num_seqs, 1], 0, dtype="int32")
        if self.fd_config.parallel_config.enable_expert_parallel:
            self.share_inputs["seq_lens_this_time"] = paddle.full([max_num_seqs, 1], 0, dtype="int32")
        self.share_inputs["seq_lens_encoder"] = paddle.full([max_num_seqs, 1], 0, dtype="int32")
        self.share_inputs["seq_lens_decoder"] = paddle.full([max_num_seqs, 1], 0, dtype="int32")
        self.share_inputs["step_seq_lens_encoder"] = paddle.full([max_num_seqs, 1], 0, dtype="int32")
        self.share_inputs["step_seq_lens_decoder"] = paddle.full([max_num_seqs, 1], 0, dtype="int32")
        self.share_inputs["prompt_lens"] = paddle.full([max_num_seqs, 1], 0, dtype="int64")
        self.share_inputs["step_idx"] = paddle.full([max_num_seqs, 1], 0, dtype="int64")
        self.share_inputs["not_need_stop"] = paddle.full([1], False, dtype="bool").cpu()
        self.share_inputs["stop_flags"] = paddle.full([max_num_seqs, 1], True, dtype="bool")
        self.share_inputs["stop_nums"] = paddle.full([1], max_num_seqs, dtype="int64")

        self.share_inputs["bad_tokens"] = paddle.full([max_num_seqs, self.model_config.vocab_size], -1, dtype="int64")
        self.share_inputs["bad_tokens_len"] = paddle.full([max_num_seqs], 1, dtype="int64")
        self.share_inputs["next_tokens"] = paddle.full([max_num_seqs, 1], -1, dtype="int64")
        self.share_inputs["is_block_step"] = paddle.full([max_num_seqs], False, dtype="bool")
        self.share_inputs["encoder_block_lens"] = paddle.full([max_num_seqs], 0, dtype="int32")
        self.share_inputs["step_block_list"] = paddle.full([max_num_seqs], -1, dtype="int32")
        self.share_inputs["step_lens"] = paddle.full([1], 0, dtype="int32")
        self.share_inputs["recover_block_list"] = paddle.full([max_num_seqs], -1, dtype="int32")
        self.share_inputs["recover_lens"] = paddle.full([1], 0, dtype="int32")
        self.share_inputs["need_block_list"] = paddle.full([max_num_seqs], -1, dtype="int32")
        self.share_inputs["need_block_len"] = paddle.full([1], 0, dtype="int32")
        self.share_inputs["used_list_len"] = paddle.full([max_num_seqs], 0, dtype="int32")
        self.share_inputs["infer_seed"] = paddle.full([max_num_seqs, 1], 0, dtype="int64").cpu()
        self.share_inputs["first_token_ids"] = paddle.full([max_num_seqs, 1], -1, dtype="int64")
        self.share_inputs["ori_seq_lens_encoder"] = paddle.full([max_num_seqs, 1], 0, dtype="int32")
        self.share_inputs["system_lens"] = paddle.full([max_num_seqs, 1], 0, dtype="int32")
        self.share_inputs["system_ids"] = paddle.full([max_num_seqs, 1], -1, dtype="int32")

        self.share_inputs["ids_remove_padding"] = paddle.full(
            [max_num_seqs * self.model_config.max_model_len],
            0,
            dtype="int64",
        )
        self.share_inputs["batch_id_per_token"] = paddle.full(
            [max_num_seqs * self.model_config.max_model_len, 1], 0, dtype="int32"
        )
        self.share_inputs["cu_seqlens_q"] = paddle.full([max_num_seqs + 1, 1], 0, dtype="int32")
        self.share_inputs["cu_seqlens_k"] = paddle.full([max_num_seqs + 1, 1], 0, dtype="int32")

        # Declare AttentionBackend buffers
        self.share_inputs["decoder_batch_ids"] = None
        self.share_inputs["decoder_tile_ids_per_batch"] = None
        self.share_inputs["decoder_num_blocks_cpu"] = None  # Pinning Memory
        self.share_inputs["decoder_num_blocks_device"] = None
        self.share_inputs["decoder_chunk_size_device"] = None
        self.share_inputs["max_len_tensor_cpu"] = None  # CPU
        self.share_inputs["encoder_batch_ids"] = None
        self.share_inputs["encoder_tile_ids_per_batch"] = None
        self.share_inputs["encoder_num_blocks_x_cpu"] = None  # CPU
        self.share_inputs["kv_batch_ids"] = None
        self.share_inputs["kv_tile_ids_per_batch"] = None
        self.share_inputs["kv_num_blocks_x_cpu"] = None  # CPU

        # Initialize thinking related buffers
        self.share_inputs["max_think_lens"] = paddle.full(shape=[max_num_seqs, 1], fill_value=-1, dtype="int32")
        self.share_inputs["limit_think_status"] = paddle.full(shape=[max_num_seqs, 1], fill_value=0, dtype="int32")

        # Initialize rotary position embedding
        if not self.enable_mm:
            self.share_inputs["rope_emb"] = get_rope(
                rotary_dim=self.model_config.head_dim,
                position_ids=paddle.arange(self.model_config.max_model_len).reshape((1, -1)),
                base=self.model_config.rope_theta,
                model_config=self.model_config,
                partial_rotary_factor=self.model_config.partial_rotary_factor,
            )

        # Set block tables
        pre_max_block_num = (
            self.model_config.max_model_len + self.cache_config.block_size - 1
        ) // self.cache_config.block_size + self.cache_config.enc_dec_block_num
        self.share_inputs["block_tables"] = paddle.full([max_num_seqs, pre_max_block_num], -1, dtype="int32")

        # Initialize free list
        free_list = list(
            range(
                self.cache_config.total_block_num - 1,
                int(self.cache_config.total_block_num * self.cache_config.kv_cache_ratio) - 1,
                -1,
            )
        )
        self.free_list_len = len(free_list)
        self.share_inputs["free_list"] = paddle.to_tensor(free_list, dtype="int32")
        self.share_inputs["free_list_len"] = paddle.full([1], self.free_list_len, dtype="int32")

        # Initialize stop seqs
        self.share_inputs["stop_seqs_len"] = paddle.full(
            [max_num_seqs, self.model_config.max_stop_seqs_num], 0, dtype="int32"
        )
        self.share_inputs["stop_seqs"] = paddle.full(
            [
                max_num_seqs,
                self.model_config.max_stop_seqs_num,
                self.model_config.stop_seqs_max_len,
            ],
            -1,
            dtype="int64",
        )
        if self.speculative_decoding:
            max_draft_token_num = self.speculative_config.num_speculative_tokens
            self.share_inputs["input_ids_cpu"] = paddle.full(
                shape=[max_num_seqs, self.model_config.max_model_len],
                fill_value=1,
                dtype="int64",
            ).cpu()
            self.share_inputs["accept_tokens"] = paddle.full(
                shape=[max_num_seqs, max_draft_token_num + 1],
                fill_value=0,
                dtype="int64",
            )
            self.share_inputs["accept_num"] = paddle.full(shape=[max_num_seqs], fill_value=0, dtype="int32")
            self.share_inputs["draft_tokens"] = paddle.full(
                shape=[max_num_seqs, max_draft_token_num + 1],
                fill_value=0,
                dtype="int64",
            )

            self.share_inputs["actual_draft_token_num"] = paddle.full(
                shape=[max_num_seqs],
                fill_value=max_draft_token_num,
                dtype="int32",
            )
            self.share_inputs["output_cum_offsets"] = paddle.full(shape=[max_num_seqs, 1], fill_value=0, dtype="int32")
            self.share_inputs["output_padding_offset"] = paddle.full(
                shape=[max_num_seqs * (max_draft_token_num + 1)],
                fill_value=0,
                dtype="int32",
            )
            # For V1_KVCACHE_SCHEDULER
            self.share_inputs["step_draft_tokens"] = paddle.full(
                shape=[max_num_seqs, max_draft_token_num + 1],
                fill_value=0,
                dtype="int64",
            )
            self.share_inputs["step_seq_lens_this_time"] = paddle.full([max_num_seqs, 1], 0, dtype="int32")
            # For MTP Logprob
            self.share_inputs["draft_logits"] = paddle.full(
                [max_num_seqs * (self.speculative_config.num_speculative_tokens + 1), self.model_config.vocab_size],
                -1,
                dtype="float32",
            )
            self.share_inputs["cu_batch_token_offset"] = paddle.full(
                shape=[max_num_seqs + 1], fill_value=0, dtype="int32"
            )

        if self.enable_mm:
            head_dim = self.model_config.head_dim
            if (
                "qwen" in self.model_config.model_type or "paddleocr" in self.model_config.model_type
            ):  # neox style = True
                rope_head_dim = head_dim
            else:  # neox style = False
                rope_head_dim = head_dim // 2

            self.share_inputs["rope_emb"] = paddle.full(
                shape=[
                    max_num_seqs,
                    2,
                    1,
                    self.model_config.max_model_len,
                    1,
                    rope_head_dim,
                ],
                fill_value=0,
                dtype="float32",
            )
            self.share_inputs["image_features"] = None

        # For logits processors
        self.share_inputs["logits_processors"] = build_logits_processors(self.fd_config)
        self.share_inputs["logits_processors_args"] = [{} for _ in range(max_num_seqs)]
        logger.info(f"Enabled logits processors: {self.share_inputs['logits_processors']}")

        self.share_inputs["mask_rollback"] = paddle.full(shape=[max_num_seqs, 1], fill_value=0, dtype="int32")

    def _prepare_inputs(self, is_dummy_or_profile_run=False) -> None:
        """Prepare the model inputs"""
        if envs.ENABLE_V1_KVCACHE_SCHEDULER:
            recover_decode_task(
                self.share_inputs["stop_flags"],
                self.share_inputs["seq_lens_this_time"],
                self.share_inputs["seq_lens_encoder"],
                self.share_inputs["seq_lens_decoder"],
                self.share_inputs["step_seq_lens_decoder"],
                self.share_inputs["block_tables"],
                self.share_inputs["is_block_step"],
                self.share_inputs["draft_tokens"] if self.speculative_decoding else None,
                self.share_inputs["step_draft_tokens"] if self.speculative_decoding else None,
                self.share_inputs["step_seq_lens_this_time"] if self.speculative_decoding else None,
                self.cache_config.block_size,
                self.speculative_config.num_speculative_tokens if self.speculative_decoding else 0,
            )

        # Remove padding
        (
            ids_remove_padding,
            batch_id_per_token,
            cu_seqlens_q,
            cu_seqlens_k,
            output_cum_offsets,
            output_padding_offset,
        ) = pre_process(
            self.share_inputs["input_ids"],
            self.share_inputs["seq_lens_this_time"],
            self.speculative_decoding,
            (self.share_inputs["draft_tokens"] if self.speculative_decoding else None),
            self.share_inputs["seq_lens_encoder"],
            self.share_inputs["seq_lens_decoder"],
        )

        self.share_inputs["ids_remove_padding"].copy_(ids_remove_padding, False)
        # NOTE: (changwenbin) Initialized to max_num_seq '-1' before copying, marking illegal positions
        self.share_inputs["batch_id_per_token"][:] = -1
        self.share_inputs["batch_id_per_token"].copy_(batch_id_per_token, False)
        self.share_inputs["cu_seqlens_q"].copy_(cu_seqlens_q, False)
        self.share_inputs["cu_seqlens_k"].copy_(cu_seqlens_k, False)

        # For speculative decoding
        if self.speculative_decoding:
            self.share_inputs["output_cum_offsets"].copy_(output_cum_offsets, False)
            self.share_inputs["output_padding_offset"].copy_(output_padding_offset, False)

        # Update bad tokens len
        max_bad_tokens_len = np.max(self.share_inputs["bad_tokens_len"].numpy())

        # Initialize forward meta data
        self.initialize_forward_meta(is_dummy_or_profile_run=is_dummy_or_profile_run)

        # Get sampling metadata
        self.sampling_metadata = SamplingMetadata(
            temperature=self.share_inputs["temperature"],
            top_p=self.share_inputs["top_p"],
            top_k=self.share_inputs["top_k"],
            top_k_list=self.share_inputs["top_k_list"],
            min_p=self.share_inputs["min_p"],
            min_p_list=self.share_inputs["min_p_list"],
            seed=self.share_inputs["infer_seed"],
            step_idx=self.share_inputs["step_idx"],
            pre_token_ids=self.share_inputs["pre_ids"],
            prompt_ids=self.share_inputs["prompt_ids"],
            prompt_lens=self.share_inputs["prompt_lens"],
            frequency_penalties=self.share_inputs["frequency_score"],
            presence_penalties=self.share_inputs["presence_score"],
            repetition_penalties=self.share_inputs["penalty_score"],
            min_dec_lens=self.share_inputs["min_dec_len"],
            bad_words_token_ids=self.share_inputs["bad_tokens"][:, :max_bad_tokens_len],
            eos_token_ids=self.share_inputs["eos_token_id"],
            max_num_logprobs=self.max_logprobs if self.enable_logprob else None,
            enable_early_stop=self.enable_early_stop,
            stop_flags=self.share_inputs["stop_flags"],
            temp_scaled_logprobs=self.share_inputs["temp_scaled_logprobs"],
            top_p_normalized_logprobs=self.share_inputs["top_p_normalized_logprobs"],
            logits_processors=self.share_inputs["logits_processors"],
            share_inputs=self.share_inputs,
        )

    def load_model(self) -> None:
        """load or download model"""
        logger.info(f"Starting to load model {self.model_config.architectures[0]}")
        # 1. Load original model
        model_loader = get_model_loader(load_config=self.fd_config.load_config)
        self.model = model_loader.load_model(fd_config=self.fd_config)

        # 1.1 Load RL dynamic model
        if self.fd_config.load_config.dynamic_load_weight:
            from fastdeploy.rl.dynamic_weight_manager import DynamicWeightManager

            self.dynamic_weight_manager = DynamicWeightManager(self.fd_config, self.model)

        # 2. Load lora model

        # 3. Load drafter model(for speculative decoding)

        # 4. Init proposer for speculative method
        self._init_speculative_proposer()

    def get_model(self) -> nn.Layer:
        """Get current model"""
        return self.model

    def initialize_forward_meta(self, is_dummy_or_profile_run=False):
        """
        Initialize forward meta and attention meta data
        """
        # Initialize forward meta
        self.forward_meta = ForwardMeta(
            ids_remove_padding=self.share_inputs["ids_remove_padding"],
            rotary_embs=self.share_inputs["rope_emb"],
            attn_backend=self.attn_backends[0],
            decoder_batch_ids=self.share_inputs["decoder_batch_ids"],
            decoder_tile_ids_per_batch=self.share_inputs["decoder_tile_ids_per_batch"],
            decoder_num_blocks_cpu=self.share_inputs["decoder_num_blocks_cpu"],
            # NOTE: (changwenbin) MLA kernel only needs decoder_num_blocks_device in place of GPU tensor,
            # adapted to cudagraph.
            decoder_num_blocks_device=self.share_inputs["decoder_num_blocks_device"],
            decoder_chunk_size_device=self.share_inputs["decoder_chunk_size_device"],
            max_len_tensor_cpu=self.share_inputs["max_len_tensor_cpu"],
            seq_lens_encoder=self.share_inputs["seq_lens_encoder"],
            seq_lens_decoder=self.share_inputs["seq_lens_decoder"],
            seq_lens_this_time=self.share_inputs["seq_lens_this_time"],
            batch_id_per_token=self.share_inputs["batch_id_per_token"],
            cu_seqlens_q=self.share_inputs["cu_seqlens_q"],
            cu_seqlens_k=self.share_inputs["cu_seqlens_k"],
            block_tables=self.share_inputs["block_tables"],
            caches=self.share_inputs["caches"],
            encoder_batch_ids=self.share_inputs["encoder_batch_ids"],
            encoder_tile_ids_per_batch=self.share_inputs["encoder_tile_ids_per_batch"],
            encoder_num_blocks_x_cpu=self.share_inputs["encoder_num_blocks_x_cpu"],
            kv_batch_ids=self.share_inputs["kv_batch_ids"],
            kv_tile_ids_per_batch=self.share_inputs["kv_tile_ids_per_batch"],
            kv_num_blocks_x_cpu=self.share_inputs["kv_num_blocks_x_cpu"],
        )

        # Update Batch type for cuda graph for only_decode_batch
        if_only_decode = self.only_decode()
        only_decode_use_cudagraph = self.use_cudagraph and if_only_decode

        # Update config about moe for better performance
        # TODO(wanglongzhi):Modifying the config at runtime is not appropriate; it needs to be moved to forward_meta. It will be used in MoEMethodBase.apply()
        if self.fd_config.parallel_config.use_ep and self.fd_config.scheduler_config.splitwise_role == "mixed":
            self.fd_config.model_config.moe_phase.phase = "decode" if if_only_decode else "prefill"
            if self.speculative_decoding:
                self.proposer.fd_config.parallel_config.moe_phase.phase = "decode" if if_only_decode else "prefill"

        # Update Batch type for cuda graph for only_prefill_batch
        only_prefill_use_cudagraph = self.use_cudagraph and self.cudagraph_only_prefill and self.only_prefill()

        # When support capture both prefill-only and decode-only, this will use [only_prefill_use_cudagraph or only_decode_use_cudagraph]
        self.forward_meta.step_use_cudagraph = (
            only_prefill_use_cudagraph if self.cudagraph_only_prefill else only_decode_use_cudagraph
        )

        # Set forward_meta.is_dummy_or_profile_run to True to skip init_kv_signal_per_query for attention backends
        self.forward_meta.is_dummy_or_profile_run = is_dummy_or_profile_run

        # Initialzie attention meta data
        for attn_backend in self.attn_backends:
            attn_backend.init_attention_metadata(self.forward_meta)

    def initialize_kv_cache(self, profile: bool = False) -> None:
        """
        Initialize kv cache
        """
        # cache_kvs = {}
        max_block_num = self.num_gpu_blocks

        # Get kv cache dtype
        cache_type = self.model_config.dtype
        kv_cache_quant_type = None
        if (
            self.quant_config
            and hasattr(self.quant_config, "kv_cache_quant_type")
            and self.quant_config.kv_cache_quant_type is not None
        ):
            cache_type = "uint8"
            kv_cache_quant_type = self.quant_config.kv_cache_quant_type

        # Get kv cache shape
        key_cache_shape, value_cache_shape = self.attn_backends[0].get_kv_cache_shape(
            max_num_blocks=max_block_num, kv_cache_quant_type=kv_cache_quant_type
        )
        if kv_cache_quant_type == "block_wise_fp8":
            kv_cache_scale_shape = [key_cache_shape[0], key_cache_shape[1], key_cache_shape[2]]
        local_rank = self.local_rank % self.parallel_config.tensor_parallel_size

        cache_ready_signal_data = np.zeros(shape=[self.parallel_config.tensor_parallel_size], dtype=np.int32)
        cache_ready_signal = IPCSignal(
            name="cache_ready_signal",
            array=cache_ready_signal_data,
            dtype=np.int32,
            suffix=self.parallel_config.engine_worker_queue_port,
            create=False,
        )

        # Check if gpu runner needs to create kv cache
        # 1. During profiling, it creates its own kv cache.
        # 2. GPU runner creates kv cache tensor unless p/d disaggregation is enabled.
        create_cache_tensor = profile or self.scheduler_config.splitwise_role == "mixed"

        if not create_cache_tensor:
            logger.info(f"Waiting for cache managers to create kv cache.. {cache_ready_signal.value}")
            while cache_ready_signal.value[local_rank] != 1:
                time.sleep(1)
            logger.info(f"OK! Stop waiting. {cache_ready_signal.value}")

        logger.info(f"Initializing kv cache for all layers. {cache_ready_signal.value}")
        cache_kvs_list = []

        # NOTE:(changwenbin) Determine whether it is Multi-Head Latent Attention,
        # To rationalize the allocation of kvcache.
        from fastdeploy import envs

        self.mla_cache = envs.FD_ATTENTION_BACKEND == "MLA_ATTN"
        for i in range(self.model_config.num_hidden_layers):
            # init key cache
            key_cache_name = f"key_caches_{i}_rank{local_rank}.device{self.device_id}"
            key_cache_scales_name = f"key_cache_scales_{i}_rank{local_rank}.device{self.device}"
            if value_cache_shape:
                val_cache_name = f"value_caches_{i}_rank{local_rank}.device{self.device_id}"
                value_cache_scales_name = f"value_cache_scales_{i}_rank{local_rank}.device{self.device}"
            if create_cache_tensor:
                logger.info(f"..creating kv cache for layer {i}: key:{key_cache_shape}, value:{value_cache_shape}")
                key_cache = paddle.full(shape=key_cache_shape, fill_value=0, dtype=cache_type)
                set_data_ipc(key_cache, key_cache_name)
                if value_cache_shape:
                    val_cache = paddle.full(shape=value_cache_shape, fill_value=0, dtype=cache_type)
                    set_data_ipc(val_cache, val_cache_name)
                    cache_kvs_list.extend([key_cache, val_cache])
                else:
                    cache_kvs_list.extend([key_cache])
                if kv_cache_quant_type == "block_wise_fp8":
                    key_cache_scales = paddle.full(
                        shape=kv_cache_scale_shape, fill_value=0, dtype=paddle.get_default_dtype()
                    )
                    if value_cache_shape:
                        val_cache_scales = paddle.full(
                            shape=kv_cache_scale_shape, fill_value=0, dtype=paddle.get_default_dtype()
                        )
                        cache_kvs_list.extend([key_cache_scales, val_cache_scales])
                    else:
                        cache_kvs_list.extend([key_cache_scales])
            else:
                logger.info(f"..attaching kv cache for layer {i}: key:{key_cache_shape}, value:{value_cache_shape}")
                key_cache = paddle.empty(shape=[], dtype=cache_type)
                key_cache = share_external_data(key_cache, key_cache_name, key_cache_shape)
                if kv_cache_quant_type == "block_wise_fp8":
                    key_cache_scales = paddle.empty(shape=[], dtype=paddle.get_default_dtype())
                    key_cache_scales = share_external_data(
                        key_cache_scales, key_cache_scales_name, kv_cache_scale_shape
                    )
                if value_cache_shape:
                    val_cache = paddle.empty(shape=[], dtype=cache_type)
                    val_cache = share_external_data(val_cache, val_cache_name, value_cache_shape)
                    cache_kvs_list.extend([key_cache, val_cache])
                    if kv_cache_quant_type == "block_wise_fp8":
                        val_cache_scales = paddle.empty(shape=[], dtype=paddle.get_default_dtype())
                        val_cache_scales = share_external_data(
                            val_cache_scales, value_cache_scales_name, kv_cache_scale_shape
                        )
                        cache_kvs_list.extend([key_cache_scales, val_cache_scales])
                else:
                    cache_kvs_list.extend([key_cache])
                    if kv_cache_quant_type == "block_wise_fp8":
                        cache_kvs_list.extend([key_cache_scales])

        self.share_inputs["caches"] = cache_kvs_list

        if not profile and create_cache_tensor:
            cache_ready_signal.value[local_rank] = 1
            logger.info(f"✅ kv cache is ready! {cache_ready_signal.value}")

        paddle.device.cuda.empty_cache()

    def _initialize_attn_backend(self) -> None:
        """
        Initialize attention backends
        """
        assert len(self.attn_backends) == 0

        num_heads = self.model_config.num_attention_heads // self.parallel_config.tensor_parallel_size
        self.model_config.kv_num_heads = max(
            1,
            int(self.model_config.num_key_value_heads) // self.parallel_config.tensor_parallel_size,
        )
        head_dim = self.model_config.head_dim

        encoder_block_shape_q = 64
        decoder_block_shape_q = 16

        res_buffer = allocate_launch_related_buffer(
            max_batch_size=self.scheduler_config.max_num_seqs,
            max_model_len=self.model_config.max_model_len,
            encoder_block_shape_q=encoder_block_shape_q,
            decoder_block_shape_q=decoder_block_shape_q,
            decoder_step_token_num=self.speculative_config.num_speculative_tokens + 1,
            num_heads=num_heads,
            kv_num_heads=self.model_config.kv_num_heads,
            block_size=self.fd_config.cache_config.block_size,
        )
        self.share_inputs.update(res_buffer)

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

        self.attn_backends.append(attn_backend)

    def _dummy_pooler_run_task(
        self,
        hidden_states: paddle.Tensor,
        task: PoolingTask,
    ) -> PoolerOutput:
        num_tokens = hidden_states.shape[0]
        max_num_seqs = self.scheduler_config.max_num_seqs
        num_reqs = min(num_tokens, max_num_seqs)
        min_tokens_per_req = num_tokens // num_reqs
        num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
        num_scheduled_tokens_list[-1] += num_tokens % num_reqs
        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs

        req_num_tokens = num_tokens // num_reqs
        dummy_prompt_lens = paddle.to_tensor(num_scheduled_tokens_list, dtype="int64", place=paddle.CPUPlace())
        dummy_token_ids = paddle.zeros([num_reqs, req_num_tokens], dtype="int64", device=hidden_states.place)
        model = cast(FdModelForPooling, self.get_model())
        dummy_pooling_params = PoolingParams(task=task)
        to_update = model.pooler.get_pooling_updates(task)
        to_update.apply(dummy_pooling_params)

        dummy_metadata = PoolingMetadata(
            prompt_lens=dummy_prompt_lens,
            prompt_token_ids=dummy_token_ids,
            pooling_params=[dummy_pooling_params] * num_reqs,
        )
        dummy_metadata.build_pooling_cursor(num_scheduled_tokens_list, device=hidden_states.place)

        try:
            return model.pooler(hidden_states=hidden_states, pooling_metadata=dummy_metadata)
        except RuntimeError as e:
            if "out of memory" in str(e):
                raise RuntimeError(
                    "CUDA out of memory occurred when warming up pooler "
                    f"({task=}) with {num_reqs} dummy requests. Please try "
                    "lowering `max_num_seqs` or `gpu_memory_utilization` when "
                    "initializing the engine."
                ) from e
            else:
                raise e

    def _dummy_pooler_run(
        self,
        hidden_states: paddle.Tensor,
        model_output: paddle.Tensor,
    ) -> PoolerOutput:
        output_size = dict[PoolingTask, float]()
        for task in self.get_supported_pooling_tasks():

            output = self._dummy_pooler_run_task(hidden_states, task)
            output_size[task] = sum(o.numel() * o.element_size() if hasattr(o, "numel") else 0 for o in output)
            del output

        max_task = max(output_size.items(), key=lambda x: x[1])[0]
        pooler_output = self._dummy_pooler_run_task(hidden_states, max_task)

        model_output_data = ModelOutputData(
            next_tokens=self.share_inputs["next_tokens"],
            stop_flags=self.share_inputs["stop_flags"],
            step_idx=self.share_inputs["step_idx"],
            max_dec_len=self.share_inputs["max_dec_len"],
            pre_ids=self.share_inputs["pre_ids"],
            seq_lens_this_time=self.share_inputs["seq_lens_this_time"],
            eos_token_id=self.share_inputs["eos_token_id"],
            not_need_stop=self.share_inputs["not_need_stop"],
            input_ids=self.share_inputs["input_ids"],
            stop_nums=self.share_inputs["stop_nums"],
            seq_lens_encoder=self.share_inputs["seq_lens_encoder"],
            seq_lens_decoder=self.share_inputs["seq_lens_decoder"],
            is_block_step=self.share_inputs["is_block_step"],
            full_hidden_states=model_output,
            msg_queue_id=self.parallel_config.msg_queue_id,
            mp_rank=self.parallel_config.tensor_parallel_rank,
            use_ep=self.parallel_config.use_ep,
            draft_tokens=(self.share_inputs["draft_tokens"] if self.speculative_decoding else None),
            actual_draft_token_num=(
                self.share_inputs["actual_draft_token_num"] if self.speculative_decoding else None
            ),
            accept_tokens=(self.share_inputs["accept_tokens"] if self.speculative_decoding else None),
            accept_num=(self.share_inputs["accept_num"] if self.speculative_decoding else None),
            stop_token_ids=self.share_inputs["stop_seqs"],
            stop_seqs_len=self.share_inputs["stop_seqs_len"],
            prompt_lens=self.share_inputs["prompt_lens"],
        )

        post_process(
            sampler_or_pooler_output=pooler_output,
            model_output=model_output_data,
            share_inputs=self.share_inputs,
            block_size=self.cache_config.block_size,
            speculative_decoding=self.speculative_decoding,
            skip_save_output=True,
            async_output_queue=self.async_output_queue,
            think_end_id=self.model_config.think_end_id,
            line_break_id=self.model_config.line_break_id,
        )
        return pooler_output

    def _dummy_sampler_run(
        self,
        hidden_states: paddle.Tensor,
        model_output: paddle.Tensor,
        accept_all_drafts=False,
        reject_all_drafts=False,
    ) -> paddle.Tensor:
        logits = self.model.compute_logits(hidden_states)

        if not self.speculative_decoding:
            set_value_by_flags_and_idx(
                self.share_inputs["pre_ids"],
                self.share_inputs["input_ids"],
                self.share_inputs["seq_lens_this_time"],
                self.share_inputs["seq_lens_encoder"],
                self.share_inputs["seq_lens_decoder"],
                self.share_inputs["step_idx"],
                self.share_inputs["stop_flags"],
            )
            sampler_output = self.sampler(logits, self.sampling_metadata)
            if self.parallel_config.tensor_parallel_size > 1:
                paddle.distributed.broadcast(
                    sampler_output.sampled_token_ids,
                    self.parallel_config.data_parallel_rank * self.parallel_config.tensor_parallel_size,
                    group=self.parallel_config.tp_group,
                )
        else:
            self.sampler(
                logits,
                self.sampling_metadata,
                self.model_config.max_model_len,
                self.share_inputs,
                accept_all_drafts,
                reject_all_drafts,
            )
            sampler_output = None
            if self.parallel_config.tensor_parallel_size > 1:
                paddle.distributed.broadcast(
                    self.share_inputs["accept_tokens"],
                    self.parallel_config.data_parallel_rank * self.parallel_config.tensor_parallel_size,
                    group=self.parallel_config.tp_group,
                )
                paddle.distributed.broadcast(
                    self.share_inputs["accept_num"],
                    self.parallel_config.data_parallel_rank * self.parallel_config.tensor_parallel_size,
                    group=self.parallel_config.tp_group,
                )
                paddle.distributed.broadcast(
                    self.share_inputs["step_idx"],
                    self.parallel_config.data_parallel_rank * self.parallel_config.tensor_parallel_size,
                    group=self.parallel_config.tp_group,
                )
                paddle.distributed.broadcast(
                    self.share_inputs["stop_flags"],
                    self.parallel_config.data_parallel_rank * self.parallel_config.tensor_parallel_size,
                    group=self.parallel_config.tp_group,
                )
        # 5. post process
        model_output_data = ModelOutputData(
            next_tokens=self.share_inputs["next_tokens"],
            stop_flags=self.share_inputs["stop_flags"],
            step_idx=self.share_inputs["step_idx"],
            max_dec_len=self.share_inputs["max_dec_len"],
            pre_ids=self.share_inputs["pre_ids"],
            seq_lens_this_time=self.share_inputs["seq_lens_this_time"],
            eos_token_id=self.share_inputs["eos_token_id"],
            not_need_stop=self.share_inputs["not_need_stop"],
            input_ids=self.share_inputs["input_ids"],
            stop_nums=self.share_inputs["stop_nums"],
            seq_lens_encoder=self.share_inputs["seq_lens_encoder"],
            seq_lens_decoder=self.share_inputs["seq_lens_decoder"],
            is_block_step=self.share_inputs["is_block_step"],
            full_hidden_states=model_output,
            msg_queue_id=self.parallel_config.msg_queue_id,
            mp_rank=self.parallel_config.tensor_parallel_rank,
            use_ep=self.parallel_config.use_ep,
            draft_tokens=(self.share_inputs["draft_tokens"] if self.speculative_decoding else None),
            actual_draft_token_num=(
                self.share_inputs["actual_draft_token_num"] if self.speculative_decoding else None
            ),
            accept_tokens=(self.share_inputs["accept_tokens"] if self.speculative_decoding else None),
            accept_num=(self.share_inputs["accept_num"] if self.speculative_decoding else None),
            stop_token_ids=self.share_inputs["stop_seqs"],
            stop_seqs_len=self.share_inputs["stop_seqs_len"],
            prompt_lens=self.share_inputs["prompt_lens"],
            mask_rollback=self.share_inputs["mask_rollback"],
        )

        post_process(
            sampler_or_pooler_output=sampler_output,
            model_output=model_output_data,
            share_inputs=self.share_inputs,
            block_size=self.cache_config.block_size,
            speculative_decoding=self.speculative_decoding,
            skip_save_output=True,
            async_output_queue=self.async_output_queue,
            think_end_id=self.model_config.think_end_id,
            line_break_id=self.model_config.line_break_id,
        )
        if self.speculative_decoding:
            if self.speculative_method == "mtp":
                self.proposer.run(
                    full_hidden_states=model_output,
                    step_use_cudagraph=self.forward_meta.step_use_cudagraph,
                    is_dummy_run=True,
                )
            else:
                self.proposer.run(share_inputs=self.share_inputs)

        return sampler_output

    def _dummy_run(
        self,
        num_tokens: paddle.Tensor,
        batch_size: paddle.Tensor,
        expected_decode_len: int = 1,
        in_capturing: bool = False,
        capture_prefill: bool = False,
        accept_all_drafts: bool = False,
        reject_all_drafts: bool = False,
    ) -> paddle.Tensor:
        """
        Use dummy inputs to run before formal execution.
        Args:
            num_tokens:
            expected_decode_len: Expected number of tokens generated
            in_capturing: Is cuda graph in capturing state
            capture_prefill: Capture pure prefill for cuda graph
            accept_all_drafts: Target model will accept all draft tokens
            reject_all_drafts: Target model will reject all draft tokens
        """
        input_length_list, max_dec_len_list, block_num = self.get_input_length_list(
            num_tokens=num_tokens,
            batch_size=batch_size,
            expected_decode_len=expected_decode_len,
            capture_prefill=capture_prefill,
        )
        self._dummy_prefill_inputs(
            input_length_list=input_length_list,
            max_dec_len_list=max_dec_len_list,
            block_num=block_num,
        )
        if self.speculative_method in ["mtp"]:
            self.proposer.dummy_prefill_inputs(
                num_tokens=num_tokens,
                batch_size=batch_size,
                expected_decode_len=expected_decode_len,
            )

        while True:
            # 1. Initialize forward meta and attention meta data
            self._prepare_inputs(is_dummy_or_profile_run=True)

            # 2. Padding inputs for cuda graph
            self.forward_meta.step_use_cudagraph = in_capturing and self.forward_meta.step_use_cudagraph
            self.padding_cudagraph_inputs()

            # 3. Run model
            if self.enable_mm:
                model_output = self.model(
                    self.share_inputs["ids_remove_padding"],
                    self.share_inputs["image_features"],
                    self.forward_meta,
                )
            else:
                model_output = self.model(
                    ids_remove_padding=self.share_inputs["ids_remove_padding"],
                    forward_meta=self.forward_meta,
                )
            if self.use_cudagraph:
                model_output = model_output[: self.real_token_num]

            if self.is_pooling_model:
                hidden_states = model_output
                self._dummy_pooler_run(hidden_states, model_output)
                break
            else:
                hidden_states = rebuild_padding(
                    model_output,
                    self.share_inputs["cu_seqlens_q"],
                    self.share_inputs["seq_lens_this_time"],
                    self.share_inputs["seq_lens_decoder"],
                    self.share_inputs["seq_lens_encoder"],
                    (
                        self.share_inputs["output_padding_offset"] if self.speculative_decoding else None
                    ),  # speculative decoding requires
                    self.model_config.max_model_len,
                )
                self._dummy_sampler_run(hidden_states, model_output, accept_all_drafts, reject_all_drafts)

            # 7. Updata 'infer_seed' and step_cuda()
            self.share_inputs["infer_seed"].add_(self.infer_seed_increment)
            self.share_inputs["infer_seed"][:] %= self.MAX_INFER_SEED
            step_cuda(
                self.share_inputs,
                self.cache_config.block_size,
                self.cache_config.enc_dec_block_num,
                self.speculative_config,
                self.cache_config.enable_prefix_caching,
            )
            if int((self.share_inputs["seq_lens_this_time"] > 0).sum()) == 0:
                break

    def _update_chunked_prefill(self, tasks):
        """
        Update chunked prefill related parameters
        """
        if not self.cache_config.enable_chunked_prefill:
            return

        if tasks is not None:
            for task in tasks:
                if task.get("prefill_chunk_info", None) is None:
                    continue

                if task.chunk_idx > len(task.prefill_chunk_info):
                    continue
                self.restore_chunked_prefill_request[task.request_id] = task

        for id, task in list(self.restore_chunked_prefill_request.items()):
            idx = task.idx
            logger.debug(f"{task.request_id} chunked prefill {task.chunk_idx}/{len(task.prefill_chunk_info)}")
            if not self.enable_mm:
                start_idx = sum(task.prefill_chunk_info[: task.chunk_idx])
            if task.chunk_idx == len(task.prefill_chunk_info):
                self.share_inputs["seq_lens_this_time"][idx : idx + 1] = 1
                self.share_inputs["seq_lens_encoder"][idx : idx + 1] = 0
                self.share_inputs["step_idx"][idx : idx + 1] = 1
                if self.enable_mm:
                    self.share_inputs["seq_lens_decoder"][idx : idx + 1] = task.start_idx
                else:
                    self.share_inputs["seq_lens_decoder"][idx : idx + 1] = start_idx + task.get("seq_lens_decoder", 0)
                del self.restore_chunked_prefill_request[task.request_id]
            else:
                token_chunk_size = task.prefill_chunk_info[task.chunk_idx]
                if self.enable_mm:
                    inputs = self._preprocess_mm_task(task.prefill_chunk_info[task.chunk_idx])
                    if inputs.get("images") is not None:
                        self.share_inputs["image_features"] = self.extract_vision_features(inputs)
                    else:
                        # Compatible with the situation that lacks images and videos
                        self.share_inputs["image_features"] = None
                    token_chunk_size = inputs["input_ids"].shape[1]
                    self.share_inputs["input_ids"][idx : idx + 1, :token_chunk_size] = inputs["input_ids"]
                    self.share_inputs["prompt_ids"][
                        idx : idx + 1,
                        self.share_inputs["prompt_lens"][idx : idx + 1] : self.share_inputs["prompt_lens"][
                            idx : idx + 1
                        ]
                        + token_chunk_size,
                    ] = inputs["input_ids"]
                    self.share_inputs["seq_lens_decoder"][idx : idx + 1] = task.start_idx
                    task.start_idx += token_chunk_size
                else:
                    self.share_inputs["input_ids"][idx, :token_chunk_size] = np.array(
                        task.prompt_token_ids[start_idx : start_idx + token_chunk_size]
                    )
                    self.share_inputs["seq_lens_decoder"][idx : idx + 1] = start_idx + task.get("seq_lens_decoder", 0)
                self.share_inputs["seq_lens_this_time"][idx : idx + 1] = token_chunk_size
                self.share_inputs["seq_lens_encoder"][idx : idx + 1] = token_chunk_size
                self.share_inputs["prompt_lens"][idx : idx + 1] += token_chunk_size
                self.share_inputs["step_idx"][idx : idx + 1] = 0

            if self.speculative_decoding and self.proposer.is_chunk_prefill_enabled():
                self.proposer.update_task_chunk_prefill(task)
            task.chunk_idx += 1

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
            elif self.speculative_decoding and self.speculative_method == "mtp":
                # Capture Target Model without bsz 1
                for batch_size in sorted(capture_sizes, reverse=True):
                    if batch_size == 1:
                        logger.info("Skip token_num = 1, when capture target model for mtp")
                    else:
                        assert batch_size % 2 == 0
                        self._dummy_run(
                            num_tokens=(
                                self.scheduler_config.max_num_seqs
                                * (self.speculative_config.num_speculative_tokens + 1)
                                if self.scheduler_config.splitwise_role == "decode"
                                else self.scheduler_config.max_num_batched_tokens
                            ),
                            batch_size=int(batch_size / 2),
                            in_capturing=True,
                            expected_decode_len=1,
                        )
                        logger.info(
                            f"Warm up the Target model with the num_tokens:{batch_size}, expected_decode_len:{1}"
                        )
                if self.graph_opt_config.draft_model_use_cudagraph:
                    # Capture Draft Model without bsz 1
                    # NOTE(liujundong): expected_decode_len = 1, will affect mtp capture in cudagraph
                    for batch_size in sorted(capture_sizes, reverse=True):
                        if batch_size == 1:
                            logger.info("Skip token_num = 1, when capture Draft model for mtp")
                        else:
                            assert batch_size % 2 == 0
                            self._dummy_run(
                                num_tokens=(
                                    self.scheduler_config.max_num_seqs
                                    if self.scheduler_config.splitwise_role == "decode"
                                    else self.scheduler_config.max_num_batched_tokens
                                ),
                                batch_size=int(batch_size / 2),
                                in_capturing=True,
                                expected_decode_len=3,
                                accept_all_drafts=True,
                            )
                            logger.info(
                                f"Warm up the Draft model with the num_tokens:{batch_size}, expected_decode_len:{3}"
                            )
                    # Capture Draft Model with bsz 1
                    if 1 in capture_sizes:
                        self._dummy_run(
                            num_tokens=(
                                self.scheduler_config.max_num_seqs
                                if self.scheduler_config.splitwise_role == "decode"
                                else self.scheduler_config.max_num_batched_tokens
                            ),
                            batch_size=int(1),
                            in_capturing=True,
                            expected_decode_len=3,
                            accept_all_drafts=False,
                            reject_all_drafts=True,
                        )
                        logger.info(
                            f"Warm up the Draft model with the num_tokens:{batch_size}, expected_decode_len:{3}"
                        )
            else:
                for batch_size in sorted(capture_sizes, reverse=True):
                    self._dummy_run(
                        num_tokens=(
                            self.scheduler_config.max_num_seqs
                            if self.scheduler_config.splitwise_role == "decode"
                            else self.scheduler_config.max_num_batched_tokens
                        ),
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

    @sot_warmup_guard(True)
    def sot_warmup(self) -> None:
        start_time = time.perf_counter()
        for batch_size in self.sot_warmup_sizes:
            self._dummy_run(
                num_tokens=(
                    self.scheduler_config.max_num_seqs
                    if self.scheduler_config.splitwise_role == "decode"
                    else self.scheduler_config.max_num_batched_tokens
                ),
                batch_size=batch_size,
            )
            logger.info(f"SOT warmup the model with the batch size:{batch_size}")
        logger.info(f"SOT warmup took {time.perf_counter() - start_time} seconds")

    def _get_p_done_idxs_gd(self, model_forward_batch: Optional[List[Request]], num_running_requests: int):
        """
        Get indices for guided decoding.
        When Prefill is done, async compiled logits_processor must be joined.
        """
        if self.guided_backend is None:
            return []

        prefill_done_idxs = []
        for idx in range(0, num_running_requests):
            if self.share_inputs["step_idx"][idx] == 0:
                prefill_done_idxs.append(idx)

        if envs.ENABLE_V1_KVCACHE_SCHEDULER:
            if model_forward_batch is None:
                return prefill_done_idxs

            for task in model_forward_batch:
                if task.task_type.value != RequestType.PREFILL.value:
                    continue
                # in chunk prefill
                if self.cache_config.enable_chunked_prefill:
                    if hasattr(task, "prefill_end_index") and hasattr(task, "prompt_token_ids"):
                        if len(task.prompt_token_ids) > task.prefill_end_index and task.idx in prefill_done_idxs:
                            prefill_done_idxs.remove(task.idx)

            return prefill_done_idxs

        if self.cache_config.enable_chunked_prefill:
            if model_forward_batch is not None:
                for task in model_forward_batch:
                    # new Request with ChunkPrefill, unfinished, store
                    if task.chunk_idx < len(task.prefill_chunk_info):
                        if task.request_id not in self.restore_chunked_prefill_request:
                            self.restore_chunked_prefill_request[task.request_id] = task

            for id, task in list(self.restore_chunked_prefill_request.items()):
                # unfinished, remove
                if task.chunk_idx < len(task.prefill_chunk_info) and task.idx in prefill_done_idxs:
                    prefill_done_idxs.remove(task.idx)
                # finished, add
                if task.chunk_idx == len(task.prefill_chunk_info) and task.idx not in prefill_done_idxs:
                    prefill_done_idxs.append(task.idx)

        return prefill_done_idxs

    def execute_model(
        self,
        model_forward_batch: Optional[List[Request]] = None,
        num_running_requests: int = None,
    ) -> None:
        """
        The Entrance of model execute.
        Args:
            model_forward_batch: 'Request' contains information related to prompt and is an abstract
            class at the server level, which is too granular for ModelRunner.
            We plan to replace it with 'ModelForwardBatch'.
            intermediate_tensors:
            num_running_requests: batch_size
        """
        # 1. Prepare inputs of model and sampler.
        p_done_idxs = self._get_p_done_idxs_gd(model_forward_batch, num_running_requests)

        self._prepare_inputs()
        self.sampler.pre_process(p_done_idxs)

        # 1.1 Update state of logits processor
        for proc in self.sampling_metadata.logits_processors:
            proc.update_state(self.share_inputs)

        # NOTE(wufeisheng): If `not_need_stop`` is False, it means the current worker is in an idle state.
        # This logic is not used in TP (Tensor Parallelism) mode. However, in EP (Expert Parallelism) mode,
        # when there is data on other runner, the current runner is required to execute part of the model.
        if not self.not_need_stop():
            self._execute_empty_input()
            return None

        # 2. Padding inputs for cuda graph
        self.padding_cudagraph_inputs()

        # 3. Execute model
        if self.enable_mm:
            model_output = self.model(
                self.share_inputs["ids_remove_padding"],
                self.share_inputs["image_features"],
                self.forward_meta,
            )
        else:
            model_output = self.model(
                ids_remove_padding=self.share_inputs["ids_remove_padding"],
                forward_meta=self.forward_meta,
            )
        if self.use_cudagraph:
            model_output = model_output[: self.real_token_num]

        prompt_logprobs_list = self._get_prompt_logprobs_list(model_output)

        if self.is_pooling_model:
            hidden_states = model_output
            pooler_output = self._pool(hidden_states, num_running_requests)

            model_output_data = ModelOutputData(
                next_tokens=self.share_inputs["next_tokens"],
                stop_flags=self.share_inputs["stop_flags"],
                step_idx=self.share_inputs["step_idx"],
                max_dec_len=self.share_inputs["max_dec_len"],
                pre_ids=self.share_inputs["pre_ids"],
                seq_lens_this_time=self.share_inputs["seq_lens_this_time"],
                eos_token_id=self.share_inputs["eos_token_id"],
                not_need_stop=self.share_inputs["not_need_stop"],
                input_ids=self.share_inputs["input_ids"],
                stop_nums=self.share_inputs["stop_nums"],
                seq_lens_encoder=self.share_inputs["seq_lens_encoder"],
                seq_lens_decoder=self.share_inputs["seq_lens_decoder"],
                is_block_step=self.share_inputs["is_block_step"],
                full_hidden_states=model_output,
                msg_queue_id=self.parallel_config.msg_queue_id,
                mp_rank=self.parallel_config.tensor_parallel_rank,
                use_ep=self.parallel_config.use_ep,
                draft_tokens=(self.share_inputs["draft_tokens"] if self.speculative_decoding else None),
                actual_draft_token_num=(
                    self.share_inputs["actual_draft_token_num"] if self.speculative_decoding else None
                ),
                accept_tokens=(self.share_inputs["accept_tokens"] if self.speculative_decoding else None),
                accept_num=(self.share_inputs["accept_num"] if self.speculative_decoding else None),
                stop_token_ids=self.share_inputs["stop_seqs"],
                stop_seqs_len=self.share_inputs["stop_seqs_len"],
                prompt_lens=self.share_inputs["prompt_lens"],
            )

            post_process(
                sampler_or_pooler_output=pooler_output,
                model_output=model_output_data,
                share_inputs=self.share_inputs,
                block_size=self.cache_config.block_size,
                save_each_rank=self.parallel_config.use_ep,
                speculative_decoding=self.speculative_decoding,
                skip_save_output=False,
                async_output_queue=self.async_output_queue,
            )

            return None
        else:
            hidden_states = rebuild_padding(
                model_output,
                self.share_inputs["cu_seqlens_q"],
                self.share_inputs["seq_lens_this_time"],
                self.share_inputs["seq_lens_decoder"],
                self.share_inputs["seq_lens_encoder"],
                (self.share_inputs["output_padding_offset"] if self.speculative_decoding else None),
                self.model_config.max_model_len,
            )

            # 4. Compute logits, Sample
            logits = self.model.compute_logits(hidden_states)

            if not self.speculative_decoding:
                set_value_by_flags_and_idx(
                    self.share_inputs["pre_ids"],
                    self.share_inputs["input_ids"],
                    self.share_inputs["seq_lens_this_time"],
                    self.share_inputs["seq_lens_encoder"],
                    self.share_inputs["seq_lens_decoder"],
                    self.share_inputs["step_idx"],
                    self.share_inputs["stop_flags"],
                )
                sampler_output = self.sampler(
                    logits,
                    self.sampling_metadata,
                    p_done_idxs,
                )
                if self.parallel_config.tensor_parallel_size > 1:
                    paddle.distributed.broadcast(
                        sampler_output.sampled_token_ids,
                        self.parallel_config.data_parallel_rank * self.parallel_config.tensor_parallel_size,
                        group=self.parallel_config.tp_group,
                    )
            else:
                sampler_output = self.sampler(
                    logits,
                    self.sampling_metadata,
                    self.model_config.max_model_len,
                    self.share_inputs,
                )
                if self.parallel_config.tensor_parallel_size > 1:
                    paddle.distributed.broadcast(
                        self.share_inputs["accept_tokens"],
                        self.parallel_config.data_parallel_rank * self.parallel_config.tensor_parallel_size,
                        group=self.parallel_config.tp_group,
                    )
                    paddle.distributed.broadcast(
                        self.share_inputs["accept_num"],
                        self.parallel_config.data_parallel_rank * self.parallel_config.tensor_parallel_size,
                        group=self.parallel_config.tp_group,
                    )
                    paddle.distributed.broadcast(
                        self.share_inputs["step_idx"],
                        self.parallel_config.data_parallel_rank * self.parallel_config.tensor_parallel_size,
                        group=self.parallel_config.tp_group,
                    )
                    paddle.distributed.broadcast(
                        self.share_inputs["stop_flags"],
                        self.parallel_config.data_parallel_rank * self.parallel_config.tensor_parallel_size,
                        group=self.parallel_config.tp_group,
                    )

            # 5. Post Process
            model_output_data = ModelOutputData(
                next_tokens=self.share_inputs["next_tokens"],
                stop_flags=self.share_inputs["stop_flags"],
                step_idx=self.share_inputs["step_idx"],
                max_dec_len=self.share_inputs["max_dec_len"],
                pre_ids=self.share_inputs["pre_ids"],
                seq_lens_this_time=self.share_inputs["seq_lens_this_time"],
                eos_token_id=self.share_inputs["eos_token_id"],
                not_need_stop=self.share_inputs["not_need_stop"],
                input_ids=self.share_inputs["input_ids"],
                stop_nums=self.share_inputs["stop_nums"],
                seq_lens_encoder=self.share_inputs["seq_lens_encoder"],
                seq_lens_decoder=self.share_inputs["seq_lens_decoder"],
                is_block_step=self.share_inputs["is_block_step"],
                full_hidden_states=model_output,
                msg_queue_id=self.parallel_config.msg_queue_id,
                mp_rank=self.parallel_config.tensor_parallel_rank,
                use_ep=self.parallel_config.use_ep,
                draft_tokens=(self.share_inputs["draft_tokens"] if self.speculative_decoding else None),
                actual_draft_token_num=(
                    self.share_inputs["actual_draft_token_num"] if self.speculative_decoding else None
                ),
                accept_tokens=(self.share_inputs["accept_tokens"] if self.speculative_decoding else None),
                accept_num=(self.share_inputs["accept_num"] if self.speculative_decoding else None),
                stop_token_ids=self.share_inputs["stop_seqs"],
                stop_seqs_len=self.share_inputs["stop_seqs_len"],
                prompt_lens=self.share_inputs["prompt_lens"],
                mask_rollback=self.share_inputs["mask_rollback"],
                prompt_logprobs_list=prompt_logprobs_list,
            )

            if self.speculative_config.method in ["mtp"] and self.scheduler_config.splitwise_role == "prefill":
                skip_save_output = True
            else:
                skip_save_output = False

            post_process(
                sampler_or_pooler_output=sampler_output,
                model_output=model_output_data,
                share_inputs=self.share_inputs,
                block_size=self.cache_config.block_size,
                save_each_rank=self.parallel_config.use_ep,
                speculative_decoding=self.speculative_decoding,
                skip_save_output=skip_save_output,
                async_output_queue=self.async_output_queue,
                think_end_id=self.model_config.think_end_id,
                line_break_id=self.model_config.line_break_id,
            )
            if self.guided_backend is not None and sampler_output is not None:
                self.sampler.post_process(sampler_output.sampled_token_ids)

            # 6. Speculative decode
            if self.speculative_decoding:
                if self.speculative_method == "mtp":
                    self.proposer.run(
                        full_hidden_states=model_output, step_use_cudagraph=self.forward_meta.step_use_cudagraph
                    )
                else:
                    self.proposer.run(share_inputs=self.share_inputs)

            # 7. Update 'infer_seed' and step_cuda()
            self.share_inputs["infer_seed"].add_(self.infer_seed_increment)
            self.share_inputs["infer_seed"][:] %= self.MAX_INFER_SEED
            if not envs.ENABLE_V1_KVCACHE_SCHEDULER:
                step_cuda(
                    self.share_inputs,
                    self.cache_config.block_size,
                    self.cache_config.enc_dec_block_num,
                    self.speculative_config,
                    self.cache_config.enable_prefix_caching,
                )

                self._update_chunked_prefill(model_forward_batch)
            elif self.speculative_decoding:
                speculate_schedule_cache(
                    self.share_inputs["draft_tokens"],
                    self.share_inputs["block_tables"],
                    self.share_inputs["stop_flags"],
                    self.share_inputs["prompt_lens"],
                    self.share_inputs["seq_lens_this_time"],
                    self.share_inputs["seq_lens_encoder"],
                    self.share_inputs["seq_lens_decoder"],
                    self.share_inputs["step_seq_lens_decoder"],
                    self.share_inputs["step_draft_tokens"],
                    self.share_inputs["step_seq_lens_this_time"],
                    self.share_inputs["accept_num"],
                    self.share_inputs["accept_tokens"],
                    self.share_inputs["is_block_step"],
                    self.share_inputs["not_need_stop"],
                    self.share_inputs["stop_nums"],
                    self.cache_config.block_size,
                    self.speculative_config.num_speculative_tokens,
                )

            return None

    def _pool(self, hidden_states: paddle.Tensor, num_running_requests: int) -> Optional[ModelRunnerOutput]:

        num_scheduled_tokens = int(self.share_inputs["seq_lens_this_time"][:num_running_requests].sum())

        hidden_states = hidden_states[:num_scheduled_tokens]

        prompt_lens = self.share_inputs["prompt_lens"][:num_running_requests]
        prompt_token_ids = self.share_inputs["prompt_ids"]

        pooling_metadata = PoolingMetadata(
            prompt_lens=prompt_lens,
            prompt_token_ids=prompt_token_ids,
            pooling_params=self.pooling_params,
        )
        num_scheduled_tokens_list = [
            int(self.share_inputs["seq_lens_this_time"][i]) for i in range(num_running_requests)
        ]
        device_str = "gpu" if hidden_states.place.is_gpu_place() else "cpu"
        pooling_metadata.build_pooling_cursor(num_scheduled_tokens_list, device=device_str)

        raw_pooler_output = self.model.pooler(hidden_states=hidden_states, pooling_metadata=pooling_metadata)
        seq_lens_cpu = self.share_inputs["seq_lens_this_time"][:num_running_requests]
        pooler_output: list[Optional[paddle.Tensor]] = []
        for raw_output, seq_len, prompt_len in zip(raw_pooler_output, seq_lens_cpu, pooling_metadata.prompt_lens):
            output = raw_output.data if int(seq_len) == int(prompt_len) else None
            pooler_output.append(output)

        pooler_output = PoolerOutput(
            outputs=pooler_output,
        )

        return pooler_output

    def _execute_empty_input(self) -> None:
        """
        In certain scenarios, such as during EP,
        the runner needs to execute partial modules of the model without input data.
        This requires the model to implement the `empty_input_forward` method.
        """
        if hasattr(self.model, "empty_input_forward"):
            self.model.empty_input_forward()
        else:
            raise ValueError(f"{type(self.model)} has no attribute 'empty_input_forward")

    @profile_run_guard(True)
    def profile_run(self) -> None:
        """Execute a forward pass with dummy inputs to profile the memory usage of the model"""
        # Initialize kv cache for profile run. After profile run kv cache will be reset.
        # TODO(gongshaotian): Optimize the management logic of kvcache
        self.num_gpu_blocks = self.cache_config.total_block_num
        self.initialize_kv_cache(profile=True)
        if self.speculative_method in ["mtp"]:
            self.proposer.initialize_kv_cache(main_model_num_blocks=self.num_gpu_blocks, profile=True)

        # 1. Profile with multimodal encoder & encoder cache

        # 2. Dummy run
        self._dummy_run(
            num_tokens=(
                self.scheduler_config.max_num_seqs
                if self.scheduler_config.splitwise_role == "decode"
                else self.scheduler_config.max_num_batched_tokens
            ),
            batch_size=self.scheduler_config.max_num_seqs,
        )

        # 3. gc
        self.clear_cache()
        if self.speculative_method in ["mtp"]:
            self.proposer.clear_mtp_cache()

    def update_share_input_block_num(self, num_gpu_blocks: int) -> None:
        """
        Set a globally unified block number and update the model's shared input.
        Args:
            num_gpu_blocks:
        """
        self.num_gpu_blocks = num_gpu_blocks

        # Reset block table and kv cache with global block num
        self.initialize_kv_cache()

        # Reset free list
        free_list = list(
            range(
                self.num_gpu_blocks - 1,
                int(self.num_gpu_blocks * self.cache_config.kv_cache_ratio) - 1,
                -1,
            )
        )
        self.free_list_len = len(free_list)
        self.share_inputs.update(
            {
                "free_list": paddle.to_tensor(free_list, dtype="int32"),
                "free_list_len": paddle.full([1], self.free_list_len, dtype="int32"),
            }
        )

        if self.speculative_method in ["mtp"]:
            self.proposer.update_mtp_block_num(num_gpu_blocks)

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
            if self.speculative_method in ["mtp"]
            else self.model_config.num_hidden_layers
        )

        # NOTE:(changwenbin) Determie whether it is Multi-Head Latent Attention,
        # To rationalize the allocation of kvcache.
        if self.mla_cache:
            required_memory = (
                byte_of_dtype
                * (self.fd_config.model_config.kv_lora_rank + self.fd_config.model_config.qk_rope_head_dim)
                * (self.cache_config.block_size)
                * num_layers
            )  # compress_kv + k_pe
        else:
            required_memory = byte_of_dtype * 2 * (self.cache_config.block_size * hidden_dim) * num_layers  # k + v
        return required_memory

    def not_need_stop(self) -> bool:
        """Stop decoding if the tensor meets the termination condition"""
        return self.share_inputs["not_need_stop"][0]

    def clear_cache(self):
        """Clear cached data from shared inputs and forward metadata"""
        self.share_inputs.pop("caches", None)
        if self.forward_meta is not None:
            self.forward_meta.clear_caches()
        paddle.device.cuda.empty_cache()

    def clear_parameters(self, pid):
        """Dynamic model loader use to clear parameters use for RL"""
        # Clear CUDAGraph
        if self.use_cudagraph:
            self.model.clear_grpah_opt_backend()
        # Clear parameters and Send single
        self.dynamic_weight_manager.clear_parameters(pid)
        self.clear_cache()
        paddle.device.cuda.empty_cache()

        self.dynamic_weight_manager._log_memory("dynamic weight manager clear all memory")

    def clear_requests(self):
        """Dynamic model loader use to clear requests use for RL"""
        self.share_inputs["stop_flags"][:] = True
        # prompt_logprobs
        self.prompt_logprobs_reqs.clear()
        self.in_progress_prompt_logprobs.clear()

    def update_parameters(self, pid):
        """Dynamic model loader use to update parameters use for RL"""
        # Update parameters
        self.dynamic_weight_manager.update_parameters(pid)
        self.initialize_kv_cache()
        # Recapture CUDAGraph
        if self.use_cudagraph:
            self.capture_model()
        # Send single
        self.dynamic_weight_manager.finalize_update(pid)

        self.dynamic_weight_manager._log_memory("dynamic weight manager update all memory")

    def padding_cudagraph_inputs(self) -> None:
        """
        Clean buffers used for the CUDA graph when replaying the CUDA graph with the padded batch.
        In FastDeploy, almost all input tensors have a buffer. So, just keep the buffer clean when replaying the CUDA graph with the padded batch.
        """
        # In init_attention_metadata, the decode buffer has already been cleared

        # To adapt to CUDA Graph, keep the forward pass at the maximum batch size.
        if self.use_cudagraph:
            self.forward_meta.seq_lens_this_time = self.seq_lens_this_time_buffer
            self.real_token_num = self.forward_meta.ids_remove_padding.shape[0]
        return

    def _init_image_preprocess(self) -> None:
        processor = DataProcessor(
            tokenizer_name=self.model_config.model,
            image_preprocessor_name=str(self.model_config.model),
        )
        processor.eval()
        image_preprocess = processor.image_preprocessor
        image_preprocess.image_mean_tensor = paddle.to_tensor(image_preprocess.image_mean, dtype="float32").reshape(
            [1, 3, 1, 1]
        )
        image_preprocess.image_std_tensor = paddle.to_tensor(image_preprocess.image_std, dtype="float32").reshape(
            [1, 3, 1, 1]
        )
        image_preprocess.rescale_factor = paddle.to_tensor(image_preprocess.rescale_factor, dtype="float32")
        image_preprocess.image_mean_tensor = image_preprocess.image_mean_tensor.squeeze([-2, -1]).repeat_interleave(
            self.model_config.vision_config.patch_size**2 * 1, -1
        )
        image_preprocess.image_std_tensor = image_preprocess.image_std_tensor.squeeze([-2, -1]).repeat_interleave(
            self.model_config.vision_config.patch_size**2 * 1, -1
        )
        self.image_preprocess = image_preprocess

    def _preprocess_mm_task(self, one: dict) -> None:
        """process batch"""

        input_ids = one["input_ids"][np.newaxis, :]
        input_ids = paddle.to_tensor(input_ids, dtype=paddle.int64)
        token_type_ids = one["token_type_ids"][np.newaxis, :]
        token_type_ids = paddle.to_tensor(token_type_ids, dtype=paddle.int64)

        if one["images"] is not None:
            image_type_ids = one["image_type_ids"][np.newaxis, :]
            images = one["images"]
            image_type_ids = paddle.to_tensor(image_type_ids, dtype=paddle.int64)
            images = paddle.to_tensor(images, dtype="uint8" if "ernie" in self.model_config.model_type else "bfloat16")
            grid_thw = paddle.to_tensor(one["grid_thw"], dtype="int64")
        else:
            image_type_ids = None
            images = None
            grid_thw = None

        if one["position_ids"] is not None:
            position_ids = paddle.to_tensor(one["position_ids"], dtype="int64")
        else:
            position_ids = None

        result = dict(
            input_ids=input_ids,
            image_type_ids=image_type_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            grid_thw=grid_thw,
            images=images,
        )
        return result

    def extract_vision_features_ernie(self, inputs: list[paddle.Tensor]) -> paddle.Tensor:
        assert inputs["images"] is not None
        grid_thw = inputs["grid_thw"]
        # ernie-vl has images norm
        images = inputs["images"].cast("float32")
        images = self.image_preprocess.rescale_factor * images - self.image_preprocess.image_mean_tensor
        images = images / self.image_preprocess.image_std_tensor
        images = images.cast("bfloat16")

        token_type_ids = inputs["token_type_ids"]
        token_type_ids_w_video = token_type_ids
        input_ids = inputs["input_ids"]
        # convert to img patch id
        image_mask = input_ids == self.model_config.im_patch_id
        image_type_ids = inputs["image_type_ids"]
        with paddle.amp.auto_cast(
            True,
            custom_black_list=self.amp_black,
            custom_white_list=self.amp_white,
            level="O2",
            dtype=self.model_config.dtype,
        ):
            image_features = self.model.vision_model.extract_feature(images, grid_thw)
            if self.parallel_config.tensor_parallel_size > 1:
                S, C = image_features.shape
                image_features = image_features.reshape([-1, C * self.model_config.spatial_conv_size**2])
                image_features = ScatterOp.apply(image_features, axis=-1)  # mp 切 Fea
                image_features = image_features.reshape([S, -1])
            # ernie-vl has resampler_model
            image_features = self.model.resampler_model(
                image_features,
                image_mask,
                token_type_ids_w_video,
                image_type_ids,
                grid_thw,
            )
        return image_features

    def extract_vision_features_qwen(self, inputs: list[paddle.Tensor]) -> paddle.Tensor:
        assert inputs["images"] is not None
        grid_thw = inputs["grid_thw"]
        images = inputs["images"]
        with paddle.amp.auto_cast(
            True,
            custom_black_list=self.amp_black,
            custom_white_list=self.amp_white,
            level="O2",
            dtype=self.model_config.dtype,
        ):
            image_features = self.model.visual.extract_feature(images, grid_thw)

        return image_features

    def extract_vision_features_paddleocr(self, inputs: list[paddle.Tensor]) -> paddle.Tensor:
        if envs.FD_ENABLE_MAX_PREFILL:
            inputs["vit_position_ids_lst"] = np.concatenate(inputs["vit_position_ids_lst"])
            images = paddle.concat(inputs["images_lst"]).cast("bfloat16")
            grid_thw = paddle.to_tensor(inputs["grid_thw_lst"], dtype="int64")
            position_ids = paddle.to_tensor(inputs["vit_position_ids_lst"], dtype="int64")
            cu_seqlens = paddle.cumsum(paddle.to_tensor(inputs["cu_seqlens"])).cast("int32")
        else:
            assert inputs["images"] is not None
            grid_thw = inputs["grid_thw"]
            images = inputs["images"]

            position_ids = []
            cu_seqlens = [0]
            for idx, thw in enumerate(grid_thw):
                numel = np.prod(np.array(thw))
                position_ids.append(paddle.arange(numel) % np.prod(thw[1:]))
                cu_seqlens.append(cu_seqlens[-1] + numel)

            position_ids = paddle.concat(position_ids, axis=0).to(images.place)
            cu_seqlens = paddle.to_tensor(cu_seqlens, dtype=paddle.int32).to(images.place)

        with paddle.amp.auto_cast(
            True,
            custom_black_list=self.amp_black,
            custom_white_list=self.amp_white,
            level="O2",
            dtype=self.model_config.dtype,
        ):
            image_features = self.model.visual(
                pixel_values=images,
                image_grid_thw=grid_thw,
                position_ids=position_ids,
                interpolate_pos_encoding=True,
                cu_seqlens=cu_seqlens,
                use_rope=True,
                window_size=-1,
            )
            image_features = self.model.projector(image_features, grid_thw)
            image_features = paddle.concat(image_features, axis=0)

        return image_features

    @paddle.no_grad()
    def extract_vision_features(self, inputs: list[paddle.Tensor]) -> paddle.Tensor:
        """extract_vision_features"""
        if "ernie" in self.model_config.model_type:
            return self.extract_vision_features_ernie(inputs)
        elif "qwen" in self.model_config.model_type:
            return self.extract_vision_features_qwen(inputs)
        elif "paddleocr" in self.model_config.model_type:
            return self.extract_vision_features_paddleocr(inputs)
        else:
            raise ValueError(f"multiple modalities model {self.model_config.model_type} is not supported")

    @paddle.no_grad()
    def prepare_rope3d(
        self, position_ids: paddle.Tensor, max_len_lst: list[int], cumsum_seqlens: list[int]
    ) -> list[paddle.Tensor]:
        """prepare_rope3d"""

        rope_emb_lst = get_rope_3d(
            position_ids=position_ids,
            rotary_dim=self.model_config.head_dim,
            partial_rotary_factor=1.0,
            base=self.model_config.rope_theta,
            max_position=self.model_config.max_model_len,
            freq_allocation=getattr(self.model_config, "freq_allocation", 20),
            model_type=self.model_config.model_type,
            max_len_lst=max_len_lst,
            cumsum_seqlens=cumsum_seqlens,
        )
        return rope_emb_lst

    def _get_prompt_logprobs_list(
        self,
        hidden_states: paddle.Tensor,
    ) -> list[Optional[LogprobsTensors]]:
        if len(self.prompt_logprobs_reqs) > 0:
            assert (
                not self.fd_config.cache_config.enable_prefix_caching
            ), "prompt_logprobs must disable prefix caching, --no-enable-prefix-caching."
        logprobs_mode = self.fd_config.model_config.logprobs_mode
        prompt_logprobs_list: list[Optional[LogprobsTensors]] = self.scheduler_config.max_num_seqs * [None]
        completed_prefill_reqs: list[Request] = []
        for req_id, request in self.prompt_logprobs_reqs.items():
            num_prompt_logprobs = request.sampling_params.prompt_logprobs
            if request.prompt_token_ids is None or num_prompt_logprobs is None:
                continue
            if num_prompt_logprobs == -1:
                num_prompt_logprobs = self.ori_vocab_size

            num_tokens = request.prefill_end_index - request.prefill_start_index
            num_prompt_tokens = len(request.prompt_token_ids)

            logprobs_tensors = self.in_progress_prompt_logprobs.get(req_id)
            if not logprobs_tensors:
                logprobs_tensors = LogprobsTensors.empty_cpu(num_prompt_tokens - 1, num_prompt_logprobs + 1)
                self.in_progress_prompt_logprobs[req_id] = logprobs_tensors
            start_idx = request.prefill_start_index
            start_tok = start_idx + 1
            num_remaining_tokens = num_prompt_tokens - start_tok
            if num_tokens <= num_remaining_tokens:
                # This is a chunk, more tokens remain.
                # In the == case, there are no more prompt logprobs to produce
                # but we want to defer returning them to the next step where we
                # have new generated tokens to return.
                num_logits = num_tokens
            else:
                # This is the last chunk of prompt tokens to return.
                num_logits = num_remaining_tokens
                completed_prefill_reqs.append(request)
                prompt_logprobs_list[request.idx] = logprobs_tensors
            if num_logits <= 0:
                # This can happen for the final chunk if we prefilled exactly
                # (num_prompt_tokens - 1) tokens for this request in the prior
                # step. There are no more prompt logprobs to produce.
                continue
            offset = self.share_inputs["cu_seqlens_q"][request.idx]
            prompt_hidden_states = hidden_states[offset : offset + num_logits]
            logits = self.model.compute_logits(prompt_hidden_states)
            prompt_token_ids = request.prompt_token_ids[start_tok : start_tok + num_logits]
            prompt_token_ids_tensor = paddle.to_tensor(prompt_token_ids, dtype="int64")
            if logprobs_mode == "raw_logprobs":
                raw_logprobs = self.sampler.compute_logprobs(logits)
            elif logprobs_mode == "raw_logits":
                raw_logprobs = logits
            token_ids, logprobs, ranks = self.sampler.gather_logprobs(
                raw_logprobs, num_prompt_logprobs, prompt_token_ids_tensor
            )
            chunk_slice = slice(start_idx, start_idx + num_logits)
            logprobs_tensors.logprob_token_ids[chunk_slice].copy_(token_ids, False)
            logprobs_tensors.logprobs[chunk_slice].copy_(logprobs, False)
            logprobs_tensors.selected_token_ranks[chunk_slice].copy_(ranks, False)

        for req in completed_prefill_reqs:
            del self.prompt_logprobs_reqs[req.request_id]
            del self.in_progress_prompt_logprobs[req.request_id]
        return prompt_logprobs_list
