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

import copy
import os
import queue
import random
import time
from contextlib import contextmanager
from threading import Thread
from typing import List, Optional

import numpy as np
import paddle
import zmq
from paddle import nn

from fastdeploy import envs
from fastdeploy.config import FDConfig
from fastdeploy.engine.request import ImagePosition, Request, RequestType
from fastdeploy.input.ernie4_5_vl_processor import DataProcessor
from fastdeploy.inter_communicator import IPCSignal, ZmqIpcClient
from fastdeploy.model_executor.forward_meta import ForwardMeta
from fastdeploy.model_executor.graph_optimization.utils import (
    profile_run_guard,
    sot_warmup_guard,
)
from fastdeploy.model_executor.layers.attention import get_attention_backend
from fastdeploy.model_executor.layers.attention.base_attention_backend import (
    AttentionBackend,
)
from fastdeploy.model_executor.layers.rotary_embedding import get_rope, get_rope_3d
from fastdeploy.model_executor.layers.sample.meta_data import SamplingMetadata
from fastdeploy.model_executor.layers.sample.sampler import Sampler, SpeculativeSampler
from fastdeploy.model_executor.model_loader import get_model_loader
from fastdeploy.model_executor.models.ernie4_5_vl.modeling_resampler import ScatterOp
from fastdeploy.model_executor.ops.xpu import (
    create_kv_signal_sender,
    destroy_kv_signal_sender,
    recover_decode_task,
    set_data_ipc,
    share_external_data,
    speculate_schedule_cache,
)
from fastdeploy.model_executor.xpu_pre_and_post_process import (
    step_xpu,
    xpu_post_process_normal,
    xpu_post_process_specualate,
    xpu_pre_process,
    xpu_process_output,
)
from fastdeploy.spec_decode import MTPProposer
from fastdeploy.utils import get_logger
from fastdeploy.worker.model_runner_base import ModelRunnerBase
from fastdeploy.worker.output import LogprobsTensors, ModelOutputData, ModelRunnerOutput

logger = get_logger("xpu_model_runner", "xpu_model_runner.log")


@contextmanager
def kv_signal_sender_context_manager(pd_disaggregation_mode):
    sender = None
    try:
        sender = (
            create_kv_signal_sender()
            if pd_disaggregation_mode == "per_chunk" or pd_disaggregation_mode == "per_query"
            else None
        )
        yield sender
    finally:
        if sender is not None:
            destroy_kv_signal_sender(sender)


class XPUModelRunner(ModelRunnerBase):
    """ """

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
        self.enable_early_stop = self.fd_config.early_stop_config.enable_early_stop
        self.enable_logprob = fd_config.model_config.enable_logprob
        self.ori_vocab_size = self.fd_config.model_config.ori_vocab_size
        self.max_logprobs = (
            self.ori_vocab_size if fd_config.model_config.max_logprobs == -1 else fd_config.model_config.max_logprobs
        )

        # VL model config:
        if self.enable_mm:
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

        self.device_id = device_id
        self.speculative_method = self.fd_config.speculative_config.method
        self.speculative_decoding = self.speculative_method is not None

        # used by SamplingMetadata
        self.enable_logprob = fd_config.model_config.enable_logprob  # fd_config.model_config.enable_logprob
        self.enable_early_stop = self.fd_config.early_stop_config.enable_early_stop

        #  Sampler
        #  TODU(lilujia): sync with GPU
        if not self.speculative_decoding:
            self.sampler = Sampler(fd_config)
        else:
            self.sampler = SpeculativeSampler(fd_config)

        # Lazy initialize kv cache after model loading
        # self.kv_caches: list[paddle.Tensor] = []

        # Cuda Graph
        self.graph_opt_level = self.graph_opt_config.graph_opt_level
        self.use_cudagraph = False
        self.sot_warmup_sizes = self.graph_opt_config.sot_warmup_sizes
        self.input_ids = paddle.zeros(self.scheduler_config.max_num_seqs, dtype="int32")

        # Initialize share inputs
        self._init_share_inputs(self.fd_config.scheduler_config.max_num_seqs)
        self.infer_seed_increment = paddle.full(
            shape=[self.scheduler_config.max_num_seqs, 1],
            fill_value=4,
            dtype="int64",
        ).cpu()

        # Initialize attention Backend
        # NOTE(gonshaotian): Currently, all attention layers share one attention backend instance.
        # In the future, we will expand it as a list.
        self.attn_backends: list[AttentionBackend] = []
        self.initialize_attn_backend()

        # Forward meta store the global meta information of the forward
        self.forward_meta: ForwardMeta = None

        # Postprocess Env params
        os.environ["INFERENCE_MSG_QUEUE_ID"] = str(self.parallel_config.local_engine_worker_queue_port)
        logger.info(f"queue id is {str(self.parallel_config.local_engine_worker_queue_port)}")

        self.pd_disaggregation_mode: str = self.fd_config.parallel_config.pd_disaggregation_mode

        # Initialize ZMQ client for async output
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
        # prompt logprobs state
        self.prompt_logprobs_reqs: dict[str, Request] = {}
        self.in_progress_prompt_logprobs: dict[str, LogprobsTensors] = {}

    def _async_output_busy_loop(self):
        """Entrypoint for the thread which handles outputs asynchronously."""
        while True:
            try:
                if self.async_output_queue is None or self.zmq_client is None:
                    break
                output = self.async_output_queue.get()
                if self.zmq_client is not None:
                    self.zmq_client.send_pyobj(output)
            except Exception as e:
                logger.exception("Exception in async output loop: %s", e)

    def _get_prompt_logprobs_list(self, hidden_states: paddle.Tensor) -> list[Optional[LogprobsTensors]]:
        """
        Build prompt_logprobs for requests that asked for it.
        """
        if len(self.prompt_logprobs_reqs) > 0:
            assert (
                not self.fd_config.cache_config.enable_prefix_caching
            ), "prompt_logprobs must disable prefix caching, --no-enable-prefix-caching."

        if len(self.prompt_logprobs_reqs) == 0:
            return self.scheduler_config.max_num_seqs * [None]

        logprobs_mode = self.fd_config.model_config.logprobs_mode
        prompt_logprobs_list: list[Optional[LogprobsTensors]] = self.scheduler_config.max_num_seqs * [None]
        completed_prefill_reqs: list[Request] = []

        for req_id, request in self.prompt_logprobs_reqs.items():
            if not hasattr(request, "sampling_params") or request.sampling_params is None:
                continue
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
                num_logits = num_tokens
            else:
                num_logits = num_remaining_tokens
                completed_prefill_reqs.append(request)
                prompt_logprobs_list[request.idx] = logprobs_tensors
            if num_logits <= 0:
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
            else:
                raw_logprobs = self.sampler.compute_logprobs(logits)
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

    def exist_prefill(self):
        """
        check whether prefill stage exist
        """
        if int(paddle.max(self.share_inputs["seq_lens_encoder"])) != 0:
            return 1
        else:
            return 0

    def only_decode(self):
        """
        Update Batch type for if_only_decode.
        """
        if_only_decode = True
        prefill_exists = None
        if self.fd_config.parallel_config.use_ep and self.fd_config.scheduler_config.splitwise_role == "mixed":
            no_need_stop_list = []
            no_need_stop = self.not_need_stop()
            paddle.distributed.all_gather_object(no_need_stop_list, not no_need_stop)
            if_all_device_empty = all(no_need_stop_list)
            if if_all_device_empty:
                if_only_decode = False
            else:
                only_decode_batch_list = []
                prefill_exists = self.exist_prefill()
                paddle.distributed.all_gather_object(only_decode_batch_list, not prefill_exists)
                if_only_decode = all(only_decode_batch_list)

        if_only_decode = if_only_decode and not (
            prefill_exists if prefill_exists is not None else self.exist_prefill()
        )
        return if_only_decode

    def _process_mm_features(self, request_list: List[Request]):
        """
        Process and cache vision features from model
            - add image_features, extract and cache vision features from model
            - add rope_emb, rotate position embeddings
        """
        if not self.enable_mm:
            return

        self.share_inputs["image_features"] = None
        multi_vision_inputs = {
            "images_lst": [],
            "grid_thw_lst": [],
            "vit_position_ids_lst": [],
            "cu_seqlens": [0],
            "encoder_cache_info": [],
            "feature_position_list": [],
        }
        rope_3d_position_ids = {
            "position_ids_idx": [],
            "position_ids_lst": [],
            "position_ids_offset": [0],
            "max_tokens_lst": [],
        }

        for request in request_list:
            if request.task_type.value != RequestType.PREFILL.value:
                continue

            if self.encoder_cache is not None:
                evict_mm_hashes = request.get("evict_mm_hashes", None)
                if evict_mm_hashes:
                    for mm_hash in evict_mm_hashes:
                        self.encoder_cache.pop(mm_hash, None)

            position_ids = request.multimodal_inputs["position_ids"]
            rope_3d_position_ids["position_ids_idx"].append(request.idx)
            rope_3d_position_ids["position_ids_lst"].append(position_ids)
            rope_3d_position_ids["position_ids_offset"].append(
                position_ids.shape[0] + rope_3d_position_ids["position_ids_offset"][-1]
            )

            # TODO xpu currently do not support pooling model
            # if self.is_pooling_model:
            #     rope_3d_position_ids["max_tokens_lst"].append(0)
            # else:
            rope_3d_position_ids["max_tokens_lst"].append(request.get("max_tokens", 2048))

            if request.with_image:
                inputs = request.multimodal_inputs
                if self.encoder_cache is not None:
                    if envs.FD_ENABLE_MAX_PREFILL:
                        if "vit_seqlen" in inputs:
                            vit_seqlen_list = inputs["vit_seqlen"][request.num_image_start : request.num_image_end]
                        if "vit_position_ids" in inputs:
                            vit_position_ids_list = inputs["vit_position_ids"][
                                request.num_image_start : request.num_image_end
                            ]
                    grid_thw_list = inputs["grid_thw"][request.num_image_start : request.num_image_end]
                    mm_hashes_list = inputs["mm_hashes"][request.num_image_start : request.num_image_end]
                    feature_positions = self._get_feature_positions(
                        mm_positions=inputs["mm_positions"][request.num_image_start : request.num_image_end],
                        prefill_start_index=request.prefill_start_index,
                        prefill_end_index=request.prefill_end_index,
                    )
                    image_start_idx = request.num_image_start

                    logger.debug(
                        f"request {request.request_id} start process encoder info, image_start_idx: {image_start_idx} "
                        f"grid_thw_list: {grid_thw_list}, feature_positions: {feature_positions}, mm_hashes_list: {mm_hashes_list}"
                    )
                    for i, mm_hash in enumerate(mm_hashes_list):
                        image_offset = np.prod(grid_thw_list[i])
                        logger.debug(
                            f"run idx {i} with mm_hash {mm_hash} image_offset: {image_offset} grid_thw: {grid_thw_list[i]}"
                        )
                        if mm_hash in self.encoder_cache:
                            multi_vision_inputs["encoder_cache_info"].append((mm_hash, feature_positions[i], True))
                            continue

                        multi_vision_inputs["encoder_cache_info"].append((mm_hash, feature_positions[i], False))
                        if envs.FD_ENABLE_MAX_PREFILL:
                            multi_vision_inputs["images_lst"].append(
                                inputs["images"][image_start_idx : image_start_idx + image_offset].to(self.device)
                            )
                            multi_vision_inputs["grid_thw_lst"].append(paddle.to_tensor(grid_thw_list[i]))
                            multi_vision_inputs["cu_seqlens"].append(vit_seqlen_list[i])
                            multi_vision_inputs["vit_position_ids_lst"].append(vit_position_ids_list[i])
                        else:
                            multi_vision_inputs["images_lst"].append(
                                paddle.to_tensor(
                                    inputs["images"][image_start_idx : image_start_idx + image_offset],
                                    dtype="uint8" if "ernie" in self.model_config.model_type else "bfloat16",
                                )
                            )
                            multi_vision_inputs["grid_thw_lst"].append(
                                paddle.to_tensor(grid_thw_list[i], dtype=paddle.int64)
                            )
                        image_start_idx += image_offset
                else:
                    if envs.FD_ENABLE_MAX_PREFILL:
                        multi_vision_inputs["images_lst"].append(
                            inputs["images"][request.image_start : request.image_end].to(self.device)
                        )
                        multi_vision_inputs["grid_thw_lst"].extend(
                            paddle.to_tensor(inputs["grid_thw"][request.num_image_start : request.num_image_end])
                        )
                        multi_vision_inputs["cu_seqlens"].extend(
                            inputs["vit_seqlen"][request.num_image_start : request.num_image_end]
                        )
                        multi_vision_inputs["vit_position_ids_lst"].extend(
                            inputs["vit_position_ids"][request.num_image_start : request.num_image_end]
                        )
                    else:
                        multi_vision_inputs["images_lst"].append(
                            paddle.to_tensor(
                                inputs["images"][request.image_start : request.image_end],
                                dtype="uint8" if "ernie" in self.model_config.model_type else "bfloat16",
                            )
                        )
                        multi_vision_inputs["grid_thw_lst"].extend(
                            paddle.to_tensor(
                                inputs["grid_thw"][request.num_image_start : request.num_image_end],
                                dtype=paddle.int64,
                            )
                        )

                    multi_vision_inputs["feature_position_list"].extend(
                        self._get_feature_positions(
                            mm_positions=inputs["mm_positions"][request.num_image_start : request.num_image_end],
                            prefill_start_index=request.prefill_start_index,
                            prefill_end_index=request.prefill_end_index,
                        )
                    )

        if self.encoder_cache is not None:
            if len(multi_vision_inputs["images_lst"]) > 0 or len(multi_vision_inputs["encoder_cache_info"]) > 0:
                image_features_output = None
                if len(multi_vision_inputs["images_lst"]) > 0:
                    image_features_output = self.extract_vision_features(multi_vision_inputs)

                logger.debug(f"encoder_cache_info: {multi_vision_inputs['encoder_cache_info']}")
                merge_image_features, feature_idx, thw_idx = [], 0, 0
                for mm_hash, feature_position, use_cache in multi_vision_inputs["encoder_cache_info"]:
                    if use_cache:
                        assert mm_hash in self.encoder_cache, f"{mm_hash} not in encoder cache"
                        mm_feature = self.encoder_cache[mm_hash].cuda()
                    else:
                        assert (
                            image_features_output is not None
                        ), f"image_features_output is None, images_lst length: {len(multi_vision_inputs['images_lst'])}"
                        grid_thw = multi_vision_inputs["grid_thw_lst"][thw_idx]
                        mm_token_lenght = inputs["mm_num_token_func"](grid_thw=grid_thw)
                        mm_feature = image_features_output[feature_idx : feature_idx + mm_token_lenght]

                        # add feature to encoder cache
                        self.encoder_cache[mm_hash] = mm_feature.detach().cpu()
                        feature_idx += mm_token_lenght
                        thw_idx += 1

                    feature_start = feature_position.offset
                    feature_end = feature_position.offset + feature_position.length
                    merge_image_features.append(mm_feature[feature_start:feature_end])

                self.share_inputs["image_features"] = paddle.concat(merge_image_features, axis=0)
                logger.debug(
                    f"merge_image_features length: {len(merge_image_features)}, features shape: {self.share_inputs['image_features'].shape}"
                )
        elif len(multi_vision_inputs["images_lst"]) > 0:
            assert len(multi_vision_inputs["feature_position_list"]) == len(
                multi_vision_inputs["grid_thw_lst"]
            ), f"{multi_vision_inputs['feature_position_list']} != {multi_vision_inputs['grid_thw_lst']}"

            merge_image_features, feature_idx, thw_idx = [], 0, 0
            image_features_output = self.extract_vision_features(multi_vision_inputs)
            for feature_position in multi_vision_inputs["feature_position_list"]:
                grid_thw = multi_vision_inputs["grid_thw_lst"][thw_idx]
                mm_token_lenght = inputs["mm_num_token_func"](grid_thw=grid_thw)
                mm_feature = image_features_output[feature_idx : feature_idx + mm_token_lenght]

                feature_start = feature_position.offset
                feature_end = feature_position.offset + feature_position.length
                merge_image_features.append(mm_feature[feature_start:feature_end])
                feature_idx += mm_token_lenght
                thw_idx += 1
            self.share_inputs["image_features"] = paddle.concat(merge_image_features, axis=0)

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

    def _get_feature_positions(
        self, mm_positions: List[ImagePosition], prefill_start_index: int, prefill_end_index: int
    ):
        """
        Filter and adjust ImagePosition objects that fall within the specified prefill range.

        Args:
            mm_positions: List of ImagePosition objects to filter
            prefill_start_index: Start index of the prefill range
            prefill_end_index: End index of the prefill range

        Returns:
            List of ImagePosition objects that are within or intersect with the prefill range
        """
        feature_positions = []
        for position in mm_positions:
            position_start = position.offset
            position_end = position.offset + position.length
            if position_end <= prefill_start_index or position_start >= prefill_end_index:
                continue
            elif position_start >= prefill_start_index and position_end <= prefill_end_index:
                new_position = copy.deepcopy(position)
                new_position.offset = 0
                feature_positions.append(new_position)
            else:
                new_position = copy.deepcopy(position)
                # Adjust offset if it starts before prefill_start_index
                if position_start < prefill_start_index:
                    new_position.offset = prefill_start_index - position_start
                    new_position.length = min(position_end, prefill_end_index) - prefill_start_index
                # Adjust length if it extends beyond prefill_end_index
                elif position_end > prefill_end_index:
                    new_position.offset = 0
                    new_position.length = prefill_end_index - position_start
                feature_positions.append(new_position)

        logger.debug(
            f"get feature_positions, original positions: {mm_positions}, filtered positions: {feature_positions}"
        )
        return feature_positions

    def insert_tasks_v1(self, req_dicts: List[Request], num_running_requests: int):
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

        for i in range(req_len):
            request = req_dicts[i]
            idx = request.idx
            if request.task_type.value == RequestType.PREFILL.value:  # prefill task
                self.share_inputs["preempted_idx"][idx : idx + 1, :] = 0
                prefill_start_index = request.prefill_start_index
                prefill_end_index = request.prefill_end_index
                length = prefill_end_index - prefill_start_index
                if request.get("enable_thinking", False) and request.get("reasoning_max_tokens", None) is not None:
                    # Enable thinking
                    self.share_inputs["max_think_lens"][idx : idx + 1, :] = request.get("reasoning_max_tokens")
                    self.share_inputs["limit_think_status"][idx : idx + 1, :] = 0
                else:
                    # Disable thinking
                    self.share_inputs["max_think_lens"][idx : idx + 1, :] = -1
                    self.share_inputs["limit_think_status"][idx : idx + 1, :] = 0

                if (
                    hasattr(request, "sampling_params")
                    and request.sampling_params is not None
                    and request.sampling_params.prompt_logprobs is not None
                ):
                    self.prompt_logprobs_reqs[request.request_id] = request

                if len(request.output_token_ids) == 0:
                    input_ids = request.prompt_token_ids
                else:
                    input_ids = request.prompt_token_ids + request.output_token_ids
                logger.debug(
                    f"Handle prefill request {request} at idx {idx} prefill_start_index {prefill_start_index} prefill_end_index {prefill_end_index} need_prefilled_token_num {len(input_ids)}"
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
                self.share_inputs["seq_lens_this_time"][idx : idx + 1] = length
                self.share_inputs["seq_lens_encoder"][idx : idx + 1] = length
                self.share_inputs["step_seq_lens_decoder"][idx : idx + 1] = 0
                self.share_inputs["prompt_lens"][idx : idx + 1] = len(input_ids)
                self.share_inputs["is_block_step"][idx : idx + 1] = False
                self.share_inputs["step_idx"][idx : idx + 1] = (
                    len(request.output_token_ids) if prefill_end_index >= len(input_ids) else 0
                )
                self.share_inputs["pre_ids"][idx : idx + 1] = -1
                if (
                    self.fd_config.scheduler_config.splitwise_role == "decode"
                ):  # In PD, we continue to decode after P generate first token
                    self.share_inputs["seq_lens_encoder"][idx : idx + 1] = 0
                has_prefill_task = True
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
                self.share_inputs["preempted_idx"][idx : idx + 1, :] = 0
                continue
            else:  # preempted task
                logger.debug(f"Handle preempted request {request} at idx {idx}")
                self.share_inputs["preempted_idx"][idx : idx + 1, :] = 1
                self.share_inputs["block_tables"][idx : idx + 1, :] = -1
                self.share_inputs["stop_flags"][idx : idx + 1] = True
                self.share_inputs["seq_lens_this_time"][idx : idx + 1] = 0
                self.share_inputs["seq_lens_decoder"][idx : idx + 1] = 0
                self.share_inputs["seq_lens_encoder"][idx : idx + 1] = 0
                self.share_inputs["is_block_step"][idx : idx + 1] = False
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

        self._process_mm_features(req_dicts)
        if has_prefill_task or has_decode_task:
            self.share_inputs["not_need_stop"][0] = True

        if self.speculative_method in ["mtp"]:
            self.proposer.insert_tasks_v1(req_dicts, num_running_requests)

    def insert_prefill_inputs(self, req_dicts: List[Request], num_running_requests: int):
        """Process inputs for prefill tasks and update share_inputs buffer"""
        # NOTE(luotingdan): Set environment variable of prefill node
        if req_dicts[-1].disaggregate_info is not None and req_dicts[-1].disaggregate_info["role"] == "prefill":
            os.environ["PREFILL_NODE_ONE_STEP_STOP"] = "1"

        req_len = len(req_dicts)
        for i in range(req_len):
            request = req_dicts[i]
            idx = request.idx
            length = len(request.prompt_token_ids)
            assert length > 0, "The prompt requested must not be empty."

            # Is Decode Node
            if req_dicts[i].disaggregate_info is not None and req_dicts[i].disaggregate_info["role"] == "decode":
                self.share_inputs["pre_ids"][idx : idx + 1] = request.prompt_token_ids[-1]
                self.share_inputs["input_ids"][idx : idx + 1, 0] = request.prompt_token_ids[0]
                self.share_inputs["prompt_ids"][idx : idx + 1, :length] = np.array(request.prompt_token_ids)
                self.share_inputs["seq_lens_encoder"][idx : idx + 1] = 0
                self.share_inputs["seq_lens_decoder"][idx : idx + 1] = length
                self.share_inputs["seq_lens_this_time"][idx : idx + 1] = 1
                self.share_inputs["step_seq_lens_encoder"][idx : idx + 1] = 0
                self.share_inputs["step_seq_lens_decoder"][idx : idx + 1] = length
                self.share_inputs["prompt_lens"][idx : idx + 1] = length
                self.share_inputs["step_idx"][idx : idx + 1] = 1

                # TODO support MTP
                # if self.speculative_decoding:
                #     num_prefill_send_token = self.speculative_config.num_speculative_tokens + 1
                #     self.share_inputs["draft_tokens"][idx : idx + 1, 0:num_prefill_send_token] = paddle.to_tensor(
                #         request.draft_token_ids[0:num_prefill_send_token],
                #         dtype="int64",
                #     )
                #     self.seq_lens_this_time_buffer[idx : idx + 1] = num_prefill_send_token
            else:
                self.share_inputs["pre_ids"][idx : idx + 1] = -1
                self.share_inputs["step_idx"][idx : idx + 1] = 0
                self.share_inputs["input_ids"][idx : idx + 1, :length] = np.array(request.prompt_token_ids)
                self.share_inputs["prompt_ids"][idx : idx + 1, :length] = np.array(request.prompt_token_ids)
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
                self.share_inputs["seq_lens_this_time"][idx : idx + 1] = length
                self.share_inputs["step_seq_lens_encoder"][idx : idx + 1] = length
                self.share_inputs["seq_lens_encoder"][idx : idx + 1] = length
                self.share_inputs["prompt_lens"][idx : idx + 1] = length

                if self.enable_mm:
                    self.share_inputs["rope_emb"][idx : idx + 1, :] = self.prepare_rope3d(
                        position_ids, [request.get("max_tokens", 2048)], [0, position_ids.shape[0]]
                    )[0]
                    self.share_inputs["seq_lens_decoder"][idx : idx + 1] = 0

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

        self.share_inputs["not_need_stop"][0] = True

        if self.speculative_method in ["mtp"]:
            self.share_inputs["temp_scaled_logprobs"][idx : idx + 1] = get_attr_from_request(
                request, "temp_scaled_logprobs", False
            )
            self.share_inputs["top_p_normalized_logprobs"][idx : idx + 1] = get_attr_from_request(
                request, "top_p_normalized_logprobs", False
            )
            self.proposer.insert_prefill_inputs(req_dicts, num_running_requests)

    def _init_share_inputs(self, max_num_seqs: int):
        """Initialize all share buffers for model inputs.
        Note: In the future, we may abandon share buffers.
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
        # self.share_inputs["top_p"] = paddle.full([max_num_seqs, 1], self.model_config.top_p, dtype="float32")
        # self.share_inputs["top_p"] default to 0.0 on XPU for consideration of the performance
        self.share_inputs["top_p"] = paddle.full([max_num_seqs, 1], 0.0, dtype="float32")
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
        self.share_inputs["seq_lens_this_time"] = paddle.full(max_num_seqs, 0, dtype="int32")
        self.share_inputs["seq_lens_encoder"] = paddle.full([max_num_seqs, 1], 0, dtype="int32")
        self.share_inputs["seq_lens_decoder"] = paddle.full([max_num_seqs, 1], 0, dtype="int32")
        self.share_inputs["step_seq_lens_encoder"] = paddle.full([max_num_seqs, 1], 0, dtype="int32")
        self.share_inputs["step_seq_lens_decoder"] = paddle.full([max_num_seqs, 1], 0, dtype="int32")
        self.share_inputs["prompt_lens"] = paddle.full([max_num_seqs, 1], 0, dtype="int64")
        self.share_inputs["step_idx"] = paddle.full([max_num_seqs, 1], 0, dtype="int64")
        self.share_inputs["not_need_stop"] = paddle.full(
            [1], False, dtype="bool"
        ).cpu()  # TODO(gongshaotian): move to pinnd memory
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
        self.share_inputs["infer_seed"] = paddle.full([max_num_seqs, 1], 0, dtype="int64")
        self.share_inputs["first_token_ids"] = paddle.full([max_num_seqs, 1], -1, dtype="int64")
        self.share_inputs["ori_seq_lens_encoder"] = paddle.full([max_num_seqs, 1], 0, dtype="int32")
        self.share_inputs["system_lens"] = paddle.full([max_num_seqs, 1], 0, dtype="int32")
        self.share_inputs["system_ids"] = paddle.full([max_num_seqs, 1], -1, dtype="int32")

        self.share_inputs["ids_remove_padding"] = paddle.full(
            [max_num_seqs * self.model_config.max_model_len],
            0,
            dtype="int64",
        )
        self.share_inputs["batch_id_per_token"] = paddle.full([max_num_seqs, 1], 0, dtype="int32")
        self.share_inputs["cu_seqlens_q"] = paddle.full([max_num_seqs, 1], 0, dtype="int32")
        self.share_inputs["cu_seqlens_k"] = paddle.full([max_num_seqs, 1], 0, dtype="int32")

        # Initialize thinking related buffers
        self.share_inputs["max_think_lens"] = paddle.full(shape=[max_num_seqs, 1], fill_value=-1, dtype="int32")
        self.share_inputs["limit_think_status"] = paddle.full(shape=[max_num_seqs, 1], fill_value=0, dtype="int32")

        # Initialize rotary position embedding
        tmp_position_ids = paddle.arange(self.model_config.max_model_len).reshape((1, -1))

        # TODO(gongshaotian): move to models
        if not self.enable_mm:
            self.share_inputs["rope_emb"] = get_rope(
                rotary_dim=self.model_config.head_dim,
                position_ids=tmp_position_ids,
                base=self.model_config.rope_theta,
                model_config=self.model_config,
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

        if self.enable_mm:
            head_dim = self.model_config.head_dim
            if "paddleocr" in self.model_config.model_type:  # neox style = True
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
            self.share_inputs["temp_scaled_logprobs"] = paddle.full([max_num_seqs, 1], False, dtype=bool)
            self.share_inputs["top_p_normalized_logprobs"] = paddle.full([max_num_seqs, 1], False, dtype=bool)
            # For MTP Logprob
            self.share_inputs["draft_logits"] = paddle.full(
                [max_num_seqs * (self.speculative_config.num_speculative_tokens + 1), self.model_config.vocab_size],
                -1,
                dtype="float32",
            )
            self.share_inputs["cu_batch_token_offset"] = paddle.full(
                shape=[max_num_seqs + 1], fill_value=0, dtype="int32"
            )
        self.max_num_seqs = max_num_seqs
        self.share_inputs["preempted_idx"] = paddle.full(shape=[max_num_seqs, 1], fill_value=0, dtype="int32").cpu()

    def _prepare_inputs(self, is_dummy_run=False) -> None:
        """Prepare the model inputs"""
        if envs.ENABLE_V1_KVCACHE_SCHEDULER and not is_dummy_run:
            recover_decode_task(
                self.share_inputs["stop_flags"],
                self.share_inputs["seq_lens_this_time"],
                self.share_inputs["seq_lens_encoder"],
                self.share_inputs["seq_lens_decoder"],
                self.share_inputs["step_seq_lens_decoder"],
                self.share_inputs["block_tables"],
                self.share_inputs["is_block_step"],
                self.cache_config.block_size,
            )
        self.forward_meta = xpu_pre_process(
            self.share_inputs["input_ids"],
            self.share_inputs["seq_lens_this_time"],
            self.share_inputs,
            use_speculate_method=self.speculative_decoding,
            block_size=self.cache_config.block_size,
            draft_tokens=self.share_inputs["draft_tokens"] if self.speculative_decoding else None,
            seq_lens_encoder=self.share_inputs["seq_lens_encoder"],
            seq_lens_decoder=self.share_inputs["seq_lens_decoder"],
            is_profiling=is_dummy_run,
        )
        # Update bad tokens len
        max_bad_tokens_len = paddle.max(self.share_inputs["bad_tokens_len"])

        self.forward_meta.attn_backend = self.attn_backends[0]
        self.initialize_attention_backend()

        if self.pd_disaggregation_mode == "per_chunk" or self.pd_disaggregation_mode == "per_query":
            self.forward_meta.kv_signal_sender = self.share_inputs["kv_signal_sender"]

        if (
            self.fd_config.scheduler_config.splitwise_role == "mixed"
        ):  # Centralized scenario: the phase is initialized as "prefill" by default. During inference runtime, different types of batches can achieve phase switching at this point.
            if_only_decode = self.only_decode()
            self.fd_config.model_config.moe_phase.phase = "decode" if if_only_decode else "prefill"

        # Get sampling metadata
        # TODU(lilujia): sync with GPU
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
            share_inputs=self.share_inputs,
        )

    def load_model(self) -> None:
        """load or download model"""
        logger.info(f"Starting to load model {self.model_config.architectures[0]}")
        # 1. Load original model
        model_loader = get_model_loader(load_config=self.fd_config.load_config)
        self.model = model_loader.load_model(fd_config=self.fd_config)

        # 2. Load lora model

        # 3. Load drafter model(for speculative decoding)
        self._init_speculative_proposer()

    def get_model(self) -> nn.Layer:
        """Get current model"""
        return self.model

    def initialize_attention_backend(self):
        """
        Initialize attention meta data
        """
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

        if (
            self.quant_config
            and hasattr(self.quant_config, "kv_cache_quant_type")
            and self.quant_config.kv_cache_quant_type is not None
        ):
            cache_type = "int8"

        # Get kv cache shape
        key_cache_shape, value_cache_shape = self.attn_backends[0].get_kv_cache_shape(max_num_blocks=max_block_num)
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
        # 2. GPU runner creates kv cache tensor unless p/d disaggregation is enabled.
        create_cache_tensor = profile or self.scheduler_config.splitwise_role == "mixed"
        if not create_cache_tensor:
            logger.info(f"Waiting for cache managers to create kv cache.. {cache_ready_signal.value}")
            while cache_ready_signal.value[local_rank] != 1:
                time.sleep(1)
            logger.info(f"OK! Stop waiting. {cache_ready_signal.value}")

        logger.info(f"Initializing kv cache for all layers. {cache_ready_signal.value}")
        cache_kvs_list = []

        for i in range(self.model_config.num_hidden_layers):
            key_cache_name = f"key_caches_{i}_rank{local_rank}.device{self.device_id}"
            val_cache_name = f"value_caches_{i}_rank{local_rank}.device{self.device_id}"

            if create_cache_tensor:
                logger.info(f"..creating kv cache for layer {i}: {key_cache_shape} {value_cache_shape}")
                key_cache = paddle.full(shape=key_cache_shape, fill_value=0, dtype=cache_type)
                set_data_ipc(key_cache, key_cache_name)
                val_cache = paddle.full(shape=value_cache_shape, fill_value=0, dtype=cache_type)
                set_data_ipc(val_cache, val_cache_name)
                cache_kvs_list.extend([key_cache, val_cache])

            else:
                logger.info(f"..attaching kv cache for layer {i}: {key_cache_shape} {value_cache_shape}")
                key_cache = paddle.empty(shape=[], dtype=cache_type)
                key_cache = share_external_data(key_cache, key_cache_name, key_cache_shape, False)
                val_cache = paddle.empty(shape=[], dtype=cache_type)
                val_cache = share_external_data(val_cache, val_cache_name, value_cache_shape, False)
                cache_kvs_list.extend([key_cache, val_cache])

        self.share_inputs["caches"] = cache_kvs_list

        if not profile and create_cache_tensor:
            cache_ready_signal.value[local_rank] = 1
            logger.info(f"✅ kv cache is ready! {cache_ready_signal.value}")

        paddle.device.xpu.empty_cache()

    def initialize_attn_backend(self) -> None:
        """
        Initialize attention backends and forward metadata
        """
        assert len(self.attn_backends) == 0

        # TODO(gongshaotian): Get rank from config
        num_heads = self.model_config.num_attention_heads // self.parallel_config.tensor_parallel_size
        self.model_config.kv_num_heads = (
            int(self.model_config.num_key_value_heads) // self.parallel_config.tensor_parallel_size
        )
        head_dim = self.model_config.head_dim

        if self.speculative_decoding:
            # Initialize AttentionBackend buffers
            encoder_block_shape_q = 64
            decoder_block_shape_q = 16
            decoder_step_token_num = self.speculative_config.num_speculative_tokens + 1
            decode_max_tile_size = self.max_num_seqs * np.ceil(
                (decoder_step_token_num * np.ceil(num_heads / self.model_config.kv_num_heads)) / decoder_block_shape_q
            )

            group_size = np.ceil(num_heads / self.model_config.kv_num_heads)
            encode_max_tile_size = self.scheduler_config.max_num_seqs * np.ceil(
                (self.model_config.max_model_len * group_size) / encoder_block_shape_q
            )
            kv_max_tile_size = self.scheduler_config.max_num_seqs * np.ceil(
                self.model_config.max_model_len / self.fd_config.cache_config.block_size
            )
            self.share_inputs["decoder_batch_ids"] = paddle.full([int(decode_max_tile_size)], 0, dtype="int32")
            self.share_inputs["decoder_tile_ids_per_batch"] = paddle.full(
                [int(decode_max_tile_size)], 0, dtype="int32"
            )
            self.share_inputs["decoder_num_blocks_cpu"] = paddle.full([1], 0, dtype="int32").cpu()
            # NOTE: (changwenbin) MLA kernel only needs decoder_num_blocks_device in place of GPU tensor,
            # adapted to cudagraph.
            self.share_inputs["decoder_num_blocks_device"] = paddle.full([1], 0, dtype="int32")
            self.share_inputs["decoder_chunk_size_device"] = paddle.full([1], 64, dtype="int32")
            self.share_inputs["max_len_tensor_cpu"] = paddle.full([8], 0, dtype="int32").cpu()

            self.share_inputs["encoder_batch_ids"] = paddle.full([int(encode_max_tile_size)], 0, dtype="int32")
            self.share_inputs["encoder_tile_ids_per_batch"] = paddle.full(
                [int(encode_max_tile_size)], 0, dtype="int32"
            )
            self.share_inputs["encoder_num_blocks_x_cpu"] = paddle.full([1], 0, dtype="int32").cpu()

            self.share_inputs["kv_batch_ids"] = paddle.full([int(kv_max_tile_size)], 0, dtype="int32")
            self.share_inputs["kv_tile_ids_per_batch"] = paddle.full([int(kv_max_tile_size)], 0, dtype="int32")
            self.share_inputs["kv_num_blocks_x_cpu"] = paddle.full([1], 0, dtype="int32").cpu()
            self.share_inputs["max_len_kv_cpu"] = paddle.full([1], 0, dtype="int32").cpu()

        # Get the attention backend
        attn_cls = get_attention_backend()
        attn_backend = attn_cls(
            self.fd_config,
            kv_num_heads=self.model_config.kv_num_heads,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        if attn_backend is None:
            raise NotImplementedError(
                "Attention backend which you specified is not supported, please set FD_ATTENTION_BACKEND correctly."
            )
        self.attn_backends.append(attn_backend)

    def get_input_length_list(self, num_tokens: int, batch_size: int, expected_decode_len: int):
        """
        Args:
            num_tokens (int): The total number of tokens across all sequences.
            batch_size (int): The number of sequences (requests) in the batch.
            expected_decode_len (int): The expected number of tokens every sequence should be generated by the model.
        Returns:
            List[int]: A list of integers representing the sequence length for each request.
                    This list is crafted to maximize the total number of blocks.
        """
        max_dec_len = expected_decode_len + 1
        input_length = min(num_tokens // batch_size, self.model_config.max_model_len - max_dec_len)
        block_num = (
            input_length + self.cache_config.block_size - 1
        ) // self.cache_config.block_size + self.cache_config.enc_dec_block_num
        input_length_list = [input_length] * batch_size
        len_of_input_length_list = len(input_length_list)
        max_dec_len_list = [max_dec_len] * len_of_input_length_list
        return input_length_list, max_dec_len_list, block_num

    def _dummy_prefill_inputs(self, input_length_list: List[int], max_dec_len_list: List[int], block_num: int):
        """Set dummy prefill inputs to share_inputs"""
        batch_size = len(input_length_list)

        for i in range(batch_size):
            idx = i
            input_length = input_length_list[idx]
            max_dec_len = max_dec_len_list[idx]
            self.share_inputs["input_ids"][idx : idx + 1, :input_length] = np.array([5] * input_length)
            self.share_inputs["prompt_ids"][idx : idx + 1, :input_length] = np.array([5] * input_length)
            self.share_inputs["eos_token_id"][:] = np.array([2], dtype="int64").reshape(-1, 1)
            self.share_inputs["seq_lens_this_time"][idx : idx + 1] = input_length

            self.share_inputs["step_seq_lens_encoder"][idx : idx + 1] = input_length
            self.share_inputs["seq_lens_encoder"][idx : idx + 1] = input_length
            self.share_inputs["seq_lens_decoder"][idx : idx + 1] = 0
            self.share_inputs["step_idx"][idx : idx + 1] = 0
            self.share_inputs["max_dec_len"][idx : idx + 1] = max_dec_len
            self.share_inputs["stop_flags"][idx : idx + 1] = False

            self.share_inputs["first_token_ids"][idx : idx + 1] = self.share_inputs["input_ids"][idx : idx + 1, :1]
            self.share_inputs["ori_seq_lens_encoder"][idx : idx + 1] = input_length

            self.share_inputs["infer_seed"][idx : idx + 1] = random.randint(0, 922337203685477580)
            self.share_inputs["encoder_block_lens"][idx : idx + 1] = block_num
            self.share_inputs["block_tables"][idx : idx + 1, :block_num] = np.arange(
                idx * block_num, (idx + 1) * block_num, 1
            )

    def _dummy_run(
        self,
        num_tokens: paddle.Tensor,
        batch_size: paddle.Tensor,
        expected_decode_len: int = 1,
        in_capturing: bool = False,
    ) -> paddle.Tensor:
        """
        Use dummy inputs to run before formal execution.
        Args:
            num_tokens: Number of the input tokens
            batch_size: Batch size
            expected_decode_len: Expected decode length
            in_capturing: Is cuda graph in capturing state
        """
        input_length_list, max_dec_len_list, block_num = self.get_input_length_list(
            num_tokens=num_tokens,
            batch_size=batch_size,
            expected_decode_len=expected_decode_len,
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
                expected_decode_len=1,
            )

        while True:
            self.execute_model(is_dummy_run=True, in_capturing=in_capturing)

            if int((self.share_inputs["seq_lens_this_time"] > 0).sum()) == 0:
                break

    def _init_speculative_proposer(self):
        """
        Init speculative proposer
        """
        if self.speculative_method == "ngram":
            # xpu not support ngram proposer now
            # self.proposer = NgramProposer(self.fd_config)
            self.proposer = None
        elif self.speculative_method == "mtp":
            self.proposer = MTPProposer(
                self.fd_config,
                self.get_model(),
                self.local_rank,
                self.device_id,
                self.share_inputs,
            )
        else:
            self.proposer = None

    def _set_debug_level(
        self, debug_level: int = 0x1, model_forward_batch: Optional[List[Request]] = None, is_dummy_run: bool = False
    ) -> None:
        """
        Set debug level for XPU: 0x1, 0xA1, 0x1B1
        """
        request_num = 0 if model_forward_batch is None else len(model_forward_batch)
        if debug_level == 0 or request_num == 0 or is_dummy_run:
            paddle.device.xpu.set_debug_level(0)
            return

        if self.parallel_config.use_ep:
            request_num = paddle.to_tensor(request_num, dtype="int32")
            paddle.distributed.all_reduce(request_num, group=self.parallel_config.ep_group)
            logger.info(f"local_rank: {self.local_rank}, request_num: {request_num.item()}")
            if request_num.item() > 0:
                paddle.device.xpu.set_debug_level(debug_level)
        else:
            paddle.device.xpu.set_debug_level(debug_level)

    def capture_model(self) -> None:
        """
        Trigger CUDA Graph capture for all shapes in 'CudaGraphConfig.cudagraph_capture_sizes'
        """
        logger.warn("XPU not support cuda graph currently")
        pass

    @sot_warmup_guard(True)
    def sot_warmup(self) -> None:
        start_time = time.perf_counter()
        for batch_size in self.sot_warmup_sizes:
            self._dummy_run(
                num_tokens=self.parallel_config.max_num_batched_tokens,
                batch_size=batch_size,
            )
            logger.info(f"SOT warmup the model with the batch size:{batch_size}")
        logger.info(f"SOT warmup took {time.perf_counter() - start_time} seconds")

    def execute_model(
        self,
        model_forward_batch: Optional[List[Request]] = None,
        num_running_requests: int = None,
        is_dummy_run: bool = False,
        in_capturing: bool = False,
    ) -> Optional[ModelRunnerOutput]:
        """
        The Entrance of model execute.
        Args:
            model_forward_batch: 'Request' contains information related to prompt and is an abstract
            class at the server level, which is too granular for ModelRunner.
            We plan to replace it with 'ModelForwardBatch'.
            num_running_requests: batch_size
            intermediate_tensors:
        """
        # 0. set debug level
        # self._set_debug_level(0x1, model_forward_batch, is_dummy_run)
        with kv_signal_sender_context_manager(self.pd_disaggregation_mode) as sender:

            self.share_inputs["kv_signal_sender"] = sender
            # 1. Prepare inputs of model and decoder.
            self._prepare_inputs(is_dummy_run=is_dummy_run)

            # NOTE(wufeisheng): If `not_need_stop`` is False, it means the current worker is in an idle state.
            # This logic is not used in TP (Tensor Parallelism) mode. However, in EP (Expert Parallelism) mode,
            # when there is data on other runner, the current runner is required to execute part of the model.
            if not self.not_need_stop() and not is_dummy_run:
                self._execute_empty_input(self.forward_meta)
                return None

            # 2. Padding inputs for cuda grph

            # 3. Execute model
            if self.enable_mm:
                model_output = self.model(
                    self.share_inputs["ids_remove_padding"], self.share_inputs["image_features"], self.forward_meta
                )
            else:
                model_output = self.model(
                    ids_remove_padding=self.share_inputs["ids_remove_padding"],
                    forward_meta=self.forward_meta,
                )

            hidden_states = xpu_process_output(
                model_output, self.share_inputs["cum_offsets"], self.forward_meta, self.share_inputs
            )
            # 4. Compute logits, Sample
            logits = self.model.compute_logits(hidden_states)
            sampler_output = None
            if not self.speculative_decoding:
                sampler_output = self.sampler(logits, self.sampling_metadata)
            else:
                self.sampler(
                    logits,
                    self.sampling_metadata,
                    self.model_config.max_model_len,
                    self.share_inputs,
                )

            prompt_logprobs_list = None
            if not self.speculative_decoding:
                prompt_logprobs_list = self._get_prompt_logprobs_list(model_output)

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
                # 投机解码
                full_hidden_states=model_output if self.speculative_decoding else None,
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
                min_tokens=self.share_inputs["min_dec_len"],
                prompt_logprobs_list=prompt_logprobs_list,
            )
            if self.speculative_decoding:
                # base model post process
                xpu_post_process_specualate(model_output_data, False, is_dummy_run)
            else:
                xpu_post_process_normal(
                    sampler_output=sampler_output,
                    model_output=model_output_data,
                    share_inputs=self.share_inputs,
                    block_size=self.cache_config.block_size,
                    skip_save_output=is_dummy_run,
                    save_each_rank=self.parallel_config.data_parallel_size > 0,
                    async_output_queue=self.async_output_queue,
                    think_end_id=self.model_config.think_end_id,
                    line_break_id=self.model_config.line_break_id,
                )

            # 6. Draft model propose
            if self.speculative_method == "mtp":
                self.proposer.run(full_hidden_states=model_output)

            # 7. Updata 'infer_seed' and step_paddle()
            self.share_inputs["infer_seed"].add_(self.infer_seed_increment)
            self.share_inputs["infer_seed"][:] %= self.MAX_INFER_SEED

            if not envs.ENABLE_V1_KVCACHE_SCHEDULER:
                step_xpu(
                    self.share_inputs,
                    self.cache_config.block_size,
                    self.cache_config.enc_dec_block_num,
                    self.fd_config.speculative_config,
                    self.fd_config.cache_config.enable_prefix_caching,
                )
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

    def _execute_empty_input(self, forward_meta) -> None:
        """
        In certain scenarios, such as during EP,
        the runner needs to execute partial modules of the model without input data.
        This requires the model to implement the `empty_input_forward` method.
        """
        if hasattr(self.model, "empty_input_forward"):
            self.model.empty_input_forward(forward_meta)
        else:
            raise ValueError(f"{type(self.model)} has no attribute 'empty_input_forward")

    @profile_run_guard(True)
    def profile_run(self) -> None:
        """Execute a forward pass with dummy inputs to profile the memory usage of the model"""

        self.num_gpu_blocks = self.cache_config.total_block_num
        if self.speculative_method in ["mtp"]:
            self.proposer.initialize_kv_cache(main_model_num_blocks=self.num_gpu_blocks, profile=True)
        self.initialize_kv_cache(profile=True)

        self._dummy_run(
            num_tokens=int(self.scheduler_config.max_num_batched_tokens),
            batch_size=min(self.scheduler_config.max_num_seqs, 1),
        )

    def update_share_input_block_num(self, num_gpu_blocks: int) -> None:
        """
        Set a globally unified block number and update the model's shared input.
        Args:
            num_gpu_blocks:
        """
        self.num_gpu_blocks = num_gpu_blocks

        # Reset block table and kv cache with global block num
        if self.speculative_method in ["mtp"]:
            self.proposer.initialize_kv_cache(main_model_num_blocks=self.num_gpu_blocks)
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

    def clear_block_table(self) -> None:
        """
        Clear the block tables and kv cache after profiling.
        """
        if hasattr(self.share_inputs, "caches"):
            del self.share_inputs["caches"]
        if self.forward_meta is not None:
            del self.forward_meta.caches
        paddle.device.xpu.empty_cache()

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
        num_layers = self.model_config.num_hidden_layers
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
            images = paddle.to_tensor(images, dtype="uint8")
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

    def extract_vision_features_ernie(self, vision_inputs: dict[str, list[paddle.Tensor]]) -> paddle.Tensor:
        """
        vision feature extactor for ernie-vl
        """
        assert len(vision_inputs["images_lst"]) > 0, "at least one image needed"

        grid_thw = paddle.to_tensor(vision_inputs["grid_thw_lst"], dtype=paddle.int64)
        # ernie-vl has images norm
        images = paddle.concat(vision_inputs["images_lst"]).cast("float32")
        images = self.image_preprocess.rescale_factor * images - self.image_preprocess.image_mean_tensor
        images = images / self.image_preprocess.image_std_tensor
        images = images.cast("bfloat16")

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
                grid_thw,
            )
        return image_features

    def extract_vision_features_paddleocr(self, inputs: dict[str, list[paddle.Tensor]]) -> paddle.Tensor:
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
    def extract_vision_features(self, multi_vision_inputs: dict[str, list[paddle.Tensor]]) -> paddle.Tensor:
        """extract_vision_features"""
        if "ernie" in self.model_config.model_type:
            return self.extract_vision_features_ernie(multi_vision_inputs)
        # TODO support VL
        # elif "qwen" in self.model_config.model_type:
        #     return self.extract_vision_features_qwen(multi_vision_inputs)
        elif "paddleocr" in self.model_config.model_type:
            return self.extract_vision_features_paddleocr(multi_vision_inputs)
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
            rope_scaling=getattr(self.model_config, "rope_scaling", {}),
            model_type=self.model_config.model_type,
            max_len_lst=max_len_lst,
            cumsum_seqlens=cumsum_seqlens,
        )
        return rope_emb_lst
