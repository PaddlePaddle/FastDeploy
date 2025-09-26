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
from typing import List, Optional

import numpy as np
import paddle
from paddle import nn
from paddleformers.utils.log import logger

from fastdeploy.config import FDConfig
from fastdeploy.engine.request import Request, RequestType
from fastdeploy.model_executor.graph_optimization.utils import (
    profile_run_guard,
    sot_warmup_guard,
)
from fastdeploy.model_executor.guided_decoding import (
    LogitsProcessorBase,
    get_guided_backend,
)
from fastdeploy.model_executor.layers.attention import get_attention_backend
from fastdeploy.model_executor.layers.attention.base_attention_backend import (
    AttentionBackend,
)
from fastdeploy.model_executor.layers.rotary_embedding import get_rope, get_rope_3d
from fastdeploy.model_executor.layers.sample.meta_data import SamplingMetadata
from fastdeploy.model_executor.layers.sample.sampler import Sampler, SpeculativeSampler
from fastdeploy.model_executor.model_loader import get_model_loader
from fastdeploy.platforms import current_platform

if current_platform.is_iluvatar():
    from fastdeploy.model_executor.ops.iluvatar import set_value_by_flags_and_idx

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
from fastdeploy.input.ernie4_5_vl_processor import DataProcessor
from fastdeploy.inter_communicator import ZmqIpcClient
from fastdeploy.model_executor.forward_meta import ForwardMeta
from fastdeploy.model_executor.models.ernie4_5_vl.modeling_resampler import ScatterOp
from fastdeploy.worker.model_runner_base import ModelRunnerBase
from fastdeploy.worker.output import ModelOutputData, ModelRunnerOutput


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
        #  Sampler
        if not self.speculative_decoding:
            self.sampler = Sampler(fd_config)
        else:
            self.sampler = SpeculativeSampler(fd_config)

        self.guided_backend = None
        if self.fd_config.parallel_config.guided_decoding_backend != "off":
            self.guided_backend = get_guided_backend(fd_config=self.fd_config)
            self.sampler.set_reasoning_parser(self.guided_backend.get_reasoning_parser())

        # Lazy initialize kv cache after model loading
        # self.kv_caches: list[paddle.Tensor] = []

        # Cuda Graph
        self.graph_opt_level = self.graph_opt_config.graph_opt_level
        self.use_cudagraph = self.graph_opt_config.use_cudagraph
        self.cudagraph_capture_sizes = list(reversed(self.graph_opt_config.cudagraph_capture_sizes))
        self.sot_warmup_sizes = self.graph_opt_config.sot_warmup_sizes
        self.cudagraph_only_prefill = self.graph_opt_config.cudagraph_only_prefill

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
        self.initialize_attn_backend()

        # Forward meta store the global meta information of the forward
        self.forward_meta: ForwardMeta = None

        # Postprocess Env params
        os.environ["INFERENCE_MSG_QUEUE_ID"] = str(self.parallel_config.engine_worker_queue_port)
        logger.info(f"queue id is {str(self.parallel_config.engine_worker_queue_port)}")

        self.zmq_client = None
        if envs.FD_USE_GET_SAVE_OUTPUT_V1:
            logger.info(f"zmq client get_save_output_rank{local_rank}")
            self.zmq_client = ZmqIpcClient(name=f"get_save_output_rank{local_rank}", mode=zmq.PUSH)
            self.zmq_client.connect()
            self.zmq_client.socket.SNDTIMEO = 3000

    def exist_prefill(self):
        """
        check whether prefill stage exist
        """
        return int(paddle.max(self.share_inputs["seq_lens_encoder"])) > 0

    def exist_decode(self):
        """
        check whether decode stage exist
        """
        return int(paddle.max(self.share_inputs["seq_lens_decoder"])) > 0

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

    def _init_logits_processor(self, request):
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

        enable_thinking = request.get("enable_thinking", True)
        enable_thinking = enable_thinking if enable_thinking is not None else True

        return (
            self.guided_backend.get_logits_processor(
                schemata_key=schemata_key,
                enable_thinking=enable_thinking,
            ),
            schemata_key,
        )

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
        for i in range(req_len):
            request = req_dicts[i]
            idx = request.idx
            if request.task_type.value == RequestType.PREFILL.value:  # prefill task
                prefill_start_index = request.prefill_start_index
                prefill_end_index = request.prefill_end_index
                length = prefill_end_index - prefill_start_index
                if self.enable_mm:
                    inputs = request.multimodal_inputs
                    if request.with_image:
                        vision_inputs = {}
                        vision_inputs["input_ids"] = paddle.to_tensor(
                            inputs["input_ids"][prefill_start_index:prefill_end_index], dtype=paddle.int64
                        )
                        vision_inputs["token_type_ids"] = paddle.to_tensor(
                            inputs["token_type_ids"][prefill_start_index:prefill_end_index], dtype=paddle.int64
                        )
                        vision_inputs["image_type_ids"] = paddle.to_tensor(
                            inputs["image_type_ids"][request.image_type_ids_start : request.image_type_ids_end],
                            dtype=paddle.int64,
                        )
                        vision_inputs["images"] = paddle.to_tensor(
                            inputs["images"][request.image_start : request.image_end],
                            dtype="uint8" if "ernie" in self.model_config.model_type else "bfloat16",
                        )
                        vision_inputs["grid_thw"] = paddle.to_tensor(
                            inputs["grid_thw"][request.num_image_start : request.num_image_end], dtype="int64"
                        )
                        self.share_inputs["image_features"] = self.extract_vision_features(vision_inputs)
                    else:
                        self.share_inputs["image_features"] = None

                    if inputs["position_ids"] is not None:
                        position_ids = paddle.to_tensor(
                            request.multimodal_inputs["position_ids"],
                            dtype="int64",
                        ).unsqueeze([0])
                    else:
                        position_ids = None

                    self.share_inputs["rope_emb"][idx : idx + 1, :] = self.prepare_rope3d(
                        position_ids, request.get("max_tokens", 2048)
                    )

                if request.get("enable_thinking", False):
                    # Enable thinking
                    req_reasoning_max_tokens = request.get("reasoning_max_tokens")
                    req_max_tokens = request.get("max_tokens")
                    final_reasoning_tokens = (
                        req_reasoning_max_tokens if req_reasoning_max_tokens is not None else req_max_tokens
                    )

                    self.share_inputs["enable_thinking"][idx : idx + 1] = True
                    self.share_inputs["need_think_end"][idx : idx + 1, :] = 1
                    self.share_inputs["reasoning_index"][idx : idx + 1, :] = final_reasoning_tokens
                else:
                    # Disable thinking
                    self.share_inputs["enable_thinking"][idx : idx + 1] = False
                    self.share_inputs["need_think_end"][idx : idx + 1, :] = 0
                    self.share_inputs["reasoning_index"][idx : idx + 1, :] = 0

                if isinstance(request.prompt_token_ids, np.ndarray):
                    prompt_token_ids = request.prompt_token_ids.tolist()
                else:
                    prompt_token_ids = request.prompt_token_ids
                input_ids = prompt_token_ids + request.output_token_ids
                logger.debug(
                    f"Handle prefill request {request} at idx {idx}, "
                    f"{prefill_start_index=}, {prefill_end_index=}, "
                    f"need_prefilled_token_num={len(input_ids)}"
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
                continue
            else:  # preempted task
                logger.debug(f"Handle preempted request {request} at idx {idx}")
                self.share_inputs["block_tables"][idx : idx + 1, :] = -1
                self.share_inputs["stop_flags"][idx : idx + 1] = True
                self.seq_lens_this_time_buffer[idx : idx + 1] = 0
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

            prefill_tokens = []
            if (
                request.guided_json is not None
                or request.guided_regex is not None
                or request.structural_tag is not None
                or request.guided_grammar is not None
            ):
                logits_info, schemata_key = self._init_logits_processor(request)
                request.logits_processor, request.logits_cached = logits_info
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
                            ).unsqueeze([0])
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
                        position_ids, request.get("max_tokens", 2048)
                    )
                    self.share_inputs["seq_lens_decoder"][idx : idx + 1] = 0

                if request.get("enable_thinking", False):
                    # Enable thinking
                    req_reasoning_max_tokens = request.get("reasoning_max_tokens")
                    req_max_tokens = request.get("max_tokens")
                    final_reasoning_tokens = (
                        req_reasoning_max_tokens if req_reasoning_max_tokens is not None else req_max_tokens
                    )

                    self.share_inputs["enable_thinking"][idx : idx + 1] = True
                    self.share_inputs["need_think_end"][idx : idx + 1, :] = 1
                    self.share_inputs["reasoning_index"][idx : idx + 1, :] = final_reasoning_tokens
                else:
                    # Disable thinking
                    self.share_inputs["enable_thinking"][idx : idx + 1] = False
                    self.share_inputs["need_think_end"][idx : idx + 1, :] = 0
                    self.share_inputs["reasoning_index"][idx : idx + 1, :] = 0

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

            self.sampler.apply_logits_processor(idx, request.get("logits_processor"), prefill_tokens)

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
            self.parallel_config.max_model_len - max_dec_len,
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
            [max_num_seqs, self.parallel_config.max_model_len],
            -1,
            dtype="int64",
        )
        self.share_inputs["input_ids"] = paddle.full(
            [max_num_seqs, self.parallel_config.max_model_len],
            self.model_config.pad_token_id,
            dtype="int64",
        )
        self.share_inputs["prompt_ids"] = paddle.full(
            [max_num_seqs, self.parallel_config.max_model_len],
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
        self.share_inputs["min_length"] = paddle.full([max_num_seqs, 1], self.model_config.min_length, dtype="int64")
        self.share_inputs["max_length"] = paddle.full(
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
            [max_num_seqs * self.parallel_config.max_model_len],
            0,
            dtype="int64",
        )
        self.share_inputs["batch_id_per_token"] = paddle.full(
            [max_num_seqs * self.parallel_config.max_model_len, 1], 0, dtype="int32"
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
        self.share_inputs["max_len_kv_cpu"] = None  # CPU

        # Initialize rotary position embedding
        tmp_position_ids = paddle.arange(self.parallel_config.max_model_len).reshape((1, -1))

        # Initialize thinking related buffers
        self.share_inputs["need_think_end"] = paddle.full(shape=[max_num_seqs, 1], fill_value=0, dtype="int32")
        self.share_inputs["enable_thinking"] = paddle.full(shape=[max_num_seqs, 1], fill_value=False, dtype="bool")
        self.share_inputs["reasoning_index"] = paddle.full(shape=[max_num_seqs, 1], fill_value=0, dtype="int32")

        # TODO(gongshaotian): move to models
        if not self.enable_mm:
            self.share_inputs["rope_emb"] = get_rope(
                rotary_dim=self.model_config.head_dim,
                position_ids=tmp_position_ids,
                base=self.model_config.rope_theta,
                model_config=self.model_config,
                partial_rotary_factor=self.model_config.partial_rotary_factor,
            )

        # Set block tables
        pre_max_block_num = (
            self.parallel_config.max_model_len + self.cache_config.block_size - 1
        ) // self.cache_config.block_size + self.cache_config.enc_dec_block_num
        self.share_inputs["block_tables"] = paddle.full([max_num_seqs, pre_max_block_num], -1, dtype="int32")

        # Initialize free list
        free_list = list(
            range(
                self.parallel_config.total_block_num - 1,
                int(self.parallel_config.total_block_num * self.cache_config.kv_cache_ratio) - 1,
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
                shape=[max_num_seqs, self.parallel_config.max_model_len],
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

        if self.enable_mm:
            head_dim = self.model_config.head_dim
            if "qwen" in self.model_config.model_type:  # neox style = True
                rope_head_dim = head_dim
            else:  # neox style = False
                rope_head_dim = head_dim // 2

            self.share_inputs["rope_emb"] = paddle.full(
                shape=[
                    max_num_seqs,
                    2,
                    1,
                    self.parallel_config.max_model_len,
                    1,
                    rope_head_dim,
                ],
                fill_value=0,
                dtype="float32",
            )
            self.share_inputs["image_features"] = None

    def _prepare_inputs(self) -> None:
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
        max_bad_tokens_len = paddle.max(self.share_inputs["bad_tokens_len"])

        # Initialize forward meta data
        self.initialize_forward_meta()

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
            max_num_logprobs=20 if self.enable_logprob else None,
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

    def initialize_forward_meta(self):
        """
        Initialize forward meta and attention meta data
        """
        # Initialize forward meta
        self.forward_meta = ForwardMeta(
            input_ids=self.share_inputs["input_ids"],
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
            max_len_kv_cpu=self.share_inputs["max_len_kv_cpu"],
        )

        # Update Batch type for cuda graph for only_decode_batch
        if_only_decode = self.only_decode()
        only_decode_use_cudagraph = self.use_cudagraph and if_only_decode

        # Update config about moe for better performance
        # TODO(wanglongzhi):Modifying the config at runtime is not appropriate; it needs to be moved to forward_meta. It will be used in MoEMethodBase.apply()
        if self.fd_config.parallel_config.use_ep and self.fd_config.scheduler_config.splitwise_role == "mixed":
            self.fd_config.model_config.moe_phase.phase = "decode" if if_only_decode else "prefill"

        # Update Batch type for cuda graph for only_prefill_batch
        only_prefill_use_cudagraph = self.use_cudagraph and self.cudagraph_only_prefill and self.only_prefill()

        # When support capture both prefill-only and decode-only, this will use [only_prefill_use_cudagraph or only_decode_use_cudagraph]
        self.forward_meta.step_use_cudagraph = (
            only_prefill_use_cudagraph if self.cudagraph_only_prefill else only_decode_use_cudagraph
        )

        # Initialzie attention meta data
        for attn_backend in self.attn_backends:
            attn_backend.init_attention_metadata(self.forward_meta)

    def initialize_kv_cache(self, profile: bool = False) -> None:
        """
        Initialize kv cache
        """
        cache_kvs = {}
        max_block_num = self.num_gpu_blocks

        # Get kv cache dtype
        cache_type = self.parallel_config.dtype

        kv_cache_quant_type = None
        if (
            self.quant_config
            and hasattr(self.quant_config, "kv_cache_quant_type")
            and self.quant_config.kv_cache_quant_type is not None
        ):
            cache_type = "uint8"
            kv_cache_quant_type = self.quant_config.kv_cache_quant_type

        # Get kv cache shape
        kv_cache_shape = self.attn_backends[0].get_kv_cache_shape(
            max_num_blocks=max_block_num, kv_cache_quant_type=kv_cache_quant_type
        )
        if kv_cache_quant_type == "block_wise_fp8":
            kv_cache_scale_shape = [kv_cache_shape[0], kv_cache_shape[1], kv_cache_shape[2]]
        local_rank = self.local_rank % self.parallel_config.tensor_parallel_size

        if not profile and (
            self.cache_config.enable_prefix_caching or self.scheduler_config.splitwise_role != "mixed"
        ):
            cache_kvs_list = []
            for i in range(self.model_config.num_hidden_layers):
                key_cache = paddle.empty(shape=[], dtype=cache_type)
                key_cache_name = f"key_caches_{i}_rank{local_rank}.device{self.device_id}"
                val_cache_name = f"value_caches_{i}_rank{local_rank}.device{self.device_id}"
                key_cache = share_external_data(key_cache, key_cache_name, kv_cache_shape)
                cache_kvs_list.append(key_cache)
                value_cache = paddle.empty(shape=[], dtype=cache_type)
                value_cache = share_external_data(value_cache, val_cache_name, kv_cache_shape)
                cache_kvs_list.append(value_cache)

            self.share_inputs["caches"] = cache_kvs_list
        else:
            for i in range(self.model_config.num_hidden_layers):
                cache_kvs[f"key_caches_{i}"] = paddle.full(
                    shape=kv_cache_shape,
                    fill_value=0,
                    dtype=cache_type,
                )
                cache_kvs[f"value_caches_{i}"] = paddle.full(
                    shape=kv_cache_shape,
                    fill_value=0,
                    dtype=cache_type,
                )
                if kv_cache_quant_type == "block_wise_fp8":
                    cache_kvs[f"key_cache_scales_{i}"] = paddle.full(
                        shape=kv_cache_scale_shape,
                        fill_value=0,
                        dtype=paddle.get_default_dtype(),
                    )
                    cache_kvs[f"value_cache_scales_{i}"] = paddle.full(
                        shape=kv_cache_scale_shape,
                        fill_value=0,
                        dtype=paddle.get_default_dtype(),
                    )
            self.share_inputs["caches"] = list(cache_kvs.values())
            for value in cache_kvs.values():
                del value
        paddle.device.cuda.empty_cache()

    def initialize_attn_backend(self) -> None:
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

        # Initialize AttentionBackend buffers
        encoder_block_shape_q = 64
        decoder_block_shape_q = 16
        decoder_step_token_num = self.speculative_config.num_speculative_tokens + 1
        group_size = np.ceil(num_heads / self.model_config.kv_num_heads)

        # NOTE: (changwenbin) When using auto_chunk,
        # decode_max_tile_size must take into account the maximum case, where *1024 can cover 128K.
        decode_max_tile_size = (
            1024
            * self.scheduler_config.max_num_seqs
            * np.ceil((decoder_step_token_num * group_size) / decoder_block_shape_q)
        )
        encode_max_tile_size = self.scheduler_config.max_num_seqs * np.ceil(
            (self.model_config.max_model_len * group_size) / encoder_block_shape_q
        )
        kv_max_tile_size = self.scheduler_config.max_num_seqs * np.ceil(
            self.model_config.max_model_len / self.fd_config.cache_config.block_size
        )
        self.share_inputs["decoder_batch_ids"] = paddle.full([int(decode_max_tile_size)], 0, dtype="int32")
        self.share_inputs["decoder_tile_ids_per_batch"] = paddle.full([int(decode_max_tile_size)], 0, dtype="int32")
        self.share_inputs["decoder_num_blocks_cpu"] = paddle.full([1], 0, dtype="int32").pin_memory()
        # NOTE: (changwenbin) MLA kernel only needs decoder_num_blocks_device in place of GPU tensor,
        # adapted to cudagraph.
        self.share_inputs["decoder_num_blocks_device"] = paddle.full([1], 0, dtype="int32")
        self.share_inputs["decoder_chunk_size_device"] = paddle.full([1], 64, dtype="int32")
        self.share_inputs["max_len_tensor_cpu"] = paddle.full([8], 0, dtype="int32").cpu()

        self.share_inputs["encoder_batch_ids"] = paddle.full([int(encode_max_tile_size)], 0, dtype="int32")
        self.share_inputs["encoder_tile_ids_per_batch"] = paddle.full([int(encode_max_tile_size)], 0, dtype="int32")
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
            encoder_block_shape_q=encoder_block_shape_q,
            decoder_block_shape_q=decoder_block_shape_q,
        )

        self.attn_backends.append(attn_backend)

    def _dummy_run(
        self,
        num_tokens: paddle.Tensor,
        batch_size: paddle.Tensor,
        expected_decode_len: int = 1,
        in_capturing: bool = False,
        capture_prefill: bool = False,
    ) -> paddle.Tensor:
        """
        Use dummy inputs to run before formal execution.
        Args:
            num_tokens:
            expected_decode_len: Expected number of tokens generated
            in_capturing: Is cuda graph in capturing state
            capture_prefill: Capture pure prefill for cuda graph
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
            self._prepare_inputs()

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

            hidden_states = rebuild_padding(
                model_output,
                self.share_inputs["cu_seqlens_q"],
                self.share_inputs["seq_lens_this_time"],
                self.share_inputs["seq_lens_decoder"],
                self.share_inputs["seq_lens_encoder"],
                (
                    self.share_inputs["output_padding_offset"] if self.speculative_decoding else None
                ),  # speculative decoding requires
                self.parallel_config.max_model_len,
            )

            logits = None
            if hasattr(self.model, "is_pooling_model") and self.model.is_pooling_model:
                # TODO(lizexu123) The preheating the pooling function have not been implemented yet.
                pass
            else:
                # 4. Execute spec decode
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
                    self.parallel_config.max_model_len,
                    self.share_inputs,
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
                enable_thinking=self.share_inputs["enable_thinking"],
                think_end_id=self.model_config.think_end_id,
                need_think_end=self.share_inputs["need_think_end"],
                reasoning_index=self.share_inputs["reasoning_index"],
                stop_token_ids=self.share_inputs["stop_seqs"],
                stop_seqs_len=self.share_inputs["stop_seqs_len"],
            )

            post_process(
                sampler_output=sampler_output,
                model_output=model_output_data,
                share_inputs=self.share_inputs,
                block_size=self.cache_config.block_size,
                speculative_decoding=self.speculative_decoding,
                skip_save_output=True,
                zmq_client=self.zmq_client,
            )

            if self.speculative_decoding:
                if self.speculative_method == "mtp":
                    self.proposer.run(full_hidden_states=model_output)
                else:
                    self.proposer.run(share_inputs=self.share_inputs)

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
        else:
            for batch_size in sorted(capture_sizes, reverse=True):
                self._dummy_run(
                    num_tokens=self.scheduler_config.max_num_batched_tokens,
                    batch_size=batch_size,
                    in_capturing=True,
                    expected_decode_len=expected_decode_len,
                )
                logger.info(
                    f"Warm up the model with the num_tokens:{batch_size}, expected_decode_len:{expected_decode_len}"
                )

        time_after_capture = time.perf_counter()
        logger.info(f"Cuda Graph capturing took {time_after_capture - time_before_capture} seconds")

    @sot_warmup_guard(True)
    def sot_warmup(self) -> None:
        start_time = time.perf_counter()
        for batch_size in self.sot_warmup_sizes:
            self._dummy_run(
                num_tokens=self.scheduler_config.max_num_batched_tokens,
                batch_size=batch_size,
            )
            logger.info(f"SOT warmup the model with the batch size:{batch_size}")
        logger.info(f"SOT warmup took {time.perf_counter() - start_time} seconds")

    def _get_skip_idx(self, model_forward_batch: Optional[List[Request]] = None):
        """
        Get the index of the request that needs to be skipped during execution.
        Args:
            model_forward_batch: A list of requests to be executed by this runner.
        Returns:
            A list of indices corresponding to the requests that need to be skipped.
        """
        if (
            not self.cache_config.enable_chunked_prefill
            or self.guided_backend is None
            or model_forward_batch is None
            or envs.ENABLE_V1_KVCACHE_SCHEDULER
        ):
            return []

        skip_idx_list = []
        for task in model_forward_batch:
            if task.get("prefill_chunk_info", None) is None or task.chunk_idx >= len(task.prefill_chunk_info):
                continue
            skip_idx_list.append(task.idx)

        for task in self.restore_chunked_prefill_request.values():
            if task.idx in skip_idx_list or task.chunk_idx >= len(task.prefill_chunk_info):
                continue
            skip_idx_list.append(task.idx)

        return skip_idx_list

    def execute_model(
        self,
        model_forward_batch: Optional[List[Request]] = None,
        num_running_requests: int = None,
    ) -> Optional[ModelRunnerOutput]:
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
        skip_idx_list = self._get_skip_idx(model_forward_batch)
        self._prepare_inputs()
        self.sampler.pre_process(skip_idx_list)

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
        hidden_states = rebuild_padding(
            model_output,
            self.share_inputs["cu_seqlens_q"],
            self.share_inputs["seq_lens_this_time"],
            self.share_inputs["seq_lens_decoder"],
            self.share_inputs["seq_lens_encoder"],
            (self.share_inputs["output_padding_offset"] if self.speculative_decoding else None),
            self.parallel_config.max_model_len,
        )

        logits = None
        # 4. Compute logits, Sample
        if hasattr(self.model, "is_pooling_model") and self.model.is_pooling_model:
            # TODO(lizexu123) The execution of the pooling function have not been implemented yet.
            pass
        else:
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
                skip_idx_list,
            )
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
                self.parallel_config.max_model_len,
                self.share_inputs,
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
            enable_thinking=self.share_inputs["enable_thinking"],
            think_end_id=self.model_config.think_end_id,
            need_think_end=self.share_inputs["need_think_end"][:num_running_requests],
            reasoning_index=self.share_inputs["reasoning_index"][:num_running_requests],
            stop_token_ids=self.share_inputs["stop_seqs"],
            stop_seqs_len=self.share_inputs["stop_seqs_len"],
        )

        if self.speculative_config.method in ["mtp"] and self.scheduler_config.splitwise_role == "prefill":
            skip_save_output = True
        else:
            skip_save_output = False
        post_process(
            sampler_output=sampler_output,
            model_output=model_output_data,
            share_inputs=self.share_inputs,
            block_size=self.cache_config.block_size,
            save_each_rank=self.parallel_config.use_ep,
            speculative_decoding=self.speculative_decoding,
            skip_save_output=skip_save_output,
            zmq_client=self.zmq_client,
        )
        if self.guided_backend is not None and sampler_output is not None:
            self.sampler.post_process(sampler_output.sampled_token_ids, skip_idx_list)

        # 6. Speculative decode
        if self.speculative_decoding:
            if self.speculative_method == "mtp":
                self.proposer.run(full_hidden_states=model_output)
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
            self._add_cache(model_forward_batch)
        elif self.speculative_decoding:
            speculate_schedule_cache(
                self.share_inputs["draft_tokens"],
                self.share_inputs["block_tables"],
                self.share_inputs["stop_flags"],
                self.share_inputs["seq_lens_this_time"],
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

        self.seq_lens_this_time_buffer[:num_running_requests].copy_(
            self.share_inputs["seq_lens_this_time"][:num_running_requests], False
        )
        return None

    def _add_cache(self, model_forward_batch) -> None:
        """
        Add cache for guided decoding.
        """
        if self.guided_backend is None or model_forward_batch is None:
            return

        for request in model_forward_batch:
            logits_cached = request.get("logits_cached", None)
            if logits_cached is None or logits_cached:
                continue

            request.logits_cached = True
            if isinstance(request.logits_processor, LogitsProcessorBase):
                self.guided_backend.add_cache(request.schemata_key, request.logits_processor)
            else:
                self.guided_backend.add_cache(request.schemata_key, request.logits_processor.result())

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
        self.num_gpu_blocks = self.parallel_config.total_block_num
        self.initialize_kv_cache(profile=True)

        # 1. Profile with multimodal encoder & encoder cache

        # 2. Dummy run
        self._dummy_run(
            num_tokens=self.scheduler_config.max_num_batched_tokens,
            batch_size=min(self.scheduler_config.max_num_seqs, 3),
        )

        # 3. gc
        self.clear_cache()

        if self.speculative_method in ["mtp"]:
            self.proposer.clear_dummy_input()

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
            self.proposer.update_block_num(num_gpu_blocks)

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
            position_ids = paddle.to_tensor(one["position_ids"], dtype="int64").unsqueeze([0])
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
            dtype=self.parallel_config.dtype,
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
            dtype=self.parallel_config.dtype,
        ):
            image_features = self.model.visual.extract_feature(images, grid_thw)

        return image_features

    @paddle.no_grad()
    def extract_vision_features(self, inputs: list[paddle.Tensor]) -> paddle.Tensor:
        """extract_vision_features"""
        if "ernie" in self.model_config.model_type:
            return self.extract_vision_features_ernie(inputs)
        elif "qwen" in self.model_config.model_type:
            return self.extract_vision_features_qwen(inputs)
        else:
            raise ValueError(f"multiple modalities model {self.model_config.model_type} is not supported")

    @paddle.no_grad()
    def prepare_rope3d(self, position_ids: paddle.Tensor, max_len: int) -> paddle.Tensor:
        """prepare_rope3d"""

        prefix_max_position_ids = paddle.max(position_ids) + 1
        dec_pos_ids = paddle.tile(
            paddle.arange(max_len, dtype="int64").unsqueeze(0).unsqueeze(-1),
            [1, 1, 3],
        )
        dec_pos_ids = dec_pos_ids + prefix_max_position_ids
        position_ids_3d_real = paddle.concat([position_ids, dec_pos_ids], axis=1)

        rope_emb = get_rope_3d(
            position_ids=position_ids_3d_real,
            rotary_dim=self.model_config.head_dim,
            partial_rotary_factor=1.0,
            base=self.model_config.rope_theta,
            max_position=self.parallel_config.max_model_len,
            freq_allocation=getattr(self.model_config, "freq_allocation", 20),
            model_type=self.model_config.model_type,
        )
        return rope_emb
