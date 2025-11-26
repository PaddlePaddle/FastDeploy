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
from typing import Dict, List, Optional

import numpy as np
import paddle
import paddle.nn as nn
from paddleformers.utils.log import logger

from fastdeploy.config import FDConfig
from fastdeploy.distributed.communication import tensor_model_parallel_all_reduce_custom
from fastdeploy.engine.request import Request

# from fastdeploy.spec_decode import MTPProposer, NgramProposer
from fastdeploy.model_executor.forward_meta import HPUForwardMeta
from fastdeploy.model_executor.guided_decoding import get_guided_backend
from fastdeploy.model_executor.layers.attention import get_attention_backend
from fastdeploy.model_executor.layers.attention.base_attention_backend import (
    AttentionBackend,
)
from fastdeploy.model_executor.layers.rotary_embedding import get_rope
from fastdeploy.model_executor.layers.sample.meta_data import SamplingMetadata
from fastdeploy.model_executor.layers.sample.sampler import Sampler, SpeculativeSampler
from fastdeploy.model_executor.model_loader import get_model_loader
from fastdeploy.model_executor.ops.intel_hpu import (
    recover_block,
    save_output,
    step_paddle,
    update_inputs_v3,
)
from fastdeploy.utils import get_logger
from fastdeploy.worker.model_runner_base import ModelRunnerBase
from fastdeploy.worker.output import ModelOutputData, ModelRunnerOutput

hpu_model_runner_profile_logger = get_logger("hpu_model_runner_profile", "hpu_model_runner_profile.log")


def post_process_hpu(sampled_token_ids: paddle.Tensor, model_output: ModelOutputData, is_warmuping: bool) -> None:
    """Post-processing steps after completing a single token generation."""
    start_time = time.time()

    not_need_stop_hpu = model_output.not_need_stop.to(sampled_token_ids.place)
    is_block_step_hpu = model_output.is_block_step.to(sampled_token_ids.place)

    update_inputs_v3(
        model_output.stop_flags,
        model_output.step_idx,
        not_need_stop_hpu,
        model_output.seq_lens_this_time,
        model_output.seq_lens_encoder,
        model_output.seq_lens_decoder,
        model_output.max_dec_len,
        model_output.input_ids,
        model_output.stop_nums,
        sampled_token_ids,
        is_block_step_hpu,
        model_output.eos_token_id,
        model_output.next_tokens,
    )

    model_output.not_need_stop[:] = not_need_stop_hpu.cpu()
    model_output.is_block_step[:] = is_block_step_hpu.cpu()

    end_time = time.time()
    execution_time = (end_time - start_time) * 1000
    hpu_model_runner_profile_logger.info(f"post_process_hpu::update_inputs_v3 execution time(ms): {execution_time}")

    if is_warmuping:
        return
    start_time = time.time()
    save_output(
        sampled_token_ids,
        model_output.not_need_stop,
        model_output.mp_rank,
    )
    end_time = time.time()
    execution_time = (end_time - start_time) * 1000
    hpu_model_runner_profile_logger.info(f"post_process_hpu::save_output execution time(ms): {execution_time}")


def recover_block_hpu(
    recover_block_list,  # cpu
    recover_len,  # cpu
    stop_flags,  # hpu
    seq_lens_this_time,  # hpu
    ori_seq_lens_encoder,  # cpu
    ori_seq_lens_decoder,  # cpu
    seq_lens_encoder,  # hpu
    seq_lens_decoder,  # hpu
    block_tables,  # cpu
    free_list,  # cpu
    free_list_len,  # cpu
    input_ids,  # hpu
    pre_ids,  # hpu
    step_idx,  # hpu
    encoder_block_lens,  # cpu
    used_list_len,  # cpu
    next_tokens,  # hpu
    first_token_ids,
):  # hpu

    for bid in range(recover_len.item()):
        recover_id = recover_block_list[bid].item()
        ori_seq_len_encoder = ori_seq_lens_encoder[recover_id].item()
        ori_seq_len_decoder = ori_seq_lens_decoder[recover_id].item()
        step_idx_now = step_idx[recover_id].item()
        seq_len = ori_seq_len_encoder + step_idx_now
        encoder_block_len = encoder_block_lens[recover_id].item()
        decoder_used_len = used_list_len[recover_id].item()

        seq_lens_this_time[recover_id] = seq_len
        seq_lens_encoder[recover_id] = seq_len
        seq_lens_decoder[recover_id] = ori_seq_len_decoder
        stop_flags[recover_id] = False

        ori_free_list_len = free_list_len[0]
        for i in range(decoder_used_len):
            block_tables[recover_id, encoder_block_len + i] = free_list[ori_free_list_len - i - 1]
        free_list_len[0] -= decoder_used_len

        recover_block(input_ids, first_token_ids, pre_ids, next_tokens, recover_id, ori_seq_len_encoder, step_idx_now)


def step_intel_hpu(share_inputs: Dict[str, paddle.Tensor], block_size: int, max_model_len: int) -> None:
    """
    step cuda
    """
    step_paddle(
        share_inputs["stop_flags"],
        share_inputs["seq_lens_this_time"],
        share_inputs["seq_lens_encoder"],
        share_inputs["seq_lens_decoder"],
        share_inputs["block_tables"],
        share_inputs["encoder_block_lens"],
        share_inputs["is_block_step"],
        share_inputs["step_block_list"],
        share_inputs["step_lens"],
        share_inputs["recover_block_list"],
        share_inputs["recover_lens"],
        share_inputs["need_block_list"],
        share_inputs["need_block_len"],
        share_inputs["used_list_len"],
        share_inputs["free_list"],
        share_inputs["free_list_len"],
        share_inputs["first_token_ids"],
        block_size,
        max_model_len,
    )
    if share_inputs["recover_lens"].item() > 0:
        logger.info("recover block hpu happening ...")
        recover_block_hpu(
            share_inputs["recover_block_list"],
            share_inputs["recover_lens"],
            share_inputs["stop_flags"],
            share_inputs["seq_lens_this_time"],
            share_inputs["ori_seq_lens_encoder"],
            share_inputs["ori_seq_lens_decoder"],
            share_inputs["seq_lens_encoder"],
            share_inputs["seq_lens_decoder"],
            share_inputs["block_tables"],
            share_inputs["free_list"],
            share_inputs["free_list_len"],
            share_inputs["input_ids"],
            share_inputs["pre_ids"],
            share_inputs["step_idx"],
            share_inputs["encoder_block_lens"],
            share_inputs["used_list_len"],
            share_inputs["next_tokens"],
            share_inputs["first_token_ids"],
        )
        share_inputs["recover_lens"] = paddle.full([1], 0, dtype="int32").cpu()
        share_inputs["not_need_stop"][0] = True


# TODO: replace rebuild_padding_v3 in CustomDevice if we adopt this version pp optimization
def rebuild_padding_v3_1(
    tmp_out,
    batch_ids,
    total_batch,
    seq_lens_encoder,
    is_prompt=None,
):
    dim_emb = tmp_out.shape[-1]
    output_data = paddle.zeros((total_batch, dim_emb))
    if is_prompt is True:  # context
        tmp_out = tmp_out.reshape([total_batch, -1, dim_emb])
        for i in range(batch_ids.shape[0]):
            seq_len = seq_lens_encoder[batch_ids[i]].item()
            output_data[i] = tmp_out[i, seq_len - 1]
    elif is_prompt is False:
        output_data[0 : batch_ids.shape[0], :] = tmp_out[: batch_ids.shape[0], :]

    return output_data


from fastdeploy.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from fastdeploy.model_executor.ops.intel_hpu import fused_mlp


def fused_attention_forward(
    self,
    src: paddle.Tensor = None,
    qkv_proj: QKVParallelLinear = None,
    o_proj: RowParallelLinear = None,
    forward_meta: HPUForwardMeta = None,
):
    """
    The forward function of attention layer.
    args:
        src: the hidden states tensor
        residual_input: the residual tensor
        forward_meta: the forward meta data
    """
    return forward_meta.attn_backend.forward(
        src,
        qkv_proj,
        o_proj,
        self,
        forward_meta,
    )


def fused_self_atten_forward(
    self,
    forward_meta: HPUForwardMeta,
    hidden_states: paddle.Tensor,
):
    """ """
    atten_out = self.attn(
        src=hidden_states,
        qkv_proj=self.qkv_proj,
        o_proj=self.o_proj,
        forward_meta=forward_meta,
    )

    return atten_out


def fused_mlp_forward(self, x):
    """ """
    out = fused_mlp(
        x,
        self.up_gate_proj.weight,
        None,
        self.down_proj.weight,
    )

    # all_reduce
    if self.nranks > 1:
        from fastdeploy.distributed.communication import (
            tensor_model_parallel_all_reduce_custom,
        )

        tensor_model_parallel_all_reduce_custom(out)

    return out


import types

from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.models.ernie4_5_moe import (
    Ernie4_5_Attention,
    Ernie4_5_MLP,
)
from fastdeploy.model_executor.models.qwen2 import Qwen2Attention, Qwen2MLP


def convert_model(model):
    """ """
    for name, module in model.named_children():
        if len(list(module.named_children())) > 0:
            # print(f"********** model {model.__class__.__name__} has submodule: name={name}, module={module.__class__.__name__}")
            if isinstance(module, Ernie4_5_Attention):
                module.forward = types.MethodType(fused_self_atten_forward, module)
            if isinstance(module, Qwen2Attention):
                module.forward = types.MethodType(fused_self_atten_forward, module)
            if isinstance(module, Ernie4_5_MLP):
                module.forward = types.MethodType(fused_mlp_forward, module)
            if isinstance(module, Qwen2MLP):
                module.forward = types.MethodType(fused_mlp_forward, module)
            convert_model(module)
        else:
            # print(f"*********[ Leaf node]  Loading submodule: name={name} -- module: {module.__class__.__name__}")
            if isinstance(module, Attention):
                module.forward = types.MethodType(fused_attention_forward, module)

    return model


class HPUModelRunner(ModelRunnerBase):
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
        self.rank = rank
        self.local_rank = local_rank
        self.device_id = device_id
        self.speculative_method = self.fd_config.speculative_config.method
        self.speculative_decoding = self.speculative_method is not None

        self.guided_backend = None
        if self.fd_config.structured_outputs_config.guided_decoding_backend != "off":
            self.guided_backend = get_guided_backend(fd_config=self.fd_config)

        #  Sampler
        if not self.speculative_decoding:
            self.sampler = Sampler(fd_config)
        else:
            self.sampler = SpeculativeSampler(fd_config)

        # Lazy initialize kv cache after model loading
        # self.kv_caches: list[paddle.Tensor] = []

        # Cuda Graph
        self.use_cudagraph = self.graph_opt_config.use_cudagraph
        self.cudagraph_capture_sizes = list(reversed(self.graph_opt_config.cudagraph_capture_sizes))
        self.cudagraph_num_of_warmups = self.graph_opt_config.cudagraph_num_of_warmups
        self.input_ids = paddle.zeros(self.scheduler_config.max_num_seqs, dtype="int32")

        # Initialize share inputs
        self._init_share_inputs(self.scheduler_config.max_num_seqs)
        self.infer_seed_increment = paddle.full(
            shape=[self.scheduler_config.max_num_seqs, 1], fill_value=4, dtype="int64"
        ).cpu()
        self.restore_chunked_prefill_request = dict()

        # Initialize attention Backend
        # Note(gonshaotian): Currently, all attention layers share one attention backend instance.
        # In the future, we will expand it as a list.
        self.attn_backends: list[AttentionBackend] = []
        # self.attn_metadatas: list[AttentionMetadata] = []
        self.initialize_attn_backend()

        # Forward meta store the global meta information of the forward
        self.forward_meta: HPUForwardMeta = None
        self.is_warmuping = False
        self.is_hpu_perf_breakdown_sync_mode = int(os.environ.get("HPU_PERF_BREAKDOWN_SYNC_MODE", 1)) == 1
        # Postprocess Env params
        os.environ["INFERENCE_MSG_QUEUE_ID"] = str(
            self.local_rank + int(self.parallel_config.engine_worker_queue_port)
        )

        if int(os.environ.get("HABANA_PROFILE", 0)) == 1:
            step_start = int(os.environ.get("PROFILE_START", 0))
            step_end = int(os.environ.get("PROFILE_END", 4))
            import paddle.profiler as profiler

            self.prof = profiler.Profiler(
                targets=[profiler.ProfilerTarget.CPU, profiler.ProfilerTarget.CUSTOM_DEVICE],
                scheduler=(step_start, step_end),
                on_trace_ready=profiler.export_chrome_tracing("./profile"),
            )
            self.prof.start()

    def exist_prefill(self):
        """
        check whether prefill stage finished
        """
        if int(paddle.max(self.share_inputs["seq_lens_encoder"])) != 0:
            return 1
        else:
            return 0

    def init_speculative_proposer(self):
        """
        Init speculative proposer
        """
        # if self.speculative_method == "ngram":
        #     self.proposer = NgramProposer(self.fd_config)
        # elif self.speculative_method == "mtp":
        #     self.proposer = MTPProposer(self.fd_config, self.get_model(),
        #                                 self.local_rank, self.device_id,
        #                                 self.share_inputs)
        # else:
        #     self.proposer = None
        pass

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

        return self.guided_backend.get_logits_processor(schemata_key=schemata_key), schemata_key

    def insert_prefill_inputs(self, req_dicts: List[Request], num_running_requests: int = None):
        """
        Process inputs for prefill tasks and insert it to share_inputs buffer
        req_dict: A list of Request dict
        num_running_requests: batch_size
        """
        # NOTE(luotingdan): Lazy initialize kv cache
        if "caches" not in self.share_inputs:
            self.initialize_kv_cache()

        # NOTE(luotingdan): Set environment variable of prefill node
        if req_dicts[-1].disaggregate_info is not None and req_dicts[-1].disaggregate_info["role"] == "prefill":
            os.environ["PREFILL_NODE_ONE_STEP_STOP"] = "1"

        req_len = len(req_dicts)
        for i in range(req_len):
            request = req_dicts[i]
            idx = request.idx
            length = len(request.prompt_token_ids)

            prefill_tokens = []
            if (
                request.guided_json is not None
                or request.guided_regex is not None
                or request.structural_tag is not None
                or request.guided_grammar is not None
            ):
                logits_info, schemata_key = self._init_logits_processor(request)
                request.logits_processor = logits_info
                request.schemata_key = schemata_key

            # Is Decode Node
            if req_dicts[i].disaggregate_info is not None and req_dicts[i].disaggregate_info["role"] == "decode":
                prefill_tokens.append(request.prompt_token_ids[0])
                self.share_inputs["pre_ids"][idx : idx + 1] = request.prompt_token_ids[-1]
                self.share_inputs["input_ids"][idx : idx + 1, 0] = request.prompt_token_ids[0]
                self.share_inputs["seq_lens_encoder"][idx : idx + 1] = 0
                self.share_inputs["seq_lens_decoder"][idx : idx + 1] = length
                self.share_inputs["seq_lens_this_time"][idx : idx + 1] = 1
                self.share_inputs["step_seq_lens_encoder"][idx : idx + 1] = 0
                self.share_inputs["step_seq_lens_decoder"][idx : idx + 1] = length
                self.share_inputs["step_idx"][idx : idx + 1] = 1

                if self.speculative_decoding:
                    num_prefill_send_token = self.speculative_config.num_speculative_tokens + 1
                    self.share_inputs["draft_tokens"][idx : idx + 1, 0:num_prefill_send_token] = paddle.to_tensor(
                        request.draft_token_ids[0:num_prefill_send_token], dtype="int64"
                    )
                    self.share_inputs["seq_lens_this_time"][idx : idx + 1] = num_prefill_send_token
            else:
                self.share_inputs["pre_ids"][idx : idx + 1] = -1
                self.share_inputs["step_idx"][idx : idx + 1] = 0
                self.share_inputs["input_ids"][idx : idx + 1, :length] = np.array(request.prompt_token_ids)

                # Use chunked prefill
                if self.cache_config.enable_chunked_prefill:
                    request.set("chunk_idx", 1)
                    logger.info(f"prefill_chunk_info: {request.prefill_chunk_info}")
                    token_chunk_size = request.prefill_chunk_info[0]
                    self.share_inputs["seq_lens_this_time"][idx : idx + 1] = token_chunk_size
                    self.share_inputs["input_ids"][idx, :token_chunk_size] = np.array(
                        request.prompt_token_ids[:token_chunk_size]
                    )
                    self.share_inputs["step_seq_lens_encoder"][idx : idx + 1] = token_chunk_size
                    self.share_inputs["seq_lens_encoder"][idx : idx + 1] = token_chunk_size
                    self.share_inputs["seq_lens_decoder"][idx : idx + 1] = request.get("seq_lens_decoder", 0)
                    self.share_inputs["ori_seq_lens_decoder"][idx : idx + 1] = request.get("seq_lens_decoder", 0)
                    self.share_inputs["step_seq_lens_decoder"][idx : idx + 1] = request.get("seq_lens_decoder", 0)
                else:
                    self.share_inputs["seq_lens_decoder"][idx : idx + 1] = request.get("seq_lens_decoder", 0)
                    self.share_inputs["ori_seq_lens_decoder"][idx : idx + 1] = request.get("seq_lens_decoder", 0)
                    self.share_inputs["step_seq_lens_decoder"][idx : idx + 1] = request.get("seq_lens_decoder", 0)
                    self.share_inputs["seq_lens_this_time"][idx : idx + 1] = length
                    self.share_inputs["step_seq_lens_encoder"][idx : idx + 1] = length
                    self.share_inputs["seq_lens_encoder"][idx : idx + 1] = length

            if len(request.eos_token_ids) < self.model_config.eos_tokens_lens:
                request.eos_token_ids.append(request.eos_token_ids[0])
            self.share_inputs["eos_token_id"][:] = np.array(request.eos_token_ids, dtype="int64").reshape(-1, 1)

            self.share_inputs["top_p"][idx : idx + 1] = request.get("top_p", 0.7)
            self.share_inputs["temperature"][idx : idx + 1] = request.get("temperature", 0.95)
            self.share_inputs["penalty_score"][idx : idx + 1] = request.get("repetition_penalty", 1.0)
            self.share_inputs["frequency_score"][idx : idx + 1] = request.get("frequency_penalty", 0.0)
            self.share_inputs["presence_score"][idx : idx + 1] = request.get("presence_penalty", 0.0)

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

            if request.get("stop_token_ids") is not None and request.get("stop_seqs_len") is not None:
                stop_seqs_num = len(request.get("stop_seqs_len"))
                for i in range(stop_seqs_num, self.model_config.max_stop_seqs_num):
                    request.stop_seqs_len.append(0)
                self.share_inputs["stop_seqs_len"][:] = np.array(request.stop_seqs_len, dtype="int32")
                self.share_inputs["stop_seqs"][:stop_seqs_num, : len(request.get("stop_token_ids")[0])] = np.array(
                    request.get("stop_token_ids"), dtype="int64"
                )

            self.sampler.apply_logits_processor(idx, request.get("logits_processor"), prefill_tokens)

        self.share_inputs["not_need_stop"][0] = True

        if self.speculative_method in ["mtp"]:
            self.proposer.insert_prefill_inputs(req_dicts, num_running_requests)

    def _dummy_prefill_inputs(self, num_tokens: int, batch_size: int, expected_decode_len: int):
        """Set dummy prefill inputs to share_inputs"""
        # NOTE(gongshaotian): The maximum decoding length is equal to the expected decoded tokens plus the eos token
        max_dec_len = expected_decode_len + 1
        full_length = min(num_tokens // batch_size, self.model_config.max_model_len - max_dec_len)
        input_length = int(full_length * self.cache_config.kv_cache_ratio)
        block_num = (
            input_length + self.cache_config.block_size - 1
        ) // self.cache_config.block_size + self.cache_config.enc_dec_block_num

        for i in range(batch_size):
            idx = i
            self.share_inputs["input_ids"][idx : idx + 1, :input_length] = np.array([5] * input_length)
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

            self.share_inputs["encoder_block_lens"][idx : idx + 1] = block_num
            self.share_inputs["block_tables"][idx : idx + 1, :block_num] = np.arange(
                idx * block_num, (idx + 1) * block_num, 1
            )

    def _init_share_inputs(self, max_num_seqs: int):
        """Initialize all share buffers for model inputs.
        Note: In the future, we may abandon share buffers.
        """
        self.MAX_INFER_SEED = 9223372036854775806
        self.share_inputs = {}

        self.share_inputs["pre_ids"] = paddle.full([max_num_seqs, self.model_config.max_model_len], -1, dtype="int64")
        self.share_inputs["input_ids"] = paddle.full(
            [max_num_seqs, self.model_config.max_model_len], self.model_config.pad_token_id, dtype="int64"
        )
        self.share_inputs["eos_token_id"] = paddle.full([self.model_config.eos_tokens_lens, 1], 0, dtype="int64")
        self.share_inputs["top_p"] = paddle.full([max_num_seqs, 1], self.model_config.top_p, dtype="float32")
        self.share_inputs["temperature"] = paddle.full(
            [max_num_seqs, 1], self.model_config.temperature, dtype="float32"
        )
        self.share_inputs["penalty_score"] = paddle.full(
            [max_num_seqs, 1], self.model_config.penalty_score, dtype="float32"
        )
        self.share_inputs["frequency_score"] = paddle.full(
            [max_num_seqs, 1], self.model_config.frequency_score, dtype="float32"
        )
        self.share_inputs["presence_score"] = paddle.full(
            [max_num_seqs, 1], self.model_config.presence_score, dtype="float32"
        )

        self.share_inputs["min_dec_len"] = paddle.full([max_num_seqs, 1], self.model_config.min_length, dtype="int64")
        self.share_inputs["max_dec_len"] = paddle.full(
            [max_num_seqs, 1], self.model_config.max_model_len, dtype="int64"
        )
        self.share_inputs["seq_lens_this_time"] = paddle.full(max_num_seqs, 0, dtype="int32")
        self.share_inputs["seq_lens_encoder"] = paddle.full([max_num_seqs, 1], 0, dtype="int32")
        self.share_inputs["seq_lens_decoder"] = paddle.full([max_num_seqs, 1], 0, dtype="int32")
        self.share_inputs["step_seq_lens_encoder"] = paddle.full([max_num_seqs, 1], 0, dtype="int32")
        self.share_inputs["step_seq_lens_decoder"] = paddle.full([max_num_seqs, 1], 0, dtype="int32")
        self.share_inputs["step_idx"] = paddle.full([max_num_seqs, 1], 0, dtype="int64")
        self.share_inputs["not_need_stop"] = paddle.full(
            [1], False, dtype="bool"
        ).cpu()  # TODO(gongshaotian): move to pinnd memory
        self.share_inputs["stop_flags"] = paddle.full([max_num_seqs, 1], True, dtype="bool")
        self.share_inputs["stop_nums"] = paddle.full([1], max_num_seqs, dtype="int64")

        self.share_inputs["bad_tokens"] = paddle.full([1], -1, dtype="int64")
        self.share_inputs["next_tokens"] = paddle.full([max_num_seqs, 1], -1, dtype="int64")
        self.share_inputs["is_block_step"] = paddle.full([max_num_seqs], False, dtype="bool").cpu()
        self.share_inputs["encoder_block_lens"] = paddle.full([max_num_seqs], 0, dtype="int32").cpu()
        self.share_inputs["step_block_list"] = paddle.full([max_num_seqs], -1, dtype="int32").cpu()
        self.share_inputs["step_lens"] = paddle.full([1], 0, dtype="int32").cpu()
        self.share_inputs["recover_block_list"] = paddle.full([max_num_seqs], -1, dtype="int32").cpu()
        self.share_inputs["recover_lens"] = paddle.full([1], 0, dtype="int32").cpu()
        self.share_inputs["need_block_list"] = paddle.full([max_num_seqs], -1, dtype="int32").cpu()
        self.share_inputs["need_block_len"] = paddle.full([1], 0, dtype="int32").cpu()
        self.share_inputs["used_list_len"] = paddle.full([max_num_seqs], 0, dtype="int32").cpu()
        self.share_inputs["infer_seed"] = paddle.full([max_num_seqs, 1], 0, dtype="int64").cpu()
        self.share_inputs["first_token_ids"] = paddle.full([max_num_seqs, 1], -1, dtype="int64")
        self.share_inputs["ori_seq_lens_encoder"] = paddle.full([max_num_seqs, 1], 0, dtype="int32").cpu()
        self.share_inputs["ori_seq_lens_decoder"] = paddle.full([max_num_seqs, 1], 0, dtype="int32").cpu()
        self.share_inputs["system_lens"] = paddle.full([max_num_seqs, 1], 0, dtype="int32")
        self.share_inputs["system_ids"] = paddle.full([max_num_seqs, 1], -1, dtype="int32")

        self.share_inputs["ids_remove_padding"] = paddle.full(
            [max_num_seqs * self.model_config.max_model_len], 0, dtype="int64"
        )
        self.share_inputs["cum_offsets"] = paddle.full([max_num_seqs, 1], 0, dtype="int32")
        self.share_inputs["padding_offset"] = paddle.full([max_num_seqs, 1], 0, dtype="int32")
        self.share_inputs["cu_seqlens_q"] = paddle.full([max_num_seqs, 1], 0, dtype="int32")
        self.share_inputs["cu_seqlens_k"] = paddle.full([max_num_seqs, 1], 0, dtype="int32")
        # AttentionBackend buffers
        self.share_inputs["decoder_batch_ids"] = paddle.full([max_num_seqs, 1], 0, dtype="int32")
        self.share_inputs["decoder_tile_ids_per_batch"] = paddle.full([max_num_seqs, 1], 0, dtype="int32")

        # Initialize rotary position embedding
        tmp_position_ids = paddle.arange(self.model_config.max_model_len).reshape((1, -1))
        # TODO(gongshaotian): move to models
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
        self.share_inputs["block_tables"] = paddle.full([max_num_seqs, pre_max_block_num], -1, dtype="int32").cpu()

        # Initialize free list
        free_list = list(
            range(
                self.cache_config.total_block_num - 2,
                int(self.cache_config.total_block_num * self.cache_config.kv_cache_ratio) - 1,
                -1,
            )
        )
        self.free_list_len = len(free_list)
        self.share_inputs["free_list"] = paddle.to_tensor(free_list, dtype="int32").cpu()
        self.share_inputs["free_list_len"] = paddle.full([1], self.free_list_len, dtype="int32").cpu()

        # Initialize stop seqs
        self.share_inputs["stop_seqs_len"] = paddle.full([self.model_config.max_stop_seqs_num], 0, dtype="int32")
        self.share_inputs["stop_seqs"] = paddle.full(
            [self.model_config.max_stop_seqs_num, self.model_config.stop_seqs_max_len], -1, dtype="int32"
        )
        if self.speculative_decoding:
            max_draft_token_num = self.speculative_config.num_speculative_tokens
            self.share_inputs["input_ids_cpu"] = paddle.full(
                shape=[max_num_seqs, self.model_config.max_model_len], fill_value=1, dtype="int64"
            ).cpu()
            self.share_inputs["accept_tokens"] = paddle.full(
                shape=[max_num_seqs, max_draft_token_num + 1], fill_value=0, dtype="int64"
            )
            self.share_inputs["accept_num"] = paddle.full(shape=[max_num_seqs], fill_value=0, dtype="int32")
            self.share_inputs["draft_tokens"] = paddle.full(
                shape=[max_num_seqs, max_draft_token_num + 1], fill_value=0, dtype="int64"
            )

            self.share_inputs["actual_draft_token_num"] = paddle.full(
                shape=[max_num_seqs], fill_value=max_draft_token_num, dtype="int32"
            )
            self.share_inputs["output_cum_offsets"] = paddle.full(shape=[max_num_seqs, 1], fill_value=0, dtype="int32")
            self.share_inputs["output_padding_offset"] = paddle.full(
                shape=[max_num_seqs * (max_draft_token_num + 1)], fill_value=0, dtype="int32"
            )

    def _prepare_inputs(self) -> None:
        """prepare the model inputs"""
        from fastdeploy.model_executor.ops.intel_hpu import prepare_block_metadata

        (
            ids_remove_padding,
            rotary_embs,
            block_groups,
            block_list,
            block_indices,
            block_offsets,
            block_mapping,
            attention_mask,
            batch_ids,
            total_batch,
            is_prompt,
        ) = prepare_block_metadata(
            self.share_inputs["input_ids"],
            self.share_inputs["rope_emb"],
            self.share_inputs["block_tables"],
            self.share_inputs["seq_lens_encoder"],
            self.share_inputs["seq_lens_decoder"],
            self.cache_config.block_size,
            self.model_config.dtype,
            self.scheduler_config.max_num_batched_tokens,
        )
        is_prompt = is_prompt.item() == 1 if is_prompt.item() > 0 else None
        if is_prompt is True:
            attention_mask = None
        # cum_offsets = None
        self.share_inputs["ids_remove_padding"] = ids_remove_padding
        self.share_inputs["rotary_embs"] = rotary_embs
        self.share_inputs["block_groups"] = block_groups
        self.share_inputs["block_list"] = block_list
        self.share_inputs["block_indices"] = block_indices
        self.share_inputs["block_offsets"] = block_offsets
        self.share_inputs["block_mapping"] = block_mapping
        self.share_inputs["block_bias"] = attention_mask
        self.share_inputs["block_size"] = self.cache_config.block_size
        self.share_inputs["batch_ids"] = batch_ids
        self.share_inputs["total_batch"] = total_batch.item()
        self.share_inputs["is_prompt"] = is_prompt
        self.initialize_forward_meta()

    def _prepare_sampler_inputs(self, sampled_ids) -> None:
        if self.forward_meta.total_batch == self.share_inputs["temperature"].shape[0]:
            self.sampling_metadata = SamplingMetadata(
                temperature=self.share_inputs["temperature"],
                top_p=self.share_inputs["top_p"],
                step_idx=self.share_inputs["step_idx"],
                prompt_ids=self.share_inputs["input_ids"],
                pre_token_ids=self.share_inputs["pre_ids"],
                stop_flags=self.share_inputs["stop_flags"],
                seq_lens_encoder=self.share_inputs["seq_lens_encoder"],
                seq_lens_decoder=self.share_inputs["seq_lens_decoder"],
                frequency_penalties=self.share_inputs["frequency_score"],
                presence_penalties=self.share_inputs["presence_score"],
                repetition_penalties=self.share_inputs["penalty_score"],
                min_dec_lens=self.share_inputs["min_dec_len"],
                bad_words_token_ids=self.share_inputs["bad_tokens"],
                eos_token_ids=self.share_inputs["eos_token_id"],
            )
        else:
            from fastdeploy.model_executor.ops.intel_hpu import fused_index_select

            (
                temperature,
                top_p,
                step_idx,
                prompt_token_ids,
                pre_token_ids,
                stop_flags,
                seq_lens_encoder,
                seq_lens_decoder,
                frequency_penalties,
                presence_penalties,
                repetition_penalties,
                min_dec_lens,
            ) = fused_index_select(
                self.share_inputs["temperature"],
                self.share_inputs["top_p"],
                self.share_inputs["step_idx"],
                self.share_inputs["input_ids"],
                self.share_inputs["pre_ids"],
                self.share_inputs["stop_flags"],
                self.share_inputs["seq_lens_encoder"],
                self.share_inputs["seq_lens_decoder"],
                self.share_inputs["frequency_score"],
                self.share_inputs["presence_score"],
                self.share_inputs["penalty_score"],
                self.share_inputs["min_dec_len"],
                sampled_ids,
                self.forward_meta.total_batch,
            )

            self.sampling_metadata = SamplingMetadata(
                temperature=temperature,
                top_p=top_p,
                step_idx=step_idx,
                prompt_ids=prompt_token_ids,
                pre_token_ids=pre_token_ids,
                stop_flags=stop_flags,
                seq_lens_encoder=seq_lens_encoder,
                seq_lens_decoder=seq_lens_decoder,
                frequency_penalties=frequency_penalties,
                presence_penalties=presence_penalties,
                repetition_penalties=repetition_penalties,
                min_dec_lens=min_dec_lens,
                bad_words_token_ids=self.share_inputs["bad_tokens"],
                eos_token_ids=self.share_inputs["eos_token_id"],
            )

    def load_model(self) -> None:
        """load or download model"""
        logger.info(f"Starting to load model {self.model_config.architectures[0]}")
        time_before_load = time.perf_counter()
        # 1. Load original model
        model_loader = get_model_loader(load_config=self.fd_config.load_config)
        self.model = model_loader.load_model(fd_config=self.fd_config)
        # 1.1 Load RL dynamic model
        if self.fd_config.load_config.dynamic_load_weight:
            from fastdeploy.rl.dynamic_weight_manager import DynamicWeightManager

            self.dynamic_weight_manager = DynamicWeightManager(self.fd_config, self.model)

        # 2. Load lora model

        # 3. Load drafter model(for speculative decoding)

        # 4. Convert model to HPU format
        self.model = convert_model(self.model)

        time_after_load = time.perf_counter()
        logger.info(f"Model loading took {time_after_load - time_before_load} seconds")

        # 4. Init proposer for speculative method
        self.init_speculative_proposer()

    def get_model(self) -> nn.Layer:
        """get current model"""
        return self.model

    def initialize_forward_meta(self):
        """
        Initialize forward meta and attention meta data
        """
        # Initialize forward meta
        self.forward_meta = HPUForwardMeta.init_forward_meta(self.share_inputs, self.attn_backends[0])

        # Initialzie attention meta data
        for attn_backend in self.attn_backends:
            attn_backend.init_attention_metadata(self.forward_meta)

    def clear_cache(self):
        """Clear cached data from shared inputs and forward metadata."""
        self.share_inputs.pop("caches", None)
        if self.forward_meta is not None:
            self.forward_meta.clear_caches()

    def initialize_kv_cache(self) -> None:
        """
        Initialize kv cache
        """
        cache_kvs = {}
        max_block_num = self.num_gpu_blocks

        key_cache_shape, value_cache_shape = self.attn_backends[0].get_kv_cache_shape(max_num_blocks=max_block_num)

        for i in range(self.model_config.num_hidden_layers):
            cache_type = self.model_config.dtype
            cache_kvs["key_caches_{}".format(i)] = paddle.full(
                shape=key_cache_shape,
                fill_value=0,
                dtype=cache_type,
            )
            cache_kvs["value_caches_{}".format(i)] = paddle.full(
                shape=value_cache_shape,
                fill_value=0,
                dtype=cache_type,
            )
        self.share_inputs["caches"] = list(cache_kvs.values())
        for value in cache_kvs.values():
            del value

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

        # Get the attention backend
        attn_cls = get_attention_backend()
        attn_backend = attn_cls(
            self.fd_config, kv_num_heads=self.model_config.kv_num_heads, num_heads=num_heads, head_dim=head_dim
        )
        if attn_backend is None:
            raise NotImplementedError(
                "Attention backend which you specified is not supported, please set FD_ATTENTION_BACKEND correctly."
            )
        self.attn_backends.append(attn_backend)

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
            num_tokens:
            expected_decode_len: Expected number of tokens generated
        """
        self._dummy_prefill_inputs(
            num_tokens=num_tokens, batch_size=batch_size, expected_decode_len=expected_decode_len
        )
        if self.speculative_method in ["mtp"]:
            raise NotImplementedError("speculative sampling is not supported on Intel HPU.")
        while True:

            # 1. Compute real num_tokens
            self._prepare_inputs()

            # 2. Initialize attention backend and forward meta data
            model_output = self.model(self.share_inputs["ids_remove_padding"], self.forward_meta)

            hiddden_states = rebuild_padding_v3_1(
                model_output,
                self.forward_meta.batch_ids,
                self.forward_meta.total_batch,
                self.forward_meta.seq_lens_encoder,
                self.forward_meta.is_prompt,
            )
            # 5. Execute spec decode
            logits = self.model.compute_logits(hiddden_states)

            self._prepare_sampler_inputs(self.forward_meta.batch_ids)
            sampled_token_ids = self.sampler(
                logits,
                self.sampling_metadata,
                self.forward_meta.batch_ids,
                self.forward_meta.seq_lens_encoder.shape[0],
                self.rank,
                self.local_rank,
            )
            if self.parallel_config.tensor_parallel_size > 1:
                dtype = sampled_token_ids.dtype
                sampled_token_ids = sampled_token_ids.to("float32")
                tensor_model_parallel_all_reduce_custom(sampled_token_ids)
                sampled_token_ids = sampled_token_ids.to(dtype)

            # 6. post process
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
                mp_rank=self.local_rank,
                use_ep=self.parallel_config.use_ep,
                draft_tokens=self.share_inputs["draft_tokens"] if self.speculative_decoding else None,
                actual_draft_token_num=(
                    self.share_inputs["actual_draft_token_num"] if self.speculative_decoding else None
                ),
                accept_tokens=self.share_inputs["accept_tokens"] if self.speculative_decoding else None,
                accept_num=self.share_inputs["accept_num"] if self.speculative_decoding else None,
            )

            post_process_hpu(
                sampled_token_ids=sampled_token_ids, model_output=model_output_data, is_warmuping=self.is_warmuping
            )

            # 7. Updata 'infer_seed' and step_cuda()
            self.share_inputs["infer_seed"].add_(self.infer_seed_increment)
            self.share_inputs["infer_seed"][:] %= self.MAX_INFER_SEED
            step_intel_hpu(self.share_inputs, self.cache_config.block_size, self.model_config.max_model_len)

            if int((self.share_inputs["seq_lens_this_time"] > 0).sum()) == 0:
                break

    def _update_chunked_prefill(self, tasks):
        """
        更新chunked prefill相关参数
        """
        if not self.cache_config.enable_chunked_prefill:
            return

        for task in tasks:
            if task.get("prefill_chunk_info", None) is None:
                continue

            if task.chunk_idx > len(task.prefill_chunk_info):
                continue
            self.restore_chunked_prefill_request[task.request_id] = task

        for id, task in list(self.restore_chunked_prefill_request.items()):
            idx = task.idx
            logger.debug(f"{task.request_id} chunked prefill {task.chunk_idx}/{len(task.prefill_chunk_info)}")
            start_idx = sum(task.prefill_chunk_info[: task.chunk_idx])
            if task.chunk_idx == len(task.prefill_chunk_info):
                self.share_inputs["seq_lens_this_time"][idx : idx + 1] = 1
                self.share_inputs["seq_lens_encoder"][idx : idx + 1] = 0
                self.share_inputs["step_idx"][idx : idx + 1] = 1
                self.share_inputs["seq_lens_decoder"][idx : idx + 1] = start_idx + task.get("seq_lens_decoder", 0)
                del self.restore_chunked_prefill_request[task.request_id]
            else:
                token_chunk_size = task.prefill_chunk_info[task.chunk_idx]

                self.share_inputs["seq_lens_this_time"][idx : idx + 1] = token_chunk_size
                self.share_inputs["input_ids"][idx, :token_chunk_size] = np.array(
                    task.prompt_token_ids[start_idx : start_idx + token_chunk_size]
                )
                self.share_inputs["seq_lens_encoder"][idx : idx + 1] = token_chunk_size
                self.share_inputs["step_idx"][idx : idx + 1] = 0
                self.share_inputs["seq_lens_decoder"][idx : idx + 1] = start_idx + task.get("seq_lens_decoder", 0)
            if self.speculative_decoding and self.proposer.is_chunk_prefill_enabled():
                self.proposer.update_task_chunk_prefill(task)
            task.chunk_idx += 1

    def _dummy_sampler_run(self) -> paddle.Tensor:
        """ """
        pass

    def update_warmup_inputs(self, requests, is_decode=False, context_len=0) -> None:
        """
        Update the shared input tensors for warmup requests.
        Args:
            requests (list): List of request dicts containing input data.
            is_decode (bool, optional): If True, sets up inputs for decode phase. Defaults to False.
            context_len (int, optional): The length of the context (prefix) to use for prefix caching during warmup.
                If >0, this value is used to set the decoder sequence length for prefill (prefix caching).
                Typically, set to the number of tokens in the prefix to be cached. Defaults to 0 (no prefix caching).
        This parameter affects the warmup behavior for prefix caching by controlling how much of the input
        is considered as context for the decoder during the prefill phase.
        """
        for i in range(len(requests)):
            request = requests[i]
            idx = request["idx"]
            length = len(request["input_ids"])
            self.share_inputs["input_ids"][idx : idx + 1, :length] = np.array(request["input_ids"])
            if is_decode:
                self.share_inputs["seq_lens_encoder"][idx : idx + 1] = 0
                self.share_inputs["seq_lens_decoder"][idx : idx + 1] = length
                self.share_inputs["seq_lens_this_time"][idx : idx + 1] = 1
                self.share_inputs["step_seq_lens_encoder"][idx : idx + 1] = 0
                self.share_inputs["step_seq_lens_decoder"][idx : idx + 1] = length
                self.share_inputs["step_idx"][idx : idx + 1] = 1
            else:
                self.share_inputs["seq_lens_encoder"][idx : idx + 1] = length
                self.share_inputs["seq_lens_decoder"][idx : idx + 1] = context_len
                self.share_inputs["seq_lens_this_time"][idx : idx + 1] = length
                self.share_inputs["step_seq_lens_encoder"][idx : idx + 1] = length
                self.share_inputs["step_seq_lens_decoder"][idx : idx + 1] = 0
                self.share_inputs["step_idx"][idx : idx + 1] = 0

            if len(request["eos_token_ids"]) < self.model_config.eos_tokens_lens:
                request["eos_token_ids"].append(request["eos_token_ids"][0])
            self.share_inputs["eos_token_id"][:] = np.array(request["eos_token_ids"], dtype="int64").reshape(-1, 1)

            self.share_inputs["top_p"][idx : idx + 1] = request.get("top_p", 0.7)
            self.share_inputs["temperature"][idx : idx + 1] = request.get("temperature", 0.95)
            self.share_inputs["penalty_score"][idx : idx + 1] = request.get("repetition_penalty", 1.0)
            self.share_inputs["frequency_score"][idx : idx + 1] = request.get("frequency_penalty", 0.0)
            self.share_inputs["presence_score"][idx : idx + 1] = request.get("presence_penalty", 0.0)

            self.share_inputs["min_dec_len"][idx : idx + 1] = request.get("min_tokens", 1)
            self.share_inputs["max_dec_len"][idx : idx + 1] = request.get("max_tokens", 1)
            self.share_inputs["stop_flags"][idx : idx + 1] = False

            self.share_inputs["first_token_ids"][idx : idx + 1] = self.share_inputs["input_ids"][idx : idx + 1, :1]
            self.share_inputs["ori_seq_lens_encoder"][idx : idx + 1] = length

            if request.get("seed") is not None:
                self.share_inputs["infer_seed"][idx : idx + 1] = request.get("seed")
            encoder_block_num = len(request["block_tables"])
            self.share_inputs["encoder_block_lens"][idx : idx + 1] = encoder_block_num
            self.share_inputs["block_tables"][idx : idx + 1, :] = -1
            self.share_inputs["block_tables"][idx : idx + 1, :encoder_block_num] = np.array(
                request["block_tables"], dtype="int32"
            )

        self.share_inputs["not_need_stop"][0] = True

    def warm_up_bucket(self) -> None:
        max_prefill_batch = int(os.getenv("MAX_PREFILL_NUM", "3"))
        warmup_max_model_len = min(int(os.environ.get("HPU_WARMUP_MODEL_LEN", 4096)), self.model_config.max_model_len)
        prefill_batchs = []
        prefill_batch_step = int(os.environ.get("BATCH_STEP_PREFILL", 1))
        prefill_seq_step = int(os.environ.get("SEQUENCE_STEP_PREFILL", 128))
        current_prefill_batch = prefill_batch_step
        while current_prefill_batch <= max_prefill_batch:
            prefill_batchs.append(int(current_prefill_batch))
            current_prefill_batch += prefill_batch_step

        max_prefill_length = self.cache_config.block_size + warmup_max_model_len
        prefill_context_block_step = int(os.environ.get("CONTEXT_BLOCK_STEP_PREFILL", 1))
        for prefill_batch in prefill_batchs:
            for prefill_length_with_context in range(
                self.cache_config.block_size, max_prefill_length, prefill_seq_step
            ):
                if prefill_length_with_context * prefill_batch > self.scheduler_config.max_num_batched_tokens:
                    continue
                for context_len in range(
                    0, prefill_length_with_context, self.cache_config.block_size * prefill_context_block_step
                ):
                    prefill_length = prefill_length_with_context - context_len
                    logger.info(
                        f"Warmup prefill_batch: {prefill_batch}, prefill_length: {prefill_length}, context_len: {context_len} start"
                    )
                    requests = [
                        {
                            "idx": i,
                            "input_ids": [5] * (prefill_length_with_context - context_len - 1),
                            "block_tables": list(range(prefill_length_with_context // self.cache_config.block_size)),
                            "eos_token_ids": [2],
                        }
                        for i in range(prefill_batch)
                    ]
                    self.update_warmup_inputs(requests, is_decode=False, context_len=context_len)
                    self.execute_model()
                    logger.info(
                        f"warmup prefill_batch: {prefill_batch}, prefill_length: {prefill_length}, context_len: {context_len} done"
                    )
                    # when disable prefix caching, only run context_len = 0 for each prefill_batch
                    if not self.cache_config.enable_prefix_caching:
                        break

        decode_batchs = []
        decode_batch_step = int(os.environ.get("BATCH_STEP_DECODE", 4))
        current_decode_batch = decode_batch_step
        while current_decode_batch <= self.scheduler_config.max_num_seqs:
            decode_batchs.append(int(current_decode_batch))
            current_decode_batch += decode_batch_step

        decode_block_nums = []
        decode_block_num_step = int(os.environ.get("BLOCK_STEP_DECODE", 16))
        current_decode_block_num = decode_block_num_step
        pre_max_block_num = (
            warmup_max_model_len + self.cache_config.block_size - 1
        ) // self.cache_config.block_size + self.cache_config.enc_dec_block_num
        while current_decode_block_num <= min(
            self.num_gpu_blocks, pre_max_block_num * self.scheduler_config.max_num_seqs
        ):
            decode_block_nums.append(int(current_decode_block_num))
            current_decode_block_num += decode_block_num_step

        logger.info(f"warmup decode_batchs: {decode_batchs}, decode_block_nums: {decode_block_nums} start")
        for decode_batch in decode_batchs:
            for decode_block_num in decode_block_nums:
                if decode_block_num < decode_batch:
                    continue
                if decode_block_num // decode_batch * self.cache_config.block_size > warmup_max_model_len:
                    continue
                blocks = [decode_block_num // decode_batch for _ in range(decode_batch)]
                remain_block_num = decode_block_num % decode_batch
                b = 0
                while remain_block_num > 0:
                    blocks[b] += 1
                    remain_block_num -= 1
                    b += 1
                if blocks[0] * self.cache_config.block_size > warmup_max_model_len:
                    continue
                logger.info(f"warmup decode_batch: {decode_batch}, decode_block_num: {decode_block_num} start")
                requests = [
                    {
                        "idx": i,
                        "input_ids": [5] * (blocks[i] * self.cache_config.block_size - 1),
                        "block_tables": list(range(blocks[i])),
                        "eos_token_ids": [2],
                    }
                    for i in range(decode_batch)
                ]
                self.update_warmup_inputs(requests, is_decode=True)
                self.execute_model()
                logger.info(f"Warmup decode_batch: {decode_batch}, decode_block_num: {decode_block_num} done")
        self.share_inputs["not_need_stop"][0] = False
        logger.info("Warmup bucket done")

    def capture_model(self) -> None:
        """
        Trigger CUDA Graph capture for all shapes in 'CudaGraphConfig.cudagraph_capture_sizes'
        """
        if not self.use_cudagraph:
            logger.info("Skipping CUDA graph capture. Please check GraphOptimizationConfig")
            return
        time_before_capture = time.perf_counter()
        expected_decode_len = 1
        capture_sizes = self.cudagraph_capture_sizes.copy()
        for batch_size in sorted(capture_sizes, reverse=True):
            self._dummy_run(
                num_tokens=self.model_config.max_model_len,
                batch_size=batch_size,
                in_capturing=True,
                expected_decode_len=expected_decode_len,
            )
            logger.info(f"Warm up the model with the batch size:{batch_size}, num tokens:{expected_decode_len}")

        time_after_capture = time.perf_counter()
        logger.info(f"Cuda Graph capturing took {time_after_capture - time_before_capture} seconds")

    def execute_model(
        self,
        model_forward_batch: Optional[List[Request]] = None,
    ) -> Optional[ModelRunnerOutput]:
        """
        The Entrance of model execute.
        Args:
            model_forward_batch: 'Request' contains information related to prompt and is an abstract
            class at the server level, which is too granular for ModelRunner.
            We plan to replace it with 'ModelForwardBatch'.
            intermediate_tensors:
        """
        # # 1. Prepare inputs of model and decoder.
        start_time = time.time()
        self._prepare_inputs()
        # self.share_inputs["ids_remove_padding"].cpu()
        # # 2. Padding inputs for cuda grph
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000
        real_bs = self.share_inputs["ids_remove_padding"].shape[0]
        hpu_model_runner_profile_logger.info(f"_prepare_inputs time(ms): {execution_time}, BT={real_bs}")
        start_time = time.time()
        # # 3. Execute model
        model_output = self.model(self.share_inputs["ids_remove_padding"], self.forward_meta)
        if self.is_hpu_perf_breakdown_sync_mode:
            model_output.cpu()
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000
        hpu_model_runner_profile_logger.info(
            f"Model execution time(ms): {execution_time}, BT={real_bs}, block_list_shape={self.share_inputs['block_list'].shape}, block_indices_shape={self.share_inputs['block_indices'].shape}"
        )

        start_time = time.time()
        start_time0 = time.time()
        hiddden_states = rebuild_padding_v3_1(
            model_output,
            self.forward_meta.batch_ids,
            self.forward_meta.total_batch,
            self.forward_meta.seq_lens_encoder,
            self.forward_meta.is_prompt,
        )
        end_time0 = time.time()
        execution_time0 = (end_time0 - start_time0) * 1000
        hpu_model_runner_profile_logger.info(f"RebuildPadding execution time(ms): {execution_time0}, BT={real_bs}")
        # # 4. Compute logits, Sample
        start_time1 = time.time()
        logits = self.model.compute_logits(hiddden_states)
        end_time1 = time.time()
        execution_time1 = (end_time1 - start_time1) * 1000
        hpu_model_runner_profile_logger.info(f"ComputeLogits execution time(ms): {execution_time1}, BT={real_bs}")

        # data = np.random.rand(self.scheduler_config.max_num_seqs, self.model_config.vocab_size).astype(np.float32)
        # logits = paddle.to_tensor(data, dtype='bfloat16')
        start_time2 = time.time()
        self._prepare_sampler_inputs(self.forward_meta.batch_ids)
        sampled_token_ids = self.sampler(
            logits,
            self.sampling_metadata,
            self.forward_meta.batch_ids,
            self.forward_meta.seq_lens_encoder.shape[0],
            self.rank,
            self.local_rank,
        )
        if self.parallel_config.tensor_parallel_size > 1:
            dtype = sampled_token_ids.dtype
            sampled_token_ids = sampled_token_ids.to("float32")
            tensor_model_parallel_all_reduce_custom(sampled_token_ids)
            sampled_token_ids = sampled_token_ids.to(dtype)
        if self.is_hpu_perf_breakdown_sync_mode:
            sampled_token_ids.cpu()
        end_time2 = time.time()
        execution_time2 = (end_time2 - start_time2) * 1000
        hpu_model_runner_profile_logger.info(f"Sampler execution time(ms): {execution_time2}, BT={real_bs}")
        # 5. Post Process
        start_time3 = time.time()
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
            mp_rank=self.local_rank,
            use_ep=self.parallel_config.use_ep,
            draft_tokens=self.share_inputs["draft_tokens"] if self.speculative_decoding else None,
            actual_draft_token_num=self.share_inputs["actual_draft_token_num"] if self.speculative_decoding else None,
            accept_tokens=self.share_inputs["accept_tokens"] if self.speculative_decoding else None,
            accept_num=self.share_inputs["accept_num"] if self.speculative_decoding else None,
        )

        # if self.speculative_config.method in ["mtp"] and self.scheduler_config.splitwise_role == "prefill":
        #     skip_save_output = True
        # else:
        #     skip_save_output = False
        post_process_hpu(
            sampled_token_ids=sampled_token_ids, model_output=model_output_data, is_warmuping=self.is_warmuping
        )
        end_time3 = time.time()
        execution_time3 = (end_time3 - start_time3) * 1000
        hpu_model_runner_profile_logger.info(f"PostProcessHpu execution time(ms): {execution_time3}, BT={real_bs}")
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000
        hpu_model_runner_profile_logger.info(f"PostProcessing execution time(ms): {execution_time}, BT={real_bs}")

        # 6. Speculative decode
        if self.speculative_decoding:
            if self.speculative_method == "mtp":
                self.proposer.run(full_hidden_states=hiddden_states)
            else:
                self.proposer.run(share_inputs=self.share_inputs)

        # 7. Updata 'infer_seed' and step_cuda()
        self.share_inputs["infer_seed"].add_(self.infer_seed_increment)
        self.share_inputs["infer_seed"][:] %= self.MAX_INFER_SEED
        start_time = time.time()
        step_intel_hpu(self.share_inputs, self.cache_config.block_size, self.model_config.max_model_len)
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000
        hpu_model_runner_profile_logger.info(f"StepPaddle execution time(ms): {execution_time}, BT={real_bs}")
        self._update_chunked_prefill(model_forward_batch)

        if int(os.environ.get("HABANA_PROFILE", 0)) == 1:
            self.prof.step()
        return None

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

    def profile_run(self) -> None:
        """Execute a forward pass with dummy inputs to profile the memory usage of the model."""

        # Initialize kv cache for profile run. After profile run kv cache will be reset.
        # TODO(gongshaotian): Optimize the management logic of kvcache
        self.num_gpu_blocks = self.cache_config.total_block_num
        self.initialize_kv_cache()

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
            range(self.num_gpu_blocks - 2, int(self.num_gpu_blocks * self.cache_config.kv_cache_ratio) - 1, -1)
        )
        self.free_list_len = len(free_list)
        self.share_inputs.update(
            {
                "free_list": paddle.to_tensor(free_list, dtype="int32").cpu(),
                "free_list_len": paddle.full([1], self.free_list_len, dtype="int32").cpu(),
            }
        )

        self.parallel_config.do_profile = False

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
        """ """
        return self.share_inputs["not_need_stop"][0]
