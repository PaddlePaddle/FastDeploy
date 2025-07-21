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

import random
import time
from typing import Dict, List, Optional

import numpy as np
import paddle
from paddle import nn

from fastdeploy.config import FDConfig
from fastdeploy.engine.request import Request
from fastdeploy.model_executor.forward_meta import ForwardMeta, XPUForwardMeta
from fastdeploy.model_executor.layers.attention import get_attention_backend
from fastdeploy.model_executor.layers.attention.base_attention_backend import (
    AttentionBackend,
)
from fastdeploy.model_executor.layers.rotary_embedding import get_rope
from fastdeploy.model_executor.layers.sample.meta_data import SamplingMetadata
from fastdeploy.model_executor.layers.sample.sampler import Sampler
from fastdeploy.model_executor.model_loader import get_model_from_loader
from fastdeploy.utils import get_logger
from fastdeploy.worker.model_runner_base import ModelRunnerBase
from fastdeploy.worker.output import ModelOutputData, ModelRunnerOutput

logger = get_logger("xpu_model_runner", "xpu_model_runner.log")


def xpu_pre_process(
    max_len: int,
    input_ids: paddle.Tensor,
    seq_lens_this_time: int,
    share_inputs: Dict,
    use_speculate_method: bool,
    draft_tokens: Optional[paddle.Tensor] = None,
    seq_lens_encoder: Optional[paddle.Tensor] = None,
    seq_lens_decoder: Optional[paddle.Tensor] = None,
) -> XPUForwardMeta:
    """ """
    cum_offsets_now = paddle.cumsum(max_len - seq_lens_this_time)
    token_num = paddle.sum(seq_lens_this_time)
    from fastdeploy.model_executor.ops.xpu import (
        adjust_batch,
        get_infer_param,
        get_padding_offset,
    )

    (
        ids_remove_padding,
        cum_offsets,
        batch_id_per_token,
        cu_seqlens_q,
        cu_seqlens_k,
    ) = get_padding_offset(input_ids, cum_offsets_now, token_num, seq_lens_this_time)

    share_inputs["ids_remove_padding"] = None  # set this after adjust batch
    share_inputs["cum_offsets"] = cum_offsets
    share_inputs["batch_id_per_token"] = batch_id_per_token
    share_inputs["cu_seqlens_q"] = cu_seqlens_q
    share_inputs["cu_seqlens_k"] = cu_seqlens_k

    xpu_forward_meta = XPUForwardMeta(
        input_ids=share_inputs["input_ids"],
        ids_remove_padding=share_inputs["ids_remove_padding"],
        rotary_embs=share_inputs["rope_emb"],
        attn_backend=None,
        seq_lens_encoder=share_inputs["seq_lens_encoder"],
        seq_lens_decoder=share_inputs["seq_lens_decoder"],
        seq_lens_this_time=share_inputs["seq_lens_this_time"],
        cum_offsets=share_inputs["cum_offsets"],
        batch_id_per_token=share_inputs["batch_id_per_token"],
        cu_seqlens_q=share_inputs["cu_seqlens_q"],
        cu_seqlens_k=share_inputs["cu_seqlens_k"],
        block_tables=share_inputs["block_tables"],
        caches=share_inputs["caches"],
    )

    # Get xpu extra param
    (
        xpu_forward_meta.encoder_batch_map,
        xpu_forward_meta.decoder_batch_map,
        xpu_forward_meta.encoder_batch_idx,
        xpu_forward_meta.decoder_batch_idx,
        xpu_forward_meta.encoder_seq_lod,
        xpu_forward_meta.decoder_context_len,
        xpu_forward_meta.decoder_context_len_cache,
        xpu_forward_meta.encoder_batch_map_cpu,
        xpu_forward_meta.decoder_batch_map_cpu,
        xpu_forward_meta.encoder_batch_idx_cpu,
        xpu_forward_meta.decoder_batch_idx_cpu,
        xpu_forward_meta.encoder_seq_lod_cpu,
        xpu_forward_meta.decoder_context_len_cpu,
        xpu_forward_meta.decoder_context_len_cache_cpu,
        xpu_forward_meta.enc_batch,
        xpu_forward_meta.dec_batch,
        xpu_forward_meta.total_enc_len,
    ) = get_infer_param(seq_lens_encoder, seq_lens_decoder)

    # Adjust batch
    adjusted_input = adjust_batch(
        ids_remove_padding.reshape([-1, 1]),
        cum_offsets,
        xpu_forward_meta.encoder_seq_lod,
        xpu_forward_meta.encoder_batch_idx,
        xpu_forward_meta.decoder_batch_idx,
        xpu_forward_meta.encoder_seq_lod_cpu,
        xpu_forward_meta.encoder_batch_idx_cpu,
        xpu_forward_meta.decoder_batch_idx_cpu,
        xpu_forward_meta.enc_batch,
        xpu_forward_meta.dec_batch,
        None,  # output_padding_offset
        -1,  # max_input_length
    )
    adjusted_input = adjusted_input.squeeze(1)

    share_inputs["ids_remove_padding"] = adjusted_input
    xpu_forward_meta.ids_remove_padding = adjusted_input
    return xpu_forward_meta


def xpu_process_output(
    forward_output,
    cum_offsets: paddle.Tensor,
    xpu_forward_meta: XPUForwardMeta,
) -> paddle.Tensor:
    """ """
    from fastdeploy.model_executor.ops.xpu import gather_next_token

    hiddden_states = gather_next_token(
        forward_output,
        cum_offsets,
        xpu_forward_meta.encoder_seq_lod,
        xpu_forward_meta.encoder_batch_map,
        xpu_forward_meta.decoder_batch_map,
        xpu_forward_meta.encoder_seq_lod_cpu,
        xpu_forward_meta.encoder_batch_map_cpu,
        xpu_forward_meta.decoder_batch_map_cpu,
        xpu_forward_meta.enc_batch,
        xpu_forward_meta.dec_batch,
        None,  # output_padding_offset
        -1,  # max_input_length
    )
    return hiddden_states


def xpu_post_process(
    sampled_token_ids: paddle.Tensor,
    model_output: ModelOutputData,
    skip_save_output: bool,
) -> None:
    """ """
    from fastdeploy.model_executor.ops.xpu import (
        save_output,
        set_stop_value_multi_ends,
        update_inputs,
    )

    # 1. Set stop value
    paddle.assign(
        paddle.where(
            model_output.stop_flags,
            model_output.step_idx,
            model_output.step_idx + 1,
        ),
        model_output.step_idx,
    )
    length_cond = paddle.greater_equal(model_output.step_idx, model_output.max_dec_len)
    paddle.assign(
        paddle.logical_or(model_output.stop_flags, length_cond),
        model_output.stop_flags,
    )
    set_stop_value_multi_ends(
        sampled_token_ids,
        model_output.stop_flags,
        model_output.seq_lens_this_time,
        model_output.eos_token_id,
        model_output.next_tokens,
        False,
    )  # multi ends

    # 2. Update the input buffer of the model
    with paddle.framework._no_check_dy2st_diff():
        update_inputs(
            model_output.stop_flags,
            model_output.not_need_stop,
            model_output.seq_lens_this_time,
            model_output.seq_lens_encoder,
            model_output.seq_lens_decoder,
            model_output.input_ids,
            model_output.stop_nums,
            sampled_token_ids,
            model_output.is_block_step,
        )
    # 3. Transmit the model's output and stop generation signal via message queue.
    #    In the future, we will abandon this approach.
    if not skip_save_output:
        save_output(
            sampled_token_ids,
            model_output.not_need_stop,
            model_output.mp_rank,
            False,  # use_ep
        )


def step_paddle(
    share_inputs: Dict[str, paddle.Tensor],
    block_size: int,
    enc_dec_block_num: int,
) -> None:
    """
    TODO(gongshaotian): normalization name
    """
    from fastdeploy.model_executor.ops.xpu import step_paddle

    step_paddle(
        share_inputs["stop_flags"],
        share_inputs["seq_lens_this_time"],
        share_inputs["step_seq_lens_encoder"],
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
        share_inputs["input_ids"],
        share_inputs["pre_ids"],
        share_inputs["step_idx"],
        share_inputs["next_tokens"],
        share_inputs["first_token_ids"],
        block_size,
        enc_dec_block_num,
    )


class XPUModelRunner(ModelRunnerBase):
    """ """

    def __init__(self, fd_config: FDConfig, device: str, rank: int, local_rank: int):
        super().__init__(fd_config=fd_config, device=device)
        self.rank = rank
        self.local_rank = local_rank

        #  Sampler
        self.sampler = Sampler()

        # Lazy initialize kv cache after model loading
        # self.kv_caches: list[paddle.Tensor] = []

        # Cuda Graph
        self.use_cudagraph = False
        self.input_ids = paddle.zeros(self.parallel_config.max_num_seqs, dtype="int32")

        # Initialize share inputs
        self._init_share_inputs(self.fd_config.parallel_config.max_num_seqs)
        self.infer_seed_increment = paddle.full(
            shape=[self.parallel_config.max_num_seqs, 1],
            fill_value=4,
            dtype="int64",
        )

        # Initialize attention Backend
        # Note(gonshaotian): Currently, all attention layers share one attention backend instance.
        # In the future, we will expand it as a list.
        self.attn_backends: list[AttentionBackend] = []

        self.initialize_attn_backend()

        # Forward meta store the global meta information of the forward
        self.forward_meta: ForwardMeta = None

    def process_prefill_inputs(self, req_dicts: List[Request]):
        """Process inputs for prefill tasks and update share_inputs buffer"""
        req_len = len(req_dicts)
        for i in range(req_len):
            request = req_dicts[i]
            idx = request.idx
            length = request.prompt_token_ids_len
            self.share_inputs["input_ids"][idx : idx + 1, :length] = np.array(request.prompt_token_ids)
            if len(request.eos_token_ids) < self.parallel_config.eos_tokens_lens:
                request.eos_token_ids.append(request.eos_token_ids[0])
            self.share_inputs["eos_token_id"][:] = np.array(request.eos_token_ids, dtype="int64").reshape(-1, 1)
            self.share_inputs["pre_ids"][idx : idx + 1] = -1
            self.share_inputs["top_p"][idx : idx + 1] = request.get("top_p", 0.7)
            self.share_inputs["top_k"][idx : idx + 1] = request.get("top_k", 0)
            self.share_inputs["temperature"][idx : idx + 1] = request.get("temperature", 0.95)
            self.share_inputs["penalty_score"][idx : idx + 1] = request.get("repetition_penalty", 1.0)
            self.share_inputs["frequency_score"][idx : idx + 1] = request.get("frequency_penalty", 0.0)
            self.share_inputs["presence_score"][idx : idx + 1] = request.get("presence_penalty", 0.0)
            self.share_inputs["seq_lens_this_time"][idx : idx + 1] = length
            self.share_inputs["step_seq_lens_encoder"][idx : idx + 1] = length
            self.share_inputs["seq_lens_encoder"][idx : idx + 1] = length
            self.share_inputs["seq_lens_decoder"][idx : idx + 1] = 0
            self.share_inputs["step_idx"][idx : idx + 1] = 0
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

        self.share_inputs["not_need_stop"][0] = True

    def _init_share_inputs(self, max_num_seqs: int):
        """Initialize all share buffers for model inputs.
        Note: In the future, we may abandon share buffers.
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
            self.parallel_config.pad_token_id,
            dtype="int64",
        )
        self.share_inputs["eos_token_id"] = paddle.full([self.parallel_config.eos_tokens_lens, 1], 0, dtype="int64")
        self.share_inputs["top_p"] = paddle.full([max_num_seqs, 1], self.model_config.top_p, dtype="float32")
        self.share_inputs["top_k"] = paddle.full([max_num_seqs, 1], 0, dtype="int64")
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

        self.share_inputs["min_dec_len"] = paddle.full([max_num_seqs, 1], self.model_config.min_length, dtype="int64")
        self.share_inputs["max_dec_len"] = paddle.full(
            [max_num_seqs, 1], self.model_config.max_model_len, dtype="int64"
        )
        self.share_inputs["min_length"] = paddle.full([max_num_seqs, 1], self.model_config.min_length, dtype="int64")
        self.share_inputs["max_length"] = paddle.full(
            [max_num_seqs, 1], self.model_config.max_model_len, dtype="int64"
        )
        self.share_inputs["seq_lens_this_time"] = paddle.full(max_num_seqs, 0, dtype="int32")
        self.share_inputs["seq_lens_encoder"] = paddle.full([max_num_seqs, 1], 0, dtype="int32")
        self.share_inputs["seq_lens_decoder"] = paddle.full([max_num_seqs, 1], 0, dtype="int32")
        self.share_inputs["step_seq_lens_encoder"] = paddle.full([max_num_seqs, 1], 0, dtype="int32")
        self.share_inputs["step_idx"] = paddle.full([max_num_seqs, 1], 0, dtype="int64")
        self.share_inputs["not_need_stop"] = paddle.full(
            [1], False, dtype="bool"
        ).cpu()  # TODO(gongshaotian): move to pinnd memory
        self.share_inputs["stop_flags"] = paddle.full([max_num_seqs, 1], True, dtype="bool")
        self.share_inputs["stop_nums"] = paddle.full([1], max_num_seqs, dtype="int64")

        self.share_inputs["bad_tokens"] = paddle.full([1], -1, dtype="int64")
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

        # Initialize rotary position embedding
        tmp_position_ids = paddle.arange(self.parallel_config.max_model_len).reshape((1, -1))
        # TODO(gongshaotian): move to models
        self.share_inputs["rope_emb"] = get_rope(
            rotary_dim=self.model_config.head_dim,
            position_ids=tmp_position_ids,
            base=self.model_config.rope_theta,
            model_config=self.model_config,
        )

        # Set block tables
        pre_max_block_num = (
            self.parallel_config.max_model_len + self.parallel_config.block_size - 1
        ) // self.parallel_config.block_size + self.parallel_config.enc_dec_block_num
        self.share_inputs["block_tables"] = paddle.full([max_num_seqs, pre_max_block_num], -1, dtype="int32")

        # Initialize free list
        free_list = list(
            range(
                self.parallel_config.total_block_num - 1,
                int(self.parallel_config.total_block_num * self.parallel_config.kv_cache_ratio) - 1,
                -1,
            )
        )
        self.free_list_len = len(free_list)
        self.share_inputs["free_list"] = paddle.to_tensor(free_list, dtype="int32")
        self.share_inputs["free_list_len"] = paddle.full([1], self.free_list_len, dtype="int32")

        # Initialize stop seqs
        self.share_inputs["stop_seqs_len"] = paddle.full([self.model_config.max_stop_seqs_num], 0, dtype="int32")
        self.share_inputs["stop_seqs"] = paddle.full(
            [
                self.model_config.max_stop_seqs_num,
                self.model_config.stop_seqs_max_len,
            ],
            -1,
            dtype="int32",
        )

    def _prepare_inputs(self) -> None:
        """prepare the model inputs"""
        self.forward_meta = xpu_pre_process(
            self.parallel_config.max_model_len,
            self.share_inputs["input_ids"],
            self.share_inputs["seq_lens_this_time"],
            self.share_inputs,
            use_speculate_method=False,
            draft_tokens=None,
            seq_lens_encoder=self.share_inputs["seq_lens_encoder"],
            seq_lens_decoder=self.share_inputs["seq_lens_decoder"],
        )
        self.forward_meta.attn_backend = self.attn_backends[0]
        self.initialize_attention_backend()

        # Get sampling metadata
        self.sampling_metadata = SamplingMetadata(
            temperature=self.share_inputs["temperature"],
            top_p=self.share_inputs["top_p"],
            top_k=self.share_inputs["top_k"],
            step_idx=self.share_inputs["step_idx"],
            pre_token_ids=self.share_inputs["pre_ids"],
            frequency_penalties=self.share_inputs["frequency_score"],
            presence_penalties=self.share_inputs["presence_score"],
            repetition_penalties=self.share_inputs["penalty_score"],
            min_dec_lens=self.share_inputs["min_dec_len"],
            bad_words_token_ids=self.share_inputs["bad_tokens"],
            eos_token_ids=self.share_inputs["eos_token_id"],
        )

    def load_model(self) -> None:
        """load or download model"""
        logger.info(f"Starting to load model {self.model_config.architectures[0]}")
        time_before_load = time.perf_counter()
        # 1. Load original model
        self.model = get_model_from_loader(fd_config=self.fd_config)

        # 2. Load lora model

        # 3. Load drafter model(for speculative decoding)

        time_after_load = time.perf_counter()
        logger.info(f"Model loading took {time_after_load - time_before_load} seconds")

    def get_model(self) -> nn.Layer:
        """get current model"""
        return self.model

    def initialize_attention_backend(self):
        """
        Initialize attention meta data
        """
        # Initialzie attention meta data
        for attn_backend in self.attn_backends:
            attn_backend.init_attention_metadata(self.forward_meta)

    def initialize_kv_cache(self) -> None:
        """
        Initialize kv cache
        """
        cache_kvs = {}
        max_block_num = self.num_gpu_blocks

        cache_type = self.parallel_config.dtype

        if (
            self.quant_config
            and hasattr(self.quant_config, "kv_cache_quant_type")
            and self.quant_config.kv_cache_quant_type is not None
        ):
            cache_type = "uint8"

        kv_cache_shape = self.attn_backends[0].get_kv_cache_shape(max_num_blocks=max_block_num)

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
        self.share_inputs["caches"] = list(cache_kvs.values())
        for value in cache_kvs.values():
            del value
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

    def capture_model(self) -> None:
        """
        Trigger CUDA Graph capture for all shapes in 'CudaGraphConfig.cudagraph_capture_sizes'
        """
        logger.warn("XPU not support cuda graph currently")
        pass

    def prefill_finished(self):
        """
        check whether prefill stage finished
        """
        if int(paddle.max(self.share_inputs["seq_lens_encoder"])) != 0:
            return 1
        else:
            return 0

    def _dummy_prefill_inputs(self, num_tokens: int, batch_size: int):
        """Set dummy prefill inputs to share_inputs"""
        full_length = min(num_tokens // batch_size, self.parallel_config.max_model_len - 10)
        input_length = int(full_length - 512)
        block_num = (
            input_length + self.parallel_config.block_size - 1
        ) // self.parallel_config.block_size + self.parallel_config.enc_dec_block_num

        for i in range(batch_size):
            idx = i
            self.share_inputs["input_ids"][idx : idx + 1, :input_length] = np.array([5] * input_length)
            self.share_inputs["eos_token_id"][:] = np.array([2], dtype="int64").reshape(-1, 1)
            self.share_inputs["seq_lens_this_time"][idx : idx + 1] = input_length
            self.share_inputs["step_seq_lens_encoder"][idx : idx + 1] = input_length
            self.share_inputs["seq_lens_encoder"][idx : idx + 1] = input_length
            self.share_inputs["seq_lens_decoder"][idx : idx + 1] = 0
            self.share_inputs["step_idx"][idx : idx + 1] = 0
            self.share_inputs["max_dec_len"][idx : idx + 1] = 10
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
        in_capturing: bool = False,
    ) -> paddle.Tensor:
        """
        Use dummy inputs to run before formal execution.
        Args:
            num_tokens: Expected number of tokens generated
        """
        self._dummy_prefill_inputs(num_tokens, batch_size)

        while True:
            self.execute_model(None, True)

            if int((self.share_inputs["seq_lens_this_time"] > 0).sum()) == 0:
                break

    def execute_model(
        self,
        model_forward_batch: Optional[List[Request]] = None,
        is_dummy_run: bool = False,
    ) -> Optional[ModelRunnerOutput]:
        """
        The Entrance of model execute.
        Args:
            model_forward_batch: 'Request' contains information related to prompt and is an abstract
            class at the server level, which is too granular for ModelRunner.
            We plan to replace it with 'ModelForwardBatch'.
            intermediate_tensors:
        """
        # 1. Prepare inputs of model and decoder.
        self._prepare_inputs()

        # 2. Padding inputs for cuda grph

        # 3. Execute model
        model_output = self.model(self.share_inputs["ids_remove_padding"], self.forward_meta)

        hiddden_states = xpu_process_output(model_output, self.share_inputs["cum_offsets"], self.forward_meta)

        # 4. Compute logits, Sample
        logits = self.model.compute_logits(hiddden_states)

        sampler_output = self.sampler(logits, self.sampling_metadata)

        # 5. Speculative decode

        # 6. Post Process
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
            msg_queue_id=self.parallel_config.msg_queue_id,
            mp_rank=self.local_rank,
            use_ep=self.parallel_config.use_ep,
            # 投机解码
            full_hidden_states=None,
            draft_tokens=None,
            actual_draft_token_num=None,
            accept_tokens=None,
            accept_num=None,
        )
        xpu_post_process(
            sampled_token_ids=sampler_output.sampled_token_ids,
            model_output=model_output_data,
            skip_save_output=is_dummy_run,
        )

        # 7. Updata 'infer_seed' and step_paddle()
        self.share_inputs["infer_seed"].add_(self.infer_seed_increment)
        self.share_inputs["infer_seed"][:] %= self.MAX_INFER_SEED
        step_paddle(
            self.share_inputs,
            self.parallel_config.block_size,
            self.parallel_config.enc_dec_block_num,
        )

        return None

    def prepare_profile(self) -> None:
        """Prepare the profile run by setting the block number and initializing the KV cache."""
        paddle.device.xpu.empty_cache()
        self.num_gpu_blocks = self.parallel_config.total_block_num
        self.initialize_kv_cache()

    def profile_run(self) -> None:
        """Execute a forward pass with dummy inputs to profile the memory usage of the model."""

        self._dummy_run(
            num_tokens=int(self.parallel_config.max_num_batched_tokens),
            batch_size=min(self.parallel_config.max_num_seqs, 1),
        )

    def clear_block_table(self) -> None:
        """
        Clear the block tables and kv cache after profiling.
        """
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
        required_memory = (
            byte_of_dtype
            * 2  # k + v
            * (self.parallel_config.block_size * hidden_dim)
            * self.model_config.num_hidden_layers
        )
        return required_memory

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
                int(self.num_gpu_blocks * self.parallel_config.kv_cache_ratio) - 1,
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

    def not_need_stop(self) -> bool:
        """ """
        return self.share_inputs["not_need_stop"][0]
