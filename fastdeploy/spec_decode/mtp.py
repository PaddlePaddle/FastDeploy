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
from typing import List

import numpy as np
import paddle

from fastdeploy.engine.request import Request
from fastdeploy.model_executor.layers.attention import get_attention_backend
from fastdeploy.model_executor.layers.attention.base_attention_backend import \
    AttentionBackend
from fastdeploy.model_executor.layers.rotary_embedding import get_rope
from fastdeploy.model_executor.layers.sample.meta_data import SamplingMetadata
from fastdeploy.model_executor.layers.sample.sampler import MTPSampler
from fastdeploy.model_executor.ops.gpu import (draft_model_postprocess,
                                               draft_model_preprocess,
                                               draft_model_update,
                                               eagle_get_hidden_states,
                                               mtp_save_first_token,
                                               mtp_step_paddle,
                                               share_external_data)
from fastdeploy.model_executor.pre_and_post_process import (pre_process,
                                                            rebuild_padding)
from fastdeploy.worker.forward_meta import ForwardMeta

from .base import Proposer


class MTPProposer(Proposer):
    """
    Proposer for Multi-Token-Prediction(MTP)
    """

    def __init__(self, cfg, main_model, local_rank, device_id,
                 main_model_inputs):
        super().__init__(cfg)
        self.num_main_model_layers = self.model_config.num_layers
        self.local_rank = local_rank
        self.device_id = device_id
        self._update_cfg(main_model)
        self._load_model()
        self.main_model_inputs = main_model_inputs

        # [mixed, prefill, decoder]
        self.role = "mixed"
        self.sampler = MTPSampler(cfg)
        self._init_model_inputs()

        self.attn_backends: list[AttentionBackend] = []
        self._initialize_attn_backend()

    def _update_cfg(self, main_model):
        """
        Update config for MTP from global config
        """
        self.model_config.architectures[0] = self.model_config.architectures[
            0].replace("MoeForCausalLM", "MTPForCausalLM")
        self.speculative_config.sharing_model = main_model
        self.model_config.num_layers = 1
        self.parallel_config.model_name_or_path = (
            self.speculative_config.model_name_or_path)
        self.model_config.prefix_name = "ernie.mtp_block"
        if self.speculative_config.quantization != "":
            self.model_config.quantization = (
                self.speculative_config.quantization)
        self.model_config.start_layer_index = self.num_main_model_layers
        self.speculative_config.model_type = "mtp"

    def _load_model(self):
        """
        Load MTP Layer
        """
        from fastdeploy.model_executor.model_loader import \
            get_model_from_loader

        self.model = get_model_from_loader(self.cfg)

    def dummy_prefill_inputs(self, num_tokens: int, batch_size: int, expected_decode_len: int):
        """Set dummy prefill inputs to model_inputs"""
        max_dec_len = expected_decode_len + 1
        self.num_gpu_blocks = self.parallel_config.max_block_num
        self.initialize_kv_cache()
        full_length = min(num_tokens // batch_size,
                          self.parallel_config.max_model_len - max_dec_len)
        input_length = int(full_length * self.parallel_config.kv_cache_ratio)
        block_num = ((input_length + self.parallel_config.block_size - 1) //
                     self.parallel_config.block_size +
                     self.parallel_config.enc_dec_block_num)

        for i in range(batch_size):
            idx = i
            self.model_inputs["input_ids"][idx:idx +
                                           1, :input_length] = (np.array(
                                               [5] * input_length))
            self.model_inputs["eos_token_id"][:] = np.array(
                [2], dtype="int64").reshape(-1, 1)
            self.model_inputs["seq_lens_this_time"][idx:idx + 1] = input_length
            self.model_inputs["seq_lens_encoder"][idx:idx + 1] = input_length
            self.model_inputs["seq_lens_decoder"][idx:idx + 1] = 0
            self.model_inputs["step_idx"][idx:idx + 1] = 0
            self.model_inputs["max_dec_len"][idx:idx + 1] = max_dec_len
            self.model_inputs["stop_flags"][idx:idx + 1] = False

            self.model_inputs["encoder_block_lens"][idx:idx + 1] = block_num
            self.model_inputs["block_tables"][idx:idx +
                                              1, :block_num] = (np.arange(
                                                  idx * block_num,
                                                  (idx + 1) * block_num, 1))

    def initialize_kv_cache(self):
        """
        Initialize kv cache
        """
        # prompt cache
        self.cache_kvs = {}

        cache_type = self.parallel_config.dtype
        
        if (self.quant_config and 
            hasattr(self.quant_config, "kv_cache_quant_type") and 
            self.quant_config.kv_cache_quant_type is not None):
            cache_type = 'uint8'

        # Get kv cache shape
        kv_cache_shape = self.attn_backends[0].get_kv_cache_shape(
            max_num_blocks=self.num_gpu_blocks)
        if (not self.parallel_config.do_profile
                and (self.parallel_config.enable_prefix_caching
                     or self.parallel_config.splitwise_role != "mixed")):
            cache_kvs_list = []
            for i in range(
                    self.num_main_model_layers,
                    self.num_main_model_layers + self.model_config.num_layers):
                key_cache = paddle.empty(shape=[], dtype=cache_type)
                key_cache_name = f"key_caches_{i}_rank{self.local_rank}.device{self.device_id}"
                val_cache_name = f"value_caches_{i}_rank{self.local_rank}.device{self.device_id}"
                key_cache = share_external_data(key_cache, key_cache_name,
                                                kv_cache_shape)
                cache_kvs_list.append(key_cache)
                value_cache = paddle.empty(shape=[], dtype=cache_type)
                value_cache = share_external_data(value_cache, val_cache_name,
                                                  kv_cache_shape)
                cache_kvs_list.append(value_cache)

            self.model_inputs["caches"] = cache_kvs_list
        else:
            for i in range(self.model_config.num_layers):
                self.cache_kvs["key_caches_{}".format(i)] = paddle.full(
                    shape=kv_cache_shape,
                    fill_value=0,
                    dtype=cache_type,
                )
                self.cache_kvs["value_caches_{}".format(i)] = paddle.full(
                    shape=kv_cache_shape,
                    fill_value=0,
                    dtype=cache_type,
                )
            self.model_inputs["caches"] = list(self.cache_kvs.values())
            for value in self.cache_kvs.values():
                del value
        paddle.device.cuda.empty_cache()

    def _initialize_attn_backend(self, ) -> None:
        """
        Initialize attention backends and forward metadata
        """
        assert len(self.attn_backends) == 0

        # TODO(gongshaotian): Get rank from config
        num_heads = (self.model_config.num_attention_heads //
                     self.parallel_config.tensor_parallel_degree)
        self.model_config.kv_num_heads = (
            int(self.model_config.num_key_value_heads) //
            self.parallel_config.tensor_parallel_degree)
        head_dim = self.model_config.head_dim

        # Get the attention backend
        attn_cls = get_attention_backend(
            self.parallel_config.attention_backend)
        attn_backend = attn_cls(
            self.cfg,
            kv_num_heads=self.model_config.kv_num_heads,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        if attn_backend is None:
            raise NotImplementedError(
                f"{ self.parallel_config.attention_backend} attention backend"
                " is not support by GPUModelRunner")
        self.attn_backends.append(attn_backend)

    def clear_dummy_input(self):
        """
        Clear allocated cacheKV
        """
        del self.model_inputs["caches"]
        if self.forward_meta is not None:
            del self.forward_meta.caches

    def update_block_num(self, num_gpu_blocks) -> None:
        """
        Update block num by theoretical calculation
        """

        self.main_model_num_gpu_blocks = num_gpu_blocks
        self.num_gpu_blocks = int(
            num_gpu_blocks *
            self.speculative_config.num_gpu_block_expand_ratio)
        if not (self.parallel_config.enable_prefix_caching
                or self.parallel_config.splitwise_role != "mixed"):
            self.initialize_kv_cache()

        # Reset free list
        free_list = list(
            range(
                self.num_gpu_blocks - 1,
                int(self.main_model_num_gpu_blocks *
                    self.parallel_config.kv_cache_ratio) - 1,
                -1,
            ))
        self.free_list_len = len(free_list)
        self.model_inputs.update({
            "free_list":
            paddle.to_tensor(free_list, dtype="int32"),
            "free_list_len":
            paddle.full([1], self.free_list_len, dtype="int32"),
        })
        self.parallel_config.do_profile = False

    def _init_model_inputs(self):
        """
        Init model inputs
        """
        self.model_inputs = {}
        # Same shape/dytpe with base model
        self.model_inputs["block_tables"] = paddle.clone(
            self.main_model_inputs["block_tables"])
        self.model_inputs["input_ids"] = paddle.clone(
            self.main_model_inputs["input_ids"])
        self.model_inputs["seq_lens_this_time"] = paddle.clone(
            self.main_model_inputs["seq_lens_this_time"])
        self.model_inputs["seq_lens_encoder"] = paddle.clone(
            self.main_model_inputs["seq_lens_encoder"])
        self.model_inputs["seq_lens_decoder"] = paddle.clone(
            self.main_model_inputs["seq_lens_decoder"])
        self.model_inputs["step_idx"] = paddle.clone(
            self.main_model_inputs["step_idx"])
        self.model_inputs["stop_flags"] = paddle.clone(
            self.main_model_inputs["stop_flags"])
        self.model_inputs["stop_nums"] = paddle.clone(
            self.main_model_inputs["stop_nums"])
        self.model_inputs["not_need_stop"] = paddle.to_tensor([False],
                                                              dtype="bool",
                                                              place="cpu")
        self.model_inputs["pre_ids"] = paddle.clone(
            self.main_model_inputs["pre_ids"])
        self.model_inputs["ids_remove_padding"] = paddle.clone(
            self.main_model_inputs["ids_remove_padding"])
        self.model_inputs["cum_offsets"] = paddle.clone(
            self.main_model_inputs["cum_offsets"])
        self.model_inputs["padding_offset"] = paddle.clone(
            self.main_model_inputs["padding_offset"])
        self.model_inputs["cu_seqlens_q"] = paddle.clone(
            self.main_model_inputs["cu_seqlens_q"])
        self.model_inputs["cu_seqlens_k"] = paddle.clone(
            self.main_model_inputs["cu_seqlens_k"])
        self.model_inputs["decoder_batch_ids"] = paddle.clone(
            self.main_model_inputs["decoder_batch_ids"])
        self.model_inputs["decoder_tile_ids_per_batch"] = paddle.clone(
            self.main_model_inputs["decoder_tile_ids_per_batch"])

        tmp_position_ids = paddle.arange(
            self.parallel_config.max_model_len).reshape((1, -1))
        self.model_inputs["rope_emb"] = get_rope(
            rotary_dim=self.model_config.head_dim,
            position_ids=tmp_position_ids,
            base=self.model_config.rope_theta,
            model_config=self.model_config,
        )
        # self.model_inputs["caches"] = self.cache_kvs
        # Inherit generation hyperparameters from the main model for consistency
        self.model_inputs["top_p"] = self.main_model_inputs["top_p"]
        self.model_inputs["temperature"] = self.main_model_inputs[
            "temperature"]
        self.model_inputs["eos_token_id"] = self.main_model_inputs[
            "eos_token_id"]
        self.model_inputs["penalty_score"] = self.main_model_inputs[
            "penalty_score"]
        self.model_inputs["frequency_score"] = self.main_model_inputs[
            "frequency_score"]
        self.model_inputs["presence_score"] = self.main_model_inputs[
            "presence_score"]
        self.model_inputs["infer_seed"] = self.main_model_inputs["infer_seed"]

        self.model_inputs["max_dec_len"] = self.main_model_inputs[
            "max_dec_len"]
        self.model_inputs["min_dec_len"] = self.main_model_inputs[
            "min_dec_len"]

        self.model_inputs["bad_tokens"] = self.main_model_inputs["bad_tokens"]

        # Integrate the updated results in model forward
        self.model_inputs["base_model_draft_tokens"] = self.main_model_inputs[
            "draft_tokens"]
        self.model_inputs["substep"] = 0

        # Input tokens
        self.model_inputs["draft_tokens"] = paddle.full(
            shape=[self.max_num_seqs, 2], fill_value=-1, dtype="int64")

        self.model_inputs["encoder_block_lens"] = paddle.clone(
            self.main_model_inputs["encoder_block_lens"])

        self.free_list = list(
            range(
                self.parallel_config.max_block_num - 1,
                int(self.parallel_config.max_block_num *
                    self.parallel_config.kv_cache_ratio) - 1,
                -1,
            ))
        self.free_list_len = len(self.free_list)

        self.model_inputs["free_list"] = paddle.to_tensor(self.free_list,
                                                          dtype="int32")
        self.model_inputs["free_list_len"] = paddle.full(
            shape=[1], fill_value=self.free_list_len, dtype="int32")

        self.model_inputs["batch_drop"] = paddle.full(
            shape=[self.max_num_seqs, 1], fill_value=False, dtype="bool")
        self.model_inputs["used_list_len"] = paddle.full(
            shape=[self.max_num_seqs], fill_value=0, dtype="int32")

    def insert_prefill_inputs(self, req_dicts: List[Request]):
        """
        Process inputs for prefill tasks and insert it to model_inputs buffer
        """
        # NOTE: Lazy initialize kv cache
        if "caches" not in self.model_inputs:
            self.initialize_kv_cache()

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

            if (req_dicts[i].disaggregate_info is not None
                    and req_dicts[i].disaggregate_info["role"] == "decode"):
                length = len(request.prompt_token_ids)
                self.model_inputs["pre_ids"][idx:idx + 1] = (
                    request.prompt_token_ids[-1])
                prefill_token_num = self.max_draft_token_num + 1
                self.model_inputs["draft_tokens"][idx : idx + 1, \
                    0:1] = paddle.to_tensor(request.draft_token_ids[0:1], dtype='int64')

                self.model_inputs["seq_lens_encoder"][idx:idx + 1] = 0
                self.model_inputs["seq_lens_decoder"][idx:idx + 1] = length
                self.model_inputs['seq_lens_this_time'][idx:idx +
                                                        1] = prefill_token_num

                self.model_inputs["stop_flags"][idx:idx + 1] = False
                self.model_inputs["batch_drop"][idx:idx + 1] = False
                self.model_inputs["step_idx"][idx:idx + 1] = 1
                encoder_block_num = len(request.block_tables)

                self.model_inputs["encoder_block_lens"][idx:idx +
                                                        1] = encoder_block_num
                self.model_inputs["block_tables"][idx:idx + 1, :] = -1
                self.model_inputs["block_tables"][
                    idx:idx + 1, :encoder_block_num] = np.array(
                        request.block_tables, dtype="int32")

            else:
                length = len(request.prompt_token_ids)

                if length > 1:
                    self.model_inputs["input_ids"][
                        idx:idx + 1, :length -
                        1] = self.main_model_inputs["input_ids"][idx:idx + 1,
                                                                 1:length]
                self.model_inputs["pre_ids"][idx:idx + 1] = -1
                self.model_inputs["step_idx"][idx:idx + 1] = 0
                # TODO(liuzichang) finish chunked_prefill
                if self.parallel_config.enable_chunked_prefill:
                    raise NotImplementedError(
                        "MTP don't support chunked_prefill now")
                else:
                    self.model_inputs["seq_lens_encoder"][idx:idx + 1] = length
                    self.model_inputs["seq_lens_decoder"][idx:idx + 1] = (
                        request.get("seq_lens_decoder", 0))
                    self.model_inputs["seq_lens_this_time"][idx:idx +
                                                            1] = length

                self.model_inputs["stop_flags"][idx:idx + 1] = False
                self.model_inputs["batch_drop"][idx:idx + 1] = False

                encoder_block_num = len(request.get("block_tables"))
                self.model_inputs["encoder_block_lens"][idx:idx +
                                                        1] = encoder_block_num
                self.model_inputs["block_tables"][idx:idx + 1, :] = -1
                self.model_inputs["block_tables"][
                    idx:idx + 1, :encoder_block_num] = np.array(
                        request.get("block_tables"), dtype="int32")
        self.model_inputs["not_need_stop"][0] = True

    def _initialize_forward_meta(self):
        """
        Initialize forward meta and attention meta data
        """
        # Initialize forward meta
        self.forward_meta = ForwardMeta.init_forward_meta(
            self.model_inputs, self.attn_backends[0])

        # Initialzie attention meta data
        for attn_backend in self.attn_backends:
            attn_backend.init_attention_metadata(self.forward_meta)

    def _prepare_inputs(self, full_hidden_states):
        """
        Prepare MTP inputs
        """
        draft_model_preprocess(
            self.model_inputs["draft_tokens"],
            self.model_inputs["input_ids"],
            self.model_inputs["stop_flags"],
            self.model_inputs["seq_lens_this_time"],
            self.model_inputs["seq_lens_encoder"],
            self.model_inputs["seq_lens_decoder"],
            self.model_inputs["step_idx"],
            self.model_inputs["not_need_stop"],
            self.model_inputs["batch_drop"],
            self.main_model_inputs["accept_tokens"],
            self.main_model_inputs["accept_num"],
            self.main_model_inputs["seq_lens_encoder"],
            self.main_model_inputs["seq_lens_decoder"],
            self.main_model_inputs["step_idx"],
            self.main_model_inputs["stop_flags"],
            self.main_model_inputs["is_block_step"],
            self.main_model_inputs["draft_tokens"],
            self.max_draft_token_num,
            self.speculative_method in ["eagle", "mtp"],
            self.role == "prefill",
        )

        target_hidden_states = eagle_get_hidden_states(
            full_hidden_states,
            self.model_inputs["seq_lens_this_time"],
            self.model_inputs["seq_lens_encoder"],
            self.model_inputs["seq_lens_decoder"],
            self.model_inputs["stop_flags"],
            self.main_model_inputs["accept_num"],
            self.main_model_inputs["seq_lens_this_time"],
            self.main_model_inputs["seq_lens_encoder"],
            self.max_draft_token_num,
        )

        return target_hidden_states

    def _post_process(self, sampled_token_ids):
        """
        PostProcess for generation
        """
        draft_model_update(
            sampled_token_ids,
            self.model_inputs["draft_tokens"],
            self.model_inputs["pre_ids"],
            self.model_inputs["seq_lens_this_time"],
            self.model_inputs["seq_lens_encoder"],
            self.model_inputs["seq_lens_decoder"],
            self.model_inputs["step_idx"],
            self.model_inputs["output_cum_offsets"],
            self.model_inputs["stop_flags"],
            self.model_inputs["not_need_stop"],
            self.model_inputs["max_dec_len"],
            self.model_inputs["eos_token_id"],
            self.model_inputs["base_model_draft_tokens"],
            self.max_model_len,
            self.model_inputs["substep"],
        )
        if self.role == "prefill":
            mtp_save_first_token(
                self.model_inputs["base_model_draft_tokens"],
                self.model_inputs["not_need_stop"],
                self.local_rank,
                self.parallel_config.use_ep,
            )

    def _propose(self, target_hidden_states):
        """
        Main process for MTP inference
        """
        for substep in range(self.max_draft_token_num):
            if self.model_inputs["not_need_stop"]:
                self.model_inputs["substep"] = substep
                # Remove padding
                (
                    ids_remove_padding,
                    cum_offsets,
                    padding_offset,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    output_cum_offsets,
                    output_padding_offset,
                ) = pre_process(
                    self.parallel_config.max_model_len,
                    self.model_inputs["input_ids"],
                    self.model_inputs["seq_lens_this_time"],
                    True,
                    self.model_inputs["draft_tokens"],
                    self.model_inputs["seq_lens_encoder"],
                    self.model_inputs["seq_lens_decoder"],
                )
                # Initialize forward meta data
                self.model_inputs["ids_remove_padding"].copy_(
                    ids_remove_padding, False)
                self.model_inputs["cum_offsets"].copy_(cum_offsets, False)
                self.model_inputs["padding_offset"].copy_(
                    padding_offset, False)
                self.model_inputs["cu_seqlens_q"].copy_(cu_seqlens_q, False)
                self.model_inputs["cu_seqlens_k"].copy_(cu_seqlens_k, False)
                # for speculative decoding
                self.model_inputs["output_cum_offsets"] = output_cum_offsets
                self.model_inputs["output_padding_offset"] = (
                    output_padding_offset)
                self._initialize_forward_meta()

                # Get sampling metadata
                self.sampling_metadata = SamplingMetadata(
                    temperature=self.model_inputs["temperature"],
                    top_p=self.model_inputs["top_p"],
                    step_idx=self.model_inputs["step_idx"],
                    pre_token_ids=self.model_inputs["pre_ids"],
                    frequency_penalties=self.model_inputs["frequency_score"],
                    presence_penalties=self.model_inputs["presence_score"],
                    repetition_penalties=self.model_inputs["penalty_score"],
                    min_dec_lens=self.model_inputs["min_dec_len"],
                    bad_words_token_ids=self.model_inputs["bad_tokens"],
                    eos_token_ids=self.model_inputs["eos_token_id"],
                )

                model_output = self.model(
                    ids_remove_padding=self.model_inputs["ids_remove_padding"],
                    previous_hidden_states=target_hidden_states,
                    forward_meta=self.forward_meta,
                )

                hiddden_states = rebuild_padding(
                    model_output,
                    self.model_inputs["cum_offsets"],
                    self.model_inputs["seq_lens_this_time"],
                    self.model_inputs["seq_lens_decoder"],
                    self.model_inputs["seq_lens_encoder"],
                    self.model_inputs["output_padding_offset"],
                    self.parallel_config.max_model_len,
                )
                paddle.device.synchronize()

                # 4. Compute logits, Sample
                logits = self.model.compute_logits(hiddden_states)

                sampled_token_ids = self.sampler(
                    logits,
                    self.sampling_metadata,
                    self.max_model_len,
                    self.model_inputs,
                )

                if self.parallel_config.tensor_parallel_degree > 1:
                    paddle.distributed.broadcast(sampled_token_ids, 0)

                self._post_process(sampled_token_ids)

    def _update_status(self):
        """
        Update main-model's forward info in next step.
        Allocate/Free block of MPT.
        """
        draft_model_postprocess(
            self.main_model_inputs["draft_tokens"],
            self.main_model_inputs["seq_lens_this_time"],
            self.main_model_inputs["seq_lens_encoder"],
            self.main_model_inputs["stop_flags"],
        )

        mtp_step_paddle(
            self.main_model_inputs["stop_flags"],
            self.model_inputs["stop_flags"],
            self.model_inputs["batch_drop"],
            self.model_inputs["seq_lens_this_time"],
            self.model_inputs["seq_lens_encoder"],
            self.model_inputs["seq_lens_decoder"],
            self.model_inputs["block_tables"],
            self.model_inputs["encoder_block_lens"],
            self.model_inputs["used_list_len"],
            self.model_inputs["free_list"],
            self.model_inputs["free_list_len"],
            self.parallel_config.block_size,
            self.max_draft_token_num,
        )

    def _run_impl(self, full_hidden_states):
        target_hidden_states = self._prepare_inputs(full_hidden_states)
        self._propose(target_hidden_states=target_hidden_states)
        self._update_status()
