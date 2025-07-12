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
import os
import random

import numpy as np
import paddle
import paddle.distributed.fleet as fleet

from fastdeploy.config import ModelConfig
from fastdeploy.input.ernie_tokenizer import ErnieBotTokenizer
from fastdeploy.input.mm_processor import DataProcessor
from fastdeploy.model_executor.forward_meta import ForwardMeta
from fastdeploy.model_executor.layers.attention import get_attention_backend
from fastdeploy.model_executor.layers.rotary_embedding import get_rope_3d
from fastdeploy.model_executor.layers.sample.meta_data import SamplingMetadata
from fastdeploy.model_executor.layers.sample.sampler import Sampler
from fastdeploy.model_executor.models.ernie4_5_vl.configuration import \
    Ernie4_5_VLMoeConfig
from fastdeploy.model_executor.models.ernie4_5_vl.modeling_resampler import \
    ScatterOp
from fastdeploy.platforms import current_platform
from fastdeploy.worker.output import SamplerOutput
from fastdeploy.worker.utils import check_safetensors_model
from fastdeploy.worker.vl_model_runner_base import VLModelRunnerBase

if current_platform.is_cuda() and current_platform.available():
    from fastdeploy.model_executor.layers.utils import (
        remove_padding, speculate_remove_padding)

from fastdeploy.model_executor.ops.gpu import (save_output, save_output_topk,
                                               set_stop_value_multi_ends,
                                               set_value_by_flags_and_idx,
                                               update_inputs)


class GPUVLModelRunner(VLModelRunnerBase):
    """
    The GPUVLModelRunner class for vision-language tasks on GPU.
    """

    def __init__(
        self,
        config: ModelConfig,
        args: argparse.Namespace,
        nranks: int,
        rank: int,
    ) -> None:
        """
        GPUVLModelRunner init
        """
        self.nranks = nranks
        self.rank = rank

        hcg = fleet.get_hybrid_communicate_group()
        self.tensor_parallel_degree = max(hcg.get_model_parallel_world_size(),
                                          1)
        self.tensor_parallel_rank = hcg.get_model_parallel_rank()
        self.mp_src_rank = hcg.get_model_parallel_group_src_rank()
        self.mp_group = hcg.get_model_parallel_group()
        self.is_safetensors_model = check_safetensors_model(
            args.model_name_or_path)
        self.enable_logprob = args.enable_logprob

        model_path = os.path.dirname(args.model_name_or_path)
        args.llm_model_name_or_path = args.model_name_or_path
        if not self.is_safetensors_model:
            args.tokenizer = args.image_preprocessor = model_path
        else:
            args.tokenizer = args.image_preprocessor = args.model_name_or_path
        args.vision_model_name_or_path = os.path.join(
            model_path, "DFNRopeVisionTransformer")

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

        super().__init__(config, args)
        self.init_extra_input(config, args)

        self._reset_paddle_env()

        self.sampler = Sampler()

    def _reset_paddle_env(self):
        pass

    def update_chunked_prefill(self, tasks: list[any]) -> None:
        """
        update chunked prefill
        """
        if not self.args.enable_chunked_prefill:
            return

        for task in tasks:
            if task.chunk_idx > len(task.prefill_chunk_info):
                continue

            idx = task.idx
            if task.chunk_idx == len(task.prefill_chunk_info):
                self.share_inputs["seq_lens_this_time"][idx:idx + 1] = 1
                self.share_inputs['seq_lens_encoder'][idx:idx + 1] = 0
                self.share_inputs["seq_lens_decoder"][idx:idx +
                                                      1] = task.start_idx
                self.share_inputs["step_idx"][idx:idx + 1] = 1
            else:
                inputs = self._preprocess_task(
                    task.prefill_chunk_info[task.chunk_idx])
                if inputs.get("images") is not None:
                    self.share_inputs[
                        "image_features"] = self.extract_vision_features(
                            inputs)
                else:
                    # Compatible with the situation that lacks images and videos
                    self.share_inputs["image_features"] = None

                token_chunk_size = inputs["input_ids"].shape[1]
                self.share_inputs["input_ids"][
                    idx:idx + 1, :token_chunk_size] = inputs["input_ids"]
                self.share_inputs["seq_lens_this_time"][idx:idx +
                                                        1] = token_chunk_size
                self.share_inputs['seq_lens_encoder'][idx:idx +
                                                      1] = token_chunk_size
                self.share_inputs["seq_lens_decoder"][idx:idx +
                                                      1] = task.start_idx
                self.share_inputs["step_idx"][idx:idx + 1] = 0

                task.start_idx += token_chunk_size
            task.chunk_idx += 1

    def _init_image_preprocess(self, vision_config) -> None:
        processor = DataProcessor(
            tokenizer_name=self.args.tokenizer,
            image_preprocessor_name=str(self.args.image_preprocessor),
        )
        processor.eval()
        image_preprocess = processor.image_preprocessor
        image_preprocess.image_mean_tensor = paddle.to_tensor(
            image_preprocess.image_mean, dtype="float32"
        ).reshape([1, 3, 1, 1])
        image_preprocess.image_std_tensor = paddle.to_tensor(
            image_preprocess.image_std, dtype="float32"
        ).reshape([1, 3, 1, 1])
        image_preprocess.rescale_factor = paddle.to_tensor(
            image_preprocess.rescale_factor, dtype="float32"
        )
        image_preprocess.image_mean_tensor = (
            image_preprocess.image_mean_tensor.squeeze(
                [-2, -1]
            ).repeat_interleave(vision_config.patch_size**2 * 1, -1)
        )
        image_preprocess.image_std_tensor = (
            image_preprocess.image_std_tensor.squeeze(
                [-2, -1]
            ).repeat_interleave(vision_config.patch_size**2 * 1, -1)
        )
        return image_preprocess

    def _load_model(
        self,
        model_name: str,
        dynamic_load_weight: int = 0,
    ) -> None:
        """
        Load the model from the given model name.
        """

        vocab_file_names = [
            "tokenizer.model", "spm.model", "ernie_token_100k.model"
        ]
        for i in range(len(vocab_file_names)):
            if os.path.exists(
                    os.path.join(self.args.tokenizer, vocab_file_names[i])):
                ErnieBotTokenizer.resource_files_names[
                    "vocab_file"] = vocab_file_names[i]
                break

        tokenizer = ErnieBotTokenizer.from_pretrained(
            self.args.tokenizer,
            model_max_length=self.args.max_model_len,
            padding_side="right",
            use_fast=False,
        )
        tokenizer.ignored_index = -100
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token

        self.dtype = self.args.dtype
        paddle.set_default_dtype(self.dtype)

        from fastdeploy.worker.worker_process import initialize_fd_config

        fd_config = initialize_fd_config(
            self.args, self.tensor_parallel_degree, self.tensor_parallel_rank
        )
        fd_config.model_config = Ernie4_5_VLMoeConfig(
            **fd_config.model_config.__dict__
        )
        fd_config.parallel_config.column_cut = False
        vision_config = fd_config.model_config.vision_config
        vision_config.attn_sep = False
        vision_config.dtype = "bfloat16"
        vision_config.tensor_parallel_degree = self.tensor_parallel_degree
        vision_config.tensor_parallel_rank = self.tensor_parallel_rank
        fd_config.model_config.pixel_hidden_size = vision_config.hidden_size
        fd_config.model_config.im_patch_id = tokenizer.get_vocab()[
            "<|IMAGE_PLACEHOLDER|>"
        ]
        fd_config.model_config.think_end_id = tokenizer.get_vocab()["</think>"]
        fd_config.model_config.max_text_id = fd_config.model_config.im_patch_id
        fd_config.model_config.sequence_parallel = False
        # TODO (bukejiyu): Remove the assignment
        fd_config.moe_config.top_k = 8
        self.fd_config = fd_config
        self.model_cfg = self.fd_config.model_config
        self.image_preprocess = self._init_image_preprocess(
            self.fd_config.model_config.vision_config
        )
        from fastdeploy.model_executor.model_loader import \
            get_model_from_loader

        self.model = get_model_from_loader(self.fd_config)
        attn_backend_cls = get_attention_backend()
        num_heads = self.fd_config.model_config.num_attention_heads // \
            self.fd_config.parallel_config.tensor_parallel_degree
        self.fd_config.model_config.kv_num_heads = int(
            self.fd_config.model_config.num_key_value_heads
        ) // self.fd_config.parallel_config.tensor_parallel_degree
        head_dim = self.fd_config.model_config.head_dim
        self.attn_backend = attn_backend_cls(
            self.fd_config,
            kv_num_heads=self.fd_config.model_config.kv_num_heads,
            num_heads=num_heads,
            head_dim=head_dim)
        self._init_kvcache()

    def init_extra_input(self, config: ModelConfig, args: argparse.Namespace) -> None:
        """
        Initialize extra input tensors.
        """
        head_dim = self.model_cfg.head_dim
        self.share_inputs.update({
            "rope_emb":
            paddle.full(shape=[
                args.max_num_seqs, 2, 1, self.max_length, 1, head_dim // 2
            ],
                        fill_value=0,
                        dtype="float32")
        })
        self.share_inputs.update({"image_features": None})
        self.share_inputs.update({
            "need_think_end":
            paddle.full(shape=[args.max_num_seqs, 1],
                        fill_value=0,
                        dtype="int32")
        })
        self.share_inputs.update({
            "enable_thinking":
            paddle.full(shape=[1], fill_value=True, dtype="bool")
        })
        self.share_inputs.update({
            "reasoning_index":
            paddle.full(shape=[args.max_num_seqs, 1],
                        fill_value=0,
                        dtype="int32")
        })

    def init_rotary_position_embedding(self, max_model_len: int) -> None:
        """
        Init rotary position embedding
        """
        pass

    def _init_kvcache(self):
        """
        Init kv cache
        """
        cache_kvs = {}
        total_block_num = self.num_gpu_blocks
        num_layers = self.model_cfg.get("num_layers",
                                        None) or self.model_cfg.get(
                                            "num_hidden_layers", None)

        kv_num_head = self.model_cfg.get(
            "num_key_value_heads",
            self.model_cfg.num_attention_heads,
        )
        kv_num_head = kv_num_head // self.tensor_parallel_degree
        self.model_cfg.kv_num_head = kv_num_head

        for i in range(num_layers):
            cache_type = self.args.dtype
            cache_kvs["key_caches_{}".format(i)] = paddle.full(
                shape=[
                    total_block_num,
                    kv_num_head,
                    self.args.block_size,
                    self.model_cfg.head_dim,
                ],
                fill_value=0,
                dtype=cache_type,
            )
            cache_kvs["value_caches_{}".format(i)] = paddle.full(
                shape=[
                    total_block_num,
                    kv_num_head,
                    self.args.block_size,
                    self.model_cfg.head_dim,
                ],
                fill_value=0,
                dtype=cache_type,
            )

        self.share_inputs["caches"] = list(cache_kvs.values())
        for value in cache_kvs.values():
            del value
        paddle.device.cuda.empty_cache()

    def clear_parameters(self, pid: int) -> None:
        """ clear_parameters """
        if "caches" in self.share_inputs:
            self.model.clear_parameters(pid)
            del self.share_inputs["caches"]
            paddle.device.cuda.empty_cache()
            self.model.log_memory_usage("clear all memory")

    def update_parameters(self, pid: int) -> None:
        """ update_parameters """
        if "caches" not in self.share_inputs:
            self.model.update_parameters(pid)
            self._init_kvcache()
            self.model.log_memory_usage("update all memory")

    @paddle.no_grad()
    def extract_vision_features(self, inputs: list[paddle.Tensor]) -> paddle.Tensor:
        """extract_vision_features"""
        assert inputs["images"] is not None
        grid_thw = inputs["grid_thw"]

        images = inputs["images"].cast("float32")
        images = self.image_preprocess.rescale_factor * images - self.image_preprocess.image_mean_tensor
        images = images / self.image_preprocess.image_std_tensor
        images = images.cast("bfloat16")

        token_type_ids = inputs["token_type_ids"]
        token_type_ids_w_video = token_type_ids
        input_ids = inputs["input_ids"]
        # convert to img patch id
        image_mask = input_ids == self.model_cfg.im_patch_id
        image_type_ids = inputs["image_type_ids"]
        with paddle.amp.auto_cast(
                True,
                custom_black_list=self.amp_black,
                custom_white_list=self.amp_white,
                level="O2",
                dtype=self.dtype,
        ):
            image_features = self.model.vision_model.extract_feature(
                images, grid_thw)
            if self.tensor_parallel_degree > 1:
                S, C = image_features.shape
                image_features = image_features.reshape(
                    [-1, C * self.model_cfg.spatial_conv_size**2])
                image_features = ScatterOp.apply(image_features,
                                                 axis=-1)  # mp 切 Fea
                image_features = image_features.reshape([S, -1])
            image_features = self.model.resampler_model(
                image_features,
                image_mask,
                token_type_ids_w_video,
                image_type_ids,
                grid_thw,
            )
        return image_features

    @paddle.no_grad()
    def prepare_rope3d(self, position_ids: paddle.Tensor, **kwargs) -> paddle.Tensor:
        """prepare_rope3d"""

        prefix_max_position_ids = paddle.max(position_ids) + 1
        dec_pos_ids = paddle.tile(
            paddle.arange(kwargs["max_length"],
                          dtype="int64").unsqueeze(0).unsqueeze(-1), [1, 1, 3])
        dec_pos_ids = dec_pos_ids + prefix_max_position_ids
        position_ids_3d_real = paddle.concat([position_ids, dec_pos_ids],
                                             axis=1)

        rope_emb = get_rope_3d(
            position_ids=position_ids_3d_real,
            rotary_dim=self.model_cfg.head_dim,
            paritial_rotary_factor=1.0,
            base=self.model_cfg.rope_theta,
            max_position=self.args.max_model_len,
            freq_allocation=self.model_cfg.freq_allocation,
        )
        return rope_emb

    def prefill_finished(self):
        """
        Verify prefill operation completion
        """
        prefill_statue = (self.share_inputs["seq_lens_this_time"] != 0) & (
            self.share_inputs["seq_lens_this_time"] != 1)
        return not paddle.any(prefill_statue).numpy()

    def dy_input_preprocess(self, tasks: list[any]) -> None:
        """
        dynamic insertion
        """

        def get_numeric_value(task, key, default_value):
            if task.get(key, None) is not None:
                return task.get(key)
            else:
                return default_value

        for i in range(len(tasks)):
            task = tasks[i]
            idx = task.idx

            kwargs = {
                "max_length":
                get_numeric_value(task, "max_tokens", 2048),
                "top_p":
                get_numeric_value(task, "top_p", 0.8),
                "temperature":
                get_numeric_value(task, "temperature", 0.2),
                "top_k":
                get_numeric_value(task, "top_k", 0),
                "penalty_score":
                get_numeric_value(task, "repetition_penalty", 1.0),
                "frequency_score":
                get_numeric_value(task, "frequency_penalty", 0.0),
                "presence_score":
                get_numeric_value(task, "presence_penalty", 0.0),
                "decode_strategy":
                "sampling",
                "pad_token_id":
                self.args.pad_token_id,
                "enable_thinking":
                get_numeric_value(task, "enable_thinking", True),
                "reasoning_max_tokens":
                get_numeric_value(task, "reasoning_max_tokens", 2048),
            }

            if self.args.enable_chunked_prefill:
                task.set("chunk_idx", 1)
                inputs = self._preprocess_task(task.prefill_chunk_info[0])
                if inputs.get("images") is not None:
                    self.share_inputs[
                        "image_features"] = self.extract_vision_features(
                            inputs)
                else:
                    # Compatible with the situation that lacks images and videos
                    self.share_inputs["image_features"] = None
                if task.multimodal_inputs["position_ids"] is not None:
                    position_ids = paddle.to_tensor(
                        task.multimodal_inputs["position_ids"],
                        dtype="int64").unsqueeze([0])
                else:
                    position_ids = None

                token_chunk_size = inputs["input_ids"].shape[1]
                task.set("start_idx", token_chunk_size)
                self.share_inputs["input_ids"][
                    idx:idx + 1, :token_chunk_size] = inputs["input_ids"]
                self.share_inputs["seq_lens_this_time"][idx:idx +
                                                        1] = token_chunk_size
                self.share_inputs["seq_lens_encoder"][idx:idx +
                                                      1] = token_chunk_size
                self.share_inputs["step_seq_lens_encoder"][
                    idx:idx + 1] = token_chunk_size
            else:
                inputs = self._preprocess_task(task.multimodal_inputs)
                if inputs.get("images") is not None:
                    self.share_inputs[
                        "image_features"] = self.extract_vision_features(
                            inputs)
                else:
                    # Compatible with the situation that lacks images and videos
                    self.share_inputs["image_features"] = None
                position_ids = inputs["position_ids"]

                length = inputs["input_ids"].shape[1]
                self.share_inputs["input_ids"][
                    idx:idx + 1, :length] = inputs["input_ids"]
                self.share_inputs["seq_lens_this_time"][idx:idx + 1] = length
                self.share_inputs["seq_lens_encoder"][idx:idx + 1] = length
                self.share_inputs["step_seq_lens_encoder"][idx:idx +
                                                           1] = length

            # force </think>
            self.share_inputs["enable_thinking"][:] = kwargs["enable_thinking"]
            self.share_inputs["need_think_end"][
                idx:idx + 1, :] = 1 if kwargs["enable_thinking"] else 0

            self.share_inputs["reasoning_index"][
                idx:idx + 1, :] = kwargs["reasoning_max_tokens"]

            self.share_inputs["rope_emb"][idx:idx +
                                          1, :] = self.prepare_rope3d(
                                              position_ids, **kwargs)

            self.share_inputs["top_p"][idx:idx + 1] = kwargs["top_p"]
            self.share_inputs["temperature"][idx:idx +
                                             1] = kwargs["temperature"]
            self.share_inputs["eos_token_id"][:] = np.array(
                task.eos_token_ids).astype("int64").reshape(-1, 1)
            self.share_inputs["penalty_score"][idx:idx +
                                               1] = kwargs["penalty_score"]
            self.share_inputs["frequency_score"][idx:idx +
                                                 1] = kwargs["frequency_score"]
            self.share_inputs["presence_score"][idx:idx +
                                                1] = kwargs["presence_score"]
            self.share_inputs["seq_lens_decoder"][idx:idx + 1] = 0
            self.share_inputs["step_idx"][idx:idx + 1] = 0
            self.share_inputs["min_dec_len"][idx:idx + 1] = 1
            self.share_inputs["max_dec_len"][idx:idx +
                                             1] = kwargs["max_length"]
            self.share_inputs["stop_flags"][idx:idx + 1] = False
            self.share_inputs["pre_ids"][idx:idx + 1] = -1
            encoder_block_num = len(task.get("block_tables"))
            self.share_inputs["encoder_block_lens"][idx:idx +
                                                    1] = encoder_block_num
            self.share_inputs["block_tables"][idx:idx + 1, :] = -1
            self.share_inputs["block_tables"][
                idx:idx + 1, :encoder_block_num] = np.array(task.block_tables,
                                                            dtype="int32")

    def pre_process(self) -> None:
        """
        pre_process
        """
        if current_platform.is_cuda():
            if self.args.speculative_method is not None:
                (
                    ids_remove_padding,
                    padding_offset,
                    cum_offsets,
                    cu_seqlens_q,
                    cu_seqlens_k,
                ) = speculate_remove_padding(
                    max_len=self.args.max_model_len,
                    input_ids=self.share_inputs["input_ids"],
                    seq_lens_this_time=self.share_inputs["seq_lens_this_time"],
                    draft_tokens=self.share_inputs["draft_tokens"],
                    seq_lens_encoder=self.share_inputs["seq_lens_encoder"])
            else:
                (
                    ids_remove_padding,
                    padding_offset,
                    cum_offsets,
                    cu_seqlens_q,
                    cu_seqlens_k,
                ) = remove_padding(
                    max_len=self.args.max_model_len,
                    input_ids=self.share_inputs["input_ids"],
                    seq_lens_this_time=self.share_inputs["seq_lens_this_time"])
        self.share_inputs["ids_remove_padding"] = ids_remove_padding
        self.share_inputs["padding_offset"] = padding_offset
        self.share_inputs["cum_offsets"] = cum_offsets
        self.share_inputs["cu_seqlens_q"] = cu_seqlens_q
        self.share_inputs["cu_seqlens_k"] = cu_seqlens_k
        self.share_inputs["decoder_batch_ids"] = paddle.full(
            [self.fd_config.parallel_config.max_num_seqs, 1], 0, dtype='int32')
        self.share_inputs["decoder_tile_ids_per_batch"] = paddle.full(
            [self.fd_config.parallel_config.max_num_seqs, 1], 0, dtype='int32')
        # initialize_forward_meta
        self.forward_meta = ForwardMeta(
            input_ids=self.share_inputs["input_ids"],
            ids_remove_padding=self.share_inputs["ids_remove_padding"],
            rotary_embs=self.share_inputs["rope_emb"],
            attn_backend=self.attn_backend,
            decoder_batch_ids=self.share_inputs["decoder_batch_ids"],
            decoder_tile_ids_per_batch=self.share_inputs["decoder_tile_ids_per_batch"],
            seq_lens_encoder=self.share_inputs["seq_lens_encoder"],
            seq_lens_decoder=self.share_inputs["seq_lens_decoder"],
            seq_lens_this_time=self.share_inputs["seq_lens_this_time"],
            cum_offsets=self.share_inputs["cum_offsets"],
            padding_offset=self.share_inputs["padding_offset"],
            cu_seqlens_q=self.share_inputs["cu_seqlens_q"],
            cu_seqlens_k=self.share_inputs["cu_seqlens_k"],
            block_tables=self.share_inputs["block_tables"],
            caches=self.share_inputs["caches"]
        )
        self.attn_backend.init_attention_metadata(self.forward_meta)

        self.sampling_metadata = SamplingMetadata(
            temperature=self.share_inputs["temperature"],
            top_p=self.share_inputs["top_p"],
            step_idx=self.share_inputs["step_idx"],
            pre_token_ids=self.share_inputs["pre_ids"],
            frequency_penalties=self.share_inputs["frequency_score"],
            presence_penalties=self.share_inputs["presence_score"],
            repetition_penalties=self.share_inputs["penalty_score"],
            min_dec_lens=self.share_inputs["min_dec_len"],
            bad_words_token_ids=self.share_inputs["bad_tokens"],
            eos_token_ids=self.share_inputs["eos_token_id"],
            max_num_logprobs=20 if self.enable_logprob else None,
        )

    def generate(self) -> None:
        """
        generate
        """
        self.pre_process()
        hiddden_states = self.model(self.share_inputs["ids_remove_padding"],
                                    self.share_inputs["image_features"],
                                    self.forward_meta)
        logits = self.model.compute_logits(hiddden_states)
        set_value_by_flags_and_idx(
            self.share_inputs["pre_ids"],
            self.share_inputs["input_ids"],
            self.share_inputs["seq_lens_this_time"],
            self.share_inputs["seq_lens_encoder"],
            self.share_inputs["seq_lens_decoder"],
            self.share_inputs["step_idx"],
            self.share_inputs["stop_flags"],
        )
        # sampler & save_output
        sampler_output = self.sampler(logits, self.sampling_metadata)
        if self.fd_config.parallel_config.tensor_parallel_degree > 1:
            paddle.distributed.broadcast(sampler_output.sampled_token_ids, 0)
        self.post_process(sampler_output)

    def post_process(self, sampler_output: SamplerOutput) -> None:
        """
        post_process
        """
        if self.share_inputs["enable_thinking"]:
            exists_think_end = sampler_output.sampled_token_ids == self.model_cfg.think_end_id
            paddle.assign(
                paddle.where(
                    exists_think_end,
                    self.share_inputs["need_think_end"] - 1,
                    self.share_inputs["need_think_end"],
                ), self.share_inputs["need_think_end"])

            paddle.assign(
                paddle.where(
                    self.share_inputs["need_think_end"].cast("bool"),
                    self.share_inputs["reasoning_index"] - 1,
                    self.share_inputs["reasoning_index"],
                ), self.share_inputs["reasoning_index"])

            stop_wo_think = (
                (sampler_output.sampled_token_ids == self.share_inputs["eos_token_id"]) |
                (self.share_inputs["reasoning_index"] == 0)) & (
                    self.share_inputs["need_think_end"] > 0)
            sampler_output.sampled_token_ids = paddle.where(stop_wo_think,
                                       self.model_cfg.think_end_id,
                                       sampler_output.sampled_token_ids)
            paddle.assign(
                paddle.where(
                    stop_wo_think,
                    self.share_inputs["need_think_end"] - 1,
                    self.share_inputs["need_think_end"],
                ), self.share_inputs["need_think_end"])
        paddle.assign(
            paddle.where(
                self.share_inputs["stop_flags"],
                self.share_inputs["step_idx"],
                self.share_inputs["step_idx"] + 1,
            ),
            self.share_inputs["step_idx"],
        )
        length_cond = paddle.greater_equal(self.share_inputs["step_idx"],
                                           self.share_inputs["max_dec_len"])
        paddle.assign(
            paddle.logical_or(self.share_inputs["stop_flags"], length_cond),
            self.share_inputs["stop_flags"],
        )

        set_stop_value_multi_ends(
            sampler_output.sampled_token_ids,
            self.share_inputs["stop_flags"],
            self.share_inputs["seq_lens_this_time"],
            self.share_inputs["eos_token_id"],
            self.share_inputs["next_tokens"],
            False,
        )  # multi ends
        # update inputs
        update_inputs(
            self.share_inputs["stop_flags"],
            self.share_inputs["not_need_stop"],
            self.share_inputs["seq_lens_this_time"],
            self.share_inputs["seq_lens_encoder"],
            self.share_inputs["seq_lens_decoder"],
            self.share_inputs["input_ids"],
            self.share_inputs["stop_nums"],
            sampler_output.sampled_token_ids,
            self.share_inputs["is_block_step"],
        )
        if sampler_output.logprobs_tensors is None:
            save_output(
                sampler_output.sampled_token_ids,
                self.share_inputs["not_need_stop"],
                self.rank,
                False,  # use_ep
            )
        else:
            save_output_topk(
                sampler_output.sampled_token_ids,
                sampler_output.logprobs_tensors.logprob_token_ids,
                sampler_output.logprobs_tensors.logprobs,
                sampler_output.logprobs_tensors.selected_token_ranks,
                self.share_inputs["not_need_stop"],
                self.rank,
            )

    def _cal_theortical_kvcache(self):
        """
        Calculate the size of kvcache for computational theory
        """
        num_layers = self.model_cfg.get("num_layers",
                                        None) or self.model_cfg.get(
                                            "num_hidden_layers", None)
        byte_of_cache = 2
        # support c8 c4

        hidden_dim = self.model_cfg.head_dim * self.model_cfg.kv_num_head
        theoretical_kv_cache_memory = (2 * byte_of_cache *
                                       self.args.block_size * num_layers *
                                       hidden_dim)
        return theoretical_kv_cache_memory

    def _update_share_input_block_num(self):
        """
        Update share_inputs['block_tables'] and share_inputs['free_list']
        """
        num_gpu_blocks = self.num_gpu_blocks

        del self.share_inputs["caches"]
        self._init_kvcache()

        del self.share_inputs["block_tables"]
        self.share_inputs["block_tables"] = paddle.full(
            [self.args.max_num_seqs, num_gpu_blocks], -1, dtype="int32")

        # Init free list
        free_list = list(
            range(num_gpu_blocks - 1,
                  int(num_gpu_blocks * self.args.kv_cache_ratio) - 1, -1))
        self.free_list_len = len(free_list)
        self.share_inputs.update({
            "free_list":
            paddle.to_tensor(free_list, dtype="int32"),
            "free_list_len":
            paddle.full([1], self.free_list_len, dtype="int32"),
        })

    def dummy_input(self, num_total_tokens: int, number_of_tasks: int) -> None:
        """
        fake input to profile
        """
        input_length = min(num_total_tokens // number_of_tasks,
                           self.args.max_model_len - 10)
        block_num = (input_length + self.args.block_size - 1 ) // self.args.block_size \
                    + self.args.enc_dec_block_num
        self.share_inputs["free_list"] = paddle.to_tensor([], dtype="int32")
        self.share_inputs["free_list_len"][0] = 0

        for i in range(number_of_tasks):
            idx = i
            self.share_inputs["input_ids"][idx:idx +
                                           1, :input_length] = np.array(
                                               [5] * input_length)
            self.share_inputs["eos_token_id"][:] = np.array(
                [2], dtype="int64").reshape(-1, 1)
            self.share_inputs["seq_lens_this_time"][idx:idx + 1] = input_length
            self.share_inputs["step_seq_lens_encoder"][idx:idx +
                                                       1] = input_length
            self.share_inputs["seq_lens_encoder"][idx:idx + 1] = input_length
            self.share_inputs["seq_lens_decoder"][idx:idx + 1] = 0
            self.share_inputs["step_idx"][idx:idx + 1] = 0
            self.share_inputs["max_dec_len"][idx:idx + 1] = 10
            self.share_inputs["stop_flags"][idx:idx + 1] = False

            self.share_inputs["first_token_ids"][
                idx:idx + 1] = self.share_inputs["input_ids"][idx:idx + 1, :1]
            self.share_inputs["ori_seq_lens_encoder"][idx:idx +
                                                      1] = input_length

            self.share_inputs["infer_seed"][idx:idx + 1] = random.randint(
                0, 922337203685477580)
            self.share_inputs["encoder_block_lens"][idx:idx + 1] = block_num
            self.share_inputs["block_tables"][idx : idx + 1, :block_num] = np.arange(idx * block_num, \
                                                                                (idx + 1) * block_num, 1)

    def _preprocess_task(self, one: dict) -> None:
        """process batch"""

        input_ids = one["input_ids"][np.newaxis, :]
        input_ids = paddle.to_tensor(input_ids, dtype=paddle.int64)
        token_type_ids = one["token_type_ids"][np.newaxis, :]
        token_type_ids = paddle.to_tensor(token_type_ids, dtype=paddle.int64)

        if one["images"] is not None:
            image_type_ids = one["image_type_ids"][np.newaxis, :]
            images = one["images"]
            image_type_ids = paddle.to_tensor(image_type_ids,
                                              dtype=paddle.int64)
            images = paddle.to_tensor(images, dtype="uint8")
            grid_thw = paddle.to_tensor(one["grid_thw"], dtype="int64")
        else:
            image_type_ids = None
            images = None
            grid_thw = None

        if one["position_ids"] is not None:
            position_ids = paddle.to_tensor(one["position_ids"],
                                            dtype="int64").unsqueeze([0])
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
