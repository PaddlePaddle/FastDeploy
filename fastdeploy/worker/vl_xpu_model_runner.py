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
import paddle

from fastdeploy.model_executor.layers.sample.meta_data import SamplingMetadata
from fastdeploy.model_executor.ops.xpu import set_value_by_flags_and_idx
from fastdeploy.worker.output import ModelOutputData
from fastdeploy.worker.vl_gpu_model_runner import GPUVLModelRunner
from fastdeploy.worker.xpu_model_runner import (xpu_post_process,
                                                xpu_pre_process,
                                                xpu_process_output)


class XPUVLModelRunner(GPUVLModelRunner):
    """
    The XPUVLModelRunner class for vision-language tasks on XPU.
    """

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
        paddle.device.xpu.empty_cache()

    def clear_parameters(self, pid: int) -> None:
        """ clear_parameters """
        if "caches" in self.share_inputs:
            self.model.clear_parameters(pid)
            del self.share_inputs["caches"]
            paddle.device.xpu.empty_cache()
            self.model.log_memory_usage("clear all memory")

    def _prepare_inputs(self) -> None:
        """ prepare the model inputs """
        self.forward_meta = xpu_pre_process(
            self.args.max_model_len,
            self.share_inputs["input_ids"],
            self.share_inputs["seq_lens_this_time"],
            self.share_inputs,
            use_speculate_method=False,
            draft_tokens=None,
            seq_lens_encoder=self.share_inputs["seq_lens_encoder"],
            seq_lens_decoder=self.share_inputs["seq_lens_decoder"],
            attn_backend=self.attn_backend
        )
        self.forward_meta.pos_emb_type = "HALF_HEAD_DIM"
        self.share_inputs["decoder_batch_ids"] = paddle.full(
            [self.fd_config.parallel_config.max_num_seqs, 1], 0, dtype='int32')
        self.share_inputs["decoder_tile_ids_per_batch"] = paddle.full(
            [self.fd_config.parallel_config.max_num_seqs, 1], 0, dtype='int32')

        self.attn_backend.init_attention_metadata(self.forward_meta)

        # Get sampling metadata
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
        )


    def generate(self) -> None:
        """
        generate
        """
        self._prepare_inputs()
        hidden_states = self.model(self.share_inputs["ids_remove_padding"],
                                    self.share_inputs["image_features"],
                                    self.forward_meta)

        hidden_states = xpu_process_output(hidden_states,
                                            self.share_inputs["cum_offsets"],
                                            self.forward_meta)

        logits = self.model.compute_logits(hidden_states)

        set_value_by_flags_and_idx(
            self.share_inputs["pre_ids"],
            self.share_inputs["input_ids"],
            self.share_inputs["seq_lens_this_time"],
            self.share_inputs["seq_lens_encoder"],
            self.share_inputs["seq_lens_decoder"],
            self.share_inputs["step_idx"],
            self.share_inputs["stop_flags"],
        )

        next_tokens = self.sampler(logits, self.sampling_metadata)

        if self.fd_config.parallel_config.tensor_parallel_degree > 1:
            paddle.distributed.broadcast(next_tokens, 0)

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
            msg_queue_id=self.fd_config.parallel_config.msg_queue_id,
            mp_rank=self.rank,
            use_ep=self.fd_config.parallel_config.use_ep,
            # 投机解码
            full_hidden_states=None,
            draft_tokens=None,
            actual_draft_token_num=None,
            accept_tokens=None,
            accept_num=None,
        )

        xpu_post_process(sampled_token_ids=next_tokens,
                    model_output=model_output_data)

        return None
