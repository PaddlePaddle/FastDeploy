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

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.forward_meta import DCUForwardMeta
from fastdeploy.worker.gpu_model_runner import GPUModelRunner


class DCUModelRunner(GPUModelRunner):
    def __init__(
        self,
        fd_config: FDConfig,
        device: str,  # logic device
        device_id: int,  # physical device id
        rank: int,
        local_rank: int,
    ):
        super(DCUModelRunner, self).__init__(
            fd_config=fd_config, device=device, device_id=device_id, rank=rank, local_rank=local_rank
        )

    def initialize_forward_meta(self):
        """
        Initialize forward meta and attention meta data
        """
        # Initialize forward meta
        self.forward_meta = DCUForwardMeta(
            input_ids=self.share_inputs["input_ids"],
            ids_remove_padding=self.share_inputs["ids_remove_padding"],
            rotary_embs=self.share_inputs["rope_emb"],
            attn_backend=self.attn_backends[0],
            decoder_batch_ids=self.share_inputs["decoder_batch_ids"],
            decoder_tile_ids_per_batch=self.share_inputs["decoder_tile_ids_per_batch"],
            decoder_num_blocks_cpu=self.share_inputs["decoder_num_blocks_cpu"],
            max_len_tensor_cpu=self.share_inputs["max_len_tensor_cpu"],
            seq_lens_encoder=self.share_inputs["seq_lens_encoder"],
            seq_lens_decoder=self.share_inputs["seq_lens_decoder"],
            seq_lens_this_time=self.share_inputs["seq_lens_this_time"],
            batch_id_per_token=self.share_inputs["batch_id_per_token"],
            cum_offsets=self.share_inputs["cum_offsets"],
            cu_seqlens_q=self.share_inputs["cu_seqlens_q"],
            cu_seqlens_k=self.share_inputs["cu_seqlens_k"],
            block_tables=self.share_inputs["block_tables"],
            caches=self.share_inputs["caches"],
        )

        # Update Batch type for cuda graph
        only_decode_batch = True
        prefill_exists = None
        # mix ep in single node
        if self.fd_config.parallel_config.use_ep and self.fd_config.scheduler_config.splitwise_role == "mixed":
            only_decode_batch_list = []
            prefill_exists = self.exist_prefill()
            paddle.distributed.all_gather_object(only_decode_batch_list, not prefill_exists)
            only_decode_batch = all(only_decode_batch_list)
            self.fd_config.model_config.moe_phase.phase = "decode" if only_decode_batch else "prefill"

        self.forward_meta.step_use_cudagraph = (
            self.use_cudagraph
            and only_decode_batch
            and not (prefill_exists if prefill_exists is not None else self.exist_prefill())
        )

        # Initialzie attention meta data
        for attn_backend in self.attn_backends:
            attn_backend.init_attention_metadata(self.forward_meta)
