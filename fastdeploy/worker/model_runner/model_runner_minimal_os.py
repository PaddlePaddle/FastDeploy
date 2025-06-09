"""
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastdeploy import InferenceArgs
    from fastdeploy.config import ModelConfig
    from fastdeploy.worker.model_runner import MHATokenToKVPool


class MinimalModelRunner:
    """Fake ModelRunner implementing minimal functionality for inference testing."""

    def __init__(self, model_config: 'ModelConfig', gpu_id: int, tp_rank: int,
                 tp_size: int, server_args: 'InferenceArgs'):
        import paddle

        # Parse args
        self.model_config = model_config
        self.device = "cuda" if paddle.device.cuda.device_count(
        ) > 0 else "cpu"
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.server_args = server_args
        self.should_log = tp_rank == 0

        self.dtype = paddle.float16
        # Max batch size for the test.
        max_batch_size = 160
        # Total tokens(prefix + extend + decode) in the test should not exceed this length.
        max_context_len = 2048

        self.sliding_window_size = None
        self.device = self.device
        # Create a large enough req_to_token_pool to fit the test usage.
        self.req_to_token_pool = type(
            "TokenPool",
            (),
            {
                # A typical max_bs * max_context_len for cuda graph decode
                "size":
                max_batch_size,
                # Add req_to_token attribute
                "req_to_token":
                paddle.zeros([max_batch_size, max_context_len],
                             dtype=paddle.int32),
            },
        )
        self.page_size = server_args.get("attentioin_page_size", 1)
        max_total_num_tokens = max_batch_size * max_context_len
        self.token_to_kv_pool = MHATokenToKVPool(
            size=max_total_num_tokens,
            page_size=self.page_size,
            dtype=self.dtype,
            head_num=self.model_config.num_attention_heads,
            head_dim=self.model_config.hidden_size //
            self.model_config.num_attention_heads,
            layer_num=1,
            device=self.device,
        )
