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

from typing import Any, Dict, Optional

from fastdeploy.worker.worker_process import initialize_fd_config


class RolloutModelConfig:
    def __init__(
        self,
        model_name_or_path: str,
        max_model_len: int = 32768,
        tensor_parallel_size: int = 4,
        dynamic_load_weight: bool = True,
        load_strategy: str = "meta",
        enable_mm: bool = False,
        # Default values for all other parameters
        max_num_seqs: int = 34,
        total_block_num: int = 2000,
        block_size: int = 64,
        engine_worker_queue_port: str = "8002",
        device_ids: str = "0",
        dtype: str = "bfloat16",
        enc_dec_block_num: int = 1,
        kv_cache_ratio: float = 0.7,
        first_token_id: int = 1,
        gpu_memory_utilization: float = 0.9,
        engine_pid: int = None,
        do_profile: bool = False,
        pad_token_id: int = -1,
        eos_tokens_lens: int = 2,
        enable_chunked_prefill: bool = False,
        speculative_method: str = None,
        speculative_max_draft_token_num: int = 1,
        speculative_model_name_or_path: str = "",
        speculative_model_quantization: str = "WINT8",
        max_num_batched_tokens: int = 2048,
        enable_prefix_caching: bool = False,
        splitwise_role: str = "mixed",
        expert_parallel_size: int = 1,
        enable_expert_parallel: bool = False,
        ori_vocab_size: int = None,
        quantization: Optional[Dict[str, Any]] = None,
        guided_decoding_backend: str = "off",
        disable_any_whitespace: bool = True,
        enable_logprob: bool = False,
        graph_optimization_config: str = None,
        early_stop_config: str = None,
        local_rank: int = 0,
        plas_attention_config: str = None,
        data_parallel_size: int = 1,
        num_nextn_predict_layers: int = 0,
        eplb_config: str = {},
    ):
        # Required parameters
        self.model = model_name_or_path
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        self.dynamic_load_weight = dynamic_load_weight
        self.load_strategy = load_strategy
        self.enable_mm = enable_mm

        # Optional parameters with defaults
        self.max_num_seqs = max_num_seqs
        self.total_block_num = total_block_num
        self.block_size = block_size
        self.engine_worker_queue_port = engine_worker_queue_port
        self.device_ids = device_ids
        self.dtype = dtype
        self.enc_dec_block_num = enc_dec_block_num
        self.kv_cache_ratio = kv_cache_ratio
        self.first_token_id = first_token_id
        self.gpu_memory_utilization = gpu_memory_utilization
        self.engine_pid = engine_pid
        self.do_profile = do_profile
        self.pad_token_id = pad_token_id
        self.eos_tokens_lens = eos_tokens_lens
        self.enable_chunked_prefill = enable_chunked_prefill
        self.speculative_config = {}
        self.speculative_config["method"] = speculative_method
        self.speculative_config["max_draft_token_num"] = speculative_max_draft_token_num
        self.speculative_config["model"] = speculative_model_name_or_path
        self.speculative_config["quantization"] = speculative_model_quantization
        self.max_num_batched_tokens = max_num_batched_tokens
        self.enable_prefix_caching = enable_prefix_caching
        self.splitwise_role = splitwise_role
        self.expert_parallel_size = expert_parallel_size
        self.enable_expert_parallel = enable_expert_parallel
        self.data_parallel_size = data_parallel_size
        self.ori_vocab_size = ori_vocab_size
        self.quantization = quantization
        self.guided_decoding_backend = guided_decoding_backend
        self.disable_any_whitespace = disable_any_whitespace
        self.enable_logprob = enable_logprob
        self.graph_optimization_config = graph_optimization_config
        self.local_rank = local_rank
        self.early_stop_config = early_stop_config
        self.ips = None
        self.plas_attention_config = plas_attention_config
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.eplb_config = eplb_config

    def __str__(self):
        return "\n".join(f"{k}: {v}" for k, v in self.__dict__.items())

    def initialize(self):
        """Initialize the final fd config"""
        return initialize_fd_config(self, ranks=self.tensor_parallel_size, local_rank=self.local_rank)
