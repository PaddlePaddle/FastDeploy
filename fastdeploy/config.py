"""
# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Literal, Optional

import paddle
from paddleformers.transformers.configuration_utils import PretrainedConfig
from paddleformers.trl import llm_utils

from fastdeploy import envs
from fastdeploy.model_executor.layers.quantization.quant_base import \
    QuantConfigBase
from fastdeploy.platforms import current_platform
from fastdeploy.scheduler import SchedulerConfig
from fastdeploy.utils import (ceil_div, check_unified_ckpt, get_logger,
                              llm_logger)

logger = get_logger("config", "config.log")

class MoEPhase(Enum):
    """
    The generation phase of the moe.
    """

    PREFILL = 1
    DECODER = 2

PRETRAINED_INIT_CONFIGURATION = {
    "temperature":1.0,
    "top_p":0.0,
    "rope_theta": 10000.0,
    "penalty_score": 1.0,
    "frequency_score":0.0,
    "presence_score":0.0,
    "num_key_value_heads":-1,
    "start_layer_index": 0,
    "moe_num_shared_experts":0,
    "moe_layer_start_index": 0,
    "num_max_dispatch_tokens_per_rank":256,
    "moe_use_aux_free":False,
    "vocab_size": -1,
    "use_rope": True,
    "hidden_dropout_prob":0.0,
    "initializer_range":0.02,
    "max_position_embeddings":512,
    "quantization_config":None,
    "use_recompute_resampler":False,
    "use_temporal_conv":True,
    "resampler_fuse_rms_norm":False,
    "freq_allocation":20,
    "tie_word_embeddings":False,
    "rms_norm_eps":1e-5,
    "min_length":1,
    "im_patch_id":(100295),
}


class ModelConfig:
    """
    The configuration class to store the configuration of a `LLM`.
    Args:
        args (dict): A dictionary containing key and value mappings for config fields.
    """
    def __init__(
        self,
        args: dict,
    ):
        self.model_name_or_path = ""
        self.is_quantized = False
        self.dtype = ""
        self.enable_logprob = False
        self.quantization = None
        self.tokenizer = None

        for key, value in args.items():
            if hasattr(self, key):
                setattr(self, key, value)

        pretrained_config, _ = PretrainedConfig.get_config_dict(self.model_name_or_path)
        self.pretrained_config = PretrainedConfig.from_dict(pretrained_config)

        # set attribute from pretrained_config
        for key, value in pretrained_config.items():
            setattr(self, key, value)

        # we need set default value when not exist
        for key, value in PRETRAINED_INIT_CONFIGURATION.items():
            if not hasattr(self, key):
                setattr(self, key, value)

        if not hasattr(self, "head_dim"):
            self.head_dim = self.hidden_size // self.num_attention_heads

        if hasattr(self, "vision_config"):
            self.vision_config = PretrainedConfig.from_dict(self.vision_config)

        self.ori_vocab_size = self.vocab_size

        if isinstance(self.architectures, list):
            self.architectures = self.architectures[0]

        if self.architectures in ["Ernie4_5_ForCausalLM","Ernie4_5_MoeForCausalLM"]:
            self.ori_vocab_size = args["ori_vocab_size"]

        self.is_unified_ckpt = check_unified_ckpt(self.model_name_or_path)
        self.override_name_from_config()
        self.read_from_env()

    def override_name_from_config(self):
        """
        Override attribute names from the exported model's configuration.
        """

        if not self.is_unified_ckpt and hasattr(self, "infer_model_mp_num"):
            self.tensor_parallel_size = self.infer_model_mp_num
            del self.infer_model_mp_num

        if hasattr(self, "num_hidden_layers"):
            if hasattr(self, "remove_tail_layer"):
                if self.remove_tail_layer is True:
                    self.num_hidden_layers -= 1
                elif isinstance(self.remove_tail_layer, int):
                    self.num_hidden_layers -= self.remove_tail_layer

            self.num_layers = self.num_hidden_layers
            del self.num_hidden_layers

        if not hasattr(self, "mla_use_absorb"):
            self.mla_use_absorb = False
        if not hasattr(self, "head_dim"):
            assert hasattr(self, "hidden_size") and hasattr(
                self, "num_attention_heads")
            self.head_dim = self.hidden_size // self.num_attention_heads

    def read_from_env(self):
        """
        Read configuration information from environment variables and update the object's attributes.

        If an attribute is not present or is an empty string in the environment variables, use the default value.
        """
        self.max_stop_seqs_num = int(envs.FD_MAX_STOP_SEQS_NUM)
        self.stop_seqs_max_len = int(envs.FD_STOP_SEQS_MAX_LEN)

        def reset_config_value(key, value):
            if not hasattr(self, key.lower()):
                if os.getenv(key, None):
                    value = eval(os.getenv(key))
                    llm_logger.info(
                        f"Get parameter `{key}` = {value} from environment.")
                else:
                    llm_logger.info(
                        f"Parameter `{key}` will use default value {value}.")
                setattr(self, key.lower(), value)

        reset_config_value("COMPRESSION_RATIO", 1.0)
        reset_config_value("ROPE_THETA", 10000)

    def _get_download_model(self, model_name, model_type="default"):
        # TODO: Provide dynamic graph for self-downloading and save to the specified download directory.
        pass

    def print(self):
        """
        Print all configuration information.
        """
        llm_logger.info("Model Configuration Information :")
        for k, v in self.__dict__.items():
            llm_logger.info("{:<20}:{:<6}{}".format(k, "", v))
        llm_logger.info(
            "=============================================================")



class ParallelConfig:
    """Configuration for the distributed execution."""
    def __init__(
        self,
        args,
    ):
        self.sequence_parallel = False  # Whether to enable sequence parallelism.
        self.use_ep = False  # Whether to enable Expert Parallelism
        self.moe_phase = MoEPhase.PREFILL  # Generation phase
        self.msg_queue_id = 1  # mesage queue id

        tensor_parallel_rank, tensor_parallel_size = llm_utils.init_dist_env()
        self.tensor_parallel_rank = tensor_parallel_rank  # TP rank ID
        self.tensor_parallel_size = tensor_parallel_size  # TP degree
        self.expert_parallel_rank = int(tensor_parallel_rank / tensor_parallel_size)  # EP rank ID
        self.expert_parallel_size = 1  # EP degree

        # Set default block num for profile run
        self.max_block_num: int = 2000


        # Encoder's decoder num
        self.enc_dec_block_num: int = 1
        # KV cache ratio for input
        self.kv_cache_ratio: float = 0.7
        # First token id
        self.first_token_id: int = 1
        # Gpu memory utilization
        self.gpu_memory_utilization: float = 0.9
        # Process ID of engine
        self.engine_pid: Optional[int] = None
        # Do profile or not
        self.do_profile: bool = False
        #
        self.pad_token_id: int = -1
        #
        self.eos_tokens_lens: int = 2

        # enable prefix cache
        self.enable_prefix_caching = None

        self.data_parallel_size = 1
        self.enable_expert_parallel = False
        self.local_data_parallel_id = 0
        # enable the custom all-reduce kernel and fall back to NCCL(dist.all_reduce).
        self.enable_custom_all_reduce: bool = False

        self.max_prefill_batch = 3
        for key, value in args.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.use_ep = args["expert_parallel_size"] > 1

        # TODO(@wufeisheng): TP and EP need to be supported simultaneously.
        assert (self.tensor_parallel_size == 1
                and self.expert_parallel_size
                >= 1) or (self.tensor_parallel_size >= 1
                          and self.expert_parallel_size
                          == 1), "TP and EP cannot be enabled at the same time"

        self.num_ranks = self.tensor_parallel_size * self.expert_parallel_size
        self.max_chips_per_node = 16 if current_platform.is_iluvatar() else 8
        if self.num_ranks > self.max_chips_per_node:
            self.worker_num_per_node = self.max_chips_per_node
        else:
            self.worker_num_per_node = self.num_ranks

    def check(self):
        assert (
            self.max_chips_per_node >= self.tensor_parallel_size > 0
        ), f"tensor_parallel_size: {self.tensor_parallel_size} should be between 1 and {self.max_chips_per_node}"


    def print(self):
        """
        print all config

        """
        llm_logger.info("Parallel Configuration Information :")
        for k, v in self.__dict__.items():
            llm_logger.info("{:<20}:{:<6}{}".format(k, "", v))
        llm_logger.info(
            "=============================================================")

class SpeculativeConfig:
    """
    Configuration for speculative decoding.
    """
    def __init__(
        self,
        args,
    ):
        # speculative method, choose in [None, "ngram_match", "mtp"]
        self.method: Optional[str] = None
        # the max length of speculative tokens
        self.num_speculative_tokens: int = 1
        # the max length of candidate tokens for speculative method
        self.max_candidate_len: int = 5
        # the max length of verify window for speculative method
        self.verify_window: int = 2
        # ngram match
        self.max_ngram_size: int = 5
        # model for mtp/eagle/draft_model
        self.model_name_or_path: Optional[str] = None
        # quantization of model
        self.quantization: Optional[str] = None
        # allocate more blocks to prevent mtp from finishing the block earlier than the main model
        # Fixed now
        self.num_gpu_block_expand_ratio: Optional[float] = 1
        # To distinguish the main model and draft model(mtp/eagle/draftmodel)
        # ["main", "mtp"]
        self.model_type: Optional[str] = "main"
        # TODO(liuzichang): To reduce memory usage, MTP shares the main model's lm_head and embedding layers.
        # A trick method is currently used to enable this sharing.
        # This will be replaced with a more standardized solution in the future.
        self.sharing_model = None
        # During benchmarking, we need to enforce that the number of accepted tokens is 1.
        # This means no tokens from MTP are accepted.
        # This ensures that the specified simulation acceptance rate is not affected.
        self.benchmark_mode: bool = False

        self.num_extra_cache_layer = 0

        for key, value in args.items():
            if hasattr(self, key):
                setattr(self, key, value)

        self.read_model_config()
        self.reset()

    def read_model_config(self):
        """
        Read configuration from file.
        """
        self.model_config = {}
        if not self.enabled_speculative_decoding():
            return

        self.is_unified_ckpt = check_unified_ckpt(self.model_name_or_path)
        if self.model_name_or_path is None:
            return

        self.config_path = os.path.join(self.model_name_or_path, "config.json")
        if os.path.exists(self.config_path):
            self.model_config = json.load(
                open(self.config_path, 'r', encoding='utf-8'))

    def reset(self):
        """
        Reset configuration.
        """

        def reset_value(cls, value_name, key=None, default=None):
            if key is not None and key in cls.model_config:
                setattr(cls, value_name, cls.model_config[key])
            elif getattr(cls, value_name, None) is None:
                setattr(cls, value_name, default)

        if not self.enabled_speculative_decoding():
            return

        # NOTE(liuzichang): We will support multi-layer in future
        if self.method in ["mtp"]:
            self.num_extra_cache_layer = 1

    def enabled_speculative_decoding(self):
        """
        Check if speculative decoding is enabled.
        """
        if self.method is None:
            return False
        return True

    def to_json_string(self):
        """
        Convert speculative_config to json string.
        """
        return json.dumps({
            key: value
            for key, value in self.__dict__.items() if value is not None
        })

    def print(self):
        """
        print all config

        """
        llm_logger.info("Speculative Decoding Configuration Information :")
        for k, v in self.__dict__.items():
            llm_logger.info("{:<20}:{:<6}{}".format(k, "", v))
        llm_logger.info(
            "=============================================================")


class DeviceConfig:
    """
    Configuration for device settings.
    """
    def __init__(
        self,
        args,
    ):
        self.type = "cuda"
        self.ids: str = "0" # Visible devices ids

        self.ids = os.getenv("CUDA_VISIBLE_DEVICES", None)
        if current_platform.is_xpu():
            self.ids = os.getenv("XPU_VISIBLE_DEVICES", None)
        for key, value in args.items():
            if hasattr(self, key):
                setattr(self, key, value)

class GraphOptimizationConfig:
    def init_with_cudagrpah_size(self,
                                 cudagraph_capture_sizes: list[int]) -> None:
        """To complete the initialization of config,
        we need to know the cudagraph sizes"""
        if self.cudagraph_capture_sizes is None:
            self.cudagraph_capture_sizes = cudagraph_capture_sizes
        else:
            dedup_sizes = list(set(self.cudagraph_capture_sizes))
            if len(dedup_sizes) < len(self.cudagraph_capture_sizes):
                logger.info(("cudagraph sizes specified by model runner"
                             " %s is overridden by config %s"),
                            cudagraph_capture_sizes, dedup_sizes)
            self.cudagraph_capture_sizes = dedup_sizes

        # sort to make sure cudagraph capture sizes are in descending order
        self.cudagraph_capture_sizes.sort(reverse=True)
        self.max_capture_size = self.cudagraph_capture_sizes[
            0] if self.cudagraph_capture_sizes else 0

        # pre-compute the mapping from batch size to padded graph size
        self.batch_size_to_captured_size = {}
        for end, start in zip(self.cudagraph_capture_sizes,
                              self.cudagraph_capture_sizes[1:] + [0]):
            for bs in range(start, end):
                if bs == start:
                    self.batch_size_to_captured_size[bs] = start
                else:
                    self.batch_size_to_captured_size[bs] = end
        self.batch_size_to_captured_size[
            self.max_capture_size] = self.max_capture_size

    def __init__(self,
                 enable_static_graph_inference: bool = False,
                 max_capture_batch_size: int = 64,
                 args = None):
        """The Top-level graph optimization contral corresponds to different backends.
        - 0: dyncmic graph
        - 1: static graph
        - 2: static graph + cinn compilation backend
        """
        self.graph_opt_level: int = 0

        # CUDA Graph Config
        """ Whether to use cudagraph.
        - False: cudagraph is not used.
        - True: cudagraph is used.
            It requires that all input buffers have fixed addresses, and all
            splitting ops write their outputs to input buffers.
            - With dyncmic graph backend: ...
            - With static grpah backend: WIP
        """
        self.use_cudagraph: bool = False
        """Sizes to capture cudagraph.
        - None (default): capture sizes are inferred from llm config.
        - list[int]: capture sizes are specified as given."""
        self.cudagraph_capture_sizes: Optional[list[int]] = None
        """ Number of warmup runs for cudagraph. """
        self.cudagraph_num_of_warmups: int = 2
        """Whether to copy input tensors for cudagraph.
        If the caller can guarantee that the same input buffers
        are always used, it can set this to False. Otherwise, it should
        set this to True."""
        self.cudagraph_copy_inputs: bool = False
        """ In static graph, this is an operation list that does not need to be captured by the CUDA graph.
        CudaGraphBackend will split these operations from the static graph.
        Example usage:
            cudagraph_splitting_ops = ["paddle.unified_attention"]

        Note: If want to use subgraph capture functionality in a dynamic graph,
        can manually split the model into multiple layers and apply the @support_cuda_graph decorator
        only to the layer where CUDA graph functionality is required.
        """
        self.cudagraph_splitting_ops = Optional[list[str]]
        """"whether to use a full cuda graph for the entire forward pass rather than
        splitting certain operations such as attention into subgraphs.
        Thus this flag cannot be used together with splitting_ops."""
        self.full_cuda_graph: bool = False

        self.max_capture_size: int = field(default=None, init=False)  # type: ignore
        self.batch_size_to_captured_size: dict[int,
                                        int] = field(default=None,
                                                    init=False)  # type: ignore

        # CINN Config ...

        for key, value in args.items():
            if hasattr(self, key):
                setattr(self, key, value)
        capture_size = [i for i in range(1, max_capture_batch_size + 1)]
        self.init_with_cudagrpah_size(cudagraph_capture_sizes=capture_size)
        #TODO(wangmingkai02): change graph_opt_level=2 when using static mode with cinn
        if enable_static_graph_inference:
            self.graph_opt_level = 1

class LoadConfig:
    """
    Configuration for dynamic weight loading strategies

    Attributes:
        dynamic_load_weight: Whether to enable dynamic weight loading
        load_strategy: Specifies the weight loading method when enabled:
            - 'ipc': Real-time IPC streaming with automatic resharding
            - 'ipc_no_reshard': Real-time IPC streaming without weight process
            - 'ipc_snapshot': Load from disk snapshot of IPC weights
            - 'meta': provide RL traing worker, no_weights_load
            - None: No dynamic loading
    """
    def __init__(
        self,
        args,
    ):
        self.use_fastsafetensor = int(envs.FD_USE_FASTSAFETENSOR) == 1
        self.dynamic_load_weight: bool = False
        self.load_strategy: Optional[Literal['ipc', 'ipc_no_reshard', 'ipc_snapshot', 'meta']] = None
        for key, value in args.items():
            if hasattr(self, key):
                setattr(self, key, value)

class LoRAConfig:
    """ LoRA Config """
    pass


class DecodingConfig:
    """
    Configuration for decoding
    """
    def __init__(
        self,
        args,
    ):
        self.reasoning_parser = None
        # guided decoding backend
        self.guided_decoding_backend: str = None
        # disable any whitespace for guided decoding
        self.disable_any_whitespace: bool = True
        for key, value in args.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def check(self):
        if self.guided_decoding_backend is not None:
            assert self.guided_decoding_backend in ["xgrammar", "XGrammar", "auto", "off"], \
                f"Only support xgrammar、auto guided decoding backend, but got {self.guided_decoding_backend}."

            if self.guided_decoding_backend != "off":
                # TODO: mm support guided_decoding
                assert self.enable_mm is False, "Multimodal model currently do not support guided_decoding"

                # TODO: speculative decoding support guided_decoding

                # TODO: xpu support guided_decoding
                assert not current_platform.is_xpu(
                ), "XPU currently do not support guided_decoding"

                try:
                    import xgrammar  # noqa
                except Exception as e:
                    raise Exception(
                        f"import XGrammar failed, please install XGrammar use `pip install xgrammar==0.1.19`. \n\t {e}"
                    )


class CacheConfig:
    """
    Configuration for the KV cache.

    Attributes:
        block_size (int): Size of a cache block in number of tokens.
        gpu_memory_utilization (float): Fraction of GPU memory to use for model execution.
        cache_dtype (str): Data type for kv cache storage. Default is 'bfloat16'.
        num_gpu_blocks_override (Optional[int]): Number of GPU blocks to use.
        Overrides profiled num_gpu_blocks if provided.
        kv_cache_ratio (float): Ratio for calculating the maximum block number.
        enc_dec_block_num (int): Number of encoder-decoder blocks.
        enable_prefix_caching (bool): Flag to enable prefix caching.
    """

    def __init__(
        self,
        block_size: int,
        gpu_memory_utilization: float,
        cache_dtype: str = "bfloat16",
        num_gpu_blocks_override: Optional[int] = None,
        swap_space: Optional[int] = None,
        kv_cache_ratio: float = 0.75,
        enc_dec_block_num: int = 2,
        tensor_parallel_size: int = 1,
        enable_prefix_caching=False,
        enable_ssd_cache=False,
        model_cfg=None,
        cache_queue_port=None,
        rdma_comm_ports=None,
        cache_transfer_protocol=None,
        pd_comm_port=None,
    ):
        """
        Initialize the CacheConfig class.

        Args:
            block_size (int): Size of a cache block in number of tokens.
            gpu_memory_utilization (float): Fraction of GPU memory to use.
            cache_dtype (str): Data type for cache storage. Default is 'bfloat16'.
            num_gpu_blocks_override (Optional[int]): Override for number of GPU blocks.
            num_cpu_blocks (Optional[int]): Number of CPU blocks.
            kv_cache_ratio (float): Ratio for max block calculation.
            enc_dec_block_num (int): Number of encoder-decoder blocks.
            enable_prefix_caching (bool): Enable prefix caching.
        """
        self.block_size = block_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.num_gpu_blocks_override = num_gpu_blocks_override
        self.kv_cache_ratio = kv_cache_ratio
        self.enc_dec_block_num = enc_dec_block_num
        self.cache_dtype = cache_dtype
        if hasattr(model_cfg, "quantization_config"):
            self.cache_dtype = model_cfg.quantization_config.get(
                "kv_cache_quant_type", cache_dtype)

        self.rdma_comm_ports = rdma_comm_ports
        self.cache_transfer_protocol = cache_transfer_protocol
        self.pd_comm_port = pd_comm_port

        if rdma_comm_ports is not None and isinstance(rdma_comm_ports, str):
            self.rdma_comm_ports = rdma_comm_ports.split(',')

        if pd_comm_port is not None and isinstance(pd_comm_port, str):
            self.pd_comm_port = [int(port) for port in pd_comm_port.split(",")]

        self.enable_prefix_caching = enable_prefix_caching
        if swap_space is None:
            self.enable_hierarchical_cache = False
        else:
            self.enable_hierarchical_cache = True

        self.enable_ssd_cache = enable_ssd_cache
        self.model_cfg = model_cfg
        self.cache_queue_port = cache_queue_port
        self.swap_space = swap_space

        if (hasattr(self.model_cfg, "num_key_value_heads")
                and hasattr(self.model_cfg, "num_key_value_heads")
                and self.model_cfg.num_key_value_heads is not None
                and int(self.model_cfg.num_key_value_heads) > 0):
            kv_num_head = int(self.model_cfg.num_key_value_heads)
        else:
            kv_num_head = self.model_cfg.num_attention_heads
        self.model_cfg.kv_num_head = kv_num_head

        # TODO check name
        if "int4" in self.cache_dtype.lower(
        ) or "float4" in self.cache_dtype.lower():
            byte_size = 0.5
            self.cache_dtype = "uint8"
        elif "int8" in self.cache_dtype.lower(
        ) or "float8" in self.cache_dtype.lower():
            self.cache_dtype = "uint8"
            byte_size = 1
        else:
            byte_size = 2

        self.each_token_cache_space = int(
            self.model_cfg.num_layers * kv_num_head * self.model_cfg.head_dim *
            byte_size)
        self.bytes_per_block = int(self.each_token_cache_space *
                                   self.block_size)
        self.bytes_per_layer_per_block = int(
            self.block_size * self.model_cfg.kv_num_head *
            self.model_cfg.head_dim // tensor_parallel_size * byte_size)

        if self.swap_space is None:
            self.num_cpu_blocks = 0
        else:
            self.num_cpu_blocks = int(self.swap_space * 1024**3 /
                                      self.bytes_per_block)
        self._verify_args()

    def metrics_info(self):
        """Convert cache_config to dict(key: str, value: str) for prometheus metrics info."""
        return {key: str(value) for key, value in self.__dict__.items()}

    def _verify_args(self):
        if self.gpu_memory_utilization > 1.0:
            raise ValueError(
                "GPU memory utilization must be less than 1.0. Got "
                f"{self.gpu_memory_utilization}.")
        if self.kv_cache_ratio > 1.0:
            raise ValueError("KV cache ratio must be less than 1.0. Got "
                             f"{self.kv_cache_ratio}.")

    def postprocess(self, num_total_tokens, number_of_tasks):
        """
        calculate block num
        """
        self.dec_token_num = self.enc_dec_block_num * self.block_size
        if self.num_gpu_blocks_override is not None:
            self.total_block_num = self.num_gpu_blocks_override
            self.prefill_kvcache_block_num = int(self.total_block_num *
                                                 self.kv_cache_ratio)
        else:
            length = num_total_tokens // number_of_tasks
            block_num = (length + self.block_size - 1 +
                         self.dec_token_num) // self.block_size
            self.total_block_num = block_num * number_of_tasks
            self.prefill_kvcache_block_num = self.total_block_num
            llm_logger.info(
                f"Doing profile, the total_block_num:{self.total_block_num}")

    def reset(self, num_gpu_blocks):
        """
        reset gpu block number
        """
        self.total_block_num = num_gpu_blocks
        self.prefill_kvcache_block_num = int(self.total_block_num *
                                             self.kv_cache_ratio)
        llm_logger.info(
            (f"Reset block num, the total_block_num:{self.total_block_num},"
             f" prefill_kvcache_block_num:{self.prefill_kvcache_block_num}"))

    def print(self):
        """
        print all config

        """
        llm_logger.info("Cache Configuration Information :")
        for k, v in self.__dict__.items():
            llm_logger.info("{:<20}:{:<6}{}".format(k, "", v))
        llm_logger.info(
            "=============================================================")


class CommitConfig:
    """
    Configuration for tracking version information from version.txt

    Attributes:
        fastdeploy_commit: Full FastDeploy git commit hash
        paddle_version: PaddlePaddle version string
        paddle_commit: PaddlePaddle git commit hash
        cuda_version: CUDA version string
        compiler_version: CXX compiler version string
    """
    def __init__(
        self,
    ):
        self.fastdeploy_commit: str = ""
        self.paddle_version: str = ""
        self.paddle_commit = paddle.version.commit
        self.cuda_version: str = ""
        self.compiler_version: str = ""
        self._load_from_version_file()

    def _load_from_version_file(self, file_path: str = "fastdeploy/version.txt"):
        """Internal method to load version info from file"""
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("fastdeploy GIT COMMIT ID:"):
                        self.fastdeploy_commit = line.split(":")[1].strip()
                    elif line.startswith("Paddle version:"):
                        self.paddle_version = line.split(":")[1].strip()
                    elif line.startswith("Paddle GIT COMMIT ID:"):
                        self.paddle_commit = line.split(":")[1].strip()
                    elif line.startswith("CUDA version:"):
                        self.cuda_version = line.split(":")[1].strip()
                    elif line.startswith("CXX compiler version:"):
                        self.compiler_version = line.split(":")[1].strip()
        except FileNotFoundError:
            llm_logger.info(f"Warning: Version file not found at {file_path}")
        except Exception as e:
            llm_logger.info(f"Warning: Could not read version file - {str(e)}")

    def print(self):
        """
        print all config

        """
        llm_logger.info("Fasedeploy Commit Information :")
        for k, v in self.__dict__.items():
            llm_logger.info("{:<20}:{:<6}{}".format(k, "", v))
        llm_logger.info(
            "=============================================================")


class MultiModalConfig:
    """Controls the behavior of multimodal models."""
    def __init__(
        self,
        args,
    ):
        self.limit_mm_per_prompt = None,
        self.mm_processor_kwargs = None,
        self.enable_mm = False,

        for key, value in args.items():
            if hasattr(self, key):
                setattr(self, key, value)

@dataclass
class FDConfig:
    """
    The configuration class which contains all fastdeploy-related configuration. This
    simplifies passing around the distinct configurations in the codebase.
    """
    def __init__(
        self,
        model_config: ModelConfig = None,
        cache_config: CacheConfig = None,
        scheduler_config: SchedulerConfig = None,
        parallel_config: ParallelConfig = None,
        decoding_config: DecodingConfig = None,
        speculative_config: SpeculativeConfig = None,
        device_config: DeviceConfig = None,
        load_config: LoadConfig = None,
        quant_config: QuantConfigBase = None,
        graph_opt_config: GraphOptimizationConfig = None,
        multi_modal_config: MultiModalConfig = None,
        commit_config: CommitConfig = None,
    ):
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.speculative_config = speculative_config
        self.device_config = device_config
        self.load_config = load_config
        self.quant_config = quant_config
        self.graph_opt_config = graph_opt_config
        self.cache_config = cache_config
        self.scheduler_config = scheduler_config
        self.multi_modal_config = multi_modal_config
        self.decoding_config = decoding_config

        self.read_from_config()
        self.postprocess()
        self.check()
        self.print()


    def postprocess(self):
        """
        calculate some parameters
        """
        if current_platform.is_xpu():
            self.parallel_config.max_prefill_batch = 1
        if self.multi_modal_config.enable_mm:
            self.parallel_config.max_prefill_batch = 1  # TODO:当前多模prefill阶段只支持并行度为1,待优化

        if self.device_config.ids is None:
            self.device_config.ids = ",".join([str(i) for i in range(self.parallel_config.worker_num_per_node)])

        assert self.device_config.ids.split(',').__len__() == self.parallel_config.worker_num_per_node, \
        f"invalid CUDA_VISIBLE_DEVICES, should be equal to {self.parallel_config.worker_num_per_node}"

        assert self.parallel_config.worker_num_per_node % self.parallel_config.tensor_parallel_size == 0, \
            f"tensor_parallel_size: {self.parallel_config.tensor_parallel_size} should be divisible by worker_num_per_node: {self.parallel_config.worker_num_per_node}"

        self.device_config.local_device_ids = self.device_config.ids.split(
            ',')[:self.parallel_config.tensor_parallel_size]

        if self.scheduler_config.long_prefill_token_threshold == 0:
            self.scheduler_config.long_prefill_token_threshold = int(self.max_model_len * 0.04)

        self.cache_config.postprocess(self.scheduler_config.max_num_batched_tokens,
                                      self.scheduler_config.max_num_seqs)
        self.cache_config.max_block_num_per_seq = int(
            self.scheduler_config.max_model_len // self.cache_config.block_size)

        if self.decoding_config.guided_decoding_backend == "auto":
            if self.multi_modal_config.enable_mm:
                self.decoding_config.guided_decoding_backend = "off"
            else:
                self.decoding_config.guided_decoding_backend = "xgrammar"

        if self.scheduler_config.splitwise_role == "mixed":
            self.model_config.moe_phase = MoEPhase.PREFILL
        elif self.scheduler_config.splitwise_role == "prefill":
            self.model_config.moe_phase = MoEPhase.PREFILL
        elif self.scheduler_config.splitwise_role == "decode":
            self.model_config.moe_phase = MoEPhase.DECODER
        else:
            raise NotImplementedError

    def check(self):
        """
        check the legality of config
        """
        nnode = ceil_div(self.parallel_config.num_ranks, self.parallel_config.worker_num_per_node)
        assert nnode == self.scheduler_config.nnode, \
            f"nnode: {nnode}, but got {self.nnode}"

        self.decoding_config.check()
        self.parallel_config.check()
        self.scheduler_config.check()

    def print(self, file=None):
        """
        print all config

        Args:
            file (str): the path of file to save config
        """
        llm_logger.info(
            "=================== Configuration Information ===============")
        for k, v in self.__dict__.items():
            if k == "generation_config" and v is not None:
                for gck, gcv in v.to_dict().items():
                    llm_logger.info("{:<20}:{:<6}{}".format(gck, "", gcv))
            elif (k == "cache_config" or
                  k == "model_config" or
                  k == "scheduler_config" or
                  k == "parallel_config" or
                  k == "commit_config"):
                v.print()
            else:
                llm_logger.info("{:<20}:{:<6}{}".format(k, "", v))
        llm_logger.info(
            "=============================================================")
        if file is not None:
            f = open(file, "a")
            now_time = datetime.now()
            f.write(f"{now_time} configuration information as below,\n")
            for k, v in self.__dict__.items():
                f.write("{:<20}:{:<6}{}\n".format(k, "", v))
            f.close()

    def init_cache_info(self):
        """
        initialize cache info
        """
        disaggregate_info = {}
        if self.scheduler_config.splitwise_role != "mixed":
            disaggregate_info["role"] = self.scheduler_config.splitwise_role
            disaggregate_info["cache_info"] = dict()
            current_protocol = self.cache_config.cache_transfer_protocol.split(
                ",")
            disaggregate_info["transfer_protocol"] = current_protocol
            for protocol in current_protocol:
                if protocol == "ipc":
                    disaggregate_info["cache_info"][protocol] = {
                        "ip": self.scheduler_config.host_ip,
                        "port": self.scheduler_config.engine_worker_queue_port,
                        "device_ids": self.device_config.local_device_ids
                    }
                elif protocol == "rdma":
                    disaggregate_info["cache_info"][protocol] = {
                        "ip": self.scheduler_config.host_ip,
                        "port": self.cache_config.pd_comm_port[0],
                        "rdma_port": self.cache_config.rdma_comm_ports,
                    }
        self.scheduler_config.disaggregate_info = disaggregate_info
        llm_logger.info(f"disaggregate_info: {disaggregate_info}")

    def read_from_config(self):
        """
        reset config from json file
        """
        pass

    def _check_master(self):
        return self.scheduler_config.is_master

    def _str_to_list(self, attr_name, default_type):
        if hasattr(self, attr_name):
            val = getattr(self, attr_name)
            if type(val) is str:
                setattr(self, attr_name, [default_type(i) for i in val.split(",")])
            else:
                setattr(self, attr_name, val)

    def __str__(self) -> str:
        return json.dumps(self.__dict__, indent=4)
