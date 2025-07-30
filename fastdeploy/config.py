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
from enum import Enum
from typing import Literal, Optional, Union

from paddleformers.transformers.configuration_utils import PretrainedConfig

import fastdeploy
from fastdeploy import envs
from fastdeploy.model_executor.layers.quantization.quant_base import QuantConfigBase
from fastdeploy.utils import check_unified_ckpt, get_logger

logger = get_logger("config", "config.log")

TaskOption = Literal["generate"]


class MoEPhase:
    """
    The generation phase of the moe.
    """

    def __init__(self, phase="prefill"):
        self._phase = phase

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, value):
        if value not in ["prefill", "decode"]:
            raise ValueError(f"The moe_phase is invalid, only support prefill and decode, but got {value}")
        else:
            self._phase = value


class ErnieArchitectures:
    """Helper class for ERNIE architecture check."""

    ARCHITECTURES = {
        "Ernie4_5_ForCausalLM",
        "Ernie4_5_MoeForCausalLM",
        "Ernie4_5_VLMoeForConditionalGeneration",
    }

    @classmethod
    def contains_ernie_arch(cls, architectures):
        """Check if any ERNIE architecture is present in the given architectures."""
        return any(arch in architectures for arch in cls.ARCHITECTURES)

    @classmethod
    def is_ernie_arch(cls, architecture):
        """Check if the given architecture is an ERNIE architecture."""
        return architecture in cls.ARCHITECTURES


PRETRAINED_INIT_CONFIGURATION = {
    "top_p": 1.0,
    "temperature": 1.0,
    "rope_theta": 10000.0,
    "penalty_score": 1.0,
    "frequency_score": 0.0,
    "presence_score": 0.0,
    "min_length": 1,
    "num_key_value_heads": -1,
    "start_layer_index": 0,
    "moe_num_shared_experts": 0,
    "moe_layer_start_index": 0,
    "num_max_dispatch_tokens_per_rank": 256,
    "moe_use_aux_free": False,
    "vocab_size": -1,
    "hidden_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "max_position_embeddings": 512,
    "quantization_config": None,
    "tie_word_embeddings": False,
    "rms_norm_eps": 1e-5,
    "moe_num_experts": None,
    "moe_layer_end_index": None,
}


class ModelConfig:
    """
    The configuration class to store the configuration of a `LLM`.
    """

    def __init__(
        self,
        args,
    ):
        self.model = ""
        self.is_quantized = False
        self.max_model_len = 0
        self.dtype = ""
        self.enable_logprob = False
        self.enable_mm = False
        self.enable_redundant_experts = False
        self.redundant_experts_num = 0
        self.quantization = None
        for key, value in args.items():
            if hasattr(self, key):
                setattr(self, key, value)

        assert self.model != ""
        pretrained_config, _ = PretrainedConfig.get_config_dict(self.model)
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
        if ErnieArchitectures.contains_ernie_arch(self.architectures):
            self.ori_vocab_size = args.get("ori_vocab_size", self.ori_vocab_size)

        self.is_unified_ckpt = check_unified_ckpt(self.model)

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

        if not hasattr(self, "mla_use_absorb"):
            self.mla_use_absorb = False

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
                    logger.info(f"Get parameter `{key}` = {value} from environment.")
                else:
                    logger.info(f"Parameter `{key}` will use default value {value}.")
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
        logger.info("Model Configuration Information :")
        for k, v in self.__dict__.items():
            logger.info("{:<20}:{:<6}{}".format(k, "", v))
        logger.info("=============================================================")


class ParallelConfig:
    """Configuration for the distributed execution."""

    def __init__(
        self,
        args,
    ):
        self.sequence_parallel = False  # Whether to enable sequence parallelism.
        self.use_ep = False  # Whether to enable Expert Parallelism
        self.moe_phase = MoEPhase("prefill")  # Generation phase
        self.msg_queue_id = 1  # mesage queue id

        self.tensor_parallel_rank = 0  # TP rank ID
        self.tensor_parallel_size = 1  # TP degree
        self.expert_parallel_rank = 0  # EP rank ID
        self.expert_parallel_size = 1  # EP degree
        self.data_parallel_size = 1  # DP degree
        self.enable_expert_parallel = False
        self.local_data_parallel_id = 0
        # The embedding weight distributed on your gpu cards is divided by row or column.
        # Defaults to False means divide by row. When vocab_size can not be divided by world_size
        # but hidden_size can, we can consider split embedding weight by column.
        """
        From old wersion worker args
        TODO(gongshaotian): Reclassify
        """
        self.max_num_seqs: int = 34
        # Set default block num for profile run
        self.total_block_num: int = 2000
        # block size
        self.block_size: int = 64
        # Engine worker queue port
        self.engine_worker_queue_port: int = 9923
        # Max model len
        self.max_model_len: int = 3072  # max_seq_len
        # cuda visible devices
        self.device_ids: str = "0"
        # Input dtype
        self.dtype: str = "bfloat16"
        # Encoder's decoder num
        self.enc_dec_block_num: int = 1
        # First token id
        self.first_token_id: int = 1
        # Process ID of engine
        self.engine_pid: Optional[int] = None
        # Do profile or not
        self.do_profile: bool = False
        #
        self.pad_token_id: int = -1
        #
        self.eos_tokens_lens: int = 2

        self.max_num_batched_tokens: int = 2048
        # splitwise role
        self.splitwise_role: str = "mixed"
        # guided decoding backend
        self.guided_decoding_backend: str = None
        # disable any whitespace for guided decoding
        self.disable_any_whitespace: bool = True
        self.pod_ip: str = None
        # enable the custom all-reduce kernel and fall back to NCCL(dist.all_reduce).
        self.enable_custom_all_reduce: bool = False
        for key, value in args.items():
            if hasattr(self, key):
                setattr(self, key, value)

        self.use_ep = self.expert_parallel_size > 1
        if self.splitwise_role == "mixed":
            self.moe_phase = MoEPhase(phase="prefill")
        elif self.splitwise_role == "prefill":
            self.moe_phase = MoEPhase(phase="prefill")
        elif self.splitwise_role == "decode":
            self.moe_phase = MoEPhase(phase="decode")
        else:
            raise NotImplementedError

        # pd_disaggregation
        use_pd_disaggregation: int = int(os.getenv("FLAGS_use_pd_disaggregation", 0))
        use_pd_disaggregation_per_chunk: int = int(os.getenv("FLAGS_use_pd_disaggregation_per_chunk", 0))
        if use_pd_disaggregation_per_chunk:
            self.pd_disaggregation_mode = "per_chunk"
        elif use_pd_disaggregation:
            self.pd_disaggregation_mode = "per_query"
        else:
            self.pd_disaggregation_mode = "None"

    def print(self):
        """
        print all config

        """
        logger.info("Parallel Configuration Information :")
        for k, v in self.__dict__.items():
            logger.info("{:<20}:{:<6}{}".format(k, "", v))
        logger.info("=============================================================")


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
        self.model: Optional[str] = None
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

        self.is_unified_ckpt = check_unified_ckpt(self.model)
        if self.model is None:
            return

        self.config_path = os.path.join(self.model, "config.json")
        if os.path.exists(self.config_path):
            self.model_config = json.load(open(self.config_path, "r", encoding="utf-8"))

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
        return json.dumps({key: value for key, value in self.__dict__.items() if value is not None})

    def print(self):
        """
        print all config

        """
        logger.info("Speculative Decoding Configuration Information :")
        for k, v in self.__dict__.items():
            logger.info("{:<20}:{:<6}{}".format(k, "", v))
        logger.info("=============================================================")

    def __str__(self) -> str:
        return self.to_json_string()


class DeviceConfig:
    """
    Configuration for device settings.
    """

    def __init__(
        self,
        args,
    ):
        self.device_type = "cuda"
        for key, value in args.items():
            if hasattr(self, key):
                setattr(self, key, value)


class GraphOptimizationConfig:
    """
    Configuration for compute graph level optimization.
    """

    def __init__(
        self,
        args,
    ):
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
        self.sot_warmup_sizes: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 32, 64, 128]
        """  Number of warmup runs for SOT warmup. """
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
        can manually split the model into multiple layers and apply the @support_graph_optimization decorator
        only to the layer where CUDA graph functionality is required.
        """
        self.cudagraph_splitting_ops: list[str] = []
        """ Whether to use a full cuda graph for the entire forward pass rather than
        splitting certain operations such as attention into subgraphs.
        Thus this flag cannot be used together with splitting_ops."""
        self.full_cuda_graph: bool = True

        self.max_capture_size: int = None
        self.batch_size_to_captured_size: dict[int, int] = None
        # CINN Config ...
        if args is not None:
            for key, value in args.items():
                if hasattr(self, key):
                    setattr(self, key, value)

        self.check_legality_parameters()

    def init_with_cudagrpah_size(self, max_num_seqs: int = 0) -> None:
        """
        Initialize cuda graph capture sizes and
        pre-compute the mapping from batch size to padded graph size
        """
        # Regular capture sizes
        self.cudagraph_capture_sizes = [size for size in self.cudagraph_capture_sizes if size <= max_num_seqs]
        dedup_sizes = list(set(self.cudagraph_capture_sizes))
        if len(dedup_sizes) < len(self.cudagraph_capture_sizes):
            logger.info(
                ("cudagraph sizes specified by model runner" " %s is overridden by config %s"),
                self.cudagraph_capture_sizes,
                dedup_sizes,
            )
        self.cudagraph_capture_sizes = dedup_sizes

        # Sort to make sure cudagraph capture sizes are in descending order
        self.cudagraph_capture_sizes.sort(reverse=True)
        self.max_capture_size = self.cudagraph_capture_sizes[0] if self.cudagraph_capture_sizes else 0

        # Pre-compute the mapping from batch size to padded graph size
        self.batch_size_to_captured_size = {}
        for end, start in zip(self.cudagraph_capture_sizes, self.cudagraph_capture_sizes[1:] + [0]):
            for bs in range(start, end):
                if bs == start:
                    self.batch_size_to_captured_size[bs] = start
                else:
                    self.batch_size_to_captured_size[bs] = end
        self.batch_size_to_captured_size[self.max_capture_size] = self.max_capture_size

    def _set_cudagraph_sizes(self, max_num_seqs: int = 0):
        """
        Calculate a series of candidate capture batch sizes,
        and then extract a portion of them as the capture list for the CUDA graph based on user input.
        """
        # Batch Size [1, 2, 4, 8, 16, ... 120, 128]
        draft_capture_sizes = [1, 2, 4] + [8 * i for i in range(1, 17)]
        # Batch Size [128, 144, ... 240, 256]
        draft_capture_sizes += [16 * i for i in range(9, 17)]
        # Batch Size [256, 288, ... 992, 1024]
        draft_capture_sizes += [32 * i for i in range(17, 33)]

        draft_capture_sizes.append(max_num_seqs)
        self.cudagraph_capture_sizes = sorted(draft_capture_sizes)

    def to_json_string(self):
        """
        Convert speculative_config to json string.
        """
        return json.dumps({key: value for key, value in self.__dict__.items()})

    def __str__(self) -> str:
        return self.to_json_string()

    def check_legality_parameters(
        self,
    ) -> None:
        """Check the legality of parameters passed in from the command line"""

        if self.graph_opt_level is not None:
            assert self.graph_opt_level in [
                0,
                1,
                2,
            ], "In graph optimization config, graph_opt_level can only take the values of 0, 1 and 2."
        if self.use_cudagraph is not None:
            assert (
                type(self.use_cudagraph) is bool
            ), "In graph optimization config, type of use_cudagraph must is bool."
        if self.cudagraph_capture_sizes is not None:
            assert (
                type(self.cudagraph_capture_sizes) is list
            ), "In graph optimization config, type of cudagraph_capture_sizes must is list."
            assert (
                len(self.cudagraph_capture_sizes) > 0
            ), "In graph optimization config, When opening the CUDA graph, it is forbidden to set the capture sizes to an empty list."

    def update_use_cudagraph(self, argument: bool):
        """
        Unified user specifies the use_cudagraph parameter through two methods,
        '--use-cudagraph' and '--graph-optimization-config'
        """
        if self.use_cudagraph is None:
            # User only set '--use-cudagraph'
            self.use_cudagraph = argument
        else:
            # User both set '--use-cudagraph' and '--graph-optimization-config'
            if self.use_cudagraph is False and argument is True:
                raise ValueError(
                    "Invalid parameter: Cannot set --use-cudagraph and --graph-optimization-config '{\"use_cudagraph\":false}' simultaneously."
                )
            argument = self.use_cudagraph


class EarlyStopConfig:
    def __init__(
        self,
        args,
    ):
        """
        Early Stop Configuration class.

        Attributes:
            window_size: size of the window
            threshold: trigger early stop when the ratio of probs exceeds the threshold
        """
        """enable to use early stop"""
        self.enable_early_stop: bool = False
        """strategy for early stop, the strategy lists are ['repetition']"""
        self.strategy: str = "repetition"
        """ the maximum length of verify window for early stop """
        self.window_size: int = 3000
        """ the probs threshold for early stop """
        self.threshold: float = 0.99

        if args is not None:
            for key, value in args.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        self.check_legality_parameters()

    def to_json_string(self):
        """
        Convert early_stop_config to json string.
        """
        return json.dumps({key: value for key, value in self.__dict__.items()})

    def __str__(self) -> str:
        return self.to_json_string()

    def check_legality_parameters(
        self,
    ) -> None:
        """Check the legality of parameters passed in from the command line"""
        if self.enable_early_stop is not None:
            assert isinstance(
                self.enable_early_stop, bool
            ), "In early stop config, type of enable_early_stop must is bool."
        if self.window_size is not None:
            assert isinstance(self.window_size, int), "In early stop config, type of window_size must be int."
            assert self.window_size > 0, "window_size must large than 0"
        if self.threshold is not None:
            assert isinstance(self.threshold, float), "In early stop config, type of threshold must be float."
            assert self.threshold >= 0 and self.threshold <= 1, "threshold must between 0 and 1"

    def update_enable_early_stop(self, argument: bool):
        """
        Unified user specifies the enable_early_stop parameter through two methods,
        '--enable-early-stop' and '--early-stop-config'
        """
        if self.enable_early_stop is None:
            # User only set '--enable-early-stop'
            self.enable_early_stop = argument
        else:
            # User both set '--enable-early-stop' and '--early-stop-config'
            if self.enable_early_stop is False and argument is True:
                raise ValueError(
                    "Invalid parameter: Cannot set ---enable-early-stop and --early-stop-config '{\"enable_early_stop\":false}' simultaneously."
                )
            argument = self.enable_early_stop


class LoadChoices(str, Enum):
    """LoadChoices"""

    DEFAULT = "default"
    # only support qwen3-bf16 now
    NEW_LOADER = "new_loader"


class LoadConfig:
    """
    Configuration for dynamic weight loading strategies

    Attributes:
        dynamic_load_weight: Whether to enable dynamic weight loading
        load_strategy: Specifies the weight loading method when enabled:
            - 'ipc': Real-time IPC streaming with automatic resharding
            - 'ipc_snapshot': Load from disk snapshot of IPC weights
            - None: No dynamic loading
    """

    def __init__(
        self,
        args,
    ):
        self.load_choices: Union[str, LoadChoices] = LoadChoices.DEFAULT.value
        self.use_fastsafetensor = int(envs.FD_USE_FASTSAFETENSOR) == 1
        self.dynamic_load_weight: bool = False
        self.load_strategy: Optional[Literal["ipc", "ipc_snapshot"]] = None
        for key, value in args.items():
            if hasattr(self, key):
                setattr(self, key, value)


class LoRAConfig:
    """LoRA Config"""

    pass


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
        prealloc_dec_block_slot_num_threshold (int): Number of token slot threadshold to allocate next blocks for decoding.
        enable_prefix_caching (bool): Flag to enable prefix caching.
    """

    def __init__(self, args):
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
            prealloc_dec_block_slot_num_threshold (int): Number of token slot threadshold to allocate next blocks for decoding, used when ENABLE_V1_KVCACHE_SCHEDULER=1.
            enable_prefix_caching (bool): Enable prefix caching.
        """
        self.block_size = 64
        self.gpu_memory_utilization = 0.9
        self.num_gpu_blocks_override = None
        self.kv_cache_ratio = 0.75
        self.enc_dec_block_num = 2
        self.prealloc_dec_block_slot_num_threshold = 5
        self.cache_dtype = "bfloat16"
        self.model_cfg = None
        self.enable_chunked_prefill = False
        self.rdma_comm_ports = None
        self.cache_transfer_protocol = None
        self.pd_comm_port = None
        self.enable_prefix_caching = False
        self.enable_ssd_cache = False
        self.cache_queue_port = None
        self.swap_space = None
        for key, value in args.items():
            if hasattr(self, key):
                setattr(self, key, value)

        if self.rdma_comm_ports is not None and isinstance(self.rdma_comm_ports, str):
            self.rdma_comm_ports = self.rdma_comm_ports.split(",")

        if self.pd_comm_port is not None and isinstance(self.pd_comm_port, str):
            self.pd_comm_port = [int(port) for port in self.pd_comm_port.split(",")]

        if self.swap_space is None:
            self.enable_hierarchical_cache = False
        else:
            self.enable_hierarchical_cache = True

        if self.model_cfg is not None:
            if self.model_cfg.quantization_config is not None:
                self.cache_dtype = self.model_cfg.quantization_config.get("kv_cache_quant_type", self.cache_dtype)
            if (
                hasattr(self.model_cfg, "num_key_value_heads")
                and hasattr(self.model_cfg, "num_key_value_heads")
                and self.model_cfg.num_key_value_heads is not None
                and int(self.model_cfg.num_key_value_heads) > 0
            ):
                kv_num_head = int(self.model_cfg.num_key_value_heads)
            else:
                kv_num_head = self.model_cfg.num_attention_heads
            self.model_cfg.kv_num_head = kv_num_head
            # TODO check name
            if "int4" in self.cache_dtype.lower() or "float4" in self.cache_dtype.lower():
                byte_size = 0.5
                self.cache_dtype = "uint8"
            elif "int8" in self.cache_dtype.lower() or "float8" in self.cache_dtype.lower():
                self.cache_dtype = "uint8"
                byte_size = 1
            else:
                byte_size = 2
            self.each_token_cache_space = int(
                self.model_cfg.num_hidden_layers * kv_num_head * self.model_cfg.head_dim * byte_size
            )
            self.bytes_per_block = int(self.each_token_cache_space * self.block_size)
            self.bytes_per_layer_per_block = int(
                self.block_size
                * self.model_cfg.kv_num_head
                * self.model_cfg.head_dim
                // args["tensor_parallel_size"]
                * byte_size
            )

        if self.swap_space is None:
            self.num_cpu_blocks = 0
        else:
            self.num_cpu_blocks = int(self.swap_space * 1024**3 / self.bytes_per_block)
        self._verify_args()

    def metrics_info(self):
        """Convert cache_config to dict(key: str, value: str) for prometheus metrics info."""
        return {key: str(value) for key, value in self.__dict__.items()}

    def _verify_args(self):
        if self.gpu_memory_utilization > 1.0:
            raise ValueError("GPU memory utilization must be less than 1.0. Got " f"{self.gpu_memory_utilization}.")
        if self.kv_cache_ratio > 1.0:
            raise ValueError("KV cache ratio must be less than 1.0. Got " f"{self.kv_cache_ratio}.")

    def postprocess(self, num_total_tokens, number_of_tasks):
        """
        calculate block num
        """
        self.dec_token_num = self.enc_dec_block_num * self.block_size
        if self.num_gpu_blocks_override is not None:
            self.total_block_num = self.num_gpu_blocks_override
            self.prefill_kvcache_block_num = int(self.total_block_num * self.kv_cache_ratio)
        else:
            length = num_total_tokens // number_of_tasks
            block_num = (length + self.block_size - 1 + self.dec_token_num) // self.block_size
            self.total_block_num = block_num * number_of_tasks
            self.prefill_kvcache_block_num = self.total_block_num
            logger.info(f"Doing profile, the total_block_num:{self.total_block_num}")

    def reset(self, num_gpu_blocks):
        """
        reset gpu block number
        """
        self.total_block_num = num_gpu_blocks
        self.prefill_kvcache_block_num = int(self.total_block_num * self.kv_cache_ratio)
        logger.info(
            f"Reset block num, the total_block_num:{self.total_block_num},"
            f" prefill_kvcache_block_num:{self.prefill_kvcache_block_num}"
        )

    def print(self):
        """
        print all config

        """
        logger.info("Cache Configuration Information :")
        for k, v in self.__dict__.items():
            logger.info("{:<20}:{:<6}{}".format(k, "", v))
        logger.info("=============================================================")


class DecodingConfig:
    """
    Configuration for decoding
    """

    def __init__(
        self,
        args,
    ):
        self.pad_token_id = None
        for key, value in args.items():
            if hasattr(self, key):
                setattr(self, key, value)


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
        self.paddle_commit: str = ""
        self.cuda_version: str = ""
        self.compiler_version: str = ""

        self._load_from_version_file()

    def _load_from_version_file(self, file_path: str = None):
        """Internal method to load version info from file"""
        if file_path is None:
            file_path = os.path.join(fastdeploy.__path__[0], "version.txt")
        try:
            with open(file_path, "r") as f:
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
            logger.info(f"Warning: Version file not found at {file_path}")
        except Exception as e:
            logger.info(f"Warning: Could not read version file - {e!s}")

    def print(self):
        """
        print all config

        """
        logger.info("Fasedeploy Commit Information :")
        for k, v in self.__dict__.items():
            logger.info("{:<20}:{:<6}{}".format(k, "", v))
        logger.info("=============================================================")


@dataclass
class FDConfig:
    """
    The configuration class which contains all fastdeploy-related configuration. This
    simplifies passing around the distinct configurations in the codebase.
    """

    model_config: ModelConfig = field(default=None, init=True)  # type: ignore

    parallel_config: ParallelConfig = field(default=None, init=True)
    speculative_config: SpeculativeConfig = field(default=None, init=True)  # type: ignore
    device_config: DeviceConfig = field(default=None, init=True)  # type: ignore
    load_config: LoadConfig = field(default=None, init=True)
    quant_config: Optional[QuantConfigBase] = None
    graph_opt_config: Optional[GraphOptimizationConfig] = None
    early_stop_config: Optional[EarlyStopConfig] = None
    decoding_config: DecodingConfig = field(default=None, init=True)  # type: ignore
    cache_config: CacheConfig = field(default=None, init=True)  # type: ignore

    def __post_init__(self):
        # Initialize cuda graph capture list
        if self.graph_opt_config.cudagraph_capture_sizes is None:
            self.graph_opt_config._set_cudagraph_sizes(max_num_seqs=self.parallel_config.max_num_seqs)
        self.graph_opt_config.init_with_cudagrpah_size(max_num_seqs=self.parallel_config.max_num_seqs)

        # TODO(wangmingkai02): change graph_opt_level=2 when using static mode with cinn
        if self.graph_opt_config.graph_opt_level == 2:
            self.graph_opt_config.graph_opt_level = 1
