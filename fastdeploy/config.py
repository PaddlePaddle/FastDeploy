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

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Optional

from paddleformers.transformers.configuration_utils import PretrainedConfig

from fastdeploy import envs
from fastdeploy.model_executor.layers.quantization.quant_base import QuantConfigBase
from fastdeploy.utils import get_logger

logger = get_logger("config", "config.log")


class MoEPhase(Enum):
    """
    The generation phase of the moe.
    """

    PREFILL = 1
    DECODER = 2


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
    "rope_theta": 10000.0,
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
        self.max_stop_seqs_num = 5
        self.stop_seqs_max_len = 8

        # NOTE(gongshaotain): form _load_model_init_val()
        self.top_p = 1.0
        self.temperature = 1.0
        self.rope_theta = 10000.0
        self.penalty_score = 1.0
        self.frequency_score = 0.0
        self.presence_score = 0.0
        self.min_length = 1
        self.model_name_or_path = ""

        self.is_quantized = False
        self.max_model_len = 0
        self.dtype = ""
        self.enable_logprob = False
        self.enable_mm = False

        for key, value in args.items():
            if hasattr(self, key):
                setattr(self, key, value)

        assert self.model_name_or_path != ""
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
        if ErnieArchitectures.contains_ernie_arch(self.architectures):
            self.ori_vocab_size = args.get("ori_vocab_size", self.ori_vocab_size)


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

        self.tensor_parallel_rank = 0  # TP rank ID
        self.tensor_parallel_size = 1  # TP degree
        self.expert_parallel_rank = 0  # EP rank ID
        self.expert_parallel_size = 1  # EP degree
        # The embedding weight distributed on your gpu cards is divided by row or column.
        # Defaults to False means divide by row. When vocab_size can not be divided by world_size
        # but hidden_size can, we can consider split embedding weight by column.
        """
        From old wersion worker args
        TODO(gongshaotian): Reclassify
        """
        self.model_name_or_path: str = "./output"
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
        # Enable chunked prefill
        self.enable_chunked_prefill: bool = False

        self.max_num_batched_tokens: int = 2048
        # enable prefix cache
        self.enable_prefix_caching = None
        # splitwise role
        self.splitwise_role: str = "mixed"
        # guided decoding backend
        self.guided_decoding_backend: str = None
        # disable any whitespace for guided decoding
        self.disable_any_whitespace: bool = True
        self.pod_ip: str = None
        for key, value in args.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.use_ep = args["expert_parallel_size"] > 1
        if self.splitwise_role == "mixed":
            self.moe_phase = MoEPhase.PREFILL
        elif self.splitwise_role == "prefill":
            self.moe_phase = MoEPhase.PREFILL
        elif self.splitwise_role == "decode":
            self.moe_phase = MoEPhase.DECODER
        else:
            raise NotImplementedError
        # enable the custom all-reduce kernel and fall back to NCCL(dist.all_reduce).
        self.enable_custom_all_reduce: bool = False

        # pd_disaggregation
        use_pd_disaggregation: int = int(os.getenv("FLAGS_use_pd_disaggregation", 0))
        use_pd_disaggregation_per_chunk: int = int(os.getenv("FLAGS_use_pd_disaggregation_per_chunk", 0))
        if use_pd_disaggregation_per_chunk:
            self.pd_disaggregation_mode = "per_chunk"
        elif use_pd_disaggregation:
            self.pd_disaggregation_mode = "per_query"
        else:
            self.pd_disaggregation_mode = "None"


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

        # TODO(YuanRisheng): The name of the server args is different from the name of the SpeculativeConfig.
        # We temperately add the name map here and will delete it in future.
        name_map = {
            "speculative_method": "method",
            "speculative_max_draft_token_num": "num_speculative_tokens",
            "speculative_model_name_or_path": "model_name_or_path",
            "speculative_model_quantization": "quantization",
            "speculative_benchmark_mode": "benchmark_mode",
        }

        for key, value in args.items():
            if key in name_map.keys() and hasattr(self, name_map[key]):
                setattr(self, name_map[key], value)


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


@dataclass
class GraphOptimizationConfig:
    """
    Configuration for compute graph level optimization.
    """

    """The Top-level graph optimization contral corresponds to different backends.
    - 0: dyncmic graph
    - 1: static graph
    - 2: static graph + cinn compilation backend
    """
    graph_opt_level: int = 0

    # CUDA Graph Config
    """ Whether to use cudagraph.
    - False: cudagraph is not used.
    - True: cudagraph is used.
        It requires that all input buffers have fixed addresses, and all
        splitting ops write their outputs to input buffers.
        - With dyncmic graph backend: ...
        - With static grpah backend: WIP
    """
    use_cudagraph: bool = False
    """Sizes to capture cudagraph.
    - None (default): capture sizes are inferred from llm config.
    - list[int]: capture sizes are specified as given."""
    cudagraph_capture_sizes: Optional[list[int]] = None
    """ Number of warmup runs for cudagraph. """
    cudagraph_num_of_warmups: int = 2
    """Whether to copy input tensors for cudagraph.
    If the caller can guarantee that the same input buffers
    are always used, it can set this to False. Otherwise, it should
    set this to True."""
    cudagraph_copy_inputs: bool = False
    """ In static graph, this is an operation list that does not need to be captured by the CUDA graph.
    CudaGraphBackend will split these operations from the static graph.
    Example usage:
        cudagraph_splitting_ops = ["paddle.unified_attention"]

    Note: If want to use subgraph capture functionality in a dynamic graph,
    can manually split the model into multiple layers and apply the @support_cuda_graph decorator
    only to the layer where CUDA graph functionality is required.
    """
    cudagraph_splitting_ops = Optional[list[str]]
    """" Whether to use a full cuda graph for the entire forward pass rather than
    splitting certain operations such as attention into subgraphs.
    Thus this flag cannot be used together with splitting_ops."""
    full_cuda_graph: bool = True

    max_capture_size: int = field(default=None, init=False)  # type: ignore
    batch_size_to_captured_size: dict[int, int] = field(default=None, init=False)  # type: ignore
    # CINN Config ...

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
        self.use_fastsafetensor = int(envs.FD_USE_FASTSAFETENSOR) == 1
        self.dynamic_load_weight: bool = False
        self.load_strategy: Optional[Literal["ipc", "ipc_snapshot"]] = None
        for key, value in args.items():
            if hasattr(self, key):
                setattr(self, key, value)


class LoRAConfig:
    """LoRA Config"""

    pass


class KVCacheConfig:
    """KV Cache Config"""

    cache_quant_dtype: str = "none"


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
    decoding_config: DecodingConfig = field(default=None, init=True)  # type: ignore
    kv_cache_config: KVCacheConfig = field(default=None, init=True)  # type: ignore

    def __post_init__(self):
        # Initialize cuda graph capture list
        if self.graph_opt_config.cudagraph_capture_sizes is None:
            self.graph_opt_config._set_cudagraph_sizes(max_num_seqs=self.parallel_config.max_num_seqs)
        self.graph_opt_config.init_with_cudagrpah_size(max_num_seqs=self.parallel_config.max_num_seqs)

        # TODO(wangmingkai02): change graph_opt_level=2 when using static mode with cinn
        if self.graph_opt_config.graph_opt_level == 2:
            self.graph_opt_config.graph_opt_level = 1
