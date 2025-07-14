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

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Optional, Union

from paddleformers.transformers.configuration_utils import PretrainedConfig
from paddleformers.trl import llm_utils

from fastdeploy import envs
from fastdeploy.model_executor.layers.quantization.quant_base import \
    QuantConfigBase
from fastdeploy.utils import get_logger

logger = get_logger("config", "config.log")

class MoEPhase(Enum):
    """
    The generation phase of the moe.
    """

    PREFILL = 1
    DECODER = 2

PRETRAINED_INIT_CONFIGURATION = {
    "rope_theta": 10000.0,
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
        self.top_p = 0.0
        self.temperature = 1.0
        self.rope_theta = 10000.0
        self.penalty_score = 1.0
        self.frequency_score = 0.0
        self.presence_score = 0.0
        self.min_length = 1
        self.model_name_or_path = ""

        self.im_patch_id = (
            100295  # multimodality, TODO(liuyuanle): read from config.json
        )
        self.is_quantized = False
        self.max_model_len = 0
        self.dtype = ""
        self.enable_logprob = False

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
        if "Ernie4_5_ForCausalLM" in self.architectures or "Ernie4_5_MoeForCausalLM" in self.architectures:
            self.ori_vocab_size = args["ori_vocab_size"]

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
        self.max_block_num: int = 2000
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

        #TODO(YuanRisheng): The name of the server args is different from the name of the SpeculativeConfig.
        #We temperately add the name map here and will delete it in future.
        name_map = {"speculative_method": "method",
                   "speculative_max_draft_token_num": "num_speculative_tokens",
                   "speculative_model_name_or_path": "model_name_or_path",
                   "speculative_model_quantization": "quantization",
                   "speculative_benchmark_mode": "benchmark_mode"}

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


class KVCacheConfig:
    """ KV Cache Config """
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
    speculative_config: SpeculativeConfig = field(default=None,
                                                  init=True)  # type: ignore
    device_config: DeviceConfig = field(default=None,
                                        init=True)  # type: ignore
    load_config: LoadConfig = field(default=None, init=True)
    quant_config: Optional[QuantConfigBase] = None
    graph_opt_config: Optional[GraphOptimizationConfig] = None
    decoding_config: DecodingConfig = field(default=None,
                                            init=True)  # type: ignore
    kv_cache_config: KVCacheConfig = field(default=None,
                                           init=True)  # type: ignore
