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
from typing import Optional

from paddleformers.transformers.configuration_utils import PretrainedConfig

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


class ModelConfig(PretrainedConfig):
    """
    The configuration class to store the configuration of a `LLM`.
    """
    max_stop_seqs_num = 5
    stop_seqs_max_len = 8

    architectures: list[str] = []

    # NOTE(gongshaotain): form _load_model_init_val()
    top_p = 0.0
    temperature = 1.0
    rope_theta = 10000.0
    rope_scaling = None
    penalty_score = 1.0
    frequency_score = 0.0
    presence_score = 0.0
    min_length = 1

    def __init__(
        self,
        vocab_size: int = 100224,
        hidden_size: int = 4096,
        num_layers: int = 48,
        num_attention_heads: int = 32,
        num_key_value_heads: Optional[int] = None,
        hidden_act: str = "swiglu",
        hidden_dropout_prob: float = 0.0,
        max_position_embeddings: int = 512,
        max_seq_len: int = 512,
        initializer_range: float = 0.02,
        use_rope=True,
        use_fast_ffn: bool = False,
        rope_theta: int = 10000,
        rope_3d: bool = False,
        ori_vocab_size: int | None = None,
        moe_layer_start_index: int | None = None,
        moe_layer_end_index: int | None = None,
        num_hidden_layers: int | None = None,
        prefix_name="",
        freeze_embedding=False,
        rope_head_dim=None,
        ffn_hidden_size: Optional[int] = None,
        dtype="bfloat16",
        start_layer_index: int = 0,
        head_dim: Optional[int] = None,
        tie_word_embeddings: bool = False,
        is_quantized: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if num_hidden_layers is not None:
            self.num_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        if head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        else:
            self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.use_rope = use_rope
        self.use_fast_ffn = use_fast_ffn
        self.rope_theta = rope_theta
        self.ori_vocab_size = ori_vocab_size or vocab_size
        self.max_seq_len = max_seq_len
        self.prefix_name = prefix_name
        self.freeze_embedding = freeze_embedding
        self.rope_head_dim = rope_head_dim
        moe_num_experts = kwargs.get("moe_num_experts", 0)
        if moe_layer_start_index is not None:
            self.moe_layer_start_index = moe_layer_start_index
        elif moe_num_experts == 0:
            self.moe_layer_start_index = self.num_layers
            self.moe_num_experts = 0
        if moe_layer_end_index is not None:
            self.moe_layer_end_index = moe_layer_end_index
        self.ffn_hidden_size = ffn_hidden_size
        self.rope_3d = rope_3d
        self.start_layer_index = start_layer_index
        self.dtype = dtype
        self.tie_word_embeddings = tie_word_embeddings
        self.is_quantized = is_quantized


@dataclass
class MoEConfig:
    """
    Configuration for MoE.
    """
    num_experts: int = -1
    top_k: int = 8
    moe_intermediate_size: int = -1
    num_experts_per_rank: int = -1
    num_experts_start_offset: int = -1

    moe_num_shared_experts = (0, )
    moe_layer_start_index = 0
    moe_layer_end_index = None
    num_max_dispatch_tokens_per_rank = 256
    im_patch_id = (
        100295  # multimodality, TODO(liuyuanle): read from config.json
    )


@dataclass
class ParallelConfig:
    """Configuration for the distributed execution."""
    block_size = 16  # The block size for processing.
    sequence_parallel = False  # Whether to enable sequence parallelism.
    use_ep = False  # Whether to enable Expert Parallelism
    moe_phase = MoEPhase.PREFILL  # Generation phase
    msg_queue_id = 1  # mesage queue id
    tensor_parallel_rank = None  # TP rank ID
    tensor_parallel_degree = None  # TP degree
    expert_parallel_rank = None  # EP rank ID
    expert_parallel_degree = None  # EP degree
    # The embedding weight distributed on your gpu cards is divided by row or column.
    # Defaults to False means divide by row. When vocab_size can not be divided by world_size
    # but hidden_size can, we can consider split embedding weight by column.
    column_cut = False  # (bool, optional)
    """
    From old wersion worker args
    TODO(gongshaotian): Reclassify
    """
    model_name_or_path: str = "./output"
    max_num_seqs: int = 34
    # Set default block num for profile run
    max_block_num: int = 2000
    # block size
    block_size: int = 64
    # Engine worker queue port
    engine_worker_queue_port: int = 9923
    # Max model len
    max_model_len: int = 3072  # max_seq_len
    # cuda visible devices
    device_ids: str = "0"
    # Input dtype
    dtype: str = "bfloat16"
    # Encoder's decoder num
    enc_dec_block_num: int = 1
    # KV cache ratio for input
    kv_cache_ratio: float = 0.7
    # First token id
    first_token_id: int = 1
    # Gpu memory utilization
    gpu_memory_utilization: float = 0.9
    # Process ID of engine
    engine_pid: Optional[int] = None
    # Do profile or not
    do_profile: bool = False
    # Dynamic load weight or not
    dynamic_load_weight: bool = False
    #
    pad_token_id: int = -1
    #
    eos_tokens_lens: int = 2
    # Enable chunked prefill
    enable_chunked_prefill: str = "store_true"
    """
    - APPEND_ATTN:
    """
    attention_backend: str = "APPEND_ATTN"
    max_num_batched_tokens: int = 2048
    # enable prefix cache
    enable_prefix_caching = None
    # splitwise role
    splitwise_role: str = "mixed"
    # guided decoding backend
    guided_decoding_backend: str = None
    # disable any whitespace for guided decoding
    disable_any_whitespace: bool = True


@dataclass
class SpeculativeConfig:
    """
    Configuration for speculative decoding.
    """
    # speculative method, choose in [None, "ngram_match", "mtp"]
    method: Optional[str] = None
    # the max length of speculative tokens
    num_speculative_tokens: int = 1
    # the max length of candidate tokens for speculative method
    max_candidate_len: int = 5
    # the max length of verify window for speculative method
    verify_window: int = 2
    # ngram match
    max_ngram_size: int = 5
    # model for mtp/eagle/draft_model
    model_name_or_path: Optional[str] = None
    # quantization of model
    quantization: Optional[str] = None
    # allocate more blocks to prevent mtp from finishing the block earlier than the main model
    # Fixed now
    num_gpu_block_expand_ratio: Optional[float] = 1
    # To distinguish the main model and draft model(mtp/eagle/draftmodel)
    # ["main", "mtp"]
    model_type: Optional[str] = "main"
    # TODO(liuzichang): To reduce memory usage, MTP shares the main model's lm_head and embedding layers.
    # A trick method is currently used to enable this sharing.
    # This will be replaced with a more standardized solution in the future.
    sharing_model = None


@dataclass
class DeviceConfig:
    """
    Configuration for device settings.
    """
    device_type = "cuda"


class GraphOptimizationConfig:
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
    """"whether to use a full cuda graph for the entire forward pass rather than
    splitting certain operations such as attention into subgraphs.
    Thus this flag cannot be used together with splitting_ops."""
    full_cuda_graph: bool = False

    max_capture_size: int = field(default=None, init=False)  # type: ignore
    batch_size_to_captured_size: dict[int,
                                      int] = field(default=None,
                                                   init=False)  # type: ignore

    # CINN Config ...

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
                 use_cudagraph: bool = False,
                 max_capture_batch_size: int = 64):
        """ """
        capture_size = [i for i in range(1, max_capture_batch_size + 1)]
        self.init_with_cudagrpah_size(cudagraph_capture_sizes=capture_size)
        self.use_cudagraph = use_cudagraph
        #TODO(wangmingkai02): change graph_opt_level=2 when using static mode with cinn
        if enable_static_graph_inference:
            self.graph_opt_level = 1


@dataclass
class LoadConfig:
    """
    Configuration for loading parameter
    """
    pass


@dataclass
class LoRAConfig:
    """ LoRA Config """
    pass


@dataclass
class KVCacheConfig:
    """ KV Cache Config """
    cache_quant_dtype: str = "none"


@dataclass
class DecodingConfig:
    """
    Configuration for decoding
    """
    pad_token_id = None


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
    load_config: LoadConfig = field(default=None, init=True)  # type: ignore
    quant_config: Optional[QuantConfigBase] = None
    graph_opt_config: Optional[GraphOptimizationConfig] = None
    moe_config: MoEConfig = field(default=None, init=True)  # type: ignore
    decoding_config: DecodingConfig = field(default=None,
                                            init=True)  # type: ignore
    kv_cache_config: KVCacheConfig = field(default=None,
                                           init=True)  # type: ignore
