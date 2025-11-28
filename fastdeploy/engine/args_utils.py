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
import json
import os
from dataclasses import asdict, dataclass
from dataclasses import fields as dataclass_fields
from typing import Any, Dict, List, Optional, Union

from fastdeploy import envs
from fastdeploy.config import (
    CacheConfig,
    ConvertOption,
    EarlyStopConfig,
    EPLBConfig,
    FDConfig,
    GraphOptimizationConfig,
    LoadConfig,
    ModelConfig,
    ParallelConfig,
    PlasAttentionConfig,
    PoolerConfig,
    RouterConfig,
    RunnerOption,
    SpeculativeConfig,
    StructuredOutputsConfig,
    TaskOption,
)
from fastdeploy.platforms import current_platform
from fastdeploy.scheduler.config import SchedulerConfig
from fastdeploy.utils import (
    DeprecatedOptionWarning,
    FlexibleArgumentParser,
    console_logger,
    is_port_available,
    parse_quantization,
)


def nullable_str(x: str) -> Optional[str]:
    """
    Convert an empty string to None, preserving other string values.
    """
    return x if x else None


def get_model_architecture(model: str, model_config_name: Optional[str] = "config.json") -> Optional[str]:
    config_path = os.path.join(model, model_config_name)
    if os.path.exists(config_path):
        model_config = json.load(open(config_path, "r", encoding="utf-8"))
        architecture = model_config["architectures"][0]
        return architecture
    else:
        return model


@dataclass
class EngineArgs:
    # Model configuration parameters
    model: str = "baidu/ernie-45-turbo"
    """
    The name or path of the model to be used.
    """
    port: Optional[str] = None
    """
    Port for api server.
    """
    served_model_name: Optional[str] = None
    """
    The name of the model being served.
    """
    revision: Optional[str] = "master"
    """
    The revision for downloading models.
    """
    model_config_name: Optional[str] = "config.json"
    """
    The name of the model configuration file.
    """
    tokenizer: str = None
    """
    The name or path of the tokenizer (defaults to model path if not provided).
    """
    tokenizer_base_url: str = None
    """
    The base URL of the remote tokenizer service (used instead of local tokenizer if provided).
    """
    max_model_len: int = 2048
    """
    Maximum context length supported by the model.
    """
    tensor_parallel_size: int = 1
    """
    Degree of tensor parallelism.
    """
    block_size: int = 64
    """
    Number of tokens in one processing block.
    """
    task: TaskOption = "generate"
    """
    The task to be executed by the model.
    """
    runner: RunnerOption = "auto"
    """
    The type of model runner to use.Each FD instance only supports one model runner.
    even if the same model can be used for multiple types.
    """
    convert: ConvertOption = "auto"
    """
    Convert the model using adapters. The most common use case is to
    adapt a text generation model to be used for pooling tasks.
    """
    override_pooler_config: Optional[Union[dict, PoolerConfig]] = None
    """
    Override configuration for the pooler.
    """
    max_num_seqs: int = 8
    """
    Maximum number of sequences per iteration.
    """
    mm_processor_kwargs: Optional[Dict[str, Any]] = None
    """
    Additional keyword arguments for the multi-modal processor.
    """
    limit_mm_per_prompt: Optional[Dict[str, Any]] = None
    """
    Limitation of numbers of multi-modal data.
    """
    max_encoder_cache: int = -1
    """
    Maximum number of tokens in the encoder cache.
    """
    max_processor_cache: float = -1
    """
    Maximum number of bytes(in GiB) in the processor cache.
    """
    reasoning_parser: str = None
    """
    specifies the reasoning parser to use for extracting reasoning content from the model output
    """
    chat_template: str = None
    """
    chat template or chat template file path
    """
    tool_call_parser: str = None
    """
    specifies the tool call parser  to use for extracting tool call from the model output
    """
    tool_parser_plugin: str = None
    """
    tool parser plugin used to register user defined tool parsers
    """
    enable_mm: bool = False
    """
    Flags to enable multi-modal model
    """
    speculative_config: Optional[Dict[str, Any]] = None
    """
    Configuration for speculative execution.
    """
    dynamic_load_weight: bool = False
    """
    dynamic load weight
    """
    load_strategy: str = "normal"
    """
    dynamic load weight strategy
    """
    quantization: Optional[Dict[str, Any]] = None
    guided_decoding_backend: str = "off"
    """
    Guided decoding backend.
    """
    guided_decoding_disable_any_whitespace: bool = False
    """
    Disable any whitespace in guided decoding.
    """

    # Inference configuration parameters
    gpu_memory_utilization: float = 0.9
    """
    The fraction of GPU memory to be utilized.
    """
    num_gpu_blocks_override: Optional[int] = None
    """
    Override for the number of GPU blocks.
    """
    max_num_batched_tokens: Optional[int] = None
    """
    Maximum number of tokens to batch together.
    """
    kv_cache_ratio: float = 0.75
    """
    Ratio of tokens to process in a block.
    """
    prealloc_dec_block_slot_num_threshold: int = 12
    """
    Token slot threshold for preallocating decoder blocks.
    """
    ips: Optional[List[str]] = None
    """
    The ips of multinode deployment

    """

    swap_space: float = None
    """
    The amount of CPU memory to offload to.
    """

    cache_queue_port: str = "0"
    """
    Port for cache queue.
    """

    # System configuration parameters
    use_warmup: int = 0
    """
    Flag to indicate whether to use warm-up before inference.
    """
    enable_prefix_caching: bool = True
    """
    Flag to enable prefix caching.
    """

    disable_custom_all_reduce: bool = False
    """
    Flag to disable the custom all-reduce kernel.
    """

    use_internode_ll_two_stage: bool = False
    """
    Flag to use the internode_ll_two_stage kernel.
    """

    disable_sequence_parallel_moe: bool = False
    """
    # The all_reduce at the end of attention (during o_proj) means that
    # inputs are replicated across each rank of the tensor parallel group.
    # If using expert-parallelism with DeepEP All2All ops, replicated
    # tokens results in useless duplicate computation and communication.
    #
    # In this case, ensure the input to the experts is sequence parallel
    # to avoid the excess work.
    #
    # This optimization is enabled by default, and can be disabled by using this flag.
    """

    engine_worker_queue_port: str = "0"
    """
    Port for worker queue communication.
    """

    splitwise_role: str = "mixed"
    """
    Splitwise role: prefill, decode or mixed
    """

    data_parallel_size: int = 1
    """
    Number of data parallelism.
    """

    local_data_parallel_id: int = 0
    """
    Local data parallel id.
    """

    enable_expert_parallel: bool = False
    """
    Enable expert parallelism.
    """

    cache_transfer_protocol: str = "ipc"
    """
    Protocol to use for cache transfer.
    """

    pd_comm_port: Optional[List[int]] = None
    """
    Port for splitwise communication.
    """

    rdma_comm_ports: Optional[List[int]] = None
    """
    Ports for rdma communication.
    """

    enable_chunked_prefill: bool = False
    """
    Flag to enable chunked prefilling.
    """
    max_num_partial_prefills: int = 1
    """
    For chunked prefill, the max number of concurrent partial prefills.
    """
    max_long_partial_prefills: int = 1
    """
    For chunked prefill, the maximum number of prompts longer than –long-prefill-token-threshold
    that will be prefilled concurrently.
    """
    long_prefill_token_threshold: int = 0
    """
    For chunked prefill, a request is considered long if the prompt is longer than this number of tokens.
    """
    static_decode_blocks: int = 2
    """
    additional decode block num
    """
    disable_chunked_mm_input: bool = False
    """
    Disable chunked_mm_input for multi-model inference.
    """

    scheduler_name: str = "local"
    """
    Scheduler name to be used
    """
    scheduler_max_size: int = -1
    """
    Size of scheduler
    """
    scheduler_ttl: int = 900
    """
    TTL of request
    """
    scheduler_host: str = "127.0.0.1"
    """
    Host of redis
    """
    scheduler_port: int = 6379
    """
    Port of redis
    """
    scheduler_db: int = 0
    """
    DB of redis
    """
    scheduler_password: Optional[str] = None
    """
    Password of redis
    """
    scheduler_topic: str = "default"
    """
    Topic of scheduler
    """
    scheduler_min_load_score: float = 3
    """
    Minimum load score for task assignment
    """
    scheduler_load_shards_num: int = 1
    """
    Number of shards for load balancing table
    """
    scheduler_sync_period: int = 5
    """
    SplitWise Use, node load sync period
    """
    scheduler_expire_period: int = 3000
    """
    SplitWise Use, node will not be scheduled after expire_period ms not sync load
    """
    scheduler_release_load_expire_period: int = 600
    """
    SplitWise Use, scheduler will release req load after expire period(s)
    """
    scheduler_reader_parallel: int = 4
    """
    SplitWise Use, Results Reader Sync Parallel
    """
    scheduler_writer_parallel: int = 4
    """
    SplitWise Use, Results Writer Sync Parallel
    """
    scheduler_reader_batch_size: int = 200
    """
    SplitWise Use, Results Reader Batch Size
    """
    scheduler_writer_batch_size: int = 200
    """
    SplitWise Use, Results Writer Batch Size
    """
    graph_optimization_config: Optional[Dict[str, Any]] = None
    """
    Configuration for graph optimization backend execution.
    """
    plas_attention_config: Optional[Dict[str, Any]] = None
    """
    Configuration for plas attention.
    """

    enable_logprob: bool = False
    """
    Flag to enable logprob output. Default is False (disabled).
    Must be explicitly enabled via the `--enable-logprob` startup parameter to output logprob values.
    """

    max_logprobs: int = 20
    """
    Maximum number of log probabilities to return when `enable_logprob` is True. The default value comes the default for the
    OpenAI Chat Completions API. -1 means no cap, i.e. all (output_length * vocab_size) logprobs are allowed to be returned and it may cause OOM.
    """

    logprobs_mode: str = "raw_logprobs"
    """
    Indicates the content returned in the logprobs.
    Supported mode:
    1) raw_logprobs, 2) processed_logprobs, 3) raw_logits, 4) processed_logits.
    Raw means the values before applying logit processors, like bad words.
    Processed means the values after applying such processors.
    """

    seed: int = 0
    """
    Random seed to use for initialization. If not set, defaults to 0.
    """

    enable_early_stop: bool = False
    """
    Flag to enable early stop. Default is False (disabled).
    """

    early_stop_config: Optional[Dict[str, Any]] = None
    """
    Configuration for early stop.
    """

    load_choices: str = "default_v1"
    """The format of the model weights to load.
        Options include:
        - "default": default loader.
        - "default_v1": default_v1 loader.
    """

    lm_head_fp32: bool = False
    """
    Flag to specify the dtype of lm_head as FP32. Default is False (Using model default dtype).
    """

    logits_processors: Optional[List[str]] = None
    """
    A list of FQCNs (Fully Qualified Class Names) of logits processors supported by the service.
    A fully qualified class name (FQCN) is a string that uniquely identifies a class within a Python module.

    - To enable builtin logits processors, add builtin module paths and class names to the list. Currently support:
        - fastdeploy.model_executor.logits_processor:LogitBiasLogitsProcessor
    - To enable custom logits processors, add your dotted paths to module and class names to the list.
    """

    router: Optional[str] = None
    """
    Url for router server, such as `0.0.0.0:30000`.
    """

    enable_eplb: bool = False
    """
    Flag to enable eplb
    """

    eplb_config: Optional[Dict[str, Any]] = None
    """
    Configuration for eplb.
    """

    def __post_init__(self):
        """
        Post-initialization processing to set default tokenizer if not provided.
        """

        if not self.tokenizer:
            self.tokenizer = self.model
        if self.splitwise_role == "decode":
            self.enable_prefix_caching = False
        if self.speculative_config is not None:
            self.enable_prefix_caching = False
        if not current_platform.is_cuda() and not current_platform.is_xpu() and not current_platform.is_intel_hpu():
            self.enable_prefix_caching = False
        # if self.dynamic_load_weight:
        #     self.enable_prefix_caching = False
        if self.enable_logprob:
            if not current_platform.is_cuda():
                raise NotImplementedError("Only CUDA platform supports logprob.")
            if self.speculative_config is not None and self.logprobs_mode.startswith("processed"):
                raise NotImplementedError("processed_logprobs not support in speculative.")
            if self.speculative_config is not None and self.max_logprobs == -1:
                raise NotImplementedError("max_logprobs=-1 not support in speculative.")
            if not envs.FD_USE_GET_SAVE_OUTPUT_V1 and (self.max_logprobs == -1 or self.max_logprobs > 20):
                self.max_logprobs = 20
                console_logger.warning("Set max_logprobs=20 when FD_USE_GET_SAVE_OUTPUT_V1=0")
            if self.max_logprobs == -1 and not envs.ENABLE_V1_KVCACHE_SCHEDULER:
                raise NotImplementedError("Only ENABLE_V1_KVCACHE_SCHEDULER=1 support max_logprobs=-1")

        if self.splitwise_role != "mixed":
            if self.scheduler_name == "local" and self.router is None:
                raise ValueError(
                    f"When using {self.splitwise_role} role and the {self.scheduler_name} "
                    f"scheduler, please provide --router argument."
                )

            if "rdma" in self.cache_transfer_protocol:
                if self.rdma_comm_ports is None:
                    raise ValueError(
                        "Please set --rdma_comm_ports argument when using " "rdma cache transfer protocol."
                    )
                num_nodes = len(self.ips) if self.ips else 1
                if self.data_parallel_size % num_nodes != 0:
                    raise ValueError(
                        f"data_parallel_size ({self.data_parallel_size}) must be divisible by "
                        f"num_nodes ({num_nodes})."
                    )
                dp_per_node = self.data_parallel_size // num_nodes
                expected_ports = self.tensor_parallel_size * dp_per_node
                if len(self.rdma_comm_ports) != expected_ports:
                    raise ValueError(
                        f"The number of rdma_comm_ports must equal "
                        f"tensor_parallel_size * (data_parallel_size / num_nodes) = "
                        f"{self.tensor_parallel_size} * ({self.data_parallel_size} / {num_nodes}) "
                        f"= {expected_ports}, but got {len(self.rdma_comm_ports)}."
                    )

        if not (current_platform.is_cuda() or current_platform.is_xpu() or current_platform.is_maca()):
            envs.ENABLE_V1_KVCACHE_SCHEDULER = 0

        if "PaddleOCR" in get_model_architecture(self.model, self.model_config_name):
            envs.FD_ENABLE_MAX_PREFILL = 1
            self.enable_prefix_caching = False
            self.max_encoder_cache = 0

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        """
        Add command line interface arguments to the parser.
        """
        # Model parameters group
        model_group = parser.add_argument_group("Model Configuration")
        model_group.add_argument(
            "--model",
            type=str,
            default=EngineArgs.model,
            help="Model name or path to be used.",
        )
        model_group.add_argument(
            "--served-model-name",
            type=nullable_str,
            default=EngineArgs.served_model_name,
            help="Served model name",
        )
        model_group.add_argument(
            "--revision",
            type=nullable_str,
            default=EngineArgs.revision,
            help="Revision for downloading models",
        )
        model_group.add_argument(
            "--model-config-name",
            type=nullable_str,
            default=EngineArgs.model_config_name,
            help="The model configuration file name.",
        )
        model_group.add_argument(
            "--tokenizer",
            type=nullable_str,
            default=EngineArgs.tokenizer,
            help="Tokenizer name or path (defaults to model path if not specified).",
        )
        model_group.add_argument(
            "--tokenizer-base-url",
            type=nullable_str,
            default=EngineArgs.tokenizer_base_url,
            help="The base URL of the remote tokenizer service (used instead of local tokenizer if provided).",
        )
        model_group.add_argument(
            "--max-model-len",
            type=int,
            default=EngineArgs.max_model_len,
            help="Maximum context length supported by the model.",
        )
        model_group.add_argument(
            "--block-size",
            type=int,
            default=EngineArgs.block_size,
            help="Number of tokens processed in one block.",
        )
        model_group.add_argument(
            "--task",
            type=str,
            default=EngineArgs.task,
            help="Task to be executed by the model.",
        )
        model_group.add_argument(
            "--runner",
            type=str,
            default=EngineArgs.runner,
            help="The type of model runner to use",
        )
        model_group.add_argument(
            "--convert", type=str, default=EngineArgs.convert, help="Convert the model using adapters"
        )
        model_group.add_argument(
            "--override-pooler-config",
            type=json.loads,
            default=EngineArgs.override_pooler_config,
            help="Override the pooler configuration with a JSON string.",
        )
        model_group.add_argument(
            "--use-warmup",
            type=int,
            default=EngineArgs.use_warmup,
            help="Flag to indicate whether to use warm-up before inference.",
        )
        model_group.add_argument(
            "--limit-mm-per-prompt",
            default=EngineArgs.limit_mm_per_prompt,
            type=json.loads,
            help="Limitation of numbers of multi-modal data.",
        )
        model_group.add_argument(
            "--mm-processor-kwargs",
            default=EngineArgs.mm_processor_kwargs,
            type=json.loads,
            help="Additional keyword arguments for the multi-modal processor.",
        )
        model_group.add_argument(
            "--max-encoder-cache",
            default=EngineArgs.max_encoder_cache,
            type=int,
            help="Maximum encoder cache tokens(use 0 to disable).",
        )
        model_group.add_argument(
            "--max-processor-cache",
            default=EngineArgs.max_processor_cache,
            type=float,
            help="Maximum processor cache bytes(use 0 to disable).",
        )
        model_group.add_argument(
            "--enable-mm",
            action=DeprecatedOptionWarning,
            default=EngineArgs.enable_mm,
            help="Flag to enable multi-modal model.",
        )
        model_group.add_argument(
            "--reasoning-parser",
            type=str,
            default=EngineArgs.reasoning_parser,
            help="Flag specifies the reasoning parser to use for extracting "
            "reasoning content from the model output",
        )
        model_group.add_argument(
            "--chat-template",
            type=str,
            default=EngineArgs.chat_template,
            help="chat template or chat template file path",
        )
        model_group.add_argument(
            "--tool-call-parser",
            type=str,
            default=EngineArgs.tool_call_parser,
            help="Flag specifies the tool call parser to use for extracting" "tool call from the model output",
        )
        model_group.add_argument(
            "--tool-parser-plugin",
            type=str,
            default=EngineArgs.tool_parser_plugin,
            help="tool parser plugin used to register user defined tool parsers",
        )
        model_group.add_argument(
            "--speculative-config",
            type=json.loads,
            default=EngineArgs.speculative_config,
            help="Configuration for speculative execution.",
        )
        model_group.add_argument(
            "--dynamic-load-weight",
            action="store_true",
            default=EngineArgs.dynamic_load_weight,
            help="Flag to indicate whether to load weight dynamically.",
        )
        model_group.add_argument(
            "--load-strategy",
            type=str,
            default=EngineArgs.load_strategy,
            help="Flag to dynamic load strategy.",
        )
        model_group.add_argument(
            "--engine-worker-queue-port",
            type=lambda s: s.split(",") if s else None,
            default=EngineArgs.engine_worker_queue_port,
            help="port for engine worker queue",
        )
        model_group.add_argument(
            "--quantization",
            type=parse_quantization,
            default=EngineArgs.quantization,
            help="Quantization name for the model, currently support "
            "'wint8', 'wint4',"
            "default is None. The priority of this configuration "
            "is lower than that of the config file. "
            "More complex quantization methods need to be configured via the config file.",
        )
        model_group.add_argument(
            "--graph-optimization-config",
            type=json.loads,
            default=EngineArgs.graph_optimization_config,
            help="Configuration for graph optimization",
        )
        model_group.add_argument(
            "--plas-attention-config",
            type=json.loads,
            default=EngineArgs.plas_attention_config,
            help="",
        )
        model_group.add_argument(
            "--guided-decoding-backend",
            type=str,
            default=EngineArgs.guided_decoding_backend,
            help="Guided Decoding Backend",
        )
        model_group.add_argument(
            "--guided-decoding-disable-any-whitespace",
            type=str,
            default=EngineArgs.guided_decoding_disable_any_whitespace,
            help="Disabled any whitespaces when using guided decoding backend XGrammar.",
        )
        model_group.add_argument(
            "--enable-logprob",
            action="store_true",
            default=EngineArgs.enable_logprob,
            help="Enable output of token-level log probabilities.",
        )
        model_group.add_argument(
            "--max-logprobs",
            type=int,
            default=EngineArgs.max_logprobs,
            help="Maximum number of log probabilities.",
        )
        model_group.add_argument(
            "--logprobs-mode",
            type=str,
            choices=["raw_logprobs", "raw_logits", "processed_logprobs", "processed_logits"],
            default=EngineArgs.logprobs_mode,
            help="Indicates the content returned in the logprobs.",
        )
        model_group.add_argument(
            "--seed",
            type=int,
            default=EngineArgs.seed,
            help="Random seed for initialization. If not specified, defaults to 0.",
        )
        model_group.add_argument(
            "--enable-early-stop",
            action="store_true",
            default=EngineArgs.enable_early_stop,
            help="Enable early stopping during generation.",
        )
        model_group.add_argument(
            "--early-stop-config",
            type=json.loads,
            default=EngineArgs.early_stop_config,
            help="the config for early stop.",
        )
        model_group.add_argument(
            "--lm_head-fp32",
            action="store_true",
            default=EngineArgs.lm_head_fp32,
            help="Specify the dtype of lm_head weight as float32.",
        )
        model_group.add_argument(
            "--logits-processors",
            type=str,
            nargs="+",
            default=EngineArgs.logits_processors,
            help="FQCNs (Fully Qualified Class Names) of logits processors supported by the service.",
        )

        # Parallel processing parameters group
        parallel_group = parser.add_argument_group("Parallel Configuration")
        parallel_group.add_argument(
            "--tensor-parallel-size",
            "-tp",
            type=int,
            default=EngineArgs.tensor_parallel_size,
            help="Degree of tensor parallelism.",
        )
        parallel_group.add_argument(
            "--disable-custom-all-reduce",
            action="store_true",
            default=EngineArgs.disable_custom_all_reduce,
            help="Flag to disable custom all-reduce.",
        )
        parallel_group.add_argument(
            "--use-internode-ll-two-stage",
            action="store_true",
            default=EngineArgs.use_internode_ll_two_stage,
            help="Flag to use the internode_ll_two_stage kernel.",
        )
        parallel_group.add_argument(
            "--disable-sequence-parallel-moe",
            action="store_true",
            default=EngineArgs.disable_sequence_parallel_moe,
            help="Flag to disable disable the sequence parallel moe.",
        )
        parallel_group.add_argument(
            "--max-num-seqs",
            type=int,
            default=EngineArgs.max_num_seqs,
            help="Maximum number of sequences per iteration.",
        )
        parallel_group.add_argument(
            "--num-gpu-blocks-override",
            type=int,
            default=EngineArgs.num_gpu_blocks_override,
            help="Override for the number of GPU blocks.",
        )
        parallel_group.add_argument(
            "--max-num-batched-tokens",
            type=int,
            default=EngineArgs.max_num_batched_tokens,
            help="Maximum number of tokens to batch together.",
        )
        parallel_group.add_argument(
            "--gpu-memory-utilization",
            type=float,
            default=EngineArgs.gpu_memory_utilization,
            help="Fraction of GPU memory to be utilized.",
        )

        parallel_group.add_argument(
            "--data-parallel-size",
            type=int,
            default=EngineArgs.data_parallel_size,
            help="Degree of data parallelism.",
        )

        parallel_group.add_argument(
            "--local-data-parallel-id",
            type=int,
            default=EngineArgs.local_data_parallel_id,
            help="the rank of data parallelism.",
        )
        parallel_group.add_argument(
            "--enable-expert-parallel",
            action="store_true",
            default=EngineArgs.enable_expert_parallel,
            help="Enable expert parallelism.",
        )
        parallel_group.add_argument(
            "--enable-eplb",
            action="store_true",
            default=EngineArgs.enable_eplb,
            help="Enable eplb.",
        )
        parallel_group.add_argument(
            "--eplb-config",
            type=json.loads,
            default=EngineArgs.eplb_config,
            help="Config of eplb.",
        )

        # Load group
        load_group = parser.add_argument_group("Load Configuration")
        load_group.add_argument(
            "--load-choices",
            type=str,
            default=EngineArgs.load_choices,
            help="The format of the model weights to load.\
                 default/default_v1.",
        )

        # CacheConfig parameters group
        cache_group = parser.add_argument_group("Cache Configuration")

        cache_group.add_argument(
            "--kv-cache-ratio",
            type=float,
            default=EngineArgs.kv_cache_ratio,
            help="Ratio of tokens to process in a block.",
        )

        cache_group.add_argument(
            "--swap-space", type=float, default=EngineArgs.swap_space, help="The amount of CPU memory to offload to."
        )

        cache_group.add_argument(
            "--prealloc-dec-block-slot-num-threshold",
            type=int,
            default=EngineArgs.prealloc_dec_block_slot_num_threshold,
            help="Number of token slot threadshold to allocate next blocks for decoding.",
        )

        cache_group.add_argument(
            "--cache-queue-port",
            type=lambda s: [int(item.strip()) for item in s.split(",")] if s else None,
            default=EngineArgs.cache_queue_port,
            help="port for cache queue",
        )
        cache_group.add_argument(
            "--static-decode-blocks",
            type=int,
            default=EngineArgs.static_decode_blocks,
            help="Static decoding blocks num.",
        )

        # Cluster system parameters group
        system_group = parser.add_argument_group("System Configuration")
        system_group.add_argument(
            "--ips",
            type=lambda s: s.split(",") if s else None,
            default=EngineArgs.ips,
            help="IP addresses of all nodes participating in distributed inference.",
        )

        # Performance tuning parameters group
        perf_group = parser.add_argument_group("Performance Tuning")
        perf_group.add_argument(
            "--enable-prefix-caching",
            action=argparse.BooleanOptionalAction,
            default=EngineArgs.enable_prefix_caching,
            help="Flag to enable prefix caching.",
        )

        perf_group.add_argument(
            "--enable-chunked-prefill",
            action="store_true",
            default=EngineArgs.enable_chunked_prefill,
            help="Flag to enable chunked prefill.",
        )
        perf_group.add_argument(
            "--max-num-partial-prefills",
            type=int,
            default=EngineArgs.max_num_partial_prefills,
            help="For chunked prefill, Maximum number \
            of concurrent partial prefill requests.",
        )
        perf_group.add_argument(
            "--max-long-partial-prefills",
            type=int,
            default=EngineArgs.max_long_partial_prefills,
            help=(
                "For chunked prefill, the maximum number of prompts longer than long-prefill-token-threshold"
                "that will be prefilled concurrently."
            ),
        )
        perf_group.add_argument(
            "--long-prefill-token-threshold",
            type=int,
            default=EngineArgs.long_prefill_token_threshold,
            help=("For chunked prefill, the threshold number of" " tokens for a prompt to be considered long."),
        )

        # Splitwise deployment parameters group
        splitwise_group = parser.add_argument_group("Splitwise Deployment")
        splitwise_group.add_argument(
            "--splitwise-role",
            type=str,
            default=EngineArgs.splitwise_role,
            help="Role of splitwise. Default is \
            'mixed'. (prefill, decode, mixed)",
        )

        splitwise_group.add_argument(
            "--cache-transfer-protocol",
            type=str,
            default=EngineArgs.cache_transfer_protocol,
            help="support protocol list (ipc or rdma), comma separated, default is ipc",
        )

        splitwise_group.add_argument(
            "--pd-comm-port",
            type=lambda s: s.split(",") if s else None,
            default=EngineArgs.pd_comm_port,
            help="port for splitwise communication.",
        )

        splitwise_group.add_argument(
            "--rdma-comm-ports",
            type=lambda s: s.split(",") if s else None,
            default=EngineArgs.rdma_comm_ports,
            help="ports for rdma communication.",
        )

        perf_group.add_argument(
            "--disable-chunked-mm-input",
            action="store_true",
            default=EngineArgs.disable_chunked_mm_input,
            help="Disable chunked mm input.",
        )

        # Router parameters group
        router_group = parser.add_argument_group("Router")
        router_group.add_argument(
            "--router",
            type=str,
            default=EngineArgs.router,
            help="url for router server.",
        )

        # Scheduler parameters group
        scheduler_group = parser.add_argument_group("Scheduler")
        scheduler_group.add_argument(
            "--scheduler-name",
            default=EngineArgs.scheduler_name,
            help=f"Scheduler name to be used. Default is {EngineArgs.scheduler_name}. (local,global)",
        )
        scheduler_group.add_argument(
            "--scheduler-max-size",
            type=int,
            default=EngineArgs.scheduler_max_size,
            help=f"Size of scheduler. Default is {EngineArgs.scheduler_max_size}. (Local)",
        )
        scheduler_group.add_argument(
            "--scheduler-ttl",
            type=int,
            default=EngineArgs.scheduler_ttl,
            help=f"TTL of request. Default is {EngineArgs.scheduler_ttl} seconds. (local,global)",
        )
        scheduler_group.add_argument(
            "--scheduler-host",
            default=EngineArgs.scheduler_host,
            help=f"Host address of redis. Default is {EngineArgs.scheduler_host}. (global)",
        )
        scheduler_group.add_argument(
            "--scheduler-port",
            type=int,
            default=EngineArgs.scheduler_port,
            help=f"Port of redis. Default is {EngineArgs.scheduler_port}. (global)",
        )
        scheduler_group.add_argument(
            "--scheduler-db",
            type=int,
            default=EngineArgs.scheduler_db,
            help=f"DB of redis. Default is {EngineArgs.scheduler_db}. (global)",
        )
        scheduler_group.add_argument(
            "--scheduler-password",
            default=EngineArgs.scheduler_password,
            help=f"Password of redis. Default is {EngineArgs.scheduler_password}. (global)",
        )
        scheduler_group.add_argument(
            "--scheduler-topic",
            default=EngineArgs.scheduler_topic,
            help=f"Topic of scheduler. Default is {EngineArgs.scheduler_topic}. (global)",
        )
        scheduler_group.add_argument(
            "--scheduler-min-load-score",
            type=float,
            default=EngineArgs.scheduler_min_load_score,
            help=f"Minimum load score for task assignment. Default is {EngineArgs.scheduler_min_load_score} (global)",
        )
        scheduler_group.add_argument(
            "--scheduler-load-shards-num",
            type=int,
            default=EngineArgs.scheduler_load_shards_num,
            help=(
                "Number of shards for load balancing table. Default is "
                f"{EngineArgs.scheduler_load_shards_num} (global)"
            ),
        )
        scheduler_group.add_argument(
            "--scheduler-sync-period",
            type=int,
            default=EngineArgs.scheduler_sync_period,
            help=f"SplitWise Use, node load sync period, "
            f"Default is {EngineArgs.scheduler_sync_period}ms. (global)",
        )
        scheduler_group.add_argument(
            "--scheduler-expire-period",
            type=int,
            default=EngineArgs.scheduler_expire_period,
            help=f"SplitWise Use, node will not be scheduled after "
            f"expire-period ms not sync load, Default is "
            f"{EngineArgs.scheduler_expire_period}ms. (global)",
        )
        scheduler_group.add_argument(
            "--scheduler-release-load-expire-period",
            type=int,
            default=EngineArgs.scheduler_release_load_expire_period,
            help=f"SplitWise Use, scheduler will release req load after "
            f"expire period(s). Default is "
            f"{EngineArgs.scheduler_release_load_expire_period}. (global)",
        )
        scheduler_group.add_argument(
            "--scheduler-reader-parallel",
            type=int,
            default=EngineArgs.scheduler_reader_parallel,
            help=f"SplitWise Use, Results Reader Sync Parallel, "
            f"Default is {EngineArgs.scheduler_reader_parallel}. (global)",
        )
        scheduler_group.add_argument(
            "--scheduler-writer-parallel",
            type=int,
            default=EngineArgs.scheduler_writer_parallel,
            help=f"SplitWise Use, Results Writer Sync Parallel, "
            f"Default is {EngineArgs.scheduler_writer_parallel}. (global)",
        )
        scheduler_group.add_argument(
            "--scheduler-reader-batch-size",
            type=int,
            default=EngineArgs.scheduler_reader_batch_size,
            help=f"SplitWise Use, Results Reader Batch Size, "
            f"Default is {EngineArgs.scheduler_reader_batch_size}. (global)",
        )
        scheduler_group.add_argument(
            "--scheduler-writer-batch-size",
            type=int,
            default=EngineArgs.scheduler_writer_batch_size,
            help=f"SplitWise Use, Results Writer Batch Size, "
            f"Default is {EngineArgs.scheduler_writer_batch_size}. (global)",
        )

        return parser

    @classmethod
    def from_cli_args(cls, args: FlexibleArgumentParser) -> "EngineArgs":
        """
        Create an instance of EngineArgs from command line arguments.
        """
        args_dict = {}
        for field in dataclass_fields(cls):
            if hasattr(args, field.name):
                args_dict[field.name] = getattr(args, field.name)
        return cls(**args_dict)

    def create_speculative_config(self) -> SpeculativeConfig:
        """ """
        speculative_args = asdict(self)
        if self.speculative_config is not None:
            for k, v in self.speculative_config.items():
                speculative_args[k] = v

        return SpeculativeConfig(speculative_args)

    def create_scheduler_config(self) -> SchedulerConfig:
        """
        Create and return a SchedulerConfig object based on the current settings.
        """
        prefix = "scheduler_"
        prefix_len = len(prefix)

        all = asdict(self)
        all.pop("port")  # port and scheduler_port are not the same
        params = dict()
        for k, v in all.items():
            if k[:prefix_len] == prefix:
                params[k[prefix_len:]] = v
            else:
                params[k] = v
        return SchedulerConfig(params)

    def create_graph_optimization_config(self) -> GraphOptimizationConfig:
        """
        Create and retuan a GraphOptimizationConfig object based on the current settings.
        """
        graph_optimization_args = asdict(self)
        if self.graph_optimization_config is not None:
            for k, v in self.graph_optimization_config.items():
                graph_optimization_args[k] = v
        return GraphOptimizationConfig(graph_optimization_args)

    def create_plas_attention_config(self) -> PlasAttentionConfig:
        """
        Create and retuan a PlasAttentionConfig object based on the current settings.
        """
        attention_args = asdict(self)
        if self.plas_attention_config is not None:
            for k, v in self.plas_attention_config.items():
                attention_args[k] = v
            return PlasAttentionConfig(attention_args)
        else:
            return PlasAttentionConfig(None)

    def create_early_stop_config(self) -> EarlyStopConfig:
        """
        Create and retuan an EarlyStopConfig object based on the current settings.
        """
        early_stop_args = asdict(self)
        if self.early_stop_config is not None:
            for k, v in self.early_stop_config.items():
                early_stop_args[k] = v
        return EarlyStopConfig(early_stop_args)

    def create_eplb_config(self) -> EPLBConfig:
        """
        Create and retuan an EPLBConfig object based on the current settings.
        """
        eplb_args = asdict(self)
        if self.eplb_config is not None:
            for k, v in self.eplb_config.items():
                eplb_args[k] = v
        eplb_args["enable_eplb"] = self.enable_eplb
        return EPLBConfig(eplb_args)

    def create_engine_config(self, port_availability_check=True) -> FDConfig:
        """
        Create and return a Config object based on the current settings.
        """
        all_dict = asdict(self)
        model_cfg = ModelConfig(all_dict)

        # XPU currently disable prefix cache for VL model
        if current_platform.is_xpu() and (self.enable_mm or model_cfg.enable_mm):
            self.enable_prefix_caching = False

        if not model_cfg.is_unified_ckpt and hasattr(model_cfg, "tensor_parallel_size"):
            self.tensor_parallel_size = model_cfg.tensor_parallel_size

        speculative_cfg = self.create_speculative_config()
        if not self.enable_chunked_prefill:
            if current_platform.is_cuda() and self.splitwise_role == "mixed":
                # default enable chunked prefill
                self.enable_chunked_prefill = True

            self.disable_chunked_prefill = int(envs.FD_DISABLE_CHUNKED_PREFILL)
            if self.disable_chunked_prefill:
                self.enable_chunked_prefill = False

        if self.max_num_batched_tokens is None:
            if int(envs.ENABLE_V1_KVCACHE_SCHEDULER):
                self.max_num_batched_tokens = 8192  # if set to max_model_len, it's easy to be OOM
            else:
                if self.enable_chunked_prefill:
                    self.max_num_batched_tokens = 2048
                else:
                    self.max_num_batched_tokens = self.max_model_len

        if isinstance(self.engine_worker_queue_port, int):
            self.engine_worker_queue_port = str(self.engine_worker_queue_port)
        if isinstance(self.engine_worker_queue_port, str):
            self.engine_worker_queue_port = self.engine_worker_queue_port.split(",")

        all_dict = asdict(self)
        all_dict["model_cfg"] = model_cfg
        cache_cfg = CacheConfig(all_dict)
        load_cfg = LoadConfig(all_dict)
        parallel_cfg = ParallelConfig(all_dict)
        scheduler_cfg = self.create_scheduler_config()
        graph_opt_cfg = self.create_graph_optimization_config()
        plas_attention_config = self.create_plas_attention_config()
        eplb_cfg = self.create_eplb_config()
        router_config = RouterConfig(all_dict)

        early_stop_cfg = self.create_early_stop_config()
        early_stop_cfg.update_enable_early_stop(self.enable_early_stop)
        structured_outputs_config: StructuredOutputsConfig = StructuredOutputsConfig(args=all_dict)
        if port_availability_check:
            assert is_port_available(
                "0.0.0.0", int(self.engine_worker_queue_port[parallel_cfg.local_data_parallel_id])
            ), f"The parameter `engine_worker_queue_port`:{self.engine_worker_queue_port} is already in use."

        return FDConfig(
            model_config=model_cfg,
            scheduler_config=scheduler_cfg,
            tokenizer=self.tokenizer,
            cache_config=cache_cfg,
            load_config=load_cfg,
            parallel_config=parallel_cfg,
            speculative_config=speculative_cfg,
            eplb_config=eplb_cfg,
            structured_outputs_config=structured_outputs_config,
            router_config=router_config,
            ips=self.ips,
            use_warmup=self.use_warmup,
            limit_mm_per_prompt=self.limit_mm_per_prompt,
            mm_processor_kwargs=self.mm_processor_kwargs,
            tool_parser=self.tool_call_parser,
            max_num_partial_prefills=self.max_num_partial_prefills,
            max_long_partial_prefills=self.max_long_partial_prefills,
            long_prefill_token_threshold=self.long_prefill_token_threshold,
            graph_opt_config=graph_opt_cfg,
            plas_attention_config=plas_attention_config,
            early_stop_config=early_stop_cfg,
        )
