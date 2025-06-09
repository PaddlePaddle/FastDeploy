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

from dataclasses import asdict, dataclass
from dataclasses import fields as dataclass_fields
from typing import Any, Dict, List, Optional

from fastdeploy.engine.config import (CacheConfig, Config, ModelConfig,
                                      TaskOption)
from fastdeploy.scheduler.config import SchedulerConfig
from fastdeploy.utils import FlexibleArgumentParser


def nullable_str(x: str) -> Optional[str]:
    """
    Convert an empty string to None, preserving other string values.
    """
    return x if x else None


@dataclass
class EngineArgs:
    # Model configuration parameters
    model: str = ""
    """
    The name or path of the model to be used.
    """
    model_config_name: Optional[str] = "config.json"
    """
    The name of the model configuration file.
    """
    tokenizer: str = None
    """
    The name or path of the tokenizer (defaults to model path if not provided).
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
    max_num_seqs: int = 8
    """
    Maximum number of sequences per iteration.
    """
    mm_processor_kwargs: Optional[Dict[str, Any]] = None
    """
    Additional keyword arguments for the multi-modal processor.
    """
    enable_mm: bool = False
    """
    Flags to enable multi-modal model
    """
    speculative_config: Optional[Dict[str, Any]] = None
    """
    Configuration for speculative execution.
    """
    dynamic_load_weight: int = 0
    """
    dynamic load weight
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
    nnode: int = 1
    """
    Number of nodes in the cluster.
    """
    pod_ips: Optional[List[str]] = None
    """
    List of IP addresses for nodes in the cluster.
    """

    # System configuration parameters
    use_warmup: int = 0
    """
    Flag to indicate whether to use warm-up before inference.
    """
    enable_prefix_caching: bool = False
    """
    Flag to enable prefix caching.
    """
    engine_worker_queue_port: int = 8002
    enable_chunked_prefill: bool = False
    """
    Flag to enable chunked prefilling.
    """

    """
    Scheduler name to be used
    """
    scheduler_name: str = "local"
    """
    Size of scheduler
    """
    scheduler_max_size: int = -1
    """
    TTL of request
    """
    scheduler_ttl: int = 900
    """
    Timeout for waiting for response
    """
    scheduler_wait_response_timeout: float = 0.001
    """
    Host of redis
    """
    scheduler_host: str = "127.0.0.1"
    """
    Port of redis
    """
    scheduler_port: int = 6379
    """
    DB of redis
    """
    scheduler_db: int = 0
    """
    Password of redis
    """
    scheduler_password: Optional[str] = None
    """
    Topic of scheduler
    """
    scheduler_topic: str = "default"
    """
    Max write time of redis
    """
    scheduler_remote_write_time: int = 3

    def __post_init__(self):
        """
        Post-initialization processing to set default tokenizer if not provided.
        """
        if not self.tokenizer:
            self.tokenizer = self.model

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        """
        Add command line interface arguments to the parser.
        """
        # Model parameters group
        model_group = parser.add_argument_group("Model Configuration")
        model_group.add_argument("--model",
                                 type=str,
                                 default=EngineArgs.model,
                                 help="Model name or path to be used.")
        model_group.add_argument("--model-config-name",
                                 type=nullable_str,
                                 default=EngineArgs.model_config_name,
                                 help="The model configuration file name.")
        model_group.add_argument(
            "--tokenizer",
            type=nullable_str,
            default=EngineArgs.tokenizer,
            help=
            "Tokenizer name or path (defaults to model path if not specified)."
        )
        model_group.add_argument(
            "--max-model-len",
            type=int,
            default=EngineArgs.max_model_len,
            help="Maximum context length supported by the model.")
        model_group.add_argument(
            "--block-size",
            type=int,
            default=EngineArgs.block_size,
            help="Number of tokens processed in one block.")
        model_group.add_argument("--task",
                                 type=str,
                                 default=EngineArgs.task,
                                 help="Task to be executed by the model.")
        model_group.add_argument(
            "--use-warmup",
            type=int,
            default=EngineArgs.use_warmup,
            help="Flag to indicate whether to use warm-up before inference.")
        model_group.add_argument(
            "--mm_processor_kwargs",
            default=None,
            help="Additional keyword arguments for the multi-modal processor.")
        model_group.add_argument("--enable-mm",
                                 action='store_true',
                                 default=EngineArgs.enable_mm,
                                 help="Flag to enable multi-modal model.")
        model_group.add_argument(
            "--speculative_config",
            default=None,
            help="Configuration for speculative execution.")

        model_group.add_argument(
            "--dynamic_load_weight",
            type=int,
            default=EngineArgs.dynamic_load_weight,
            help="Flag to indicate whether to load weight dynamically.")

        model_group.add_argument("--engine-worker-queue-port",
                                 type=int,
                                 default=EngineArgs.engine_worker_queue_port,
                                 help="port for engine worker queue")

        # Parallel processing parameters group
        parallel_group = parser.add_argument_group("Parallel Configuration")
        parallel_group.add_argument("--tensor-parallel-size",
                                    "-tp",
                                    type=int,
                                    default=EngineArgs.tensor_parallel_size,
                                    help="Degree of tensor parallelism.")
        parallel_group.add_argument(
            "--max-num-seqs",
            type=int,
            default=EngineArgs.max_num_seqs,
            help="Maximum number of sequences per iteration.")
        parallel_group.add_argument(
            "--num-gpu-blocks-override",
            type=int,
            default=EngineArgs.num_gpu_blocks_override,
            help="Override for the number of GPU blocks.")
        parallel_group.add_argument(
            "--max-num-batched-tokens",
            type=int,
            default=EngineArgs.max_num_batched_tokens,
            help="Maximum number of tokens to batch together.")
        parallel_group.add_argument(
            "--gpu-memory-utilization",
            type=float,
            default=EngineArgs.gpu_memory_utilization,
            help="Fraction of GPU memory to be utilized.")
        parallel_group.add_argument(
            "--kv-cache-ratio",
            type=float,
            default=EngineArgs.kv_cache_ratio,
            help="Ratio of tokens to process in a block.")

        # Cluster system parameters group
        system_group = parser.add_argument_group("System Configuration")
        system_group.add_argument(
            "--pod-ips",
            type=lambda s: s.split(",") if s else None,
            default=EngineArgs.pod_ips,
            help=
            "List of IP addresses for nodes in the cluster (comma-separated).")
        system_group.add_argument("--nnode",
                                  type=int,
                                  default=EngineArgs.nnode,
                                  help="Number of nodes in the cluster.")

        # Performance tuning parameters group
        perf_group = parser.add_argument_group("Performance Tuning")
        perf_group.add_argument(
            "--enable-prefix-caching",
            action='store_true',
            default=EngineArgs.enable_prefix_caching,
            help="Flag to enable prefix caching."
        )
        perf_group.add_argument(
            "--enable-chunked-prefill",
            action='store_true',
            default=EngineArgs.enable_chunked_prefill,
            help="Flag to enable chunked prefill."
        )

        # Scheduler parameters group
        scheduler_group = parser.add_argument_group("Scheduler")
        scheduler_group.add_argument(
            "--scheduler-name",
            default=EngineArgs.scheduler_name,
            help=
            f"Scheduler name to be used. Default is {EngineArgs.scheduler_name}. (local,global)"
        )
        scheduler_group.add_argument(
            "--scheduler-max-size",
            type=int,
            default=EngineArgs.scheduler_max_size,
            help=
            f"Size of scheduler. Default is {EngineArgs.scheduler_max_size}. (Local)"
        )
        scheduler_group.add_argument(
            "--scheduler-ttl",
            type=int,
            default=EngineArgs.scheduler_ttl,
            help=
            f"TTL of request. Default is {EngineArgs.scheduler_ttl} seconds. (local,global)"
        )
        scheduler_group.add_argument(
            "--scheduler-wait-response-timeout",
            type=float,
            default=EngineArgs.scheduler_wait_response_timeout,
            help=
            ("Timeout for waiting for response. Default is "
             f"{EngineArgs.scheduler_wait_response_timeout} seconds. (local,global)"
             ))
        scheduler_group.add_argument(
            "--scheduler-host",
            default=EngineArgs.scheduler_host,
            help=
            f"Host address of redis. Default is {EngineArgs.scheduler_host}. (global)"
        )
        scheduler_group.add_argument(
            "--scheduler-port",
            type=int,
            default=EngineArgs.scheduler_port,
            help=
            f"Port of redis. Default is {EngineArgs.scheduler_port}. (global)")
        scheduler_group.add_argument(
            "--scheduler-db",
            type=int,
            default=EngineArgs.scheduler_db,
            help=f"DB of redis. Default is {EngineArgs.scheduler_db}. (global)"
        )
        scheduler_group.add_argument(
            "--scheduler-password",
            default=EngineArgs.scheduler_password,
            help=
            f"Password of redis. Default is {EngineArgs.scheduler_password}. (global)"
        )
        scheduler_group.add_argument(
            "--scheduler-topic",
            default=EngineArgs.scheduler_topic,
            help=
            f"Topic of scheduler. Defaule is {EngineArgs.scheduler_topic}. (global)"
        )
        scheduler_group.add_argument(
            "--scheduler-remote-write-time",
            type=int,
            default=EngineArgs.scheduler_remote_write_time,
            help=
            f"Max write time of redis. Default is {EngineArgs.scheduler_remote_write_time} seconds (global)"
        )

        return parser

    @classmethod
    def from_cli_args(cls, args: FlexibleArgumentParser) -> "EngineArgs":
        """
        Create an instance of EngineArgs from command line arguments.
        """
        return cls(
            **{
                field.name: getattr(args, field.name)
                for field in dataclass_fields(cls)
            })

    def create_model_config(self) -> ModelConfig:
        """
        Create and return a ModelConfig object based on the current settings.
        """
        return ModelConfig(model_name_or_path=self.model,
                           config_json_file=self.model_config_name,
                           dynamic_load_weight=self.dynamic_load_weight)

    def create_cache_config(self) -> CacheConfig:
        """
        Create and return a CacheConfig object based on the current settings.
        """
        return CacheConfig(
            block_size=self.block_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            num_gpu_blocks_override=self.num_gpu_blocks_override,
            kv_cache_ratio=self.kv_cache_ratio,
            enable_prefix_caching=self.enable_prefix_caching)

    def create_scheduler_config(self) -> SchedulerConfig:
        """
        Create and retuan a SchedulerConfig object based on the current settings.
        """
        prefix = "scheduler_"
        prefix_len = len(prefix)

        all = asdict(self)
        params = dict()
        for k, v in all.items():
            if k[:prefix_len] == prefix:
                params[k[prefix_len:]] = v
        return SchedulerConfig(**params)

    def create_engine_config(self) -> Config:
        """
        Create and return a Config object based on the current settings.
        """
        model_cfg = self.create_model_config()
        if not model_cfg.is_unified_ckpt and hasattr(model_cfg,
                                                     'tensor_parallel_size'):
            self.tensor_parallel_size = model_cfg.tensor_parallel_size
        if self.max_num_batched_tokens is None:
            if self.enable_chunked_prefill:
                self.max_num_batched_tokens = 2048
            else:
                self.max_num_batched_tokens = self.max_model_len
        scheduler_cfg = self.create_scheduler_config()
        return Config(
            model_name_or_path=self.model,
            model_config=model_cfg,
            scheduler_config=scheduler_cfg,
            tokenizer=self.tokenizer,
            cache_config=self.create_cache_config(),
            max_model_len=self.max_model_len,
            tensor_parallel_size=self.tensor_parallel_size,
            max_num_seqs=self.max_num_seqs,
            mm_processor_kwargs=self.mm_processor_kwargs,
            speculative_config=self.speculative_config,
            max_num_batched_tokens=self.max_num_batched_tokens,
            nnode=self.nnode,
            pod_ips=self.pod_ips,
            use_warmup=self.use_warmup,
            engine_worker_queue_port=self.engine_worker_queue_port,
            enable_mm=self.enable_mm,
            enable_chunked_prefill=self.enable_chunked_prefill,
        )
