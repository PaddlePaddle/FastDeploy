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

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from fastdeploy import envs
from fastdeploy.platforms import current_platform
from fastdeploy.scheduler import SchedulerConfig
from fastdeploy.utils import (ceil_div, check_unified_ckpt, get_host_ip,
                              is_port_available, llm_logger)

TaskOption = Literal["generate"]


class ModelConfig:
    """
    Configuration class for the model.

    Attributes:
        model_dir (str): Directory path to the model.
        is_unified_ckpt (bool): Flag indicating if the checkpoint is unified.
        model_name_or_path (str): Name or path of the model.
    """

    def __init__(self,
                 model_name_or_path: str,
                 config_json_file: str = "config.json",
                 dynamic_load_weight: bool = False,
                 load_strategy: str="meta",
                 quantization: str = None,
                 download_dir: Optional[str] = None):
        """
        Initialize the ModelConfig class.

        Args:
            model_name_or_path (str): Name or path of the model.
            config_json_file (str): Path to the configuration JSON file. Default is 'config.json'.
            download_dir (Optional[str]): Directory to download model files. Default is None.
        """
        self.model_dir = model_name_or_path
        self.is_unified_ckpt = check_unified_ckpt(self.model_dir)
        self.dynamic_load_weight = dynamic_load_weight
        self.load_strategy = load_strategy
        self.quantization = quantization

        config_file = os.path.join(model_name_or_path, config_json_file)
        if os.path.isfile(model_name_or_path):
            try:
                from paddleformers.transformers import AutoConfig
                config = AutoConfig.from_pretrained(model_name_or_path)
                config_dict = {
                    k: v
                    for k, v in vars(config).items() if not k.startswith('_')
                }
                for key, value in config_dict.items():
                    setattr(self, key, value)
            except Exception:
                llm_logger.error(
                    "Don't support the current model, you can use `paddleformers` to register your model."
                )
                raise ValueError(
                    "Don't support the current model, you can use `paddleformers` to register your model."
                )
        else:
            with open(config_file, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
                for key, value in config_dict.items():
                    try:
                        setattr(self, key, value)
                    except Exception:
                        continue

        if isinstance(self.architectures, list):
            self.architectures = self.architectures[0]
        self.model_name_or_path = model_name_or_path
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
        enable_chunked_prefill=False,
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

        self.enable_chunked_prefill = enable_chunked_prefill
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


class SpeculativeConfig:
    """
    Speculative Decoding Configuration class.

    Attributes:
        method (Optional[str]): Method used for speculative decoding.
        num_speculative_tokens (int): Maximum draft tokens, default is 1.
        model_name_or_path (Optional[str]): Path of the model.
        quantization (str): Quantization method for draft model, default is WINT8.
        max_model_len: Optional[int]: Maximum model length for draft model.
    """

    def __init__(self,
                 method: Optional[str] = None,
                 num_speculative_tokens: Optional[int] = 1,
                 model: Optional[str] = None,
                 quantization: Optional[str] = "WINT8",
                 max_model_len: Optional[int] = None,
                 **kwargs):
        self.model_name_or_path = model
        self.method = method
        self.num_speculative_tokens = num_speculative_tokens
        self.quantization = quantization
        self.max_model_len = max_model_len
        # Fixed now
        self.num_gpu_block_expand_ratio = 1
        self.num_extra_cache_layer = 0

        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except Exception:
                continue

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


class ParallelConfig:
    """
    Configuration for parallelism.

    Attributes:
        tensor_parallel_size (int): Size of tensor parallelism.
        data_parallel_size (int): Size of data parallelism.
        local_data_parallel_id (int): ID of local data parallel.
        enable_expert_parallel (bool): Whether to enable expert parallel.
    """

    def __init__(
        self,
        tensor_parallel_size: int = 1,
        data_parallel_size: int = 1,
        enable_expert_parallel: bool = False,
    ):
        """
        Initialize the ParallelConfig class.

        Args:
            tensor_parallel_size (int): Size of tensor parallelism.
            data_parallel_size (int): Size of data parallelism.
            local_data_parallel_id (int): ID of local data parallel.
            enable_expert_parallel (bool): Whether to enable expert parallel.
        """
        self.tensor_parallel_size = tensor_parallel_size
        self.data_parallel_size = data_parallel_size
        self.enable_expert_parallel = enable_expert_parallel
        self.expert_parallel_size = data_parallel_size
        self.local_data_parallel_id = 0

    def print(self):
        """
        print all config

        """
        llm_logger.info("Parallel Configuration Information :")
        for k, v in self.__dict__.items():
            llm_logger.info("{:<20}:{:<6}{}".format(k, "", v))
        llm_logger.info("==================")


class Config:
    """
    Initial configuration class.

    Attributes:
        model_config (ModelConfig): Model configuration object.
        cache_config (CacheConfig): Cache configuration object.
        model_name_or_path (str): Directory path to the model or the model name.
        tokenizer (Optional[str]): Default is the model.
        max_num_batched_tokens (Optional[int]): Maximum number of batched tokens.
        tensor_parallel_size (int): Tensor parallel size.
        nnode (int): Number of nodes.
        max_model_len (int): Maximum model length. Default is 8192.
        max_num_seqs (int): Maximum number of sequences. Default is 8.
        mm_processor_kwargs (Optional[Dict[str, Any]]): Additional arguments for multi-modal processor.
        speculative_config (Optional[Dict[str, Any]]): Speculative execution configuration.
        use_warmup (bool): Flag to use warmup.
        engine_worker_queue_port (int): Port for engine worker queue.
        enable_mm (bool): Flag to enable multi-modal processing.
        reasoning_parser(str): Flag specifies the reasoning parser to use for
            extracting reasoning content from the model output
        splitwise_role (str): Splitwise role.
        innode_prefill_ports (Optional[List[int]]): Innode prefill ports.
            Temporary configuration, will be removed in the future.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        scheduler_config: SchedulerConfig,
        parallel_config: ParallelConfig,
        model_name_or_path: str = None,
        tokenizer: str = None,
        tensor_parallel_size: int = 8,
        nnode: int = 1,
        max_model_len: int = 8192,
        max_num_seqs: int = 8,
        max_num_batched_tokens: Optional[int] = None,
        pod_ips: Optional[List[str]] = None,
        speculative_config: Optional[Dict[str, Any]] = None,
        use_warmup: bool = False,
        engine_worker_queue_port: int = 8002,
        limit_mm_per_prompt: Optional[Dict[str, Any]] = None,
        mm_processor_kwargs: Optional[Dict[str, Any]] = None,
        enable_mm: bool = False,
        splitwise_role: str = "mixed",
        innode_prefill_ports: Optional[List[int]] = None,
        max_num_partial_prefills: int = 1,
        max_long_partial_prefills: int = 1,
        long_prefill_token_threshold: int = 0,
        reasoning_parser: str = None,
        enable_static_graph_inference: bool = False,
        use_cudagraph: bool = False,
        max_capture_batch_size: int = 64,
        guided_decoding_backend: Optional[str] = None,
        disable_any_whitespace: bool = False,
    ):
        """
        Initialize the Config class.

        Args:
            model_config (ModelConfig): Model configuration object.
            cache_config (CacheConfig): Cache configuration object.
            parallel_config (ParallelConfig): Parallel configuration object.
            scheduler_config (SchedulerConfig): Scheduler configuration object.
            model_name_or_path (str): Model directory path or model name.
            tokenizer (str): Default is the model.
            tensor_parallel_size (int): Tensor parallel size. Default is 8.
            nnode (int): Number of nodes. Default is 1.
            max_model_len (int): Maximum model length. Default is 8192.
            max_num_seqs (int): Maximum number of sequences. Default is 8.
            max_num_batched_tokens (Optional[int]): Maximum number of batched tokens. Default is None.
            pod_ips (Optional[List[str]]): List of POD IPs. Default is None.
            mm_processor_kwargs (Optional[Dict[str, Any]]): Additional arguments for multi-modal processor. Default is None.
            speculative_config (Optional[Dict[str, Any]]): Speculative execution configuration. Default is None.
            use_warmup (bool): Flag to use warmup. Default is False.
            engine_worker_queue_port (int): Engine worker queue port. Default is 8002.
            enable_mm (bool): Flag to enable multi-modal processing. Default is False.
            splitwise_role (str): Splitwise role. Default is "mixed".
            innode_prefill_ports (Optional[List[int]]): Innode prefill ports. Default is None.
            reasoning_parser (str): Flag specifies the reasoning parser to use for
                   extracting reasoning content from the model output. Default is None.
            guided_decoding_backend(str): Guided decoding backend. Default is None.
            disable_any_whitespace(bool): Disable any whitespace when using guided decoding.
                Default is False.
        """
        self.model_config = model_config
        self.cache_config = cache_config
        self.scheduler_config = scheduler_config
        self.parallel_config = parallel_config
        self.model_name_or_path = model_name_or_path
        self.tokenizer = tokenizer
        self.max_num_batched_tokens = max_num_batched_tokens
        self.tensor_parallel_size = tensor_parallel_size
        self.nnode = nnode
        self.pod_ips = pod_ips
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.limit_mm_per_prompt = limit_mm_per_prompt
        self.mm_processor_kwargs = mm_processor_kwargs
        self.enable_mm = enable_mm
        self.speculative_config = speculative_config
        self.use_warmup = use_warmup
        self.splitwise_role = splitwise_role
        self.innode_prefill_ports = innode_prefill_ports
        self.max_num_partial_prefills = max_num_partial_prefills
        self.max_long_partial_prefills = max_long_partial_prefills
        self.long_prefill_token_threshold = long_prefill_token_threshold
        self.reasoning_parser = reasoning_parser
        self.enable_static_graph_inference = enable_static_graph_inference
        self.use_cudagraph = use_cudagraph
        self.max_capture_batch_size = max_capture_batch_size
        self.guided_decoding_backend = guided_decoding_backend
        self.disable_any_whitespace = disable_any_whitespace

        if self.innode_prefill_ports is not None:
            if not isinstance(self.innode_prefill_ports, list):
                ports = str(self.innode_prefill_ports).split(',')
                self.innode_prefill_ports = [int(port) for port in ports]

        assert self.splitwise_role in ["mixed", "prefill", "decode"]

        # TODO
        self.max_prefill_batch = 3
        if current_platform.is_xpu():
            self.max_prefill_batch = 1
        if enable_mm:
            self.max_prefill_batch = 1  # TODO:当前多模prefill阶段只支持并行度为1,待优化

        # TODO(@wufeisheng): TP and EP need to be supported simultaneously.
        assert (self.tensor_parallel_size == 1
                and self.parallel_config.expert_parallel_size
                >= 1) or (self.tensor_parallel_size >= 1
                          and self.parallel_config.expert_parallel_size
                          == 1), "TP and EP cannot be enabled at the same time"

        num_ranks = self.tensor_parallel_size * self.parallel_config.expert_parallel_size
        if num_ranks > 8:
            local_num_ranks = 8
            self.nnode = ceil_div(num_ranks, local_num_ranks)
        else:
            local_num_ranks = num_ranks

        self.engine_worker_queue_port = engine_worker_queue_port
        self.device_ids = ",".join([str(i) for i in range(min((self.tensor_parallel_size * \
                                        self.parallel_config.expert_parallel_size), 8))])
        self.device_ids = os.getenv("CUDA_VISIBLE_DEVICES", self.device_ids)

        self.read_from_config()
        self.postprocess()
        self.check()
        self.print()

    def postprocess(self):
        """
        calculate some parameters
        """
        total_rank = self.tensor_parallel_size * self.parallel_config.expert_parallel_size
        assert self.device_ids.split(',').__len__() == min(total_rank, 8), \
        f"invalid CUDA_VISIBLE_DEVICES, should be equal to {min(total_rank, 8)}"
        self.local_device_ids = self.device_ids.split(
            ',')[:self.tensor_parallel_size]
        assert self.tensor_parallel_size % self.nnode == 0, \
        f"tensor_parallel_size: {self.tensor_parallel_size} should be divisible by nnode: {self.nnode}"
        self.worker_num_per_node = total_rank // self.nnode
        self.host_ip = get_host_ip()

        import paddle
        self.paddle_commit_id = paddle.version.commit

        if self.max_num_batched_tokens is None:
            if self.cache_config.enable_chunked_prefill:
                self.max_num_batched_tokens = 2048
            else:
                self.max_num_batched_tokens = self.max_model_len

        if self.long_prefill_token_threshold == 0:
            self.long_prefill_token_threshold = int(self.max_model_len * 0.04)

        self.cache_config.postprocess(self.max_num_batched_tokens,
                                      self.max_num_seqs)
        self.cache_config.max_block_num_per_seq = int(
            self.max_model_len // self.cache_config.block_size)

        if self.guided_decoding_backend == "auto":
            if self.enable_mm:
                self.guided_decoding_backend = "off"
            else:
                self.guided_decoding_backend = "xgrammar"

    def check(self):
        """
        check the legality of config
        """
        assert (
            self.max_num_seqs <= 256
        ), "The parameter `max_num_seqs` is not allowed to exceed 256, " "but now it's {}.".format(
            self.max_num_seqs)
        assert (
            is_port_available('0.0.0.0', self.engine_worker_queue_port)
        ), f"The parameter `engine_worker_queue_port`:{self.engine_worker_queue_port} is already in use."
        assert (
            8 >= self.tensor_parallel_size > 0
        ), f"tensor_parallel_size: {self.tensor_parallel_size} should be between 1 and 8"
        assert (self.nnode >= 1), f"nnode: {self.nnode} should no less than 1"
        assert (
            self.max_model_len >= 16
        ), f"max_model_len: {self.max_model_len} should be larger than 16"
        assert (
            self.max_num_seqs
            >= 1), f"max_num_seqs: {self.max_num_seqs} should be larger than 1"
        assert (
            self.max_num_batched_tokens >= self.max_num_seqs
        ), f"max_num_batched_tokens: {self.max_num_batched_tokens} " \
            f"should be larger than or equal to max_num_seqs: {self.max_num_seqs}"
        assert (self.max_num_batched_tokens <= self.max_model_len * self.max_num_seqs), \
                f"max_num_batched_tokens: {self.max_num_batched_tokens} should be larger" \
                f"than or equal to max_num_seqs: {self.max_num_seqs} * max_model_len: {self.max_model_len}"
        assert (
            self.max_num_partial_prefills >= 1
        ), f"max_num_partial_prefills: {self.max_num_partial_prefills} should be larger than or equal to 1"

        assert (
            self.max_long_partial_prefills >= 1
        ), f"max_long_partial_prefills: {self.max_long_partial_prefills} should be larger than or equal to 1"
        assert (self.max_long_partial_prefills <= self.max_num_partial_prefills), \
                f"max_long_partial_prefills: {self.max_long_partial_prefills} should " \
                f"be less than or equal to max_num_partial_prefills: {self.max_num_partial_prefills}"

        if not self.cache_config.enable_chunked_prefill:
            assert (
                self.max_num_batched_tokens >= self.max_model_len
            ), f"max_num_batched_tokens: {self.max_num_batched_tokens} " \
                f"should be larger than or equal to max_model_len: {self.max_model_len}"

        if self.max_num_partial_prefills > 1:
            assert (self.cache_config.enable_chunked_prefill is True), \
            "Chunked prefill must be enabled to set max_num_partial_prefills > 1"
            assert (self.long_prefill_token_threshold < self.max_model_len), \
            f"long_prefill_token_threshold: {self.long_prefill_token_threshold} should be less than"\
            f" max_model_len: {self.max_model_len}"

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
                    import xgrammar
                except Exception as e:
                    raise Exception(
                        f"import XGrammar failed, please install XGrammar use `pip install xgrammar==0.1.19`. \n\t {e}"
                    )

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
            elif k == "cache_config" or k == "model_config" or k == "scheduler_config" or k == "parallel_config":
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
        if self.splitwise_role != "mixed":
            disaggregate_info["role"] = self.splitwise_role
            disaggregate_info["cache_info"] = dict()
            current_protocol = self.cache_config.cache_transfer_protocol.split(
                ",")
            disaggregate_info["transfer_protocol"] = current_protocol
            for protocol in current_protocol:
                if protocol == "ipc":
                    disaggregate_info["cache_info"][protocol] = {
                        "ip": self.host_ip,
                        "port": self.engine_worker_queue_port,
                        "device_ids": self.local_device_ids
                    }
                elif protocol == "rdma":
                    disaggregate_info["cache_info"][protocol] = {
                        "ip": self.host_ip,
                        "port": self.cache_config.pd_comm_port[0],
                        "rdma_port": self.cache_config.rdma_comm_ports,
                    }
        self.disaggregate_info = disaggregate_info
        llm_logger.info(f"disaggregate_info: {self.disaggregate_info}")

    def read_from_config(self):
        """
        reset model config from json file
        """

        def reset_value(cls, value_name, key):
            if hasattr(cls, key):
                value = getattr(cls, key)
                setattr(cls, value_name, value)
                llm_logger.info(
                    f"Reset parameter {value_name} = {value} from configuration."
                )

        reset_value(self.cache_config, "block_size", "infer_model_block_size")
        reset_value(self.model_config, "return_full_hidden_states",
                    "return_full_hidden_states")
        reset_value(self.cache_config, "cache_dtype", "infer_model_dtype")

    def __str__(self) -> str:
        return json.dumps(self.__dict__, indent=4)
