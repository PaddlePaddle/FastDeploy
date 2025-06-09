# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from fastdeploy.scheduler import SchedulerConfig
from fastdeploy.utils import (check_unified_ckpt, get_host_ip,
                              is_port_available, llm_logger)

TaskOption = Literal["generate"]


class ModelConfig:
    """
Configuration class for model settings and parameters.

Attributes:
    model_dir (str): Path to the model directory
    is_unified_ckpt (bool): Whether the checkpoint uses unified format
    model_name_or_path (str): Model identifier or path
    dynamic_load_weight (int): Dynamic weight loading flag
"""

    def __init__(self,
                 model_name_or_path: str,
                 config_json_file: str = "config.json",
                 dynamic_load_weight: int = 0,
                 download_dir: Optional[str] = None):
        """
        Initialize model configuration.

        Args:
            model_name_or_path (str): Model identifier or path
            config_json_file (str): Model config file name (default: 'config.json')
            dynamic_load_weight (int): Dynamic weight loading mode (default: 0)
            download_dir (Optional[str]): Directory for downloaded models (default: None)
        """
        self.model_dir = model_name_or_path
        self.is_unified_ckpt = check_unified_ckpt(self.model_dir)
        self.dynamic_load_weight = dynamic_load_weight

        config_file = os.path.join(model_name_or_path, config_json_file)
        if os.path.isfile(model_name_or_path):
            try:
                from paddlenlp.transformers import AutoConfig
                config = AutoConfig.from_pretrained(model_name_or_path)
                config_dict = {
                    k: v
                    for k, v in vars(config).items() if not k.startswith('_')
                }
                for key, value in config_dict.items():
                    setattr(self, key, value)
            except Exception:
                llm_logger.error(
                    "Don't support the current model, you can use `paddlenlp` to register your model."
                )
                raise ValueError(
                    "Don't support the current model, you can use `paddlenlp` to register your model."
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
        Update attribute names from model configuration.
        Handles special cases like:
        - Renaming infer_model_mp_num to tensor_parallel_size
        - Adjusting num_hidden_layers based on remove_tail_layer
        - Setting default mla_use_absorb value
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

    def read_from_env(self):
        """
        Load configuration from environment variables.
        Sets default values if env vars not found.
        Reads:
        - MAX_STOP_SEQS_NUM (default: 5)
        - STOP_SEQS_MAX_LEN (default: 8) 
        - ELLM_DYNAMIC_QUANT_TYPE (default: 'default')
        - ELLM_DYNAMIC_USE_STOP_SEQS (default: 0)
        - COMPRESSION_RATIO (default: 1.0)
        - ROPE_THETA (default: 10000)
        """
        self.max_stop_seqs_num = int(os.getenv("MAX_STOP_SEQS_NUM", "5"))
        self.stop_seqs_max_len = int(os.getenv("STOP_SEQS_MAX_LEN", "8"))

        self.ellm_dynamic_quant_type = os.getenv("ELLM_DYNAMIC_QUANT_TYPE",
                                                 "default")
        # Whether to use stop sequences in dynamic graph inference
        self.ellm_dynamic_use_stop_seqs = int(
            os.getenv("ELLM_DYNAMIC_USE_STOP_SEQS", "0")) == 1

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
        # TODO: Implement dynamic graph for self-downloading models
        pass

    def print(self):
        """
        Print current model configuration.
        Logs all attributes and their values.
        """
        llm_logger.info("Model Configuration Information :")
        for k, v in self.__dict__.items():
            llm_logger.info("{:<20}:{:<6}{}".format(k, "", v))
        llm_logger.info(
            "=============================================================")


class CacheConfig:
    """
Configuration for key-value cache management.

Attributes:
    block_size (int): Tokens per cache block
    gpu_memory_utilization (float): GPU memory usage fraction (0-1)
    cache_dtype (str): Data type for cache (default: 'bfloat16')
    num_gpu_blocks_override (Optional[int]): Manual GPU blocks override
    kv_cache_ratio (float): Max blocks ratio (default: 0.75)
    enc_dec_block_num (int): Encoder-decoder blocks count
    enable_prefix_caching (bool): Prefix caching enable flag
    total_block_num (int): Total available blocks
    prefill_kvcache_block_num (int): Blocks allocated for prefill
"""

    def __init__(
        self,
        block_size: int,
        gpu_memory_utilization: float,
        cache_dtype: str = "bfloat16",
        num_gpu_blocks_override: Optional[int] = None,
        kv_cache_ratio: float = 0.75,
        enc_dec_block_num: int = 2,
        enable_prefix_caching: bool = False,
    ):
        """
        Initialize cache configuration.

        Args:
            block_size (int): Tokens per cache block
            gpu_memory_utilization (float): GPU memory usage target (0-1)
            cache_dtype (str): Cache data type (default: 'bfloat16')
            num_gpu_blocks_override (Optional[int]): Manual GPU blocks setting
            kv_cache_ratio (float): Max blocks ratio (default: 0.75)
            enc_dec_block_num (int): Encoder-decoder blocks count
            enable_prefix_caching (bool): Enable prefix sharing
        """
        self.block_size = block_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.num_gpu_blocks_override = num_gpu_blocks_override
        self.kv_cache_ratio = kv_cache_ratio
        self.enc_dec_block_num = enc_dec_block_num
        self.cache_dtype = cache_dtype
        self.enable_prefix_caching = enable_prefix_caching
        self._verify_args()

    def metrics_info(self):
        """
        Convert config to metrics dictionary.
        
        Returns:
            Dict[str, str]: Key-value pairs of all config attributes
        """
        return {key: str(value) for key, value in self.__dict__.items()}

    def _verify_args(self):
        """Validate configuration arguments."""
        if self.gpu_memory_utilization > 1.0:
            raise ValueError(
                "GPU memory utilization must be less than 1.0. Got "
                f"{self.gpu_memory_utilization}.")
        if self.kv_cache_ratio > 1.0:
            raise ValueError("KV cache ratio must be less than 1.0. Got "
                             f"{self.kv_cache_ratio}.")

    def postprocess(self, num_total_tokens, number_of_tasks):
        """
        Calculate block allocation based on tokens and tasks.
        
        Args:
            num_total_tokens (int): Total tokens to process
            number_of_tasks (int): Number of parallel tasks
            
        Sets:
            dec_token_num (int): Decoder tokens per block
            total_block_num (int): Total blocks needed
            prefill_kvcache_block_num (int): Blocks for prefill phase
        """
        self.dec_token_num = self.enc_dec_block_num * self.block_size
        if self.num_gpu_blocks_override is not None:
            self.total_block_num = self.num_gpu_blocks_override
            self.prefill_kvcache_block_num= int(self.total_block_num * self.kv_cache_ratio)
        else:
            length = num_total_tokens // number_of_tasks
            block_num = (length + self.block_size - 1 + self.enc_dec_block_num) // self.block_size
            self.total_block_num =  block_num * number_of_tasks
            self.prefill_kvcache_block_num= self.total_block_num
            llm_logger.info(f"Doing profile, the total_block_num:{self.total_block_num}")

    def reset(self, num_gpu_blocks):
        """
        Reset GPU block allocation.
        
        Args:
            num_gpu_blocks (int): New total blocks count
            
        Updates:
            total_block_num (int)
            prefill_kvcache_block_num (int)
        """
        self.total_block_num  = num_gpu_blocks
        self.prefill_kvcache_block_num= int(self.total_block_num * self.kv_cache_ratio)
        llm_logger.info((f"Reset block num, the total_block_num:{self.total_block_num},"
            f" prefill_kvcache_block_num:{self.prefill_kvcache_block_num}"))

    def print(self):
        """Print current cache configuration."""
        llm_logger.info("Cache Configuration Information :")
        for k, v in self.__dict__.items():
            llm_logger.info("{:<20}:{:<6}{}".format(k, "", v))
        llm_logger.info(
            "=============================================================")


class Config:
    """
Main engine configuration class combining all components.

Attributes:
    model_config (ModelConfig): Model settings
    cache_config (CacheConfig): Cache management settings
    scheduler_config (SchedulerConfig): Task scheduling settings
    model_name_or_path (str): Model identifier/path
    tokenizer (str): Tokenizer identifier
    tensor_parallel_size (int): Parallelism degree (default: 8)
    nnode (int): Node count (default: 1)
    max_model_len (int): Max sequence length (default: 8192)
    max_num_seqs (int): Max concurrent sequences (default: 8)
    max_num_batched_tokens (Optional[int]): Max batched tokens
    pod_ips (Optional[List[str]]): Cluster node IPs
    mm_processor_kwargs (Optional[Dict]): Multi-modal processor args
    speculative_config (Optional[Dict]): Speculative execution settings
    use_warmup (bool): Warmup enable flag
    enable_mm (bool): Multi-modal enable flag
    enable_chunked_prefill (bool): Chunked prefill enable flag
    device_ids (str): GPU device IDs
    tp_num_per_node (int): Tensor parallelism per node
    host_ip (str): Current host IP
    paddle_commit_id (str): PaddlePaddle version
"""

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        scheduler_config: SchedulerConfig,
        model_name_or_path: str = None,
        tokenizer: str = None,
        tensor_parallel_size: int = 8,
        nnode: int = 1,
        max_model_len: int = 8192,
        max_num_seqs: int = 8,
        max_num_batched_tokens: Optional[int] = None,
        pod_ips: Optional[List[str]] = None,
        mm_processor_kwargs: Optional[Dict[str, Any]] = None,
        speculative_config: Optional[Dict[str, Any]] = None,
        use_warmup: bool = False,
        engine_worker_queue_port: int = 8002,
        enable_mm: bool = False,
        enable_chunked_prefill: bool = False,
    ):
        """
        Initialize engine configuration.

        Args:
            model_config (ModelConfig): Model settings
            cache_config (CacheConfig): Cache settings
            scheduler_config (SchedulerConfig): Scheduler settings
            model_name_or_path (str): Model identifier (default: None)
            tokenizer (str): Tokenizer identifier (default: None)
            tensor_parallel_size (int): Parallelism degree (default: 8)
            nnode (int): Node count (default: 1)
            max_model_len (int): Max sequence length (default: 8192)
            max_num_seqs (int): Max concurrent sequences (default: 8)
            max_num_batched_tokens (Optional[int]): Max batched tokens (default: None)
            pod_ips (Optional[List[str]]): Cluster node IPs (default: None)
            mm_processor_kwargs (Optional[Dict]): Multi-modal args (default: None)
            speculative_config (Optional[Dict]): Speculative settings (default: None)
            use_warmup (bool): Warmup flag (default: False)
            engine_worker_queue_port (int): Worker queue port (default: 8002)
            enable_mm (bool): Multi-modal flag (default: False)
            enable_chunked_prefill (bool): Chunked prefill flag (default: False)
        """
        self.model_config = model_config
        self.cache_config = cache_config
        self.scheduler_config = scheduler_config
        self.model_name_or_path = model_name_or_path
        self.tokenizer = tokenizer
        self.max_num_batched_tokens = max_num_batched_tokens
        self.tensor_parallel_size = tensor_parallel_size
        self.nnode = nnode
        self.pod_ips = pod_ips
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.mm_processor_kwargs = mm_processor_kwargs
        self.enable_mm = enable_mm
        self.speculative_config = speculative_config
        self.use_warmup = use_warmup
        self.enable_chunked_prefill = enable_chunked_prefill

        # TODO
        self.max_prefill_batch = 3
        if enable_mm:
            self.max_prefill_batch = 1  # TODO: Currently multi-modal prefill only supports parallelism=1 (needs optimization)

        self.engine_worker_queue_port = engine_worker_queue_port
        self.device_ids = ",".join(
            [str(i) for i in range(self.tensor_parallel_size)])
        self.device_ids = os.getenv("CUDA_VISIBLE_DEVICES", self.device_ids)

        self.read_from_config()
        self.postprocess()
        self.check()
        self.print()

    def postprocess(self):
        """
        Calculate derived parameters:
        - Validates GPU device count matches tensor_parallel_size
        - Computes tensor parallelism per node
        - Gets host IP and Paddle version
        - Sets default max_num_batched_tokens if not provided
        - Initializes cache configuration
        """
        if len(self.device_ids.split(',')) > self.tensor_parallel_size:
            self.device_ids = ",".join(
                self.device_ids.split(',')[:self.tensor_parallel_size:])
        assert len(
            self.device_ids.split(',')
        ) == self.tensor_parallel_size, f"The number of available GPUs is {len(self.device_ids.split(','))}, which is less than the tensor parallel required {self.tensor_parallel_size}."

        assert self.tensor_parallel_size % self.nnode == 0, f"tensor_parallel_size: {self.tensor_parallel_size} should be divisible by nnode: {self.nnode}"
        self.tp_num_per_node = self.tensor_parallel_size // self.nnode
        self.host_ip = get_host_ip()

        import paddle
        self.paddle_commit_id = paddle.version.commit

        if self.max_num_batched_tokens is None:
            if self.enable_chunked_prefill:
                self.max_num_batched_tokens = 2048
            else:
                self.max_num_batched_tokens = self.max_model_len
        self.cache_config.postprocess(self.max_num_batched_tokens, self.max_num_seqs)


    def check(self):
        """
        Validate configuration values:
        - max_num_seqs <= 256
        - engine_worker_queue_port available
        - 1 <= tensor_parallel_size <= 8
        - nnode >= 1
        - max_model_len >= 16
        - max_num_seqs >= 1
        - Validates scheduler configuration
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

        self.scheduler_config.check()

    def print(self, file=None):
        """
        Print or save current configuration.
        
        Args:
            file (Optional[str]): File path to save config (default: None)
        """
        llm_logger.info(
            "=================== Configuration Information ===============")
        for k, v in self.__dict__.items():
            if k == "generation_config" and v is not None:
                for gck, gcv in v.to_dict().items():
                    llm_logger.info("{:<20}:{:<6}{}".format(gck, "", gcv))
            elif k == "cache_config" or k == "model_config" or k == "scheduler_config":
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

    def read_from_config(self):
        """
        Update configuration from model JSON file.
        Handles special cases:
        - infer_model_block_size -> block_size
        - return_full_hidden_states
        - infer_model_dtype -> cache_dtype
        """

        def reset_value(cls, value_name, key):
            if hasattr(cls, key):
                value = getattr(cls, key)
                setattr(cls, value_name, value)
                llm_logger.info(
                    f"Reset parameter {value_name} = {value} from configuration."
                )

        reset_value(self.cache_config, "block_size", "infer_model_block_size")
        reset_value(self.model_config, "return_full_hidden_states", "return_full_hidden_states")
        reset_value(self.cache_config, "cache_dtype", "infer_model_dtype")

    def __str__(self) -> str:
        return json.dumps(self.__dict__, indent=4)

