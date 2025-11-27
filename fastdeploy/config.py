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
from dataclasses import field
from enum import Enum
from typing import Any, Dict, Literal, Optional, Union

import paddle
import paddle.distributed as dist
from paddleformers.transformers.configuration_utils import PretrainedConfig
from typing_extensions import assert_never

import fastdeploy
from fastdeploy import envs
from fastdeploy.model_executor.layers.quantization.quant_base import QuantConfigBase
from fastdeploy.platforms import current_platform
from fastdeploy.scheduler import SchedulerConfig
from fastdeploy.transformer_utils.config import get_pooling_config
from fastdeploy.utils import ceil_div, check_unified_ckpt, get_host_ip, get_logger

logger = get_logger("config", "config.log")

TaskOption = Literal["auto", "generate", "embedding", "embed"]

RunnerType = Literal["generate", "pooling"]

RunnerOption = Literal["auto", "generate", "pooling"]

ConvertOption = Literal["auto", "none", "embed"]

ConvertType = Literal["none", "embed"]

_ResolvedTask = Literal["generate", "encode", "embed"]

_RUNNER_CONVERTS: dict[RunnerType, list[ConvertType]] = {
    "generate": [],
    "pooling": ["embed"],
}

# Some model suffixes are based on auto classes from Transformers:
# https://huggingface.co/docs/transformers/en/model_doc/auto
# NOTE: Items higher on this list priority over lower ones
_SUFFIX_TO_DEFAULTS: list[tuple[str, tuple[RunnerType, ConvertType]]] = [
    ("ForCausalLM", ("generate", "none")),
    ("ForConditionalGeneration", ("generate", "none")),
    ("ChatModel", ("generate", "none")),
    ("LMHeadModel", ("generate", "none")),
    ("ForTextEncoding", ("pooling", "embed")),
    ("EmbeddingModel", ("pooling", "embed")),
    ("ForSequenceClassification", ("pooling", "classify")),
    ("ForAudioClassification", ("pooling", "classify")),
    ("ForImageClassification", ("pooling", "classify")),
    ("ForVideoClassification", ("pooling", "classify")),
    ("ClassificationModel", ("pooling", "classify")),
    ("ForRewardModeling", ("pooling", "reward")),
    ("RewardModel", ("pooling", "reward")),
    # Let other `*Model`s take priority
    ("Model", ("pooling", "embed")),
]


def iter_architecture_defaults():
    yield from _SUFFIX_TO_DEFAULTS


def try_match_architecture_defaults(
    architecture: str,
    *,
    runner_type: Optional[RunnerType] = None,
    convert_type: Optional[ConvertType] = None,
):
    for suffix, (default_runner_type, default_convert_type) in iter_architecture_defaults():
        if (
            (runner_type is None or runner_type == default_runner_type)
            and (convert_type is None or convert_type == default_convert_type)
            and architecture.endswith(suffix)
        ):
            return suffix, (default_runner_type, default_convert_type)
    return None


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
        "Ernie4_5ForCausalLM",  # 0.3B-PT
        "Ernie4_5_ForCausalLM",
        "Ernie4_5_MoeForCausalLM",
        "Ernie4_5_VLMoeForConditionalGeneration",
        "Ernie4_5_VLMoeForProcessRewardModel",
    }

    @classmethod
    def register_ernie_model_arch(cls, model_class):
        if model_class.name().startswith("Ernie") and model_class.name() not in cls.ARCHITECTURES:
            cls.ARCHITECTURES.add(model_class.name())

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
    "num_max_dispatch_tokens_per_rank": 128,
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
        self.is_moe_quantized = False
        self.max_model_len = 0
        self.dtype = "bfloat16"
        self.enable_logprob = False
        self.max_logprobs = 20
        self.logprobs_mode = "raw_logprobs"
        self.redundant_experts_num = 0
        self.seed = 0
        self.quantization = None
        self.pad_token_id: int = -1
        self.eos_tokens_lens: int = 2
        self.lm_head_fp32: bool = False
        self.model_format = "auto"
        self.runner = "auto"
        self.convert = "auto"
        self.pooler_config: Optional["PoolerConfig"] = field(init=False)
        self.override_pooler_config: Optional[Union[dict, "PoolerConfig"]] = None
        self.revision = None
        self.prefix_layer_name = "layers"
        self.kv_cache_quant_scale_path = ""

        self.partial_rotary_factor: float = 1.0
        self.num_nextn_predict_layers = 0
        for key, value in args.items():
            if hasattr(self, key) and value != "None":
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

        self.ori_vocab_size = args.get("ori_vocab_size", self.vocab_size)
        self.think_end_id = args.get("think_end_id", -1)
        self.im_patch_id = args.get("image_patch_id", -1)
        self.line_break_id = args.get("line_break_id", -1)
        if self.max_logprobs < -1:
            raise ValueError(" The possible values for max_logprobs can't be less than -1 ")

        self._post_init()

    def _post_init(self):
        self.is_unified_ckpt = check_unified_ckpt(self.model)
        self.runner_type = self._get_runner_type(self.architectures, self.runner)
        self.convert_type = self._get_convert_type(self.architectures, self.runner_type, self.convert)
        registry = self.registry
        is_generative_model = registry.is_text_generation_model(self.architectures, self)
        is_pooling_model = registry.is_pooling_model(self.architectures, self)
        is_multimodal_model = registry.is_multimodal_model(self.architectures, self)
        self.is_reasoning_model = registry.is_reasoning_model(self.architectures, self)

        self.enable_mm = is_multimodal_model

        self.kv_cache_quant_scale_path = os.path.join(self.model, "kv_cache_scale.json")
        if self.runner_type == "pooling":
            os.environ["FD_USE_GET_SAVE_OUTPUT_V1"] = "1"

        if self.runner_type == "generate" and not is_generative_model:
            if is_multimodal_model:
                pass
            else:
                generate_converts = _RUNNER_CONVERTS["generate"]
                if self.convert_type not in generate_converts:
                    raise ValueError("This model does not support '--runner generate.")
        if self.runner_type == "pooling" and not is_pooling_model:
            pooling_converts = _RUNNER_CONVERTS["pooling"]
            if self.convert_type not in pooling_converts:
                convert_option = "<" + "|".join(pooling_converts) + ">"
                raise ValueError(
                    "This model does not support `--runner pooling`. "
                    f"You can pass `--convert {convert_option} to adapt "
                    "it into a pooling model."
                )

        self.supported_tasks = self._get_supported_tasks(self.architectures, self.runner_type, self.convert_type)
        model_info, arch = registry.inspect_model_cls(self.architectures, self)
        self._model_info = model_info
        self._architecture = arch

        self.pooler_config = self._init_pooler_config()
        self.override_name_from_config()
        self.read_from_env()
        self.read_model_config()

    @property
    def registry(self):
        from fastdeploy.model_executor.models.model_base import ModelRegistry

        return ModelRegistry()

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

        if hasattr(self, "num_experts") and getattr(self, "moe_num_experts") is None:
            self.moe_num_experts = self.num_experts
        if hasattr(self, "n_routed_experts") and getattr(self, "moe_num_experts") is None:
            self.moe_num_experts = self.n_routed_experts

    def read_from_env(self):
        """
        Read configuration information from environment variables and update the object's attributes.
        If an attribute is not present or is an empty string in the environment variables, use the default value.
        """
        self.max_stop_seqs_num = envs.FD_MAX_STOP_SEQS_NUM
        self.stop_seqs_max_len = envs.FD_STOP_SEQS_MAX_LEN

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

    def read_model_config(self):
        config_path = os.path.join(self.model, "config.json")
        if os.path.exists(config_path):
            self.model_config = json.load(open(config_path, "r", encoding="utf-8"))
            if "torch_dtype" in self.model_config and "dtype" in self.model_config:
                raise ValueError(
                    "Only one of 'torch_dtype' or 'dtype' should be present in config.json. "
                    "Found both, which indicates an ambiguous model format. "
                    "Please ensure your config.json contains only one dtype field."
                )
            elif "torch_dtype" in self.model_config:
                self.model_format = "torch"
                logger.info("The model format is Hugging Face")
            elif "dtype" in self.model_config:
                self.model_format = "paddle"
                logger.info("The model format is Paddle")
            else:
                raise ValueError(
                    "Unknown model format. Please ensure your config.json contains "
                    "either 'torch_dtype' (for Hugging Face models) or 'dtype' (for Paddle models) field. "
                    f"Config file path: {config_path}"
                )

    def _get_default_runner_type(
        self,
        architectures: list[str],
    ) -> RunnerType:
        registry = self.registry
        if get_pooling_config(self.model, self.revision):
            return "pooling"
        for arch in architectures:
            if arch in registry.get_supported_archs():
                if registry.is_pooling_model(architectures, self):
                    return "pooling"
                if registry.is_text_generation_model(architectures, self):
                    return "generate"
            match = try_match_architecture_defaults(arch)
            if match:
                _, (runner_type, _) = match
                return runner_type
        return "generate"

    def _get_default_convert_type(
        self,
        architectures: list[str],
        runner_type: RunnerType,
    ) -> ConvertType:
        registry = self.registry

        for arch in architectures:
            if arch in registry.get_supported_archs():
                if runner_type == "generate" and registry.is_text_generation_model(architectures, self):
                    return "none"
                if runner_type == "pooling" and registry.is_pooling_model(architectures, self):
                    return "none"
            match = try_match_architecture_defaults(arch, runner_type=runner_type)
            if match:
                _, (_, convert_type) = match
                return convert_type

        # This is to handle Sentence Transformers models that use *ForCausalLM
        # and also multi-modal pooling models which are not defined as
        # Sentence Transformers models
        if runner_type == "pooling":
            return "embed"

        return "none"

    def _get_runner_type(
        self,
        architectures: list[str],
        runner: RunnerOption,
    ) -> RunnerType:
        if runner != "auto":
            return runner

        runner_type = self._get_default_runner_type(architectures)
        if runner_type != "generate":
            logger.info(
                "Resolved `--runner auto` to `--runner %s`. " "Pass the value explicitly to silence this message.",
                runner_type,
            )

        return runner_type

    def _get_convert_type(
        self,
        architectures: list[str],
        runner_type: RunnerType,
        convert: ConvertOption,
    ) -> ConvertType:
        if convert != "auto":
            return convert

        convert_type = self._get_default_convert_type(architectures, runner_type)

        if convert_type != "none":
            logger.info(
                "Resolved `--convert auto` to `--convert %s`. " "Pass the value explicitly to silence this message.",
                convert_type,
            )

        return convert_type

    def _get_supported_generation_tasks(
        self,
        architectures: list[str],
        convert_type: ConvertType,
    ) -> list[_ResolvedTask]:
        registry = self.registry

        supported_tasks = list[_ResolvedTask]()
        if registry.is_text_generation_model(architectures, self) or convert_type in _RUNNER_CONVERTS["generate"]:
            supported_tasks.append("generate")

        # TODO:Temporarily does not support transcription.
        return supported_tasks

    def _get_default_pooling_task(
        self,
        architectures: list[str],
    ) -> Literal["embed"]:
        # Temporarily does not support classification and reward.
        for arch in architectures:
            match = try_match_architecture_defaults(arch, runner_type="pooling")
            if match:
                _, (_, convert_type) = match
                assert convert_type != "none"
                return convert_type

        return "embed"

    def _get_supported_pooling_tasks(
        self,
        architectures: list[str],
        convert_type: ConvertType,
    ) -> list[_ResolvedTask]:
        registry = self.registry

        supported_tasks = list[_ResolvedTask]()
        if registry.is_pooling_model(architectures, self) or convert_type in _RUNNER_CONVERTS["pooling"]:
            supported_tasks.append("encode")

            extra_task = self._get_default_pooling_task(architectures) if convert_type == "none" else convert_type
            supported_tasks.append(extra_task)

        return supported_tasks

    def _get_supported_tasks(
        self,
        architectures: list[str],
        runner_type: RunnerType,
        convert_type: ConvertType,
    ) -> list[_ResolvedTask]:
        if runner_type == "generate":
            return self._get_supported_generation_tasks(architectures, convert_type)
        if runner_type == "pooling":
            return self._get_supported_pooling_tasks(architectures, convert_type)

        assert_never(runner_type)

    def _init_pooler_config(self) -> Optional["PoolerConfig"]:
        if self.runner_type == "pooling":
            if isinstance(self.override_pooler_config, dict):
                self.override_pooler_config = PoolerConfig(**self.override_pooler_config)

            pooler_config = self.override_pooler_config or PoolerConfig()

            base_config = get_pooling_config(self.model, self.revision)
            if base_config is not None:
                for k, v in base_config.items():
                    if getattr(pooler_config, k) is None:
                        setattr(pooler_config, k, v)

            default_pooling_type = self._model_info.default_pooling_type
            if pooler_config.pooling_type is None:
                pooler_config.pooling_type = default_pooling_type

            return pooler_config

        return None

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
        self.msg_queue_id = 1  # message queue id

        self.tensor_parallel_rank = 0  # TP rank ID
        self.tensor_parallel_size = 1  # TP degree
        self.expert_parallel_rank = 0  # EP rank ID
        self.expert_parallel_size = 1  # EP degree
        self.data_parallel_size = 1  # DP degree
        self.enable_expert_parallel = False
        self.local_data_parallel_id = 0
        # Engine worker queue port
        self.engine_worker_queue_port: str = "9923"
        # cuda visible devices
        self.device_ids: str = "0"
        # First token id
        self.first_token_id: int = 1
        # Process ID of engine
        self.engine_pid: Optional[int] = None
        # Do profile or not
        self.do_profile: bool = False
        # Use internode_ll_two_stage or not
        self.use_internode_ll_two_stage: bool = False
        # disable sequence parallel moe
        self.disable_sequence_parallel_moe: bool = False

        self.pod_ip: str = None
        # enable the custom all-reduce kernel and fall back to NCCL(dist.all_reduce).
        self.disable_custom_all_reduce: bool = False
        for key, value in args.items():
            if hasattr(self, key):
                setattr(self, key, value)
        if isinstance(self.engine_worker_queue_port, str):
            self.engine_worker_queue_port = [int(port) for port in self.engine_worker_queue_port.split(",")]
            logger.info(f"engine_worker_queue_port: {self.engine_worker_queue_port}")
        elif isinstance(self.engine_worker_queue_port, int):
            self.engine_worker_queue_port = [self.engine_worker_queue_port]
        # currently, the expert parallel size is equal data parallel size
        if self.enable_expert_parallel:
            self.expert_parallel_size = self.data_parallel_size * self.tensor_parallel_size
        else:
            self.expert_parallel_size = 1
        self.use_ep = self.expert_parallel_size > 1

        # pd_disaggregation
        use_pd_disaggregation: int = int(os.getenv("FLAGS_use_pd_disaggregation", 0))
        use_pd_disaggregation_per_chunk: int = int(os.getenv("FLAGS_use_pd_disaggregation_per_chunk", 0))
        if use_pd_disaggregation_per_chunk:
            self.pd_disaggregation_mode = "per_chunk"
        elif use_pd_disaggregation:
            self.pd_disaggregation_mode = "per_query"
        else:
            self.pd_disaggregation_mode = "None"

        # disable_sequence_parallel_moe: qkv_linear + attn + out_linear + allreduce
        # use_sequence_parallel_moe: allgather + qkv_linear + attn + all2all + out_linear
        self.use_sequence_parallel_moe = (
            (not self.disable_sequence_parallel_moe)
            and self.expert_parallel_size > 1
            and self.tensor_parallel_size > 1
        )
        logger.info(f"use_sequence_parallel_moe: {self.use_sequence_parallel_moe}")

    def set_communicate_group(self):
        # different tp group id
        # prevent different tp_groups using the same group_id
        tp_gid_offset = envs.FD_TP_GROUP_GID_OFFSET
        dist.collective._set_custom_gid(self.data_parallel_rank + tp_gid_offset)

        self.tp_group = dist.new_group(
            range(
                self.data_parallel_rank * self.tensor_parallel_size,
                (self.data_parallel_rank + 1) * self.tensor_parallel_size,
            )
        )
        dist.collective._set_custom_gid(None)
        # same ep group id
        if self.enable_expert_parallel:
            dist.collective._set_custom_gid(self.data_parallel_size + tp_gid_offset)
            self.ep_group = dist.new_group(range(self.expert_parallel_size))
            dist.collective._set_custom_gid(None)
        logger.info(
            f"data_parallel_size: {self.data_parallel_size}, tensor_parallel_size: {self.tensor_parallel_size}, expert_parallel_size: {self.expert_parallel_size}, data_parallel_rank: {self.data_parallel_rank}, tensor_parallel_rank: {self.tensor_parallel_rank}, expert_parallel_rank: {self.expert_parallel_rank}, tp_group: {self.tp_group}."
        )

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
        self.method_list = ["ngram_match", "mtp"]
        self.mtp_strategy_list = ["default", "with_ngram"]

        # speculative method, choose in [None, "ngram_match", "mtp", "hybrid_mtp_ngram"]
        self.method: Optional[str] = None
        # mtp strategy in mtp-method
        self.mtp_strategy = "default"
        # the max length of speculative tokens
        self.num_speculative_tokens: int = 1
        # the model runner step of draft model/mtp...
        self.num_model_steps: int = 1
        # the max length of candidate tokens for speculative method
        self.max_candidate_len: int = 5
        # the max length of verify window for speculative method
        self.verify_window: int = 2
        # ngram match
        self.max_ngram_size: int = 5
        self.min_ngram_size: int = 2
        # model for mtp/eagle/draft_model
        self.model: Optional[str] = None
        # quantization of model
        self.quantization: Optional[Dict[str, Any]] = None
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

    def check_legality_parameters(
        self,
    ) -> None:
        """Check the legality of parameters passed in from the command line"""
        if self.method is not None:
            assert (
                self.method in self.method_list
            ), f"speculative method only support {self.method_list} now, but get {self.method}."

            assert (
                self.num_speculative_tokens >= 1 and self.num_speculative_tokens <= 5
            ), f"num_speculative_tokens only support in range[1, 5], but get {self.num_speculative_tokens}."
            assert (
                self.num_model_steps >= 1 and self.num_model_steps <= 5
            ), f"num_model_steps only support in range[1, 5], but get {self.num_model_steps}."

            if self.method in ["mtp", "hybrid_mtp_ngram"]:
                if self.num_speculative_tokens < self.num_model_steps:
                    logger.warning(
                        f"Get num_model_steps > num_speculative_tokens. Reset num_speculative_tokens to {self.num_model_steps}"
                    )
                    self.num_speculative_tokens = self.num_model_steps

            assert (
                self.mtp_strategy in self.mtp_strategy_list
            ), f"mtp_strategy_list only support {self.mtp_strategy_list}, but get {self.mtp_strategy}"

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
            - With static graph backend: WIP
        """
        self.sot_warmup_sizes: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 32, 64, 128]
        """  Number of warmup runs for SOT warmup. """
        self.use_cudagraph: bool = True
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
        self.cudagraph_only_prefill: bool = False
        """When cudagraph_only_prefill is False, only capture decode-only.
        When cudagraph_only_prefill is True, only capture prefill-only.
        Now don't support capture both decode-only and prefill-only"""
        self.full_cuda_graph: bool = True

        """ Maximum CUDA Graph capture size """
        self.max_capture_size: int = None
        """ Record maps mapped from real shape to captured size to reduce runtime overhead """
        self.real_shape_to_captured_size: dict[int, int] = None
        """ Whether to use shared memory pool for multi capture_size """
        self.use_unique_memory_pool: bool = True
        """ Whether to use cudagraph for draft model."""
        self.draft_model_use_cudagraph: bool = False

        # CINN Config ...
        if args is not None:
            for key, value in args.items():
                if hasattr(self, key):
                    setattr(self, key, value)

        self.check_legality_parameters()

    def init_with_cudagrpah_size(self, max_capture_size: int = 0) -> None:
        """
        Initialize cuda graph capture sizes and
        pre-compute the mapping from batch size to padded graph size
        """
        # Regular capture sizes
        self.cudagraph_capture_sizes = [size for size in self.cudagraph_capture_sizes if size <= max_capture_size]
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

        # Pre-compute the mapping from shape to padded graph size
        self.real_shape_to_captured_size = {}
        for end, start in zip(self.cudagraph_capture_sizes, self.cudagraph_capture_sizes[1:] + [0]):
            for bs in range(start, end):
                if bs == start:
                    self.real_shape_to_captured_size[bs] = start
                else:
                    self.real_shape_to_captured_size[bs] = end
        self.real_shape_to_captured_size[self.max_capture_size] = self.max_capture_size

    def _set_cudagraph_sizes(self, max_capture_size: int = 0):
        """
        Calculate a series of candidate capture sizes,
        and then extract a portion of them as the capture list for the CUDA graph based on user input.
        """
        # Shape [1, 2, 4, 8, 16, ... 120, 128]
        draft_capture_sizes = [1, 2, 4] + [8 * i for i in range(1, 17)]
        # Shape [128, 144, ... 240, 256]
        draft_capture_sizes += [16 * i for i in range(9, 17)]
        # Shape [256, 288, ... 992, 1024]
        draft_capture_sizes += [32 * i for i in range(9, 33)]

        draft_capture_sizes.append(max_capture_size)
        self.cudagraph_capture_sizes = sorted(draft_capture_sizes)

    def filter_capture_size(self, tp_size: int = 1):
        """When TSP is used, capture size must be divisible by tp size."""
        self.cudagraph_capture_sizes = [
            draft_size for draft_size in self.cudagraph_capture_sizes if (draft_size % tp_size == 0)
        ]

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


class PlasAttentionConfig:
    def __init__(
        self,
        args,
    ):
        self.plas_encoder_top_k_left: int = None
        self.plas_encoder_top_k_right: int = None
        "The sparse topk of encoder attention is located at [plas_encoder_top_k_left, plas_encoder top_k_right]"
        self.plas_decoder_top_k_left: int = None
        self.plas_decoder_top_k_right: int = None
        "The sparse topk of decoder attention is located at [plas_decoder_top_k_left, plas_decoder top_k_right]"
        self.plas_use_encoder_seq_limit: int = None
        "When the number of encdoer token is less than plas_use_encoder_seq_limit, it is not sparse"
        self.plas_use_decoder_seq_limit: int = None
        "When the number of decdoer token is less than plas_use_decoder_seq_limit, it is not sparse"
        self.plas_block_size: int = 128
        self.mlp_weight_name: str = "plas_attention_mlp_weight.safetensors"
        self.plas_max_seq_length: int = 128 * 1024
        if args is not None:
            for key, value in args.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            if self.plas_use_encoder_seq_limit is None and self.plas_encoder_top_k_left is not None:
                self.plas_use_encoder_seq_limit = self.plas_encoder_top_k_left * self.plas_block_size
            if self.plas_use_decoder_seq_limit is None and self.plas_decoder_top_k_left is not None:
                self.plas_use_decoder_seq_limit = self.plas_decoder_top_k_left * self.plas_block_size
            self.check_legality_parameters()

    def check_legality_parameters(
        self,
    ) -> None:
        if self.plas_encoder_top_k_left is not None:
            assert self.plas_encoder_top_k_left > 0, "plas_encoder_top_k_left must large than 0"

        if self.plas_encoder_top_k_right is not None:
            assert self.plas_encoder_top_k_right > 0, "plas_encoder_top_k_right must large than 0"
            assert (
                self.plas_encoder_top_k_right >= self.plas_encoder_top_k_left
            ), "plas_encoder_top_k_right must large than plas_encoder_top_k_left"

        if self.plas_decoder_top_k_left is not None:
            assert self.plas_decoder_top_k_left > 0, "plas_decoder_top_k_left must large than 0"

        if self.plas_decoder_top_k_right is not None:
            assert self.plas_decoder_top_k_right > 0, "plas_decoder_top_k_right must large than 0"
            assert (
                self.plas_decoder_top_k_right >= self.plas_decoder_top_k_left
            ), "plas_decoder_top_k_right must large than plas_decoder_top_k_left"

        if self.plas_use_encoder_seq_limit is not None and self.plas_encoder_top_k_left is not None:
            assert self.plas_use_encoder_seq_limit >= self.plas_encoder_top_k_left * self.plas_block_size
        if self.plas_use_decoder_seq_limit is not None and self.plas_decoder_top_k_left is not None:
            assert self.plas_use_decoder_seq_limit >= self.plas_decoder_top_k_left * self.plas_block_size

    def to_json_string(self):
        """
        Convert plas_attention_config to json string.
        """
        return json.dumps({key: value for key, value in self.__dict__.items() if value is not None})

    def __str__(self) -> str:
        return json.dumps({key: value for key, value in self.__dict__.items()})


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
    DEFAULT_V1 = "default_v1"


class LoadConfig:
    """
    Configuration for dynamic weight loading strategies

    Attributes:
        dynamic_load_weight: Whether to enable dynamic weight loading
        load_strategy: Specifies the weight loading method when enabled:
            - 'ipc': Real-time IPC streaming with automatic resharding
            - 'ipc_snapshot': Load from disk snapshot of IPC weights
            - 'meta': Only model meta messages
            - None: No dynamic loading
    """

    def __init__(
        self,
        args,
    ):
        self.load_choices: Union[str, LoadChoices] = LoadChoices.DEFAULT.value
        self.use_fastsafetensor = int(envs.FD_USE_FASTSAFETENSOR) == 1
        self.dynamic_load_weight: bool = False
        self.load_strategy: Optional[Literal["ipc", "ipc_snapshot", "meta", "normal"]] = "normal"
        for key, value in args.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def __str__(self) -> str:
        return json.dumps({key: value for key, value in self.__dict__.items()})


class PoolerConfig:
    """Controls the behavior of output pooling in pooling models."""

    pooling_type: Optional[str] = None
    """
    The pooling method of the pooling model.
    """
    # for embeddings models
    normalize: Optional[bool] = None
    """
    Whether to normalize the embeddings outputs. Defaults to True.
    """
    dimensions: Optional[int] = None
    """
    Reduce the dimensions of embeddings if model
    support matryoshka representation. Defaults to None.
    """
    enable_chunked_processing: Optional[bool] = None
    """
    Whether to enable chunked processing for long inputs that exceed the model's
    maximum position embeddings. When enabled, long inputs will be split into
    chunks, processed separately, and then aggregated using weighted averaging.
    This allows embedding models to handle arbitrarily long text without CUDA
    errors. Defaults to False.
    """
    max_embed_len: Optional[int] = None
    """
    Maximum input length allowed for embedding generation. When set, allows
    inputs longer than max_embed_len to be accepted for embedding models.
    When an input exceeds max_embed_len, it will be handled according to
    the original max_model_len validation logic.
    Defaults to None (i.e. set to max_model_len).
    """


class EPLBConfig:
    """
    Configuration for EPLB manager.
    """

    def __init__(
        self,
        args,
    ):
        if args is None:
            args = {}

        # enable eplb
        self.enable_eplb: bool = False
        # redundant experts num
        self.redundant_experts_num: int = 0
        # expert ip shm size
        self.redundant_expert_ip_shm_size: int = 1024
        # expert meta dir
        self.redundant_expert_meta_dir: str = "/tmp/redundant_expert_meta"
        # expert api user and password
        self.redundant_expert_api_user: str = ""
        self.redundant_expert_api_password: str = ""
        # expert eplb strategy
        self.redundant_expert_eplb_strategy: str = ""
        # expert dump workload interval
        self.redundant_expert_dump_workload_interval: int = 10
        # expert async load model shmem size gb
        self.redundant_expert_async_load_model_shmem_size_gb: int = 0
        # expert enable schedule cordon
        self.redundant_expert_enable_schedule_cordon: bool = True
        # model use safetensors
        self.model_use_safetensors: bool = True
        # model use offline quant
        self.model_use_offline_quant: bool = True
        # moe quant type
        self.moe_quant_type: str = "w4a8"
        for key, value in args.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_json_string(self):
        """
        Convert eplb_config to json string.
        """
        return json.dumps({key: value for key, value in self.__dict__.items() if value is not None})

    def print(self):
        """
        Print all configuration information.
        """
        logger.info("EPLB Configuration Information :")
        for k, v in self.__dict__.items():
            logger.info("{:<20}:{:<6}{}".format(k, "", v))
        logger.info("=============================================================")


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
            max_encoder_cache(int): Maximum number of tokens in the encoder cache.
            max_processor_cache(int): Maximum number of bytes in the processor cache.
        """
        self.block_size = 64
        self.gpu_memory_utilization = 0.9
        self.num_gpu_blocks_override = None
        if envs.ENABLE_V1_KVCACHE_SCHEDULER:
            self.kv_cache_ratio = 1.0
        else:
            self.kv_cache_ratio = 0.75
        self.enc_dec_block_num = envs.FD_ENC_DEC_BLOCK_NUM
        self.prealloc_dec_block_slot_num_threshold = 12
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
        self.max_encoder_cache = None
        self.max_processor_cache = None
        self.disable_chunked_mm_input = False
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
            if self.model_cfg.quantization is not None and isinstance(self.model_cfg.quantization, dict):
                self.cache_dtype = self.model_cfg.quantization.get("kv_cache_quant_type", self.cache_dtype)
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
            if envs.ENABLE_V1_KVCACHE_SCHEDULER:
                self.prefill_kvcache_block_num = self.total_block_num
            else:
                self.prefill_kvcache_block_num = int(self.total_block_num * self.kv_cache_ratio)
            assert (
                self.prefill_kvcache_block_num >= self.max_block_num_per_seq
            ), f"current block number :{self.prefill_kvcache_block_num} should be greater than or equal to current model len needed minimum block number :{self.max_block_num_per_seq}"
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
        if envs.ENABLE_V1_KVCACHE_SCHEDULER:
            self.prefill_kvcache_block_num = self.total_block_num
        else:
            self.prefill_kvcache_block_num = int(self.total_block_num * self.kv_cache_ratio)
        logger.info(
            f"Reset block num, the total_block_num:{self.total_block_num},"
            f" prefill_kvcache_block_num:{self.prefill_kvcache_block_num}"
        )
        assert (
            self.prefill_kvcache_block_num >= self.max_block_num_per_seq
        ), f"current block number :{self.prefill_kvcache_block_num} should be greater than or equal to current model len needed minimum block number :{self.max_block_num_per_seq}"

    def print(self):
        """
        print all config

        """
        logger.info("Cache Configuration Information :")
        for k, v in self.__dict__.items():
            logger.info("{:<20}:{:<6}{}".format(k, "", v))
        logger.info("=============================================================")


class RouterConfig:
    """
    Configuration for router
    Attributes:
        router: the url of router, such as http://127.0.0.1:8000
        api_server_host: the host ip of model server
        api_server_port: the http port of model server
    """

    def __init__(self, args: dict):
        self.router = args["router"]
        if self.router is not None and not self.router.startswith(("http://", "https://")):
            self.router = f"http://{self.router}"

        self.api_server_host = get_host_ip()
        self.api_server_port = args["port"]


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


class StructuredOutputsConfig:
    """
    Configuration for structured outputs
    """

    def __init__(
        self,
        args,
    ) -> None:
        self.reasoning_parser: Optional[str] = None
        self.guided_decoding_backend: Optional[str] = None
        # disable any whitespace for guided decoding
        self.disable_any_whitespace: bool = True
        self.logits_processors: Optional[list[str]] = None
        for key, value in args.items():
            if hasattr(self, key) and value != "None":
                setattr(self, key, value)

    def __str__(self) -> str:
        return json.dumps({key: value for key, value in self.__dict__.items()})


class FDConfig:
    """
    The configuration class which contains all fastdeploy-related configuration. This
    simplifies passing around the distinct configurations in the codebase.
    """

    def __init__(
        self,
        model_config: ModelConfig = None,
        cache_config: CacheConfig = None,
        parallel_config: ParallelConfig = None,
        load_config: LoadConfig = None,
        commit_config: CommitConfig = CommitConfig(),
        scheduler_config: SchedulerConfig = None,
        device_config: DeviceConfig = None,
        quant_config: QuantConfigBase = None,
        graph_opt_config: GraphOptimizationConfig = None,
        plas_attention_config: PlasAttentionConfig = None,
        speculative_config: SpeculativeConfig = None,
        eplb_config: EPLBConfig = None,
        structured_outputs_config: StructuredOutputsConfig = None,
        router_config: RouterConfig = None,
        tokenizer: str = None,
        ips: str = None,
        use_warmup: bool = False,
        limit_mm_per_prompt: Optional[Dict[str, Any]] = None,
        mm_processor_kwargs: Optional[Dict[str, Any]] = None,
        max_num_partial_prefills: int = 1,
        max_long_partial_prefills: int = 1,
        long_prefill_token_threshold: int = 0,
        early_stop_config: Optional[Dict[str, Any]] = None,
        tool_parser: str = None,
        test_mode=False,
    ):
        self.model_config: ModelConfig = model_config  # type: ignore
        self.cache_config: CacheConfig = cache_config  # type: ignore
        self.scheduler_config: SchedulerConfig = scheduler_config  # type: ignore
        self.parallel_config = parallel_config  # type: ignore
        self.speculative_config: SpeculativeConfig = speculative_config
        self.eplb_config: Optional[EPLBConfig] = eplb_config
        self.device_config: DeviceConfig = device_config  # type: ignore
        self.load_config: LoadConfig = load_config
        self.quant_config: Optional[QuantConfigBase] = quant_config
        self.graph_opt_config: Optional[GraphOptimizationConfig] = graph_opt_config
        self.early_stop_config: Optional[EarlyStopConfig] = early_stop_config
        self.cache_config: CacheConfig = cache_config  # type: ignore
        self.plas_attention_config: Optional[PlasAttentionConfig] = plas_attention_config
        self.structured_outputs_config: StructuredOutputsConfig = structured_outputs_config
        self.router_config: RouterConfig = router_config

        # Initialize cuda graph capture list
        max_capture_shape = self.scheduler_config.max_num_seqs
        if self.speculative_config is not None and self.speculative_config.method == "mtp":
            max_capture_shape = self.scheduler_config.max_num_seqs * (
                self.speculative_config.num_speculative_tokens + 1
            )
            assert max_capture_shape % 2 == 0, "CUDAGraph only supports capturing even token nums in MTP scenarios."
        if self.graph_opt_config.cudagraph_only_prefill:
            max_capture_shape = 512
        else:
            max_capture_shape = min(512, max_capture_shape)

        if self.graph_opt_config.cudagraph_capture_sizes is None:
            self.graph_opt_config._set_cudagraph_sizes(max_capture_size=max_capture_shape)
        self.graph_opt_config.init_with_cudagrpah_size(max_capture_size=max_capture_shape)

        self.tokenizer = tokenizer
        self.ips = ips
        self.tool_parser = tool_parser

        if self.ips is None:
            self.master_ip = "0.0.0.0"
        elif isinstance(self.ips, str):
            self.ips = self.ips.split(",")

        self.host_ip = get_host_ip()

        if self.ips is None:
            self.nnode = 1
            self.node_rank = 0
        else:
            self.nnode = len(self.ips)

            for idx, ip in enumerate(self.ips):
                if ip == self.host_ip:
                    self.node_rank = idx

        self.limit_mm_per_prompt = limit_mm_per_prompt
        self.mm_processor_kwargs = mm_processor_kwargs
        self.use_warmup = use_warmup
        self.max_num_partial_prefills = max_num_partial_prefills
        self.max_long_partial_prefills = max_long_partial_prefills
        self.long_prefill_token_threshold = long_prefill_token_threshold

        if envs.FD_FOR_TORCH_MODEL_FORMAT:
            self.model_config.model_format = "torch"

        # TODO
        if not envs.FD_ENABLE_MAX_PREFILL:
            self.max_prefill_batch = int(os.getenv("MAX_PREFILL_NUM", "3"))
            if current_platform.is_xpu():
                self.max_prefill_batch = 1
            if self.model_config is not None and self.model_config.enable_mm:
                self.max_prefill_batch = 1  # TODO:当前多模prefill阶段只支持并行度为1,待优化
        else:
            self.max_prefill_batch = self.scheduler_config.max_num_seqs

        num_ranks = self.parallel_config.tensor_parallel_size * self.parallel_config.data_parallel_size
        self.max_chips_per_node = 16 if current_platform.is_iluvatar() else 8
        if num_ranks > self.max_chips_per_node and self.load_config.load_strategy != "meta":
            self.worker_num_per_node = self.max_chips_per_node
            nnode = ceil_div(num_ranks, self.worker_num_per_node)
            assert nnode == self.nnode, f"nnode: {nnode}, but got {self.nnode}"
        else:
            self.worker_num_per_node = num_ranks

        self.parallel_config.device_ids = ",".join([str(i) for i in range(self.worker_num_per_node)])
        self.parallel_config.device_ids = os.getenv("CUDA_VISIBLE_DEVICES", self.parallel_config.device_ids)
        if current_platform.is_xpu():
            self.parallel_config.device_ids = os.getenv("XPU_VISIBLE_DEVICES", self.parallel_config.device_ids)
        if current_platform.is_intel_hpu():
            self.parallel_config.device_ids = os.getenv("HPU_VISIBLE_DEVICES", self.parallel_config.device_ids)

        self.read_from_config()
        self.postprocess()
        self.init_cache_info()
        if test_mode:
            return
        self.check()
        self.print()

    def _disable_sequence_parallel_moe_if_needed(self, mode_name):
        if self.parallel_config.use_sequence_parallel_moe and self.graph_opt_config.use_cudagraph:
            self.parallel_config.use_sequence_parallel_moe = False
            logger.warning(
                f"Sequence parallel MoE does not support {mode_name} mode with cudagraph. "
                "Setting use_sequence_parallel_moe to False."
            )

    def postprocess(self):
        """
        calculate some parameters
        """
        self.local_device_ids = self.parallel_config.device_ids.split(",")[: self.parallel_config.tensor_parallel_size]

        if self.parallel_config.tensor_parallel_size <= self.worker_num_per_node or self.node_rank == 0:
            self.is_master = True
            self.master_ip = "0.0.0.0"
        else:
            self.is_master = False
            self.master_ip = self.ips[0]

        self.paddle_commit_id = paddle.version.commit

        if self.scheduler_config.max_num_batched_tokens is None:
            if int(envs.ENABLE_V1_KVCACHE_SCHEDULER):
                if paddle.is_compiled_with_xpu():
                    self.scheduler_config.max_num_batched_tokens = self.model_config.max_model_len
                else:
                    self.scheduler_config.max_num_batched_tokens = 8192  # if set to max_model_len, it's easy to be OOM
            else:
                if self.cache_config.enable_chunked_prefill:
                    self.scheduler_config.max_num_batched_tokens = 2048
                else:
                    self.scheduler_config.max_num_batched_tokens = self.model_config.max_model_len

        if self.long_prefill_token_threshold == 0:
            self.long_prefill_token_threshold = int(self.model_config.max_model_len * 0.04)

        self.cache_config.max_block_num_per_seq = int(self.model_config.max_model_len // self.cache_config.block_size)
        self.cache_config.postprocess(self.scheduler_config.max_num_batched_tokens, self.scheduler_config.max_num_seqs)
        if self.model_config is not None and self.model_config.enable_mm and not envs.ENABLE_V1_KVCACHE_SCHEDULER:
            self.cache_config.enable_prefix_caching = False

        if (
            self.structured_outputs_config is not None
            and self.structured_outputs_config.guided_decoding_backend == "auto"
        ):
            if current_platform.is_xpu() or self.speculative_config.method is not None:
                logger.warning("Speculative Decoding and XPU currently do not support Guided decoding, set off.")
                self.structured_outputs_config.guided_decoding_backend = "off"
            else:
                self.structured_outputs_config.guided_decoding_backend = "xgrammar"

        if self.model_config.enable_mm:
            if self.cache_config.max_encoder_cache is None or self.cache_config.max_encoder_cache < 0:
                self.cache_config.max_encoder_cache = self.scheduler_config.max_num_batched_tokens
            elif self.cache_config.max_encoder_cache != 0:
                if self.cache_config.max_encoder_cache < self.scheduler_config.max_num_batched_tokens:
                    logger.warning(
                        f"max_encoder_cache{self.cache_config.max_encoder_cache} is less than "
                        f"max_num_batched_tokens{self.scheduler_config.max_num_batched_tokens}, "
                        f"set to max_num_batched_tokens."
                    )
                    self.cache_config.max_encoder_cache = self.scheduler_config.max_num_batched_tokens
        else:
            self.cache_config.max_encoder_cache = 0

        # Adjustment GraphOptConfig
        if self.scheduler_config is not None and self.scheduler_config.splitwise_role == "prefill":
            self.graph_opt_config.use_cudagraph = self.graph_opt_config.cudagraph_only_prefill
        if self.load_config is not None and self.load_config.dynamic_load_weight is True:
            self.graph_opt_config.graph_opt_level = 0
            logger.info(
                "Static Graph does not support to be started together with RL Training, and automatically switch to dynamic graph!"
            )
        if self.device_config is not None and self.device_config.device_type != "cuda":
            self.graph_opt_config.use_cudagraph = False
            logger.info(f"CUDAGraph only support on GPU, current device type is {self.device_config.device_type}!")
        if self.parallel_config.use_sequence_parallel_moe and self.graph_opt_config.use_cudagraph:
            if self.scheduler_config.max_num_seqs < self.parallel_config.tensor_parallel_size:
                self.parallel_config.use_sequence_parallel_moe = False
                logger.info(
                    "Warning: sequence parallel moe do not support max_num_seqs < tensor_parallel_size when cudagraph enabled. We set use_sequence_parallel_moe to False."
                )
            else:
                # It will hang when real batch_size < tp_size
                self.graph_opt_config.filter_capture_size(tp_size=self.parallel_config.tensor_parallel_size)
        if self.model_config.enable_mm and self.graph_opt_config.use_cudagraph:
            self.cache_config.enable_prefix_caching = False
            logger.info("Multi-modal models do not support prefix caching when using CUDAGraph!")

        if self.scheduler_config.splitwise_role == "mixed":
            self._disable_sequence_parallel_moe_if_needed("Mixed")
            self.model_config.moe_phase = MoEPhase(phase="prefill")
        elif self.scheduler_config.splitwise_role == "prefill":
            self.model_config.moe_phase = MoEPhase(phase="prefill")
        elif self.scheduler_config.splitwise_role == "decode":
            self._disable_sequence_parallel_moe_if_needed("PD's decode node")
            self.model_config.moe_phase = MoEPhase(phase="decode")
        else:
            raise NotImplementedError

    def check(self):
        """
        check the legality of config
        """
        assert self.scheduler_config.max_num_seqs <= 256, (
            "The parameter `max_num_seqs` is not allowed to exceed 256, "
            f"but now it's {self.scheduler_config.max_num_seqs}."
        )
        assert self.nnode >= 1, f"nnode: {self.nnode} should no less than 1"
        assert (
            self.model_config.max_model_len >= 16
        ), f"max_model_len: {self.model_config.max_model_len} should be larger than 16"
        assert (
            self.scheduler_config.max_num_seqs >= 1
        ), f"max_num_seqs: {self.scheduler_config.max_num_seqs} should be larger than 1"
        assert self.scheduler_config.max_num_batched_tokens >= self.scheduler_config.max_num_seqs, (
            f"max_num_batched_tokens: {self.scheduler_config.max_num_batched_tokens} "
            f"should be larger than or equal to max_num_seqs: {self.scheduler_config.max_num_seqs}"
        )
        assert (
            self.scheduler_config.max_num_batched_tokens
            <= self.model_config.max_model_len * self.scheduler_config.max_num_seqs
        ), (
            f"max_num_batched_tokens: {self.scheduler_config.max_num_batched_tokens} should be larger"
            f"than or equal to max_num_seqs: {self.scheduler_config.max_num_seqs} * max_model_len: {self.model_config.max_model_len}"
        )
        assert (
            self.max_num_partial_prefills >= 1
        ), f"max_num_partial_prefills: {self.max_num_partial_prefills} should be larger than or equal to 1"

        assert (
            self.max_long_partial_prefills >= 1
        ), f"max_long_partial_prefills: {self.max_long_partial_prefills} should be larger than or equal to 1"
        assert self.max_long_partial_prefills <= self.max_num_partial_prefills, (
            f"max_long_partial_prefills: {self.max_long_partial_prefills} should "
            f"be less than or equal to max_num_partial_prefills: {self.max_num_partial_prefills}"
        )
        assert self.scheduler_config.splitwise_role in ["mixed", "prefill", "decode"]

        if not self.cache_config.enable_chunked_prefill:
            if not envs.ENABLE_V1_KVCACHE_SCHEDULER:
                assert self.scheduler_config.max_num_batched_tokens >= self.model_config.max_model_len, (
                    f"max_num_batched_tokens: {self.scheduler_config.max_num_batched_tokens} "
                    f"should be larger than or equal to max_model_len: {self.model_config.max_model_len}"
                )
        else:
            assert self.scheduler_config.max_num_batched_tokens >= self.cache_config.block_size, (
                f"max_num_batched_tokens: {self.scheduler_config.max_num_batched_tokens} "
                f"should be larger than or equal to block_size: {self.cache_config.block_size}"
            )

        if self.max_num_partial_prefills > 1:
            assert (
                self.cache_config.enable_chunked_prefill is True
            ), "Chunked prefill must be enabled to set max_num_partial_prefills > 1"
            assert self.long_prefill_token_threshold < self.model_config.max_model_len, (
                f"long_prefill_token_threshold: {self.long_prefill_token_threshold} should be less than"
                f" max_model_len: {self.model_config.max_model_len}"
            )

        if (
            self.structured_outputs_config is not None
            and self.structured_outputs_config.guided_decoding_backend is not None
        ):
            assert self.structured_outputs_config.guided_decoding_backend in [
                "xgrammar",
                "XGrammar",
                "auto",
                "off",
            ], f"Only support xgrammar、auto guided decoding backend, but got {self.structured_outputs_config.guided_decoding_backend}."

            if self.structured_outputs_config.guided_decoding_backend != "off":
                # TODO: speculative decoding support guided_decoding
                assert (
                    self.speculative_config.method is None
                ), "speculative decoding currently do not support guided_decoding"

                # TODO: xpu support guided_decoding
                assert not current_platform.is_xpu(), "XPU currently do not support guided_decoding"

                try:
                    import xgrammar  # noqa
                except Exception as e:
                    raise Exception(
                        f"import XGrammar failed, please install XGrammar use `pip install xgrammar==0.1.19`. \n\t {e}"
                    )

        if self.scheduler_config is not None:
            self.scheduler_config.check()

        # Check graph optimization config
        if self.graph_opt_config.graph_opt_level > 0:
            if self.load_config is not None:
                assert (
                    self.load_config.dynamic_load_weight is False
                ), "Static graph cannot be used in RL scene temporarily"

        if int(envs.ENABLE_V1_KVCACHE_SCHEDULER) == 1:
            assert (
                int(envs.FD_DISABLED_RECOVER) == 0
            ), "FD_DISABLED_RECOVER is not supported while ENABLE_V1_KVCACHE_SCHEDULER is turned on."

        if self.eplb_config is not None and self.eplb_config.enable_eplb:
            try:
                import cuda  # noqa
            except ImportError:
                raise ImportError(
                    "cuda-python not installed. Install the version matching your CUDA toolkit:\n"
                    "  CUDA 12.x → pip install cuda-python==12.*\n"
                )

    def print(self):
        """
        print all config
        """
        logger.info("=================== Configuration Information ===============")
        for k, v in self.__dict__.items():
            if k == "generation_config" and v is not None:
                for gck, gcv in v.to_dict().items():
                    logger.info("{:<20}:{:<6}{}".format(gck, "", gcv))
            elif (
                k == "cache_config"
                or k == "model_config"
                or k == "scheduler_config"
                or k == "parallel_config"
                or k == "commit_config"
            ):
                if v is not None:
                    v.print()
            else:
                logger.info("{:<20}:{:<6}{}".format(k, "", v))
        logger.info("=============================================================")

    def init_cache_info(self):
        """
        initialize cache info
        """
        # TODO: group the splitiwse params
        # There are two methods for splitwise deployment:
        # 1. v0 splitwise_scheduler or dp_scheduler
        # 2. v1 local_scheduler + router
        self.splitwise_version = None
        if self.scheduler_config.name in ("splitwise", "dp"):
            self.splitwise_version = "v0"
        elif self.scheduler_config.name == "local" and self.router_config and self.router_config.router:
            self.splitwise_version = "v1"

        if isinstance(self.parallel_config.engine_worker_queue_port, (int, str)):
            engine_worker_queue_port = self.parallel_config.engine_worker_queue_port
        else:
            engine_worker_queue_port = self.parallel_config.engine_worker_queue_port[
                self.parallel_config.local_data_parallel_id
            ]
        connector_port = self.cache_config.pd_comm_port[0] if self.cache_config.pd_comm_port else None

        self.disaggregate_info = {}
        if self.scheduler_config.splitwise_role != "mixed":
            self.disaggregate_info["role"] = self.scheduler_config.splitwise_role
            self.disaggregate_info["cache_info"] = dict()
            current_protocol = self.cache_config.cache_transfer_protocol.split(",")
            self.disaggregate_info["transfer_protocol"] = current_protocol

            for protocol in current_protocol:
                if protocol == "ipc":
                    self.disaggregate_info["cache_info"][protocol] = {
                        "ip": self.host_ip,
                        "port": engine_worker_queue_port,
                        "device_ids": self.local_device_ids,
                    }
                elif protocol == "rdma":
                    self.disaggregate_info["cache_info"][protocol] = {
                        "ip": self.host_ip,
                        "port": connector_port,
                        "rdma_port": self.cache_config.rdma_comm_ports,
                    }
            logger.info(f"disaggregate_info: {self.disaggregate_info}")

        if self.router_config:
            self.register_info = {
                "role": self.scheduler_config.splitwise_role,
                "host_ip": self.host_ip,
                "port": self.router_config.api_server_port,
                "connector_port": connector_port,
                "rdma_ports": self.cache_config.rdma_comm_ports,
                "engine_worker_queue_port": engine_worker_queue_port,
                "device_ids": self.local_device_ids,
                "transfer_protocol": self.cache_config.cache_transfer_protocol.split(","),
            }
            logger.info(f"register_info: {self.register_info}")

    def read_from_config(self):
        """
        reset model config from json file
        """

        def reset_value(cls, value_name, key):
            if hasattr(cls, key):
                value = getattr(cls, key)
                setattr(cls, value_name, value)
                logger.info(f"Reset parameter {value_name} = {value} from configuration.")

        reset_value(self.cache_config, "block_size", "infer_model_block_size")
        reset_value(
            self.model_config,
            "return_full_hidden_states",
            "return_full_hidden_states",
        )
        reset_value(self.cache_config, "cache_dtype", "infer_model_dtype")

    def _check_master(self):
        return self.is_master

    def _str_to_list(self, attr_name, default_type):
        if hasattr(self, attr_name):
            val = getattr(self, attr_name)
            if val is None:
                return
            if type(val) is str:
                setattr(self, attr_name, [default_type(i) for i in val.split(",")])
            else:
                setattr(self, attr_name, [default_type(i) for i in val])

    def __str__(self) -> str:
        return json.dumps(self.__dict__, indent=4)
