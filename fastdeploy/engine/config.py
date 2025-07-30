"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastdeploy.config import (
    CacheConfig,
    CommitConfig,
    LoadConfig,
    ModelConfig,
    ParallelConfig,
)
from fastdeploy.platforms import current_platform
from fastdeploy.scheduler import SchedulerConfig
from fastdeploy.utils import ceil_div, get_host_ip, is_port_available, llm_logger


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
        load_choices(str):The format of the model weights to load. .Default is default
    """

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        scheduler_config: SchedulerConfig,
        parallel_config: ParallelConfig,
        load_config: LoadConfig,
        commit_config: CommitConfig = CommitConfig(),
        model_name_or_path: str = None,
        tokenizer: str = None,
        tensor_parallel_size: int = 8,
        max_model_len: int = 8192,
        max_num_seqs: int = 8,
        max_num_batched_tokens: Optional[int] = None,
        ips: str = None,
        speculative_config: Optional[Dict[str, Any]] = None,
        graph_optimization_config: Optional[Dict[str, Any]] = None,
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
        guided_decoding_backend: Optional[str] = None,
        disable_any_whitespace: bool = False,
        enable_logprob: bool = False,
        early_stop_config: Optional[Dict[str, Any]] = None,
        load_choices: str = "default",
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
            max_model_len (int): Maximum model length. Default is 8192.
            max_num_seqs (int): Maximum number of sequences. Default is 8.
            max_num_batched_tokens (Optional[int]): Maximum number of batched tokens. Default is None.
            mm_processor_kwargs (Optional[Dict[str, Any]]): Additional arguments for multi-modal processor. Default is None.
            speculative_config (Optional[Dict[str, Any]]): Speculative execution configuration. Default is None.
            graph_optimization_config (Optional[Dict[str, Any]]): Graph optimizaion backend execution configuration. Default is None.
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
            enable_logprob(bool): Enable logprob. Default is False.
            early_stop_config (Optional[Dict[str, Any]]): Early stop configuration. Default is None.
            load_choices(str):The format of the model weights to load. .Default is default
        """
        self.model_config = model_config
        self.cache_config = cache_config
        self.scheduler_config = scheduler_config
        self.parallel_config = parallel_config
        self.load_config = load_config
        self.commit_config = commit_config
        self.model_name_or_path = model_name_or_path
        self.tokenizer = tokenizer
        self.max_num_batched_tokens = max_num_batched_tokens
        self.tensor_parallel_size = tensor_parallel_size
        self.ips = ips

        if self.ips is None:
            self.master_ip = "0.0.0.0"
        elif isinstance(self.ips, list):
            self.master_ip = self.ips[0]
        else:
            self.ips = self.ips.split(",")
            self.master_ip = self.ips[0]

        if self.ips is None:
            self.nnode = 1
            self.node_rank = 0
        else:
            self.nnode = len(self.ips)

            for idx, ip in enumerate(self.ips):
                if ip == self.master_ip:
                    self.node_rank = idx

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
        self.graph_optimization_config = graph_optimization_config
        self.early_stop_config = early_stop_config
        self.guided_decoding_backend = guided_decoding_backend
        self.disable_any_whitespace = disable_any_whitespace
        self._str_to_list("innode_prefill_ports", int)
        self.load_choices = load_choices

        assert self.splitwise_role in ["mixed", "prefill", "decode"]

        # TODO
        self.max_prefill_batch = 3
        if current_platform.is_xpu():
            self.max_prefill_batch = 1
        if enable_mm:
            self.max_prefill_batch = 1  # TODO:当前多模prefill阶段只支持并行度为1,待优化

        # TODO(@wufeisheng): TP and EP need to be supported simultaneously.
        assert (self.tensor_parallel_size == 1 and self.parallel_config.expert_parallel_size >= 1) or (
            self.tensor_parallel_size >= 1 and self.parallel_config.expert_parallel_size == 1
        ), "TP and EP cannot be enabled at the same time"

        num_ranks = self.tensor_parallel_size * self.parallel_config.expert_parallel_size
        self.max_chips_per_node = 16 if current_platform.is_iluvatar() else 8
        if num_ranks > self.max_chips_per_node:
            self.worker_num_per_node = self.max_chips_per_node
            nnode = ceil_div(num_ranks, self.worker_num_per_node)
            assert nnode == self.nnode, f"nnode: {nnode}, but got {self.nnode}"
        else:
            self.worker_num_per_node = num_ranks

        self.engine_worker_queue_port = engine_worker_queue_port
        self.device_ids = ",".join([str(i) for i in range(self.worker_num_per_node)])
        self.device_ids = os.getenv("CUDA_VISIBLE_DEVICES", self.device_ids)
        if current_platform.is_xpu():
            self.device_ids = os.getenv("XPU_VISIBLE_DEVICES", self.device_ids)

        self.enable_logprob = enable_logprob

        self.read_from_config()
        self.postprocess()
        self.check()
        self.print()

    def postprocess(self):
        """
        calculate some parameters
        """
        assert (
            self.device_ids.split(",").__len__() == self.worker_num_per_node
        ), f"invalid CUDA_VISIBLE_DEVICES, should be equal to {self.worker_num_per_node}"

        self.local_device_ids = self.device_ids.split(",")[: self.tensor_parallel_size]

        self.host_ip = get_host_ip()

        if self.ips is None or self.host_ip == self.master_ip:
            self.is_master = True
        else:
            self.is_master = False

        import paddle

        self.paddle_commit_id = paddle.version.commit

        if self.max_num_batched_tokens is None:
            if self.cache_config.enable_chunked_prefill:
                self.max_num_batched_tokens = 2048
            else:
                self.max_num_batched_tokens = self.max_model_len

        if self.long_prefill_token_threshold == 0:
            self.long_prefill_token_threshold = int(self.max_model_len * 0.04)

        self.cache_config.postprocess(self.max_num_batched_tokens, self.max_num_seqs)
        self.cache_config.max_block_num_per_seq = int(self.max_model_len // self.cache_config.block_size)

        if self.guided_decoding_backend == "auto":
            if self.enable_mm:
                self.guided_decoding_backend = "off"
            else:
                self.guided_decoding_backend = "xgrammar"

    def check(self):
        """
        check the legality of config
        """
        assert self.max_num_seqs <= 256, (
            "The parameter `max_num_seqs` is not allowed to exceed 256, " f"but now it's {self.max_num_seqs}."
        )
        assert is_port_available(
            "0.0.0.0", self.engine_worker_queue_port
        ), f"The parameter `engine_worker_queue_port`:{self.engine_worker_queue_port} is already in use."
        assert self.nnode >= 1, f"nnode: {self.nnode} should no less than 1"
        assert self.max_model_len >= 16, f"max_model_len: {self.max_model_len} should be larger than 16"
        assert self.max_num_seqs >= 1, f"max_num_seqs: {self.max_num_seqs} should be larger than 1"
        assert self.max_num_batched_tokens >= self.max_num_seqs, (
            f"max_num_batched_tokens: {self.max_num_batched_tokens} "
            f"should be larger than or equal to max_num_seqs: {self.max_num_seqs}"
        )
        assert self.max_num_batched_tokens <= self.max_model_len * self.max_num_seqs, (
            f"max_num_batched_tokens: {self.max_num_batched_tokens} should be larger"
            f"than or equal to max_num_seqs: {self.max_num_seqs} * max_model_len: {self.max_model_len}"
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

        if not self.cache_config.enable_chunked_prefill:
            assert self.max_num_batched_tokens >= self.max_model_len, (
                f"max_num_batched_tokens: {self.max_num_batched_tokens} "
                f"should be larger than or equal to max_model_len: {self.max_model_len}"
            )
        else:
            assert self.max_num_batched_tokens >= self.cache_config.block_size, (
                f"max_num_batched_tokens: {self.max_num_batched_tokens} "
                f"should be larger than or equal to block_size: {self.cache_config.block_size}"
            )

        if self.max_num_partial_prefills > 1:
            assert (
                self.cache_config.enable_chunked_prefill is True
            ), "Chunked prefill must be enabled to set max_num_partial_prefills > 1"
            assert self.long_prefill_token_threshold < self.max_model_len, (
                f"long_prefill_token_threshold: {self.long_prefill_token_threshold} should be less than"
                f" max_model_len: {self.max_model_len}"
            )

        if self.guided_decoding_backend is not None:
            assert self.guided_decoding_backend in [
                "xgrammar",
                "XGrammar",
                "auto",
                "off",
            ], f"Only support xgrammar、auto guided decoding backend, but got {self.guided_decoding_backend}."

            if self.guided_decoding_backend != "off":
                # TODO: mm support guided_decoding
                assert self.enable_mm is False, "Multimodal model currently do not support guided_decoding"

                # TODO: speculative decoding support guided_decoding

                # TODO: xpu support guided_decoding
                assert not current_platform.is_xpu(), "XPU currently do not support guided_decoding"

                try:
                    import xgrammar  # noqa
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
        llm_logger.info("=================== Configuration Information ===============")
        for k, v in self.__dict__.items():
            if k == "generation_config" and v is not None:
                for gck, gcv in v.to_dict().items():
                    llm_logger.info("{:<20}:{:<6}{}".format(gck, "", gcv))
            elif (
                k == "cache_config"
                or k == "model_config"
                or k == "scheduler_config"
                or k == "parallel_config"
                or k == "commit_config"
            ):
                v.print()
            else:
                llm_logger.info("{:<20}:{:<6}{}".format(k, "", v))
        llm_logger.info("=============================================================")
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
            current_protocol = self.cache_config.cache_transfer_protocol.split(",")
            disaggregate_info["transfer_protocol"] = current_protocol
            for protocol in current_protocol:
                if protocol == "ipc":
                    disaggregate_info["cache_info"][protocol] = {
                        "ip": self.host_ip,
                        "port": self.engine_worker_queue_port,
                        "device_ids": self.local_device_ids,
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
                llm_logger.info(f"Reset parameter {value_name} = {value} from configuration.")

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
            if type(val) is str:
                setattr(self, attr_name, [default_type(i) for i in val.split(",")])
            else:
                setattr(self, attr_name, val)

    def __str__(self) -> str:
        return json.dumps(self.__dict__, indent=4)
