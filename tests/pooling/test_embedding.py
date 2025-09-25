# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import json
import os
import sys

import paddle
import pytest

from fastdeploy.config import (
    CacheConfig,
    FDConfig,
    GraphOptimizationConfig,
    LoadConfig,
    ModelConfig,
    ParallelConfig,
)
from fastdeploy.model_executor.models.adapters import as_embedding_model
from fastdeploy.model_executor.models.model_base import ModelRegistry
from fastdeploy.scheduler import SchedulerConfig

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tests.model_loader.utils import get_torch_model_path

test_model_configs = {
    "Qwen3-0.6B": {
        "tensor_parallel_size": 2,
        "max_model_len": 8192,
        "baseline_suffix": "standard",
    },
    "Qwen3-Embedding-0.6B": {
        "tensor_parallel_size": 2,
        "max_model_len": 8192,
        "baseline_suffix": "embedding",
    },
}


class TestModelLoader:

    @pytest.fixture(scope="session", autouse=True)
    def setup_paddle(self):
        if not paddle.is_compiled_with_cuda():
            raise AssertionError("CUDA not available")
        paddle.set_device("gpu")
        yield

    @pytest.fixture(scope="session", params=list(test_model_configs.keys()))
    def model_info(self, request):
        model_name = request.param
        try:
            torch_model_path = get_torch_model_path(model_name)
            if not os.path.exists(torch_model_path):
                raise AssertionError(f"Model path does not exist: {torch_model_path}")
            return {"name": model_name, "path": torch_model_path, "config": test_model_configs[model_name]}
        except Exception as e:
            raise AssertionError(f"Could not get torch model path for {model_name}: {e}")

    @pytest.fixture
    def model_config(self, model_info):
        if model_info is None:
            raise AssertionError("model_info is None")

        model_args = {
            "model": model_info["path"],
            "dtype": "bfloat16",
            "max_model_len": model_info["config"]["max_model_len"],
            "tensor_parallel_size": model_info["config"]["tensor_parallel_size"],
            "runner": "auto",
            "convert": "auto",
        }

        try:
            config = ModelConfig(model_args)
            return config
        except Exception as e:
            raise AssertionError(f"Could not create ModelConfig: {e}")

    @pytest.fixture
    def scheduler_config(self):
        scheduler_args = {
            "name": "local",
            "max_num_seqs": 256,
            "max_num_batched_tokens": 8192,
            "splitwise_role": "mixed",
            "max_size": -1,
            "ttl": 900,
            "max_model_len": 8192,
            "enable_chunked_prefill": False,
            "max_num_partial_prefills": 1,
            "max_long_partial_prefills": 1,
            "long_prefill_token_threshold": 0,
        }

        try:
            config = SchedulerConfig(scheduler_args)
            return config
        except Exception as e:
            raise AssertionError(f"Could not create SchedulerConfig: {e}")

    @pytest.fixture
    def fd_config(self, model_info, model_config, scheduler_config):
        if model_config is None:
            raise AssertionError("ModelConfig is None")
        if scheduler_config is None:
            raise AssertionError("SchedulerConfig is None")

        try:
            tensor_parallel_size = model_info["config"]["tensor_parallel_size"]

            cache_args = {
                "block_size": 64,
                "gpu_memory_utilization": 0.9,
                "cache_dtype": "bfloat16",
                "model_cfg": model_config,
                "tensor_parallel_size": tensor_parallel_size,
            }
            cache_config = CacheConfig(cache_args)

            parallel_args = {
                "tensor_parallel_size": tensor_parallel_size,
                "data_parallel_size": 1,
            }
            parallel_config = ParallelConfig(parallel_args)

            load_args = {}
            load_config = LoadConfig(load_args)

            graph_opt_args = {}
            graph_opt_config = GraphOptimizationConfig(graph_opt_args)

            fd_config = FDConfig(
                model_config=model_config,
                cache_config=cache_config,
                parallel_config=parallel_config,
                scheduler_config=scheduler_config,
                load_config=load_config,
                graph_opt_config=graph_opt_config,
                test_mode=True,
            )
            return fd_config

        except Exception as e:
            raise AssertionError(f"Could not create FDConfig: {e}")

    @pytest.fixture
    def model_json_config(self, model_info):
        if model_info is None:
            raise AssertionError("model_info is None")

        config_path = os.path.join(model_info["path"], "config.json")
        if not os.path.exists(config_path):
            raise AssertionError(f"Config file does not exist: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def test_embedding_with_none_convert_type(self, model_info, fd_config, model_json_config):
        if any(x is None for x in [model_info, fd_config, model_json_config]):
            raise AssertionError("Required configs not available")

        architectures = model_json_config.get("architectures", [])
        if not architectures:
            raise AssertionError("No architectures found in model config")

        fd_config.model_config.convert_type = "none"

        try:
            model_cls = ModelRegistry.get_class(architectures[0])

            if hasattr(model_cls, "__name__"):
                assert (
                    "ForEmbedding" not in model_cls.__name__
                ), f"Standard model should not have 'ForEmbedding' in name, but got: {model_cls.__name__}"

            standard_methods = set(dir(model_cls))
            assert "_init_pooler" not in standard_methods, "Standard model should not have _init_pooler method"

        except Exception as e:
            raise AssertionError(f"Error in none convert type test: {e}")

    def test_embedding_with_embed_convert_type(self, model_info, fd_config, model_json_config):
        if any(x is None for x in [model_info, fd_config, model_json_config]):
            raise AssertionError("Required configs not available")

        architectures = model_json_config.get("architectures", [])
        if not architectures:
            raise AssertionError("No architectures found in model config")

        fd_config.model_config.convert_type = "embed"

        try:
            model_cls = ModelRegistry.get_class(architectures[0])
            model_cls = as_embedding_model(model_cls)

            if hasattr(model_cls, "__name__"):
                assert (
                    "ForEmbedding" in model_cls.__name__
                ), f"Embedding model should have 'ForEmbedding' in name, but got: {model_cls.__name__}"

            embedding_methods = set(dir(model_cls))
            assert "_init_pooler" in embedding_methods, "Embedding model should have _init_pooler method"

        except Exception as e:
            raise AssertionError(f"Error in embed convert type test: {e}")
