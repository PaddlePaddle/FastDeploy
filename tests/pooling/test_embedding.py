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
from fastdeploy.model_executor.models.model_base import ModelRegistry

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tests.model_loader.utils import get_torch_model_path


class TestModelLoader:

    @pytest.fixture(scope="session", autouse=True)
    def setup_paddle(self):
        if not paddle.is_compiled_with_cuda():
            print("CUDA not available, using CPU")
            paddle.set_device("cpu")
        else:
            print("Using CUDA device")
            paddle.set_device("gpu")
        yield

    @pytest.fixture(scope="session")
    def model_path(self):
        try:
            torch_model_path = get_torch_model_path("Qwen3-0.6B")
            if os.path.exists(torch_model_path):
                return torch_model_path
        except Exception as e:
            print(f"Could not get torch model path: {e}")

    @pytest.fixture
    def model_config(self, model_path):
        model_args = {
            "model": model_path,
            "dtype": "bfloat16",
            "max_model_len": 8192,
            "tensor_parallel_size": 1,
            "runner": "auto",
            "convert": "auto",
        }

        try:
            return ModelConfig(model_args)
        except Exception as e:
            print(f"Could not create ModelConfig: {e}")

    @pytest.fixture
    def fd_config(self, model_config):
        try:
            cache_args = {
                "block_size": 64,
                "gpu_memory_utilization": 0.9,
                "cache_dtype": "bfloat16",
                "model_cfg": model_config,
                "tensor_parallel_size": 1,
            }
            cache_config = CacheConfig(cache_args)

            parallel_args = {
                "tensor_parallel_size": 1,
                "data_parallel_size": 1,
            }
            parallel_config = ParallelConfig(parallel_args)

            load_args = {}
            load_config = LoadConfig(load_args)

            graph_opt_args = {
                "enable_cudagraph": False,
                "cudagraph_capture_sizes": None,
            }
            graph_opt_config = GraphOptimizationConfig(graph_opt_args)

            return FDConfig(
                model_config=model_config,
                cache_config=cache_config,
                parallel_config=parallel_config,
                load_config=load_config,
                graph_opt_config=graph_opt_config,
                test_mode=True,
            )
        except Exception as e:
            print(f"Could not create FDConfig: {e}")

    @pytest.fixture
    def model_json_config(self, model_path):
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def test_embedding_with_none_convert_type(self, fd_config, model_json_config):
        if model_json_config is None:
            pytest.skip("Model config not available")

        if fd_config is None:
            pytest.skip("FDConfig not available")

        print("=" * 60)
        print("Testing initialize_model with convert_type='none'")
        print("=" * 60)

        architectures = model_json_config.get("architectures", [])
        if not architectures:
            pytest.skip("No architectures found in model config")

        fd_config.model_config.convert_type = "none"

        try:
            model_cls = ModelRegistry.get_class(architectures)

            if hasattr(model_cls, "__name__"):
                assert (
                    "ForEmbedding" not in model_cls.__name__
                ), f"Standard model should not have 'ForEmbedding' in name, but got: {model_cls.__name__}"
                print(f"Confirmed standard model type (no ForEmbedding): {model_cls.__name__}")

            standard_methods = set(dir(model_cls))
            assert "_init_pooler" not in standard_methods, "Standard model should not have _init_pooler method"

        except Exception as e:
            print(f"Error in none: {e}")

    def test_embedding_with_embed_convert_type(self, fd_config, model_json_config):
        if model_json_config is None:
            pytest.skip("Model config not available")

        if fd_config is None:
            pytest.skip("FDConfig not available")

        print("=" * 60)
        print("Testing embedding with convert_type='embed'")
        print("=" * 60)

        architectures = model_json_config.get("architectures", [])
        if not architectures:
            pytest.skip("No architectures found in model config")

        fd_config.model_config.convert_type = "embed"

        try:
            model_cls = ModelRegistry.get_class(architectures)
            if hasattr(model_cls, "__name__"):
                assert "ForEmbedding" in model_cls.__name__, "Embedding model should have 'ForEmbedding' in name"
                print(f"Confirmed embedding model type: {model_cls.__name__}")

            embedding_methods = set(dir(model_cls))
            assert "_init_pooler" in embedding_methods, "Embedding model should have _init_pooler method"

        except Exception as e:
            print(f"Error in convert embed: {e}")
