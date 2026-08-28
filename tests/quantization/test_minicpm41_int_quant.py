# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def read_repo_file(path):
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def load_module(module_path):
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_parse_inputs(quantization):
    args = SimpleNamespace(
        quantization=quantization,
        dynamic_load_weight=False,
        enable_mega_moe=False,
    )
    model_config = SimpleNamespace(
        architectures=["MiniCPMForCausalLM"],
        quantization_config=None,
        model_format="torch",
        is_quantized=False,
    )
    return args, model_config


def test_minicpm41_declares_only_verified_wint_quantizations():
    module = load_module(REPO_ROOT / "fastdeploy/model_executor/models/minicpm41/config_minicpm41.py")

    assert module.SUPPORTED_QUANTIZATIONS == ("wint4", "wint8")


@pytest.mark.parametrize(
    ("quant_name", "expected_class", "expected_algo"),
    [
        ("wint4", "WINT4Config", "weight_only_int4"),
        ("wint8", "WINT8Config", "weight_only_int8"),
    ],
)
def test_minicpm41_cli_wint_uses_bf16_checkpoint_online_quantization(
    monkeypatch,
    quant_name,
    expected_class,
    expected_algo,
):
    pytest.importorskip("paddle")
    from fastdeploy.model_executor.layers.quantization import parse_quant_config

    class FakeQuantConfig:
        @classmethod
        def from_config(cls, config):
            return SimpleNamespace(
                class_name=expected_class,
                quant_name=quant_name,
                algo=expected_algo,
                is_checkpoint_bf16=not config.get("is_quantized", False),
            )

    monkeypatch.setattr(
        "fastdeploy.model_executor.layers.quantization.get_quantization_config",
        lambda name: FakeQuantConfig,
    )

    args, model_config = make_parse_inputs(quant_name)
    quant_config = parse_quant_config(args, model_config, is_ernie=False, is_v1_loader=False)

    assert quant_config.class_name == expected_class
    assert quant_config.quant_name == quant_name
    assert quant_config.algo == expected_algo
    assert quant_config.is_checkpoint_bf16 is True
    assert model_config.is_quantized is False
    assert args.quantization == {"quantization": quant_name}


@pytest.mark.parametrize("quant_name", ["wint4", "wint8"])
def test_minicpm41_one_key_quant_dict_stays_online(monkeypatch, quant_name):
    pytest.importorskip("paddle")
    from fastdeploy.model_executor.layers.quantization import parse_quant_config

    class FakeQuantConfig:
        @classmethod
        def from_config(cls, config):
            return SimpleNamespace(is_checkpoint_bf16=not config.get("is_quantized", False))

    monkeypatch.setattr(
        "fastdeploy.model_executor.layers.quantization.get_quantization_config",
        lambda name: FakeQuantConfig,
    )

    args, model_config = make_parse_inputs({"quantization": quant_name})
    quant_config = parse_quant_config(args, model_config, is_ernie=False, is_v1_loader=False)

    assert quant_config.is_checkpoint_bf16 is True
    assert model_config.is_quantized is False
