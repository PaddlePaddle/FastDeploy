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

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_quality_module():
    module_path = REPO_ROOT / "scripts/evaluate_minicpm41_quant_quality.py"
    spec = importlib.util.spec_from_file_location("evaluate_minicpm41_quant_quality", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_score_output_accepts_nfkc_exact_answer():
    module = load_quality_module()

    passed, _ = module.score_output({"exact_answers": ("H2O",)}, " H₂O。 ")

    assert passed


def test_score_output_rejects_explanation_for_exact_answer():
    module = load_quality_module()

    passed, detail = module.score_output({"exact_answers": ("391",)}, "答案是 391。")

    assert not passed
    assert "391" in detail


def test_score_output_requires_every_keyword():
    module = load_quality_module()

    passed, detail = module.score_output({"required_terms": ("蓝", "散射")}, "天空看起来是蓝色的。")

    assert not passed
    assert "散射" in detail


def test_generation_config_records_deterministic_request_settings():
    module = load_quality_module()

    assert module.generation_config(96) == {
        "max_tokens": 96,
        "temperature": 0,
        "top_p": 1,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
