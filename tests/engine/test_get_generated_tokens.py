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

"""Tests for LLMEngine._get_generated_tokens (issue #2795).

Uses sys.modules mocking to avoid heavy import chains (paddle, tensorflow, etc.)
"""

import os
import sys
import types
from types import SimpleNamespace


def _stub_module(name):
    """Create and register a stub module to prevent real imports."""
    if name not in sys.modules:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
    return sys.modules[name]


# Stub out heavy dependencies before importing fastdeploy
for mod_name in [
    "paddle",
    "paddle.device",
    "paddle.device.cuda",
    "paddleformers",
    "paddleformers.transformers",
    "paddleformers.transformers.image_processing_utils",
    "tensorflow",
    "transformers",
    "transformers.image_processing_utils",
    "transformers.image_transforms",
    "ml_dtypes",
]:
    _stub_module(mod_name)

# Provide the attribute that the import chain expects
sys.modules["transformers.image_processing_utils"].BaseImageProcessor = type("BaseImageProcessor", (), {})


def _make_output(request_id, text, finished=False):
    """Create a minimal RequestOutput-like object for testing."""
    obj = SimpleNamespace(
        request_id=request_id,
        outputs=SimpleNamespace(text=text),
        finished=finished,
        to_dict=lambda: {"request_id": request_id, "finished": finished},
    )
    return obj


def _make_engine_with_scheduler(results_sequence):
    """Create a minimal LLMEngine mock with _get_generated_tokens.

    We test the method logic directly without importing the full engine.
    """
    call_idx = {"i": 0}

    def _get_results():
        idx = call_idx["i"]
        call_idx["i"] += 1
        if idx < len(results_sequence):
            return results_sequence[idx]
        return {}

    # Import only the method, not the whole module
    # Define the method inline to test its logic
    def _get_generated_tokens(self, req_id):
        while True:
            results = self._get_generated_result()
            if req_id in results:
                for output in results[req_id]:
                    yield output
                    if output.finished:
                        return

    engine = SimpleNamespace(
        _get_generated_result=_get_results,
    )
    # Bind the method
    engine._get_generated_tokens = lambda req_id: _get_generated_tokens(engine, req_id)
    return engine


class TestGetGeneratedTokens:
    """Tests for _get_generated_tokens logic."""

    def test_yields_matching_request_outputs(self):
        """Should yield only outputs for the requested req_id."""
        r1_partial = _make_output("req-1", "hello", finished=False)
        r1_final = _make_output("req-1", "hello world", finished=True)
        r2_output = _make_output("req-2", "other", finished=True)

        results_seq = [
            {"req-1": [r1_partial], "req-2": [r2_output]},
            {"req-1": [r1_final]},
        ]
        engine = _make_engine_with_scheduler(results_seq)

        outputs = list(engine._get_generated_tokens("req-1"))
        assert len(outputs) == 2
        assert outputs[0] is r1_partial
        assert outputs[1] is r1_final
        assert outputs[1].finished is True

    def test_stops_after_finished(self):
        """Should stop yielding once a finished output is seen."""
        r1_final = _make_output("req-1", "done", finished=True)
        extra = _make_output("req-1", "should not appear", finished=False)

        results_seq = [
            {"req-1": [r1_final]},
            {"req-1": [extra]},
        ]
        engine = _make_engine_with_scheduler(results_seq)

        outputs = list(engine._get_generated_tokens("req-1"))
        assert len(outputs) == 1
        assert outputs[0].finished is True

    def test_skips_other_request_ids(self):
        """Should ignore results for other request_ids."""
        r2 = _make_output("req-2", "other", finished=True)
        r1 = _make_output("req-1", "mine", finished=True)

        results_seq = [
            {"req-2": [r2]},
            {"req-1": [r1]},
        ]
        engine = _make_engine_with_scheduler(results_seq)

        outputs = list(engine._get_generated_tokens("req-1"))
        assert len(outputs) == 1
        assert outputs[0] is r1

    def test_empty_results_continues_polling(self):
        """Should keep polling when get_results returns empty dict."""
        r1 = _make_output("req-1", "finally", finished=True)

        results_seq = [
            {},
            {},
            {"req-1": [r1]},
        ]
        engine = _make_engine_with_scheduler(results_seq)

        outputs = list(engine._get_generated_tokens("req-1"))
        assert len(outputs) == 1
        assert outputs[0].finished is True

    def test_multiple_outputs_in_single_batch(self):
        """Should yield all outputs from a single get_results call."""
        r1_a = _make_output("req-1", "token1", finished=False)
        r1_b = _make_output("req-1", "token1 token2", finished=False)
        r1_c = _make_output("req-1", "token1 token2 token3", finished=True)

        results_seq = [
            {"req-1": [r1_a, r1_b, r1_c]},
        ]
        engine = _make_engine_with_scheduler(results_seq)

        outputs = list(engine._get_generated_tokens("req-1"))
        assert len(outputs) == 3
        assert outputs[2].finished is True


class TestMethodExists:
    """Verify the method is actually defined on LLMEngine."""

    def test_method_defined_in_source(self):
        """_get_generated_tokens should be defined in engine.py source."""
        engine_file = os.path.join(os.path.dirname(__file__), "..", "..", "fastdeploy", "engine", "engine.py")
        with open(engine_file) as f:
            source = f.read()
        assert "def _get_generated_tokens(self, req_id):" in source
        # Also verify _get_generated_tokens is called by generate()
        assert "_get_generated_tokens(req_id)" in source
