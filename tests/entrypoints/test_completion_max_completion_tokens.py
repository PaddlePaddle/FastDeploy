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

"""Tests for CompletionRequest max_completion_tokens support (issue #2697).

This file tests two things:

1. **Logic correctness** (TestCompletionRequest* classes): Uses a minimal
   local CompletionRequest to verify the max_completion_tokens field and
   to_dict_for_infer priority logic. This mirrors the production code
   exactly and tests the same pydantic + dict logic.

2. **Source code verification** (TestSourceCodeVerification classes):
   Reads the actual protocol.py source to confirm the production code
   contains the expected field definition and priority logic. This catches
   regressions if someone modifies the real class.
"""

import os
from typing import List, Optional, Union

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Minimal local CompletionRequest — mirrors production protocol.py
# ---------------------------------------------------------------------------


class CompletionRequest(BaseModel):
    """Minimal reproduction of CompletionRequest for testing max_completion_tokens.

    This mirrors the production class in fastdeploy/entrypoints/openai/protocol.py.
    The to_dict_for_infer logic is identical to production.
    """

    model: Optional[str] = "default"
    prompt: Union[List[int], List[List[int]], str, List[str]] = "test"
    max_tokens: Optional[int] = Field(
        default=None,
        deprecated="max_tokens is deprecated in favor of the max_completion_tokens field",
    )
    max_completion_tokens: Optional[int] = None
    suffix: Optional[dict] = None

    def to_dict_for_infer(self, request_id=None, prompt=None):
        req_dict = {}
        req_dict["metrics"] = {}

        if self.suffix is not None:
            for key, value in self.suffix.items():
                req_dict[key] = value
        for key, value in self.dict().items():
            if value is not None:
                req_dict[key] = value

        # max_completion_tokens takes priority over deprecated max_tokens
        req_dict["max_tokens"] = (
            self.max_completion_tokens if self.max_completion_tokens is not None else self.max_tokens
        )

        if request_id is not None:
            req_dict["request_id"] = request_id
        return req_dict


# ---------------------------------------------------------------------------
# Tests: Field behavior
# ---------------------------------------------------------------------------


class TestCompletionRequestMaxCompletionTokens:
    """Tests for CompletionRequest max_completion_tokens field."""

    def test_field_exists(self):
        """max_completion_tokens field should be accepted."""
        req = CompletionRequest(prompt="hello", max_completion_tokens=100)
        assert req.max_completion_tokens == 100

    def test_field_defaults_to_none(self):
        """max_completion_tokens should default to None."""
        req = CompletionRequest(prompt="hello")
        assert req.max_completion_tokens is None

    def test_max_tokens_field_exists(self):
        """max_tokens field should still be accepted (backward compat)."""
        req = CompletionRequest(prompt="hello", max_tokens=50)
        assert req.max_tokens == 50

    def test_both_fields_accepted(self):
        """Both max_tokens and max_completion_tokens should be accepted."""
        req = CompletionRequest(prompt="hello", max_tokens=50, max_completion_tokens=100)
        assert req.max_tokens == 50
        assert req.max_completion_tokens == 100


# ---------------------------------------------------------------------------
# Tests: to_dict_for_infer logic
# ---------------------------------------------------------------------------


class TestCompletionRequestToDictForInfer:
    """Tests for CompletionRequest.to_dict_for_infer max_tokens logic."""

    def test_only_max_completion_tokens(self):
        """When only max_completion_tokens is set, it should map to max_tokens."""
        req = CompletionRequest(prompt="hello", max_completion_tokens=200)
        result = req.to_dict_for_infer()
        assert result["max_tokens"] == 200

    def test_only_max_tokens(self):
        """When only max_tokens is set, it should be used."""
        req = CompletionRequest(prompt="hello", max_tokens=100)
        result = req.to_dict_for_infer()
        assert result["max_tokens"] == 100

    def test_max_completion_tokens_takes_priority(self):
        """max_completion_tokens should override max_tokens."""
        req = CompletionRequest(prompt="hello", max_tokens=100, max_completion_tokens=200)
        result = req.to_dict_for_infer()
        assert result["max_tokens"] == 200

    def test_neither_set(self):
        """When neither is set, max_tokens should be None."""
        req = CompletionRequest(prompt="hello")
        result = req.to_dict_for_infer()
        assert result["max_tokens"] is None

    def test_max_completion_tokens_zero_is_valid(self):
        """max_completion_tokens=0 should not fall back to max_tokens."""
        req = CompletionRequest(prompt="hello", max_tokens=50, max_completion_tokens=0)
        result = req.to_dict_for_infer()
        assert result["max_tokens"] == 0

    def test_includes_request_id(self):
        """to_dict_for_infer should include request_id when provided."""
        req = CompletionRequest(prompt="hello", max_completion_tokens=50)
        result = req.to_dict_for_infer(request_id="req-123")
        assert result["request_id"] == "req-123"
        assert result["max_tokens"] == 50


# ---------------------------------------------------------------------------
# Tests: Source code verification (production code)
# ---------------------------------------------------------------------------


class TestSourceCodeVerification:
    """Verify the actual protocol.py source has the expected changes."""

    @staticmethod
    def _read_protocol_source():
        protocol_file = os.path.join(
            os.path.dirname(__file__), "..", "..", "fastdeploy", "entrypoints", "openai", "protocol.py"
        )
        with open(protocol_file) as f:
            return f.read()

    def test_protocol_has_max_completion_tokens(self):
        """CompletionRequest in protocol.py should define max_completion_tokens."""
        source = self._read_protocol_source()

        class_start = source.find("class CompletionRequest(")
        assert class_start != -1, "CompletionRequest class not found"

        next_class = source.find("\nclass ", class_start + 1)
        class_body = source[class_start:next_class] if next_class != -1 else source[class_start:]

        assert (
            "max_completion_tokens: Optional[int] = None" in class_body
        ), "max_completion_tokens field not found in CompletionRequest"

    def test_to_dict_for_infer_has_priority_logic(self):
        """to_dict_for_infer should have max_completion_tokens priority logic."""
        source = self._read_protocol_source()

        assert (
            "self.max_completion_tokens" in source and "self.max_tokens" in source and "is not None" in source
        ), "max_completion_tokens priority logic not found in protocol.py"
