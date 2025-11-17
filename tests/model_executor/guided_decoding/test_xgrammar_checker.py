# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import json
import sys
import unittest
from unittest.mock import MagicMock

mock_torch = MagicMock()
mock_xgrammar = MagicMock()

sys.modules["torch"] = mock_torch
sys.modules["xgrammar"] = mock_xgrammar

from fastdeploy.engine.request import Request
from fastdeploy.model_executor.guided_decoding.xgrammar_backend import XGrammarChecker


def make_request(**kwargs) -> Request:
    """
    Construct a Request object with default fields and override with any provided keyword arguments.
    This helper function simplifies creating Request instances for testing by
    pre-filling common fields and allowing selective overrides.
    """
    base = dict(
        request_id="req-1",
        prompt="",
        prompt_token_ids=[],
        prompt_token_ids_len=0,
        messages=[],
        history=[],
        tools=[],
        system="",
        sampling_params={},
        eos_token_ids=[],
        arrival_time=0.0,
        guided_json=None,
        guided_grammar=None,
        guided_json_object=None,
        guided_choice=None,
        structural_tag=None,
        pooling_params={},
    )
    base.update(kwargs)
    return Request(**base)


class TestXGrammarChecker(unittest.TestCase):
    def setUp(self):
        self.checker = XGrammarChecker()

    def test_guided_json_valid(self):
        """
        Test that a valid guided_json passes the schema check.
        """

        request = make_request(guided_json={"type": "string"})
        request, err = self.checker.schema_format(request)
        self.assertIsNone(err)
        self.assertIsInstance(request.guided_json, str)

    def test_guided_json_object(self):
        """
        Test that guided_json_object generates a JSON object type.
        """

        request = make_request(guided_json_object=True)
        request, err = self.checker.schema_format(request)
        self.assertIsNone(err)
        self.assertEqual(request.guided_json, '{"type": "object"}')

    def test_guided_grammar_valid(self):
        """
        Test that a valid guided_grammar passes the schema check.
        """

        request = make_request(guided_grammar='root ::= "yes" | "no"')
        request, err = self.checker.schema_format(request)
        self.assertIsNone(err)
        self.assertIn("root", request.guided_grammar)

    def test_guided_choice_valid(self):
        """
        Test that a valid guided_choice is correctly converted to EBNF.
        """

        request = make_request(guided_choice=["yes", "no"])
        request, err = self.checker.schema_format(request)
        self.assertIsNone(err)
        self.assertIn("yes", request.guided_grammar)
        self.assertIn("no", request.guided_grammar)

    def test_guided_choice_invalid(self):
        """
        Test that an invalid guided_choice (containing None) raises TypeError.
        """

        request = make_request(guided_choice=[None])
        with self.assertRaises(TypeError):
            self.checker.schema_format(request)

    def test_structural_tag_valid(self):
        """
        Test that a valid structural_tag passes the schema check.
        """

        structural_tag = {
            "structures": [{"begin": "<a>", "schema": {"type": "string"}, "end": "</a>"}],
            "triggers": ["<a>"],
        }
        request = make_request(structural_tag=json.dumps(structural_tag))
        request, err = self.checker.schema_format(request)
        self.assertIsNone(err)

    def test_structural_tag_invalid(self):
        """
        Test that a structural_tag missing 'triggers' raises KeyError.
        """

        structural_tag = {"structures": [{"begin": "<a>", "schema": {"type": "string"}, "end": "</a>"}]}
        request = make_request(structural_tag=json.dumps(structural_tag))
        with self.assertRaises(KeyError):
            self.checker.schema_format(request)

    def test_regex_passthrough(self):
        """
        Test that regex is not modified by schema_format and passes through as-is.
        """

        request = make_request()
        request.regex = "^[a-z]+$"
        request, err = self.checker.schema_format(request)
        self.assertIsNone(err)
        self.assertEqual(request.regex, "^[a-z]+$")


if __name__ == "__main__":
    unittest.main(verbosity=2)
