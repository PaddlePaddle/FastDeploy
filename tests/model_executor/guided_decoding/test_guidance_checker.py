"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import json
import sys
import unittest
from unittest.mock import MagicMock, patch

import pytest

# Check if llguidance can be imported
HAS_LLGUIDANCE = False
try:
    import llguidance

    llguidance
    HAS_LLGUIDANCE = True
except ImportError:
    mock_llguidance = MagicMock()
    mock_llguidancehf = MagicMock()
    mock_llguidancetorch = MagicMock()
    mock_torch = MagicMock()
    sys.modules["llguidance"] = mock_llguidance
    sys.modules["llguidance.hf"] = mock_llguidancehf
    sys.modules["llguidance.torch"] = mock_llguidancetorch
    sys.modules["torch"] = mock_torch


@pytest.fixture
def llguidance_checker():
    """Return an LLGuidanceChecker instance for testing."""
    return LLGuidanceChecker()


@pytest.fixture
def llguidance_checker_with_options():
    """Return an LLGuidanceChecker instance configured with specific options."""
    return LLGuidanceChecker(disable_any_whitespace=True)


from fastdeploy.model_executor.guided_decoding.guidance_backend import LLGuidanceChecker


def MockRequest():
    request = MagicMock()
    request.guided_json = None
    request.guided_json_object = None
    request.structural_tag = None
    request.guided_regex = None
    request.guided_choice = None
    request.guided_grammar = None
    return request


class TestLLGuidanceCheckerMocked:
    """Test LLGuidanceChecker using Mock, suitable for environments without the llguidance library."""

    @patch("llguidance.LLMatcher.grammar_from_json_schema")
    @patch("llguidance.LLMatcher.validate_grammar")
    def test_serialize_guided_json_as_string(self, mock_validate, mock_from_schema, llguidance_checker):
        """Test processing guided_json string type."""
        mock_from_schema.return_value = "serialized_grammar"
        mock_validate.return_value = None

        request = MockRequest()
        request.guided_json = '{"type": "object", "properties": {"name": {"type": "string"}}}'

        grammar = llguidance_checker.serialize_guidance_grammar(request)

        mock_from_schema.assert_called_once()
        assert grammar == "serialized_grammar"

    @patch("llguidance.LLMatcher.grammar_from_json_schema")
    @patch("llguidance.LLMatcher.validate_grammar")
    def test_serialize_guided_json_as_dict(self, mock_validate, mock_from_schema, llguidance_checker):
        """Test processing guided_json dictionary type."""
        mock_from_schema.return_value = "serialized_grammar"
        mock_validate.return_value = None

        request = MockRequest()
        request.guided_json = {"type": "object", "properties": {"name": {"type": "string"}}}

        grammar = llguidance_checker.serialize_guidance_grammar(request)

        mock_from_schema.assert_called_once()
        assert isinstance(request.guided_json, dict)  # Verify that the dictionary has been converted to a string
        assert grammar == "serialized_grammar"

    @patch("llguidance.LLMatcher.grammar_from_json_schema")
    @patch("llguidance.LLMatcher.validate_grammar")
    def test_serialize_guided_json_object(self, mock_validate, mock_from_schema, llguidance_checker):
        """Test processing guided_json_object."""
        mock_from_schema.return_value = "serialized_grammar"
        mock_validate.return_value = None

        request = MockRequest()
        request.guided_json_object = True

        grammar = llguidance_checker.serialize_guidance_grammar(request)

        mock_from_schema.assert_called_once()
        assert request.guided_json_object
        assert grammar == "serialized_grammar"

    @patch("llguidance.grammar_from")
    @patch("llguidance.LLMatcher.validate_grammar")
    def test_serialize_guided_regex(self, mock_validate, mock_grammar_from, llguidance_checker):
        """Test processing guided_regex."""
        mock_grammar_from.return_value = "serialized_regex_grammar"
        mock_validate.return_value = None

        request = MockRequest()
        request.guided_regex = "[a-zA-Z]+"

        grammar = llguidance_checker.serialize_guidance_grammar(request)

        mock_grammar_from.assert_called_once_with("regex", "[a-zA-Z]+")
        assert grammar == "serialized_regex_grammar"

    @patch("llguidance.grammar_from")
    @patch("llguidance.LLMatcher.validate_grammar")
    def test_serialize_guided_choice(self, mock_validate, mock_grammar_from, llguidance_checker):
        """Test processing guided_choice."""
        mock_grammar_from.return_value = "serialized_choice_grammar"
        mock_validate.return_value = None

        request = MockRequest()
        request.guided_choice = ["option1", "option2"]

        grammar = llguidance_checker.serialize_guidance_grammar(request)

        mock_grammar_from.assert_called_once_with("choice", ["option1", "option2"])
        assert grammar == "serialized_choice_grammar"

    @patch("llguidance.grammar_from")
    @patch("llguidance.LLMatcher.validate_grammar")
    def test_serialize_guided_grammar(self, mock_validate, mock_grammar_from, llguidance_checker):
        """Test processing guided_grammar."""
        mock_grammar_from.return_value = "serialized_grammar_spec"
        mock_validate.return_value = None

        request = MockRequest()
        request.guided_grammar = "grammar specification"

        grammar = llguidance_checker.serialize_guidance_grammar(request)

        mock_grammar_from.assert_called_once_with("grammar", "grammar specification")
        assert grammar == "serialized_grammar_spec"

    @patch("llguidance.StructTag")
    @patch("llguidance.LLMatcher.grammar_from_json_schema")
    def test_serialize_structural_tag(self, mock_from_schema, mock_struct_tag, llguidance_checker):
        """Test processing structural_tag."""
        # Configure mock objects
        mock_from_schema.return_value = "serialized_schema"
        mock_struct_tag.to_grammar.return_value = "serialized_structural_grammar"
        struct_tag_instance = MagicMock()
        mock_struct_tag.return_value = struct_tag_instance

        request = MockRequest()
        request.structural_tag = {
            "triggers": ["<json>"],
            "structures": [{"begin": "<json>", "schema": {"type": "object"}, "end": "</json>"}],
        }

        grammar = llguidance_checker.serialize_guidance_grammar(request)

        mock_from_schema.assert_called_once()
        mock_struct_tag.assert_called_once()
        mock_struct_tag.to_grammar.assert_called_once()
        assert grammar == "serialized_structural_grammar"

    @patch("llguidance.StructTag")
    def test_serialize_structural_tag_missing_trigger(self, mock_struct_tag, llguidance_checker):
        """Test processing structural_tag when a trigger is missing."""
        request = MockRequest()
        request.structural_tag = {
            "triggers": ["<xml>"],
            "structures": [{"begin": "<json>", "schema": {"type": "object"}, "end": "</json>"}],
        }

        with pytest.raises(ValueError, match="Trigger .* not found in triggers"):
            llguidance_checker.serialize_guidance_grammar(request)

    @patch("llguidance.StructTag")
    def test_serialize_structural_tag_empty_structures(self, mock_struct_tag, llguidance_checker):
        """Test processing structural_tag when structures are empty."""
        request = MockRequest()
        request.structural_tag = {"triggers": ["<json>"], "structures": []}

        with pytest.raises(ValueError, match="No structural tags found in the grammar spec"):
            llguidance_checker.serialize_guidance_grammar(request)

    def test_serialize_invalid_grammar_type(self, llguidance_checker):
        """Test processing invalid grammar types."""
        request = MockRequest()
        # No grammar type set

        with pytest.raises(ValueError, match="grammar is not of valid supported types"):
            llguidance_checker.serialize_guidance_grammar(request)

    @patch("llguidance.LLMatcher.grammar_from_json_schema")
    @patch("llguidance.LLMatcher.validate_grammar")
    def test_schema_format_valid_json(self, mock_validate, mock_from_schema, llguidance_checker):
        """Test schema_format method processing valid JSON."""
        mock_from_schema.return_value = "serialized_grammar"
        mock_validate.return_value = None

        request = MockRequest()
        request.guided_json = '{"type": "object"}'

        result_request, error = llguidance_checker.schema_format(request)

        assert error is None
        assert result_request is request

    @patch("llguidance.LLMatcher.grammar_from_json_schema")
    @patch("llguidance.LLMatcher.validate_grammar")
    def test_schema_format_invalid_grammar(self, mock_validate, mock_from_schema, llguidance_checker):
        """Test schema_format method processing invalid grammar."""
        mock_from_schema.return_value = "serialized_grammar"
        mock_validate.return_value = "Invalid grammar"

        request = MockRequest()
        request.guided_json = '{"type": "object"}'

        result_request, error = llguidance_checker.schema_format(request)

        assert error is not None
        assert "Grammar error: Invalid grammar" in error

    @patch("llguidance.LLMatcher.grammar_from_json_schema")
    def test_schema_format_json_decode_error(self, mock_from_schema, llguidance_checker):
        """Test schema_format method processing JSON decode error."""
        mock_from_schema.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)

        request = MockRequest()
        request.guided_json = "{invalid json}"

        result_request, error = llguidance_checker.schema_format(request)

        assert error is not None
        assert "Invalid format for guided decoding" in error

    @patch("llguidance.LLMatcher.grammar_from_json_schema")
    def test_schema_format_unexpected_error(self, mock_from_schema, llguidance_checker):
        """Test schema_format method processing unexpected errors."""
        mock_from_schema.side_effect = Exception("Unexpected error")

        request = MockRequest()
        request.guided_json = '{"type": "object"}'

        result_request, error = llguidance_checker.schema_format(request)

        assert error is not None
        assert "An unexpected error occurred during schema validation" in error

    def test_init_with_disable_whitespace(self, llguidance_checker_with_options):
        """Test setting the disable_any_whitespace option during initialization."""
        assert llguidance_checker_with_options.any_whitespace is False
        assert llguidance_checker_with_options.disable_additional_properties is True
        assert LLGuidanceChecker(disable_any_whitespace=True).any_whitespace is False
        assert LLGuidanceChecker(disable_any_whitespace=False).any_whitespace is True

        # default check
        from fastdeploy.envs import FD_GUIDANCE_DISABLE_ADDITIONAL

        assert FD_GUIDANCE_DISABLE_ADDITIONAL

        assert LLGuidanceChecker().disable_additional_properties is True
        with patch("fastdeploy.model_executor.guided_decoding.guidance_backend.FD_GUIDANCE_DISABLE_ADDITIONAL", False):
            assert LLGuidanceChecker().disable_additional_properties is False


@pytest.mark.skipif(not HAS_LLGUIDANCE, reason="llguidance library not installed, skipping actual dependency tests")
class TestLLGuidanceCheckerReal:
    """Test using the actual llguidance library, suitable for development environments."""

    def test_serialize_guided_json_string_real(self, llguidance_checker):
        """Test processing guided_json string using the actual library."""
        request = MockRequest()
        request.guided_json = '{"type": "object", "properties": {"name": {"type": "string"}}}'

        grammar = llguidance_checker.serialize_guidance_grammar(request)

        # Verify if the returned grammar is a valid string
        assert isinstance(grammar, str)
        assert len(grammar) > 0
        print("grammar", grammar)

    def test_serialize_guided_json_dict_real(self, llguidance_checker):
        """Test processing guided_json dictionary using the actual library."""
        request = MockRequest()
        request.guided_json = {"type": "object", "properties": {"name": {"type": "string"}}}

        grammar = llguidance_checker.serialize_guidance_grammar(request)

        assert isinstance(request.guided_json, dict)
        assert isinstance(grammar, str)
        assert len(grammar) > 0

    def test_serialize_guided_json_object_real(self, llguidance_checker):
        """Test processing guided_json_object using the actual library."""
        request = MockRequest()
        request.guided_json_object = True

        grammar = llguidance_checker.serialize_guidance_grammar(request)

        assert request.guided_json_object
        assert isinstance(grammar, str)
        assert len(grammar) > 0

    def test_serialize_guided_regex_real(self, llguidance_checker):
        """Test processing guided_regex using the actual library."""
        request = MockRequest()
        request.guided_regex = "[a-zA-Z]+"

        grammar = llguidance_checker.serialize_guidance_grammar(request)

        assert isinstance(grammar, str)
        assert len(grammar) > 0

    def test_serialize_guided_choice_real(self, llguidance_checker):
        """Test processing guided_choice using the actual library."""
        request = MockRequest()
        request.guided_choice = ["option1", "option2"]

        grammar = llguidance_checker.serialize_guidance_grammar(request)

        assert isinstance(grammar, str)
        assert len(grammar) > 0

    def test_serialize_guided_grammar_real(self, llguidance_checker):
        """Test processing guided_grammar using the actual library."""
        request = MockRequest()
        # Use a simple CFG grammar example
        request.guided_grammar = """
        root ::= greeting name
        greeting ::= "Hello" | "Hi"
        name ::= "world" | "everyone"
        """

        grammar = llguidance_checker.serialize_guidance_grammar(request)

        assert isinstance(grammar, str)
        assert len(grammar) > 0

    def test_serialize_structural_tag_real(self, llguidance_checker):
        """Test processing structural_tag using the actual library."""
        request = MockRequest()
        request.structural_tag = {
            "triggers": ["<json>"],
            "structures": [{"begin": "<json>", "schema": {"type": "object"}, "end": "</json>"}],
        }

        grammar = llguidance_checker.serialize_guidance_grammar(request)

        assert isinstance(grammar, str)
        assert len(grammar) > 0

    def test_schema_format_valid_json_real(self, llguidance_checker):
        """Test schema_format method processing valid JSON using the actual library."""
        request = MockRequest()
        request.guided_json = '{"type": "object", "properties": {"name": {"type": "string"}}}'

        result_request, error = llguidance_checker.schema_format(request)

        assert error is None
        assert result_request is request
        assert result_request.guided_json != '{"type": "object", "properties": {"name": {"type": "string"}}}'

    def test_schema_format_invalid_json_real(self, llguidance_checker):
        """Test schema_format method processing invalid JSON using the actual library."""
        request = MockRequest()
        request.guided_json = "{invalid json}"

        result_request, error = llguidance_checker.schema_format(request)

        assert error is not None
        assert "Invalid format for guided decoding" in error

    def test_whitespace_flexibility_option_real(self):
        """Test the impact of the whitespace flexibility option using the actual library."""
        # Create two instances with different configurations
        flexible = LLGuidanceChecker(disable_any_whitespace=False)
        strict = LLGuidanceChecker(disable_any_whitespace=True)

        request_flexible = MockRequest()
        request_flexible.guided_json = '{"type": "object"}'

        request_strict = MockRequest()
        request_strict.guided_json = '{"type": "object"}'

        grammar_flexible = flexible.serialize_guidance_grammar(request_flexible)
        grammar_strict = strict.serialize_guidance_grammar(request_strict)
        print("grammar_flexible", grammar_flexible)
        print("grammar_strict", grammar_strict)

        # Expect grammars generated by the two configurations to be different
        assert grammar_flexible != grammar_strict

    def test_schema_format_guided_json_object_real(self, llguidance_checker):
        """Test schema_format processing guided_json_object."""
        request = MockRequest()
        request.guided_json_object = True

        result_request, error = llguidance_checker.schema_format(request)

        assert error is None
        assert result_request is request

    def test_schema_format_guided_regex_real(self, llguidance_checker):
        """Test schema_format processing valid regular expressions."""
        request = MockRequest()
        request.guided_regex = r"[a-zA-Z0-9]+"

        result_request, error = llguidance_checker.schema_format(request)

        assert error is None
        assert result_request is request
        assert result_request.guided_regex != r"[a-zA-Z0-9]+"  # Should be converted to grammar format

    def test_schema_format_invalid_guided_regex_real(self, llguidance_checker):
        """Test schema_format processing invalid regular expressions."""
        request = MockRequest()
        request.guided_regex = r"["  # Invalid regular expression

        result_request, error = llguidance_checker.schema_format(request)

        assert error is not None
        assert "Invalid format for guided decoding" in error

    def test_schema_format_guided_choice_real(self, llguidance_checker):
        """Test schema_format processing guided_choice."""
        request = MockRequest()
        request.guided_choice = ["option1", "option2", "option3"]

        result_request, error = llguidance_checker.schema_format(request)

        assert error is None
        assert result_request is request
        assert result_request.guided_choice != [
            "option1",
            "option2",
            "option3",
        ]  # Should be converted to grammar format

    def test_schema_format_guided_grammar_real(self, llguidance_checker):
        """Test schema_format processing guided_grammar."""
        request = MockRequest()
        # Use the correct grammar format supported by LLGuidance
        request.guided_grammar = """
        start: number
        number: DIGIT+
        DIGIT: "0"|"1"|"2"|"3"|"4"|"5"|"6"|"7"|"8"|"9"
        """

        result_request, error = llguidance_checker.schema_format(request)

        assert error is None
        assert result_request is request
        assert isinstance(result_request.guided_grammar, str)

    def test_schema_format_structural_tag_real(self, llguidance_checker):
        """Test schema_format processing structural_tag."""
        request = MockRequest()
        request.structural_tag = {
            "triggers": ["```json"],
            "structures": [
                {
                    "begin": "```json",
                    "schema": {"type": "object", "properties": {"name": {"type": "string"}}},
                    "end": "```",
                }
            ],
        }

        result_request, error = llguidance_checker.schema_format(request)

        assert error is None
        assert result_request is request

    def test_schema_format_structural_tag_string_real(self, llguidance_checker):
        """Test schema_format processing structural_tag in string format."""
        request = MockRequest()
        request.structural_tag = json.dumps(
            {
                "triggers": ["```json"],
                "structures": [
                    {
                        "begin": "```json",
                        "schema": {"type": "object", "properties": {"name": {"type": "string"}}},
                        "end": "```",
                    }
                ],
            }
        )

        result_request, error = llguidance_checker.schema_format(request)

        assert error is None
        assert result_request is request

    def test_schema_format_structural_tag_invalid_trigger_real(self, llguidance_checker):
        """Test schema_format processing structural_tag with invalid triggers."""
        request = MockRequest()
        request.structural_tag = {
            "triggers": ["```xml"],  # Trigger does not match begin
            "structures": [
                {
                    "begin": "```json",
                    "schema": {"type": "object"},
                    "end": "```",
                }  # Does not contain any prefix from triggers here
            ],
        }

        result_request, error = llguidance_checker.schema_format(request)

        assert error is not None
        assert "Invalid format for guided decoding" in error

    def test_schema_format_structural_tag_empty_structures_real(self, llguidance_checker):
        """Test schema_format processing structural_tag with empty structures."""
        request = MockRequest()
        request.structural_tag = {"triggers": ["```json"], "structures": []}  # Empty structure

        result_request, error = llguidance_checker.schema_format(request)

        assert error is not None
        assert "Invalid format for guided decoding" in error

    def test_schema_format_json_dict_real(self, llguidance_checker):
        """Test schema_format processing guided_json in dictionary format."""
        request = MockRequest()
        request.guided_json = {"type": "object", "properties": {"name": {"type": "string"}}}

        result_request, error = llguidance_checker.schema_format(request)

        assert error is None
        assert result_request is request

    def test_schema_format_disable_additional_properties_real(self):
        """Test schema_format processing disable_additional_properties parameter."""
        checker = LLGuidanceChecker(disable_additional_properties=True)
        request = MockRequest()
        request.guided_json = {"type": "object", "properties": {"name": {"type": "string"}}}

        result_request, error = checker.schema_format(request)

        assert error is None
        assert result_request is request

    def test_schema_format_unexpected_error_real(self, monkeypatch, llguidance_checker):
        """Test schema_format processing unexpected errors."""
        request = MockRequest()
        request.guided_json = '{"type": "object"}'

        # Mock unexpected exception
        def mock_serialize_grammar(*args, **kwargs):
            raise Exception("Unexpected error")

        monkeypatch.setattr(llguidance_checker, "serialize_guidance_grammar", mock_serialize_grammar)

        result_request, error = llguidance_checker.schema_format(request)

        assert error is not None
        assert "An unexpected error occurred during schema validation" in error

    def test_schema_format_no_valid_grammar_real(self, llguidance_checker):
        """Test schema_format processing requests without valid grammar."""
        request = MockRequest()
        # No grammar-related attributes set

        with pytest.raises(ValueError, match="grammar is not of valid supported types"):
            llguidance_checker.serialize_guidance_grammar(request)
        result_request, error = llguidance_checker.schema_format(request)
        assert error is not None


if __name__ == "__main__":
    unittest.main()
