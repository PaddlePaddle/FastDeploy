import json
import unittest
from unittest.mock import MagicMock, patch

import pytest

from fastdeploy.model_executor.guided_decoding.guidance_backend import LLGuidanceChecker

# 检查是否可以导入llguidance
HAS_LLGUIDANCE = False
try:
    import llguidance

    llguidance
    HAS_LLGUIDANCE = True
except ImportError:
    pass


@pytest.fixture
def llguidance_checker():
    """返回一个LLGuidanceChecker实例供测试使用"""
    return LLGuidanceChecker()


@pytest.fixture
def llguidance_checker_with_options():
    """返回一个配置了特定选项的LLGuidanceChecker实例"""
    return LLGuidanceChecker(disable_any_whitespace=True)


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
    """使用Mock测试LLGuidanceChecker，适用于没有llguidance库的环境"""

    @patch("llguidance.LLMatcher.grammar_from_json_schema")
    @patch("llguidance.LLMatcher.validate_grammar")
    def test_serialize_guided_json_as_string(self, mock_validate, mock_from_schema, llguidance_checker):
        """测试处理guided_json字符串类型"""
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
        """测试处理guided_json字典类型"""
        mock_from_schema.return_value = "serialized_grammar"
        mock_validate.return_value = None

        request = MockRequest()
        request.guided_json = {"type": "object", "properties": {"name": {"type": "string"}}}

        grammar = llguidance_checker.serialize_guidance_grammar(request)

        mock_from_schema.assert_called_once()
        assert isinstance(request.guided_json, dict)  # 验证字典已转换为字符串
        assert grammar == "serialized_grammar"

    @patch("llguidance.LLMatcher.grammar_from_json_schema")
    @patch("llguidance.LLMatcher.validate_grammar")
    def test_serialize_guided_json_object(self, mock_validate, mock_from_schema, llguidance_checker):
        """测试处理guided_json_object"""
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
        """测试处理guided_regex"""
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
        """测试处理guided_choice"""
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
        """测试处理guided_grammar"""
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
        """测试处理structural_tag"""
        # 配置mock对象
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
        """测试处理structural_tag中缺少触发器的情况"""
        request = MockRequest()
        request.structural_tag = {
            "triggers": ["<xml>"],
            "structures": [{"begin": "<json>", "schema": {"type": "object"}, "end": "</json>"}],
        }

        with pytest.raises(ValueError, match="Trigger .* not found in triggers"):
            llguidance_checker.serialize_guidance_grammar(request)

    @patch("llguidance.StructTag")
    def test_serialize_structural_tag_empty_structures(self, mock_struct_tag, llguidance_checker):
        """测试处理structural_tag中结构为空的情况"""
        request = MockRequest()
        request.structural_tag = {"triggers": ["<json>"], "structures": []}

        with pytest.raises(ValueError, match="No structural tags found in the grammar spec"):
            llguidance_checker.serialize_guidance_grammar(request)

    def test_serialize_invalid_grammar_type(self, llguidance_checker):
        """测试处理无效的语法类型"""
        request = MockRequest()
        # 没有设置任何语法类型

        with pytest.raises(ValueError, match="grammar is not of valid supported types"):
            llguidance_checker.serialize_guidance_grammar(request)

    @patch("llguidance.LLMatcher.grammar_from_json_schema")
    @patch("llguidance.LLMatcher.validate_grammar")
    def test_schema_format_valid_json(self, mock_validate, mock_from_schema, llguidance_checker):
        """测试schema_format方法处理有效的JSON"""
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
        """测试schema_format方法处理无效的语法"""
        mock_from_schema.return_value = "serialized_grammar"
        mock_validate.return_value = "Invalid grammar"

        request = MockRequest()
        request.guided_json = '{"type": "object"}'

        result_request, error = llguidance_checker.schema_format(request)

        assert error is not None
        assert "Grammar error: Invalid grammar" in error

    @patch("llguidance.LLMatcher.grammar_from_json_schema")
    def test_schema_format_json_decode_error(self, mock_from_schema, llguidance_checker):
        """测试schema_format方法处理JSON解码错误"""
        mock_from_schema.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)

        request = MockRequest()
        request.guided_json = "{invalid json}"

        result_request, error = llguidance_checker.schema_format(request)

        assert error is not None
        assert "Invalid format for guided decoding" in error

    @patch("llguidance.LLMatcher.grammar_from_json_schema")
    def test_schema_format_unexpected_error(self, mock_from_schema, llguidance_checker):
        """测试schema_format方法处理意外错误"""
        mock_from_schema.side_effect = Exception("Unexpected error")

        request = MockRequest()
        request.guided_json = '{"type": "object"}'

        result_request, error = llguidance_checker.schema_format(request)

        assert error is not None
        assert "An unexpected error occurred during schema validation" in error

    def test_init_with_disable_whitespace(self, llguidance_checker_with_options):
        """测试初始化时设置disable_any_whitespace选项"""
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


@pytest.mark.skipif(not HAS_LLGUIDANCE, reason="llguidance库未安装，跳过实际依赖测试")
class TestLLGuidanceCheckerReal:
    """使用实际的llguidance库进行测试，适用于开发环境"""

    def test_serialize_guided_json_string_real(self, llguidance_checker):
        """使用实际库测试处理guided_json字符串"""
        request = MockRequest()
        request.guided_json = '{"type": "object", "properties": {"name": {"type": "string"}}}'

        grammar = llguidance_checker.serialize_guidance_grammar(request)

        # 验证返回的grammar是否是一个有效的字符串
        assert isinstance(grammar, str)
        assert len(grammar) > 0
        print("grammar", grammar)

    def test_serialize_guided_json_dict_real(self, llguidance_checker):
        """使用实际库测试处理guided_json字典"""
        request = MockRequest()
        request.guided_json = {"type": "object", "properties": {"name": {"type": "string"}}}

        grammar = llguidance_checker.serialize_guidance_grammar(request)

        assert isinstance(request.guided_json, dict)
        assert isinstance(grammar, str)
        assert len(grammar) > 0

    def test_serialize_guided_json_object_real(self, llguidance_checker):
        """使用实际库测试处理guided_json_object"""
        request = MockRequest()
        request.guided_json_object = True

        grammar = llguidance_checker.serialize_guidance_grammar(request)

        assert request.guided_json_object
        assert isinstance(grammar, str)
        assert len(grammar) > 0

    def test_serialize_guided_regex_real(self, llguidance_checker):
        """使用实际库测试处理guided_regex"""
        request = MockRequest()
        request.guided_regex = "[a-zA-Z]+"

        grammar = llguidance_checker.serialize_guidance_grammar(request)

        assert isinstance(grammar, str)
        assert len(grammar) > 0

    def test_serialize_guided_choice_real(self, llguidance_checker):
        """使用实际库测试处理guided_choice"""
        request = MockRequest()
        request.guided_choice = ["option1", "option2"]

        grammar = llguidance_checker.serialize_guidance_grammar(request)

        assert isinstance(grammar, str)
        assert len(grammar) > 0

    def test_serialize_guided_grammar_real(self, llguidance_checker):
        """使用实际库测试处理guided_grammar"""
        request = MockRequest()
        # 使用简单的CFG文法示例
        request.guided_grammar = """
        root ::= greeting name
        greeting ::= "Hello" | "Hi"
        name ::= "world" | "everyone"
        """

        grammar = llguidance_checker.serialize_guidance_grammar(request)

        assert isinstance(grammar, str)
        assert len(grammar) > 0

    def test_serialize_structural_tag_real(self, llguidance_checker):
        """使用实际库测试处理structural_tag"""
        request = MockRequest()
        request.structural_tag = {
            "triggers": ["<json>"],
            "structures": [{"begin": "<json>", "schema": {"type": "object"}, "end": "</json>"}],
        }

        grammar = llguidance_checker.serialize_guidance_grammar(request)

        assert isinstance(grammar, str)
        assert len(grammar) > 0

    def test_schema_format_valid_json_real(self, llguidance_checker):
        """使用实际库测试schema_format方法处理有效的JSON"""
        request = MockRequest()
        request.guided_json = '{"type": "object", "properties": {"name": {"type": "string"}}}'

        result_request, error = llguidance_checker.schema_format(request)

        assert error is None
        assert result_request is request
        assert result_request.guided_json != '{"type": "object", "properties": {"name": {"type": "string"}}}'

    def test_schema_format_invalid_json_real(self, llguidance_checker):
        """使用实际库测试schema_format方法处理无效的JSON"""
        request = MockRequest()
        request.guided_json = "{invalid json}"

        result_request, error = llguidance_checker.schema_format(request)

        assert error is not None
        assert "Invalid format for guided decoding" in error

    def test_whitespace_flexibility_option_real(self):
        """使用实际库测试whitespace灵活性选项的影响"""
        # 创建两个不同配置的实例
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

        # 预期两种配置生成的语法应该不同
        assert grammar_flexible != grammar_strict

    def test_schema_format_guided_json_object_real(self, llguidance_checker):
        """测试schema_format处理guided_json_object"""
        request = MockRequest()
        request.guided_json_object = True

        result_request, error = llguidance_checker.schema_format(request)

        assert error is None
        assert result_request is request

    def test_schema_format_guided_regex_real(self, llguidance_checker):
        """测试schema_format处理有效的正则表达式"""
        request = MockRequest()
        request.guided_regex = r"[a-zA-Z0-9]+"

        result_request, error = llguidance_checker.schema_format(request)

        assert error is None
        assert result_request is request
        assert result_request.guided_regex != r"[a-zA-Z0-9]+"  # 应该被转换为grammar格式

    def test_schema_format_invalid_guided_regex_real(self, llguidance_checker):
        """测试schema_format处理无效的正则表达式"""
        request = MockRequest()
        request.guided_regex = r"["  # 无效的正则表达式

        result_request, error = llguidance_checker.schema_format(request)

        assert error is not None
        assert "Invalid format for guided decoding" in error

    def test_schema_format_guided_choice_real(self, llguidance_checker):
        """测试schema_format处理guided_choice"""
        request = MockRequest()
        request.guided_choice = ["option1", "option2", "option3"]

        result_request, error = llguidance_checker.schema_format(request)

        assert error is None
        assert result_request is request
        assert result_request.guided_choice != ["option1", "option2", "option3"]  # 应该被转换为grammar格式

    def test_schema_format_guided_grammar_real(self, llguidance_checker):
        """测试schema_format处理guided_grammar"""
        request = MockRequest()
        # 使用LLGuidance支持的正确语法格式
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
        """测试schema_format处理structural_tag"""
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
        """测试schema_format处理字符串形式的structural_tag"""
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
        """测试schema_format处理trigger无效的structural_tag"""
        request = MockRequest()
        request.structural_tag = {
            "triggers": ["```xml"],  # 触发器与begin不匹配
            "structures": [
                {"begin": "```json", "schema": {"type": "object"}, "end": "```"}  # 这里不包含任何triggers中的前缀
            ],
        }

        result_request, error = llguidance_checker.schema_format(request)

        assert error is not None
        assert "Invalid format for guided decoding" in error

    def test_schema_format_structural_tag_empty_structures_real(self, llguidance_checker):
        """测试schema_format处理空structures的structural_tag"""
        request = MockRequest()
        request.structural_tag = {"triggers": ["```json"], "structures": []}  # 空结构

        result_request, error = llguidance_checker.schema_format(request)

        assert error is not None
        assert "Invalid format for guided decoding" in error

    def test_schema_format_json_dict_real(self, llguidance_checker):
        """测试schema_format处理字典形式的guided_json"""
        request = MockRequest()
        request.guided_json = {"type": "object", "properties": {"name": {"type": "string"}}}

        result_request, error = llguidance_checker.schema_format(request)

        assert error is None
        assert result_request is request

    def test_schema_format_disable_additional_properties_real(self):
        """测试schema_format处理disable_additional_properties参数"""
        checker = LLGuidanceChecker(disable_additional_properties=True)
        request = MockRequest()
        request.guided_json = {"type": "object", "properties": {"name": {"type": "string"}}}

        result_request, error = checker.schema_format(request)

        assert error is None
        assert result_request is request

    def test_schema_format_unexpected_error_real(self, monkeypatch, llguidance_checker):
        """测试schema_format处理意外错误"""
        request = MockRequest()
        request.guided_json = '{"type": "object"}'

        # 模拟意外异常
        def mock_serialize_grammar(*args, **kwargs):
            raise Exception("Unexpected error")

        monkeypatch.setattr(llguidance_checker, "serialize_guidance_grammar", mock_serialize_grammar)

        result_request, error = llguidance_checker.schema_format(request)

        assert error is not None
        assert "An unexpected error occurred during schema validation" in error

    def test_schema_format_no_valid_grammar_real(self, llguidance_checker):
        """测试schema_format处理没有有效语法的请求"""
        request = MockRequest()
        # 没有设置任何语法相关的属性

        with pytest.raises(ValueError, match="grammar is not of valid supported types"):
            llguidance_checker.serialize_guidance_grammar(request)
        result_request, error = llguidance_checker.schema_format(request)
        assert error is not None


if __name__ == "__main__":
    unittest.main()
