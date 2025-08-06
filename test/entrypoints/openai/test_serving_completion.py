import unittest
from unittest.mock import Mock

from fastdeploy.entrypoints.openai.serving_completion import OpenAIServingCompletion


class TestOpenAIServingCompletion(unittest.TestCase):

    def test_calc_finish_reason_tool_calls(self):
        # 创建一个模拟的engine_client，并设置reasoning_parser为"ernie_x1"
        engine_client = Mock()
        engine_client.reasoning_parser = "ernie_x1"
        # 创建一个OpenAIServingCompletion实例
        serving_completion = OpenAIServingCompletion(engine_client, "pid", "ips")
        # 创建一个模拟的output，并设置finish_reason为"tool_calls"
        output = {"finish_reason": "tool_calls"}
        # 调用calc_finish_reason方法
        result = serving_completion.calc_finish_reason(None, 100, output)
        # 断言结果为"tool_calls"
        assert result == "tool_calls"

    def test_calc_finish_reason_stop(self):
        # 创建一个模拟的engine_client，并设置reasoning_parser为"ernie_x1"
        engine_client = Mock()
        engine_client.reasoning_parser = "ernie_x1"
        # 创建一个OpenAIServingCompletion实例
        serving_completion = OpenAIServingCompletion(engine_client, "pid", "ips")
        # 创建一个模拟的output，并设置finish_reason为其他值
        output = {"finish_reason": "other_reason"}
        # 调用calc_finish_reason方法
        result = serving_completion.calc_finish_reason(None, 100, output)
        # 断言结果为"stop"
        assert result == "stop"

    def test_calc_finish_reason_length(self):
        # 创建一个模拟的engine_client
        engine_client = Mock()
        # 创建一个OpenAIServingCompletion实例
        serving_completion = OpenAIServingCompletion(engine_client, "pid", "ips")
        # 创建一个模拟的output
        output = {}
        # 调用calc_finish_reason方法
        result = serving_completion.calc_finish_reason(100, 100, output)
        # 断言结果为"length"
        assert result == "length"


if __name__ == "__main__":
    unittest.main()
