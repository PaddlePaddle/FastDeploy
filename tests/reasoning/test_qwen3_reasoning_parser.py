import unittest

from fastdeploy.entrypoints.openai.protocol import ChatCompletionRequest
from fastdeploy.reasoning.qwen3_reasoning_parsers import Qwen3ReasoningParser


class MockTokenizer:
    """Minimal tokenizer with vocab for testing."""

    def __init__(self):
        self.vocab = {
            "<think>": 100,
            "</think>": 101,
        }

    def get_vocab(self):
        """Return vocab dict for testing."""
        return self.vocab


class MissingTokenTokenizer:
    def __init__(self):
        self.vocab = {
            "</think>": 100,
        }

    def get_vocab(self):
        """Return vocab dict for testing."""
        return self.vocab


class TestQwen3ReasoningParser(unittest.TestCase):
    def setUp(self):
        self.parser = Qwen3ReasoningParser(MockTokenizer())
        self.request = ChatCompletionRequest(model="test", messages=[{"role": "user", "content": "test message"}])
        self.tokenizer = MockTokenizer()

    def test_missing_token(self):
        with self.assertRaises(RuntimeError) as context:
            Qwen3ReasoningParser(MissingTokenTokenizer())
        exception_message = str(context.exception)
        expected_message_part = "Qwen3 reasoning parser could not find the following token ids"
        self.assertIn(expected_message_part, exception_message)

    def test_get_model_status(self):
        status = self.parser.get_model_status([1, 2, 100])
        self.assertEqual(status, "think_start")
        status = self.parser.get_model_status([1, 2, 101])
        self.assertEqual(status, "think_end")
        status = self.parser.get_model_status([1])
        self.assertEqual(status, "think_start")

    def test_streaming_thinking_content(self):
        msg = self.parser.extract_reasoning_content_streaming(
            previous_text="",
            current_text="a",
            delta_text="a",
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[200],
            model_status="think_start",
        )
        self.assertEqual(msg.reasoning_content, "a")

        msg = self.parser.extract_reasoning_content_streaming(
            previous_text="",
            current_text="a</think>b",
            delta_text="a</think>b",
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[99, 101, 102],
            model_status="think_start",
        )
        self.assertEqual(msg.reasoning_content, "a")
        self.assertEqual(msg.content, "b")

        msg = self.parser.extract_reasoning_content_streaming(
            previous_text="a</think>",
            current_text="a</think>b",
            delta_text="b",
            previous_token_ids=[1, 101],
            current_token_ids=[],
            delta_token_ids=[102],
            model_status="think_start",
        )
        self.assertEqual(msg.content, "b")

        msg = self.parser.extract_reasoning_content_streaming(
            previous_text="",
            current_text="a",
            delta_text="a",
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            model_status="think_start",
        )
        self.assertEqual(msg.reasoning_content, "a")

        msg = self.parser.extract_reasoning_content_streaming(
            previous_text="",
            current_text="a",
            delta_text="a",
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[200],
            model_status="think_end",
        )
        self.assertEqual(msg.content, "a")

        msg = self.parser.extract_reasoning_content_streaming(
            previous_text="hello",
            current_text="hello</think>hi",
            delta_text="</think>hi",
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[101, 200],
            model_status="think_start",
        )
        self.assertEqual(msg.content, "hi")
        self.assertEqual(msg.reasoning_content, "")

        msg = self.parser.extract_reasoning_content_streaming(
            previous_text="hello",
            current_text="hello</think>hi",
            delta_text="hi",
            previous_token_ids=[100],
            current_token_ids=[],
            delta_token_ids=[],
            model_status="think_start",
        )
        self.assertEqual(msg.content, None)
        self.assertEqual(msg.reasoning_content, "hi")

        msg = self.parser.extract_reasoning_content_streaming(
            previous_text="hello",
            current_text="hello</think>hi",
            delta_text="<think>hi",
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[100, 200],
            model_status="think_start",
        )
        self.assertEqual(msg.content, "")
        self.assertEqual(msg.reasoning_content, "hi")

    def test_none_streaming_thinking_content(self):
        reasoning_content, content = self.parser.extract_reasoning_content(
            model_output="a",
            request={},
            model_status="think_start",
        )
        self.assertEqual(reasoning_content, None)
        self.assertEqual(content, "a")

        reasoning_content, content = self.parser.extract_reasoning_content(
            model_output="a</think>b",
            request={},
            model_status="think_start",
        )
        self.assertEqual(reasoning_content, "a")
        self.assertEqual(content, "b")

        reasoning_content, content = self.parser.extract_reasoning_content(
            model_output="a",
            request={},
            model_status="think_end",
        )
        self.assertEqual(reasoning_content, None)
        self.assertEqual(content, "a")

        reasoning_content, content = self.parser.extract_reasoning_content(
            model_output="<think>a",
            request={},
            model_status="think_start",
        )
        self.assertEqual(reasoning_content, None)
        self.assertEqual(content, "<think>a")

        reasoning_content, content = self.parser.extract_reasoning_content(
            model_output="<think>a</think>b",
            request={},
            model_status="think_start",
        )
        self.assertEqual(reasoning_content, "a")
        self.assertEqual(content, "b")

        reasoning_content, content = self.parser.extract_reasoning_content(
            model_output="</think>b",
            request={},
            model_status="think_start",
        )
        self.assertEqual(reasoning_content, "")
        self.assertEqual(content, "b")


if __name__ == "__main__":
    unittest.main()
