import unittest
from unittest.mock import MagicMock, patch

from fastdeploy.entrypoints.llm import LLM


def get_patch_path(cls, method="__init__"):
    return f"{cls.__module__}.{cls.__qualname__}.{method}"


class TestCompletionEcho(unittest.TestCase):

    def setUp(self):
        """
        Set up the test environment by creating an instance of the LLM class using Mock.
        """
        patch_llm = get_patch_path(LLM)
        with patch(patch_llm, return_value=None):
            self.llm = LLM()
            # Mock OpenAI client
            self.openai_client = MagicMock()

    def test_non_streaming_prompt_echo_response(self):
        """
        Test echo option in non-streaming completion functionality.
        """
        # Test single prompt
        self.openai_client.completions.create.return_value = MagicMock()
        self.openai_client.completions.create.return_value.choices = [
            MagicMock(text="Hello, how are you? Some response")
        ]

        response = self.openai_client.completions.create(
            model="default",
            prompt="Hello, how are you?",
            temperature=1,
            max_tokens=10,
            stream=False,
            echo=True,
        )
        self.assertTrue(response.choices[0].text.startswith("Hello, how are you?"))

        # Test multiple prompts
        prompts = ["Hello, how are you?", "What is your name?"]
        self.openai_client.completions.create.return_value.choices = [
            MagicMock(text=prompt + " Some response") for prompt in prompts
        ]
        response = self.openai_client.completions.create(
            model="default",
            prompt=prompts,
            temperature=1,
            max_tokens=10,
            stream=False,
            echo=True,
        )
        for i in range(len(response.choices)):
            self.assertTrue(response.choices[i].text.startswith(prompts[i]))

    def test_streaming_prompt_echo_response(self):
        """
        Test echo option in streaming completion functionality.
        """
        # Mock streaming response
        mock_stream = [
            MagicMock(choices=[MagicMock(index=0, text="Hello, how are you?")]),
            MagicMock(choices=[MagicMock(index=0, text=" Some")]),
            MagicMock(choices=[MagicMock(index=0, text=" response")]),
        ]
        self.openai_client.completions.create.return_value = mock_stream

        response = self.openai_client.completions.create(
            model="default",
            prompt="Hello, how are you?",
            temperature=1,
            max_tokens=10,
            stream=True,
            echo=True,
        )
        output = []
        for chunk in response:
            output.append(chunk.choices[0].text)
        self.assertTrue("".join(output).startswith("Hello, how are you?"))

        # Test multiple prompts
        prompts = ["Hello, how are you?", "What is your name?"]
        mock_stream = [MagicMock(choices=[MagicMock(index=i, text=prompts[i])]) for i in range(len(prompts))] + [
            MagicMock(choices=[MagicMock(index=i, text=" Some")]) for i in range(len(prompts))
        ]
        self.openai_client.completions.create.return_value = mock_stream

        response = self.openai_client.completions.create(
            model="default",
            prompt=prompts,
            temperature=1,
            max_tokens=10,
            stream=True,
            echo=True,
        )
        outputs = {i: [] for i in range(len(prompts))}
        for chunk in response:
            for choice in chunk.choices:
                index = choice.index
                text = choice.text
                outputs[index].append(text)
                if len(outputs[index]) == 1:
                    self.assertTrue(
                        text.startswith(prompts[index]),
                        f"Prompt {index} first response '{text}' doesn't match prompt '{prompts[index]}'",
                    )


if __name__ == "__main__":
    unittest.main()
