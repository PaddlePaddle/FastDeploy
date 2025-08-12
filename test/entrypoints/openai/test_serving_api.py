import os
import signal
import socket
import subprocess
import sys
import time
import unittest

import openai

from fastdeploy.utils import get_random_port

FD_API_PORT = int(os.getenv("FD_API_PORT", get_random_port()))
FD_ENGINE_QUEUE_PORT = int(os.getenv("FD_ENGINE_QUEUE_PORT", get_random_port()))
FD_METRICS_PORT = int(os.getenv("FD_METRICS_PORT", get_random_port()))
PORTS_TO_CLEAN = [FD_API_PORT, FD_ENGINE_QUEUE_PORT, FD_METRICS_PORT]


def is_port_open(host, port, timeout=1.0):
    try:
        with socket.create_connection((host, port), timeout):
            return True
    except Exception:
        return False


def kill_process_on_port(port):
    try:
        output = subprocess.check_output(f"lsof -i:{port} -t", shell=True).decode().strip()
        for pid in output.splitlines():
            os.kill(int(pid), signal.SIGKILL)
    except subprocess.CalledProcessError:
        pass


class TestServingAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        for port in PORTS_TO_CLEAN:
            kill_process_on_port(port)

        base_path = os.getenv("MODEL_PATH", ".")
        model_path = os.path.join(base_path, "ERNIE-4.5-0.3B-Paddle")

        cmd = [
            sys.executable,
            "-m",
            "fastdeploy.entrypoints.openai.api_server",
            "--model",
            model_path,
            "--port",
            str(FD_API_PORT),
            "--tensor-parallel-size",
            "1",
            "--engine-worker-queue-port",
            str(FD_ENGINE_QUEUE_PORT),
            "--metrics-port",
            str(FD_METRICS_PORT),
            "--max-model-len",
            "32768",
            "--max-num-seqs",
            "128",
            "--use-cudagraph",
            "--graph-optimization-config",
            '{"cudagraph_capture_sizes": [1]}',
        ]

        cls.server_proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True
        )

        for _ in range(300):
            if is_port_open("127.0.0.1", FD_API_PORT):
                break
            time.sleep(1)
        else:
            raise RuntimeError("API server did not start in time")

        cls.client = openai.Client(
            base_url=f"http://0.0.0.0:{FD_API_PORT}/v1",
            api_key="EMPTY_API_KEY",
        )

    @classmethod
    def tearDownClass(cls):
        try:
            os.killpg(cls.server_proc.pid, signal.SIGTERM)
        except Exception:
            pass

    def create_chat(self, **kwargs):
        default_kwargs = dict(
            model="default",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "List 3 countries and their capitals."},
            ],
            max_tokens=10,
            temperature=1,
            top_p=0,
        )
        default_kwargs.update(**kwargs)
        return self.client.chat.completions.create(**default_kwargs)

    def create_completion(self, **kwargs):
        default_kwargs = dict(
            model="default",
            prompt="Once upon a time, in a small village by the sea, there",
            max_tokens=10,
            temperature=1,
            top_p=0,
        )
        default_kwargs.update(**kwargs)
        return self.client.completions.create(**default_kwargs)

    def test_non_streaming_chat_base(self):
        """
        Test case for basic non-streaming chat
        """
        response = self.create_chat()
        self.assertTrue(response.choices[0].message.content)

    def test_streaming_chat_base(self):
        """
        Test case for basic streaming chat
        """
        response = self.create_chat(stream=True)
        output = ""
        for chunk in response:
            output += chunk.choices[0].delta.content
        self.assertTrue(output)

    def test_non_streaming_completion_base(self):
        """
        Test case for basic non-streaming completion
        """
        response = self.create_completion()
        self.assertTrue(response.choices[0].text)

    def test_streaming_completion_base(self):
        """
        Test case for basic streaming completion
        """
        response = self.create_completion(stream=True)
        output = ""
        for chunk in response:
            output += chunk.choices[0].text
        self.assertTrue(output)

    def test_non_streaming_chat_with_stop_str(self):
        """
        Test case for setting `include_stop_str_in_output` in non-streaming chat
        """
        response = self.create_chat(extra_body={"include_stop_str_in_output": True})
        self.assertTrue(response.choices[0].message.content.endswith("</s>"))

        response = self.create_chat(extra_body={"include_stop_str_in_output": False})
        self.assertFalse(response.choices[0].message.content.endswith("</s>"))

    def test_non_streaming_completion_with_stop_str(self):
        """
        Test case for setting `include_stop_str_in_output` in non-streaming completion
        """
        response = self.create_completion(max_tokens=1024)
        self.assertFalse(response.choices[0].text.endswith("</s>"))

        response = self.create_completion(max_tokens=1024, extra_body={"include_stop_str_in_output": True})
        self.assertTrue(response.choices[0].text.endswith("</s>"))

    def test_streaming_chat_with_stop_str(self):
        """
        Test case for setting `include_stop_str_in_output` in streaming chat
        """
        response = self.create_chat(
            extra_body={"include_stop_str_in_output": True},
            stream=True,
        )
        last_token = ""
        for chunk in response:
            last_token = chunk.choices[0].delta.content
        self.assertEqual(last_token, "</s>")

        response = self.create_chat(
            extra_body={"include_stop_str_in_output": False},
            stream=True,
        )
        last_token = ""
        for chunk in response:
            last_token = chunk.choices[0].delta.content
        self.assertNotEqual(last_token, "</s>")

    def test_streaming_completion_with_stop_str(self):
        """
        Test case for setting `include_stop_str_in_output` in streaming completion
        """
        response = self.create_completion(stream=True)
        last_token = ""
        for chunk in response:
            last_token = chunk.choices[0].text
        self.assertFalse(last_token.endswith("</s>"))

        response = self.create_completion(
            extra_body={"include_stop_str_in_output": True},
            stream=True,
        )
        last_token = ""
        for chunk in response:
            last_token = chunk.choices[0].text
        self.assertTrue(last_token.endswith("</s>"))

    def test_non_streaming_chat_with_return_token_ids(self):
        """
        Test case for setting `return_token_ids` in non-streaming chat
        """
        #  enable return_token_ids
        response = self.create_chat(
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            extra_body={"return_token_ids": True},
        )
        self.assertIsInstance(response.choices[0].message.prompt_token_ids, list)
        self.assertGreater(len(response.choices[0].message.prompt_token_ids), 0)
        self.assertIsInstance(response.choices[0].message.completion_token_ids, list)
        self.assertGreater(len(response.choices[0].message.completion_token_ids), 0)

        #  disable return_token_ids
        response = self.create_chat(
            model="default",
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            max_tokens=5,
            extra_body={"return_token_ids": False},
            stream=False,
        )
        self.assertIsNone(response.choices[0].message.prompt_token_ids)
        self.assertIsNone(response.choices[0].message.completion_token_ids)

    def test_streaming_chat_with_return_token_ids(self):
        """
        Tese case for setting `return_token_ids` in streaming chat
        """
        # enable return_token_ids
        response = self.create_chat(
            extra_body={"return_token_ids": True},
            stream=True,
        )
        is_first_chunk = True
        for chunk in response:
            delta = chunk.choices[0].delta
            if is_first_chunk:
                is_first_chunk = False
                self.assertIsInstance(delta.prompt_token_ids, list)
                self.assertGreater(len(delta.prompt_token_ids), 0)
                self.assertIsNone(delta.completion_token_ids)
            else:
                self.assertIsNone(delta.prompt_token_ids)
                self.assertIsInstance(delta.completion_token_ids, list)
                self.assertGreater(len(delta.completion_token_ids), 0)

        # disable return_token_ids
        response = self.create_chat(
            extra_body={"return_token_ids": False},
            stream=True,
        )
        for chunk in response:
            delta = chunk.choices[0].delta
            self.assertIsNone(delta.prompt_token_ids)
            self.assertIsNone(delta.completion_token_ids)

    def test_non_streaming_completion_with_return_token_ids(self):
        """
        Test case for setting `return_token_ids` in non-streaming completion
        """
        # enable return_token_ids
        response = self.create_completion(
            extra_body={"return_token_ids": True},
        )
        self.assertIsInstance(response.choices[0].prompt_token_ids, list)
        self.assertGreater(len(response.choices[0].prompt_token_ids), 0)
        self.assertIsInstance(response.choices[0].completion_token_ids, list)
        self.assertGreater(len(response.choices[0].completion_token_ids), 0)

        # disable return_token_ids
        response = self.create_completion()
        self.assertIsNone(response.choices[0].prompt_token_ids)
        self.assertIsNone(response.choices[0].completion_token_ids)

    def test_streaming_completion_with_return_token_ids(self):
        """
        Test case for setting `return_token_ids` in streaming completion
        """
        # enable return_token_ids
        response = self.create_completion(
            extra_body={"return_token_ids": True},
            stream=True,
        )
        is_first_chunk = True
        for chunk in response:
            choice = chunk.choices[0]
            if is_first_chunk:
                is_first_chunk = False
                self.assertIsInstance(choice.prompt_token_ids, list)
                self.assertGreater(len(choice.prompt_token_ids), 0)
                self.assertIsNone(choice.completion_token_ids)
            else:
                self.assertIsNone(choice.prompt_token_ids)
                self.assertIsInstance(choice.completion_token_ids, list)
                self.assertGreater(len(choice.completion_token_ids), 0)

        # disable return_token_ids
        response = self.create_completion(stream=True)
        for chunk in response:
            choice = chunk.choices[0]
            self.assertIsNone(choice.prompt_token_ids)
            self.assertIsNone(choice.completion_token_ids)

    def test_non_streaming_completion_with_prompt_token_ids(self):
        """
        Test case for passing token ids via `prompt_token_ids` or `prompt` in non-streaming completion
        """
        # passing a token id list in `prompt_token_ids`
        response = self.create_completion(
            extra_body={"prompt_token_ids": [1001, 1002, 1003, 1004, 1005]},
        )
        self.assertEqual(len(response.choices), 1)
        self.assertEqual(response.usage.prompt_tokens, 5)

        # passing a batch of token id lists in `prompt_token_ids`
        response = self.create_completion(
            extra_body={"prompt_token_ids": [[1001, 1002, 1003, 1004, 1005], [1006, 1007, 1008]]},
        )
        self.assertEqual(len(response.choices), 2)
        self.assertEqual(response.usage.prompt_tokens, 8)

        # passing a token id list in `prompt`
        response = self.create_completion(
            prompt=[1001, 1002, 1003, 1004, 1005],
        )
        self.assertEqual(len(response.choices), 1)
        self.assertEqual(response.usage.prompt_tokens, 5)

        # passing a batch of token id lists in `prompt`
        response = self.create_completion(
            prompt=[[1001, 1002, 1003, 1004, 1005], [1006, 1007, 1008]],
        )
        self.assertEqual(len(response.choices), 2)
        self.assertEqual(response.usage.prompt_tokens, 8)

    def test_streaming_completion_with_prompt_token_ids(self):
        """
        Test case for passing token ids via `prompt_token_ids` or `prompt` in streaming completion
        """
        # passing a token id list in `prompt_token_ids`
        response = self.create_completion(
            extra_body={"prompt_token_ids": [1001, 1002, 1003, 1004, 1005]},
            stream=True,
            stream_options={"include_usage": True},
        )
        sum_prompt_tokens = 0
        for chunk in response:
            if len(chunk.choices) > 0:
                self.assertIsNone(chunk.usage)
            else:
                sum_prompt_tokens += chunk.usage.prompt_tokens
        self.assertEqual(sum_prompt_tokens, 5)

        # passing a batch of token id lists in `prompt_token_ids`
        response = self.create_completion(
            extra_body={"prompt_token_ids": [[1001, 1002, 1003, 1004, 1005], [1006, 1007, 1008]]},
            stream=True,
            stream_options={"include_usage": True},
        )
        sum_prompt_tokens = 0
        for chunk in response:
            if len(chunk.choices) > 0:
                self.assertIsNone(chunk.usage)
            else:
                sum_prompt_tokens += chunk.usage.prompt_tokens
        self.assertEqual(sum_prompt_tokens, 8)

        # passing a token id list in `prompt`
        response = self.create_completion(
            prompt=[1001, 1002, 1003, 1004, 1005],
            stream=True,
            stream_options={"include_usage": True},
        )
        sum_prompt_tokens = 0
        for chunk in response:
            if len(chunk.choices) > 0:
                self.assertIsNone(chunk.usage)
            else:
                sum_prompt_tokens += chunk.usage.prompt_tokens
        self.assertEqual(sum_prompt_tokens, 5)

        # passing a batch of token id lists in `prompt`
        response = self.create_completion(
            prompt=[[1001, 1002, 1003, 1004, 1005], [1006, 1007, 1008]],
            stream=True,
            stream_options={"include_usage": True},
        )
        sum_prompt_tokens = 0
        for chunk in response:
            if len(chunk.choices) > 0:
                self.assertIsNone(chunk.usage)
            else:
                sum_prompt_tokens += chunk.usage.prompt_tokens
        self.assertEqual(sum_prompt_tokens, 8)

    def test_non_streaming_chat_with_disable_chat_template(self):
        """
        Test case for setting `disable_chat_template` in non-streaming chat
        """
        enabled_response = self.create_chat(
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            extra_body={"disable_chat_template": False},
        )
        self.assertGreater(len(enabled_response.choices), 0)

        # from fastdeploy.input.ernie_tokenizer import ErnieBotTokenizer
        # tokenizer = ErnieBotTokenizer.from_pretrained("PaddlePaddle/ERNIE-4.5-0.3B-Paddle", trust_remote_code=True)
        # prompt = tokenizer.apply_chat_template([{"role": "user", "content": "Hello, how are you?"}], tokenize=False)
        prompt = "<|begin_of_sentence|>User: Hello, how are you?\nAssistant: "
        disabled_response = self.create_chat(
            messages=[{"role": "user", "content": prompt}],
            extra_body={"disable_chat_template": True},
        )
        self.assertGreater(len(disabled_response.choices), 0)
        self.assertEqual(enabled_response.choices[0].message.content, disabled_response.choices[0].message.content)

    def test_non_streaming_chat_with_min_tokens(self):
        """
        Test case for setting `min_tokens` in non-streaming chat
        """
        min_tokens = 1000
        response = self.create_chat(
            max_tokens=1010,
            extra_body={"min_tokens": min_tokens},
        )
        self.assertGreaterEqual(response.usage.completion_tokens, min_tokens)

    def test_non_streaming_chat_with_min_max_token_equals_one(self):
        """
        Test case for chat/completion when min_tokens equals max_tokens equals 1.
        Verify it returns exactly one token.
        """
        # Test non-streaming chat
        response = self.create_chat(max_tokens=1)
        self.assertIsNotNone(response.choices[0].message.content)
        # Verify usage shows exactly 1 completion token
        self.assertEqual(response.usage.completion_tokens, 1)

    def test_non_streaming_chat_with_bad_words(self):
        """
        Test case for setting `bad_words` in non-streaming chat
        """
        resp0 = self.create_chat(max_tokens=10)
        words0 = resp0.choices[0].message.content.split(" ")

        # add bad words
        resp1 = self.create_chat(
            max_tokens=10,
            extra_body={"bad_words": words0[-5:]},
        )
        words1 = resp1.choices[0].message.content.split(" ")
        for w in words0[-5:]:
            self.assertNotIn(w, words1)

    def test_streaming_chat_with_bad_words(self):
        """
        Test case for setting `bad_words` in streaming chat
        """
        resp0 = self.create_chat(
            max_tokens=10,
            stream=True,
        )
        str0 = ""
        for chunk in resp0:
            str0 += chunk.choices[0].delta.content
        words0 = str0.split(" ")
        self.assertGreater(len(words0), 0)

        resp1 = self.create_chat(
            max_tokens=10,
            stream=True,
            extra_body={"bad_words": words0[-5:]},
        )
        str1 = "'"
        for chunk in resp1:
            str1 += chunk.choices[0].delta.content
        words1 = str1.split(" ")
        self.assertGreater(len(words1), 0)
        for w in words0[-5:]:
            self.assertNotIn(w, words1)

    def test_non_streaming_completion_with_bad_words(self):
        """
        Test case for setting `bad_words` in non-streaming completion
        """
        resp0 = self.create_completion(max_tokens=10)
        words0 = resp0.choices[0].text.split(" ")

        # add bad words
        resp1 = self.create_completion(
            max_tokens=10,
            extra_body={"bad_words": words0[-5:]},
        )
        words1 = resp1.choices[0].text.split(" ")
        for w in words0[-5:]:
            self.assertNotIn(w, words1)

    def test_streaming_completion_with_bad_words(self):
        """
        Test case for setting `bad_words` in streaming completion
        """
        resp0 = self.create_completion(
            max_tokens=10,
            stream=True,
        )
        str0 = ""
        for chunk in resp0:
            str0 += chunk.choices[0].text
        words0 = str0.split(" ")
        self.assertGreater(len(words0), 0)

        resp1 = self.create_completion(
            max_tokens=10,
            stream=True,
            extra_body={"bad_words": words0[-5:]},
        )
        str1 = ""
        for chunk in resp1:
            str1 += chunk.choices[0].text
        words1 = str1.split(" ")
        self.assertGreater(len(words1), 0)

        for w in words0[-5:]:
            self.assertNotIn(w, words1)


if __name__ == "__main__":
    unittest.main()
