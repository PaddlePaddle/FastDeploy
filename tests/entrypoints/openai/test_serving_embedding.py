import time
import unittest
from unittest.mock import AsyncMock, MagicMock

from fastdeploy.engine.request import (
    PoolingOutput,
    PoolingRequestOutput,
    RequestMetrics,
)
from fastdeploy.entrypoints.openai.protocol import (
    EmbeddingChatRequest,
    EmbeddingCompletionRequest,
    EmbeddingRequest,
    EmbeddingResponse,
)
from fastdeploy.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from fastdeploy.entrypoints.openai.serving_engine import ServeContext


class TestOpenAIServingEmbedding(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_engine_client = MagicMock()
        self.mock_engine_client.semaphore.acquire = AsyncMock()
        self.mock_engine_client.semaphore.release = MagicMock()

        self.mock_engine_client.check_model_weight_status = AsyncMock(return_value=False)

        mock_dealer = MagicMock()
        mock_response_queue = MagicMock()
        self.response_data: PoolingRequestOutput = PoolingRequestOutput(
            request_id="test_request_id",
            prompt_token_ids=[1, 2, 3],
            finished=True,
            outputs=PoolingOutput(data=[0.1, 0.2, 0.3]),
            metrics=RequestMetrics(arrival_time=time.time()),
        )
        mock_response_queue.get = AsyncMock(
            return_value=[
                self.response_data.to_dict(),
            ]
        )
        self.mock_engine_client.connection_manager.get_connection = AsyncMock(
            return_value=(mock_dealer, mock_response_queue)
        )

        self.mock_engine_client.connection_manager.cleanup_request = AsyncMock()
        self.mock_engine_client.format_and_add_data = AsyncMock(return_value=[[1, 2, 3]])
        models = MagicMock()
        models.is_supported_model = MagicMock(return_value=(True, "ERNIE"))
        pid = 123
        ips = ["127.0.0.1"]
        max_waiting_time = 30
        chat_template = MagicMock()
        cfg = MagicMock()
        self.embedding_service = OpenAIServingEmbedding(
            self.mock_engine_client, models, cfg, pid, ips, max_waiting_time, chat_template
        )

    async def test_create_embedding_success(self):
        # Setup
        request = EmbeddingChatRequest(
            model="text-embedding-ada-002",
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ],
        )

        # Execute
        result: EmbeddingResponse = await self.embedding_service.create_embedding(request)

        # Assert
        self.assertEqual(result.data[0].embedding, self.response_data.outputs.data)

    def test_request_to_batch_dicts(self):
        test_cases = [
            ("string input", EmbeddingCompletionRequest(input="hello"), ["hello"], ["req-1_0"]),
            ("list of ints", EmbeddingCompletionRequest(input=[1, 2, 3]), [[1, 2, 3]], ["req-1_0"]),
            ("list of strings", EmbeddingCompletionRequest(input=["a", "b"]), ["a", "b"], ["req-1_0", "req-1_1"]),
            (
                "list of list of ints",
                EmbeddingCompletionRequest(input=[[1, 2], [3, 4]]),
                [[1, 2], [3, 4]],
                ["req-1_0", "req-1_1"],
            ),
        ]

        for name, request, expected_prompts, expected_ids in test_cases:
            with self.subTest(name=name):
                ctx = ServeContext[EmbeddingRequest](
                    request=request,
                    model_name="request.model",
                    request_id="req-1",
                )
                result = self.embedding_service._request_to_batch_dicts(ctx)
                self.assertEqual(len(result), len(expected_prompts))
                for r, prompt, rid in zip(result, expected_prompts, expected_ids):
                    # print(f"assertEqual r:{r} prompt:{prompt} rid:{rid}")
                    self.assertEqual(r["prompt"], prompt)
                    self.assertEqual(r["request_id"], rid)

        # 测试非 EmbeddingCompletionRequest 输入
        with self.subTest(name="non-embedding request"):
            with self.assertRaises(AttributeError):
                ctx = ServeContext(request={"foo": "bar"}, model_name="request.model", request_id="req-1")
                result = self.embedding_service._request_to_batch_dicts(ctx)


if __name__ == "__main__":
    unittest.main()
