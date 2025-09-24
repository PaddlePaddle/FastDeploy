import unittest
from unittest.mock import AsyncMock, MagicMock

from fastdeploy.entrypoints.openai.protocol import (
    EmbeddingChatRequest,
    EmbeddingCompletionRequest,
)
from fastdeploy.entrypoints.openai.serving_embedding import OpenAIServingEmbedding


class TestOpenAIServingEmbedding(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_engine_client = MagicMock()
        self.mock_engine_client.semaphore.acquire = AsyncMock()
        self.mock_engine_client.format_and_add_data = AsyncMock(return_value=[[1, 2, 3]])
        models = MagicMock()
        models.is_supported_model = MagicMock(return_value=(True, "ERNIE"))
        pid = 123
        ips = ["127.0.0.1"]
        max_waiting_time = 30
        self.embedding_service = OpenAIServingEmbedding(self.mock_engine_client, models, pid, ips, max_waiting_time)

    async def test_create_embedding_success(self):
        # Setup
        mock_response = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

        request = EmbeddingChatRequest(
            model="text-embedding-ada-002",
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ],
        )

        request = EmbeddingCompletionRequest(
            model="text-embedding-ada-002",
            input="Hello world",
        )

        # Execute
        result = await self.embedding_service.create_embedding(request)

        # Assert
        self.assertEqual(result, mock_response)
        # self.mock_engine_client.handle.assert_awaited_once_with(request)

    # async def test_create_embedding_multiple_inputs(self):
    #     # Setup
    #     mock_response = {"data": [{"embedding": [0.1, 0.2, 0.3]}, {"embedding": [0.4, 0.5, 0.6]}]}
    #     self.mock_engine_client.handle = AsyncMock(return_value=mock_response)

    #     request = EmbeddingCompletionRequest(input=["first text", "second text"], model="text-embedding-ada-002")

    #     # Execute
    #     result = await self.embedding_service.create_embedding(request)

    #     # Assert
    #     self.assertEqual(result, mock_response)
    #     self.assertEqual(len(result["data"]), 2)
    #     self.mock_engine_client.handle.assert_awaited_once_with(request)

    # async def test_create_embedding_unsupported_model(self):
    #     # Setup
    #     request = EmbeddingCompletionRequest(input=["test text"], model="unsupported-model")

    #     # Execute & Assert
    #     with self.assertRaises(ValueError) as context:
    #         await self.embedding_service.create_embedding(request)

    #     self.assertIn("Model unsupported-model not found", str(context.exception))
    #     self.mock_engine_client.handle.assert_not_called()

    # async def test_create_embedding_empty_input(self):
    #     # Setup
    #     mock_response = {"data": []}
    #     self.mock_engine_client.handle = AsyncMock(return_value=mock_response)

    #     request = EmbeddingCompletionRequest(input=[], model="text-embedding-ada-002")

    #     # Execute
    #     result = await self.embedding_service.create_embedding(request)

    #     # Assert
    #     self.assertEqual(result, mock_response)
    #     self.assertEqual(len(result["data"]), 0)
    #     self.mock_engine_client.handle.assert_awaited_once_with(request)


if __name__ == "__main__":
    unittest.main()
