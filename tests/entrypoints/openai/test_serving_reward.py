import time
import unittest
from unittest.mock import AsyncMock, MagicMock

from fastdeploy.engine.request import (
    PoolingOutput,
    PoolingRequestOutput,
    RequestMetrics,
)
from fastdeploy.entrypoints.openai.protocol import ChatRewardRequest, ChatRewardResponse
from fastdeploy.entrypoints.openai.serving_reward import OpenAIServingReward


class TestOpenAIServingReward(unittest.IsolatedAsyncioTestCase):
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
        self.reward_service = OpenAIServingReward(
            self.mock_engine_client, models, cfg, pid, ips, max_waiting_time, chat_template
        )

    async def test_create_reward_success(self):
        # Setup
        request = ChatRewardRequest(
            model="text-reward-ada-002",
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
        )

        # Execute
        result: ChatRewardResponse = await self.reward_service.create_reward(request)

        # Assert
        self.assertEqual(result.data[0].score, self.response_data.outputs.data)


if __name__ == "__main__":
    unittest.main()
