import asyncio
import unittest

from fastdeploy.entrypoints.openai.protocol import ModelInfo, ModelList
from fastdeploy.entrypoints.openai.serving_models import ModelPath, OpenAIServingModels
from fastdeploy.utils import get_host_ip

MODEL_NAME = "baidu/ERNIE-4.5-0.3B-PT"
MODEL_PATHS = [ModelPath(name=MODEL_NAME, model_path=MODEL_NAME)]
MAX_MODEL_LEN = 2048


async def _async_serving_models_init() -> OpenAIServingModels:
    """Asynchronously initialize an OpenAIServingModels instance."""
    return OpenAIServingModels(
        model_paths=MODEL_PATHS,
        max_model_len=MAX_MODEL_LEN,
        ips=get_host_ip(),
    )


class TestOpenAIServingModels(unittest.TestCase):
    """Unit test for OpenAIServingModels"""

    def test_serving_model_name(self):
        """Test model name retrieval"""
        # 通过 asyncio.run() 执行异步初始化
        serving_models = asyncio.run(_async_serving_models_init())
        self.assertEqual(serving_models.model_name(), MODEL_NAME)

    def test_list_models(self):
        """Test the model listing functionality"""
        serving_models = asyncio.run(_async_serving_models_init())

        # 通过 asyncio.run() 执行异步方法
        result = asyncio.run(serving_models.list_models())

        # 验证返回类型和内容
        self.assertIsInstance(result, ModelList)
        self.assertEqual(len(result.data), 1)

        model_info = result.data[0]
        self.assertIsInstance(model_info, ModelInfo)
        self.assertEqual(model_info.id, MODEL_NAME)
        self.assertEqual(model_info.max_model_len, MAX_MODEL_LEN)
        self.assertEqual(model_info.root, MODEL_PATHS[0].model_path)
        self.assertEqual(result.object, "list")


if __name__ == "__main__":
    unittest.main()
