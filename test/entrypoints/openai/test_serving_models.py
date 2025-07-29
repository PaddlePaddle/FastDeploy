from unittest.mock import MagicMock

import pytest

from fastdeploy.entrypoints.engine_client import EngineClient
from fastdeploy.entrypoints.openai.protocol import ModelInfo, ModelList
from fastdeploy.entrypoints.openai.serving_models import ModelPath, OpenAIServingModels
from fastdeploy.utils import get_host_ip

MODEL_NAME = "baidu/ERNIE-4.5-0.3B-PT"
MODEL_PATHS = [ModelPath(name=MODEL_NAME, model_path=MODEL_NAME)]
MAX_MODEL_LEN = 2048


async def _async_serving_models_init() -> OpenAIServingModels:
    mock_engine_client = MagicMock(spec=EngineClient)

    serving_models = OpenAIServingModels(
        engine_client=mock_engine_client,
        model_paths=MODEL_PATHS,
        max_model_len=MAX_MODEL_LEN,
        pid=1,
        ips=[get_host_ip()],
    )

    return serving_models


@pytest.mark.asyncio
async def test_serving_model_name():
    serving_models = await _async_serving_models_init()
    assert serving_models.model_name(None) == MODEL_NAME


@pytest.mark.asyncio
async def test_list_models(serving_models):
    serving_models = await _async_serving_models_init()
    result = serving_models.list_models()
    assert isinstance(result, ModelList)
    assert isinstance(result.data[0], ModelInfo)
    assert result.object == "list"
    assert len(result.data) == 1
    assert result.data[0].id == MODEL_NAME
    assert result.data[0].max_model_len == MAX_MODEL_LEN
    assert result.data[0].root == MODEL_PATHS[0].model_path
