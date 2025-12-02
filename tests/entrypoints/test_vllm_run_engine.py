import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from fastdeploy.engine.sampling_params import SamplingParams
from fastdeploy.entrypoints.llm import LLM
from fastdeploy.worker.output import Logprob, LogprobsTensors


class DummyModelConfig:
    def __init__(self, max_logprobs=10, ori_vocab_size=50):
        self.max_logprobs = max_logprobs
        self.ori_vocab_size = ori_vocab_size


class DummyCacheConfig:
    def __init__(self, enable_prefix_caching=False):
        self.enable_prefix_caching = enable_prefix_caching


class DummyLLMEngineConfig:
    def __init__(self, model_config=None, cache_config=None):
        self.model_config = model_config or DummyModelConfig()
        self.cache_config = cache_config or DummyCacheConfig()


class DummyLLMEngine:
    def __init__(self, model_config=None, cache_config=None):
        self.cfg = DummyLLMEngineConfig(model_config, cache_config)
        self.data_processor = MagicMock()
        # Mock tokenizer with sp_model attribute
        self.data_processor.tokenizer = MagicMock()
        self.data_processor.tokenizer.sp_model = MagicMock()
        self.data_processor.tokenizer.sp_model.__len__ = MagicMock(return_value=100)
        self.data_processor.tokenizer.vocab = MagicMock()
        self.data_processor.tokenizer.vocab.__len__ = MagicMock(return_value=100)
        self.data_processor.process_logprob_response.side_effect = lambda ids, **kwargs: f"TOKEN_{ids[0]}"
        self.add_requests = MagicMock()


@pytest.fixture
def mock_llm():
    llm = LLM.__new__(LLM)
    llm.llm_engine = DummyLLMEngine()
    return llm


@pytest.fixture
def mock_llm_with_prefix_caching():
    llm = LLM.__new__(LLM)
    llm.llm_engine = DummyLLMEngine(cache_config=DummyCacheConfig(enable_prefix_caching=True))
    return llm


def test_prompt_logprobs_not_supported_with_stream(mock_llm):
    # Set FD_USE_GET_SAVE_OUTPUT_V1=1 to enable prompt_logprobs support
    with patch.dict(os.environ, {"FD_USE_GET_SAVE_OUTPUT_V1": "1"}):
        sampling = SamplingParams(prompt_logprobs=5)
        with pytest.raises(ValueError, match="prompt_logprobs is not supported with streaming"):
            mock_llm._add_request(["hi"], sampling, stream=True)


def test_prompt_logprobs_not_supported_with_prefix_caching(mock_llm_with_prefix_caching):
    # Set FD_USE_GET_SAVE_OUTPUT_V1=1 to enable prompt_logprobs support
    with patch.dict(os.environ, {"FD_USE_GET_SAVE_OUTPUT_V1": "1"}):
        sampling = SamplingParams(prompt_logprobs=5)
        with pytest.raises(ValueError, match="prompt_logprobs is not supported with prefix caching enabled"):
            mock_llm_with_prefix_caching._add_request(["hi"], sampling)


def test_num_logprobs_exceeds_max(mock_llm):
    # Set FD_USE_GET_SAVE_OUTPUT_V1=1 to allow logprobs > 20
    with patch.dict(os.environ, {"FD_USE_GET_SAVE_OUTPUT_V1": "1"}):
        sampling = SamplingParams(logprobs=20)
        with pytest.raises(ValueError, match="Number of logprobs requested"):
            mock_llm._add_request(["hi"], sampling)


def test_max_logprobs_exceeds_vocab_size(mock_llm):
    # Test case where max_logprobs > ori_vocab_size
    mock_llm.llm_engine.cfg.model_config.max_logprobs = 150  # > vocab size (100)
    with pytest.raises(ValueError, match="max_logprobs \\(150\\) exceeds vocabulary size \\(100\\)"):
        mock_llm._add_request(["hi"], SamplingParams())


def test_max_logprobs_less_than_minus_one(mock_llm):
    # Test case where max_logprobs < -1
    mock_llm.llm_engine.cfg.model_config.max_logprobs = -2
    with pytest.raises(ValueError, match="max_logprobs \\(-2\\) can't be less than -1"):
        mock_llm._add_request(["hi"], SamplingParams())


def test_logprobs_minus_one_uses_vocab_size(mock_llm):
    # Test that logprobs=-1 uses vocab size
    with patch.dict(os.environ, {"FD_USE_GET_SAVE_OUTPUT_V1": "1"}):
        sampling = SamplingParams(logprobs=-1)
        mock_llm.llm_engine.cfg.model_config.max_logprobs = -1  # Allow unlimited
        mock_llm._add_request(["hi"], sampling)
        mock_llm.llm_engine.add_requests.assert_called_once()


def test_num_prompt_logprobs_exceeds_max(mock_llm):
    # Set FD_USE_GET_SAVE_OUTPUT_V1=1 to enable prompt_logprobs support
    with patch.dict(os.environ, {"FD_USE_GET_SAVE_OUTPUT_V1": "1"}):
        sampling = SamplingParams(prompt_logprobs=20)
        with pytest.raises(ValueError, match="Number of logprobs requested"):
            mock_llm._add_request(["hi"], sampling)


def test_logprobs_equal_to_minus_one_uses_ori_vocab_size(mock_llm):
    # Set FD_USE_GET_SAVE_OUTPUT_V1=1 to allow logprobs=-1
    with patch.dict(os.environ, {"FD_USE_GET_SAVE_OUTPUT_V1": "1"}):
        sampling = SamplingParams(logprobs=-1)
        mock_llm.llm_engine.cfg.model_config.max_logprobs = -1
        mock_llm._add_request(["hi"], sampling)
        mock_llm.llm_engine.add_requests.assert_called_once()
        # Get the first argument (tasks) which should be a dict
        call_args = mock_llm.llm_engine.add_requests.call_args
        tasks = call_args[0][0]  # First positional argument
        assert isinstance(tasks, dict)
        assert "prompt" in tasks
        assert "request_id" in tasks


def test_prompt_logprobs_equal_to_minus_one(mock_llm):
    # Set FD_USE_GET_SAVE_OUTPUT_V1=1 to enable prompt_logprobs support and allow -1
    with patch.dict(os.environ, {"FD_USE_GET_SAVE_OUTPUT_V1": "1"}):
        sampling = SamplingParams(prompt_logprobs=-1)
        mock_llm.llm_engine.cfg.model_config.max_logprobs = -1
        mock_llm._add_request(["hi"], sampling)
        mock_llm.llm_engine.add_requests.assert_called_once()


def test_dynamic_vocab_size_from_sp_model(mock_llm):
    # Test that ori_vocab_size is dynamically obtained from sp_model
    mock_llm.llm_engine.data_processor.tokenizer.sp_model.__len__.return_value = 200
    mock_llm.llm_engine.cfg.model_config.max_logprobs = -1

    with patch.dict(os.environ, {"FD_USE_GET_SAVE_OUTPUT_V1": "1"}):
        sampling = SamplingParams(logprobs=-1)
        mock_llm._add_request(["hi"], sampling)
        # Should use the dynamic vocab size (200)
        mock_llm.llm_engine.add_requests.assert_called_once()


def test_dynamic_vocab_size_from_vocab_fallback(mock_llm):
    # Test fallback to vocab when sp_model is not available
    del mock_llm.llm_engine.data_processor.tokenizer.sp_model
    mock_llm.llm_engine.data_processor.tokenizer.vocab.__len__.return_value = 300
    mock_llm.llm_engine.cfg.model_config.max_logprobs = -1

    with patch.dict(os.environ, {"FD_USE_GET_SAVE_OUTPUT_V1": "1"}):
        sampling = SamplingParams(logprobs=-1)
        mock_llm._add_request(["hi"], sampling)
        # Should use the vocab size (300)
        mock_llm.llm_engine.add_requests.assert_called_once()


def test_build_prompt_logprobs_basic(mock_llm):
    # 构造 2 个 token，每个 token 对应 3 个 logprob 值
    token_ids = np.array([[1, 2, 3], [4, 5, 6]])
    logprobs = np.array([[-0.1, -0.2, -0.3], [-0.4, -0.5, -0.6]])
    ranks = np.array([1, 2])
    tensors = LogprobsTensors(token_ids, logprobs, ranks)

    result = mock_llm._build_prompt_logprobs(tensors, num_prompt_logprobs=2)

    # 检查结果格式
    assert isinstance(result, list)
    assert len(result) == 3
    for pos_dict in result:
        if pos_dict is not None:
            assert isinstance(pos_dict, dict)
            for logprob_obj in pos_dict.values():
                assert isinstance(logprob_obj, Logprob)
                assert logprob_obj.decoded_token.startswith("TOKEN_")


def test_build_prompt_logprobs_handles_minus_one(mock_llm):
    token_ids = np.array([[7, 8]])
    logprobs = np.array([[-0.9, -1.0]])
    ranks = np.array([1])
    tensors = LogprobsTensors(token_ids, logprobs, ranks)

    result = mock_llm._build_prompt_logprobs(tensors, num_prompt_logprobs=-1)

    assert isinstance(result, list)
    assert len(result) == 2
    pos_dict = result[1]
    assert 7 in pos_dict
    assert pos_dict[7].decoded_token == "TOKEN_7"
