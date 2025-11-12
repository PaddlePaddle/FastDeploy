from unittest.mock import MagicMock

import numpy as np
import pytest

from fastdeploy.engine.sampling_params import SamplingParams
from fastdeploy.entrypoints.llm import LLM
from fastdeploy.worker.output import Logprob, LogprobsTensors


class DummyModelConfig:
    def __init__(self, max_logprobs=10, ori_vocab_size=50):
        self.max_logprobs = max_logprobs
        self.ori_vocab_size = ori_vocab_size


@pytest.fixture
def mock_llm():
    llm = LLM.__new__(LLM)
    llm.llm_engine = MagicMock()
    llm.llm_engine.add_requests = MagicMock()
    llm.llm_engine.cfg.model_config = DummyModelConfig(max_logprobs=10, ori_vocab_size=100)
    # Mock the data_processor.process_logprob_response method to return proper strings
    llm.llm_engine.data_processor = MagicMock()
    llm.llm_engine.data_processor.process_logprob_response.side_effect = lambda ids, **kwargs: f"TOKEN_{ids[0]}"
    return llm


def test_prompt_logprobs_not_supported_with_stream(mock_llm):
    sampling = SamplingParams(prompt_logprobs=5)
    with pytest.raises(ValueError, match="prompt_logprobs is not supported with streaming"):
        mock_llm._add_request(["hi"], sampling, stream=True)


def test_num_logprobs_exceeds_max(mock_llm):
    sampling = SamplingParams(logprobs=20)
    with pytest.raises(ValueError, match="Number of logprobs requested"):
        mock_llm._add_request(["hi"], sampling)


def test_num_prompt_logprobs_exceeds_max(mock_llm):
    sampling = SamplingParams(prompt_logprobs=20)
    with pytest.raises(ValueError, match="Number of logprobs requested"):
        mock_llm._add_request(["hi"], sampling)


def test_logprobs_equal_to_minus_one_uses_ori_vocab_size(mock_llm):
    sampling = SamplingParams(logprobs=-1)
    mock_llm.llm_engine.cfg.model_config.max_logprobs = -1
    mock_llm.llm_engine.cfg.model_config.ori_vocab_size = 30
    mock_llm._add_request(["hi"], sampling)
    mock_llm.llm_engine.add_requests.assert_called_once()
    # Get the first argument (tasks) which should be a dict
    call_args = mock_llm.llm_engine.add_requests.call_args
    tasks = call_args[0][0]  # First positional argument
    assert isinstance(tasks, dict)
    assert "prompt" in tasks
    assert "request_id" in tasks


def test_prompt_logprobs_equal_to_minus_one(mock_llm):
    sampling = SamplingParams(prompt_logprobs=-1)
    mock_llm.llm_engine.cfg.model_config.max_logprobs = -1
    mock_llm.llm_engine.cfg.model_config.ori_vocab_size = 25
    mock_llm._add_request(["hi"], sampling)
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
    assert len(result) == 2
    for pos_dict in result:
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
    assert len(result) == 1
    pos_dict = result[0]
    assert 7 in pos_dict
    assert pos_dict[7].decoded_token == "TOKEN_7"
