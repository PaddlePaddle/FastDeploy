# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref

import pytest

from fastdeploy.engine.request import RequestOutput
from fastdeploy.engine.sampling_params import SamplingParams
from fastdeploy.entrypoints.llm import LLM
from fastdeploy.utils import get_random_port

MODEL_NAME = "PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle"

PROMPTS = [
    "Hello, my name is",
    "The capital of China is",
    "The future of AI is",
    "人工智能是",
]

TOKEN_IDS = [
    [0],
    [0, 1],
    [0, 1, 3],
    [0, 2, 4, 6],
]


@pytest.fixture(scope="module")
def llm():
    # pytest caches the fixture so we use weakref.proxy to
    # enable garbage collection
    llm = LLM(
        model=MODEL_NAME,
        max_num_batched_tokens=4096,
        tensor_parallel_size=1,
        engine_worker_queue_port=get_random_port(),
    )
    yield weakref.proxy(llm)


def assert_outputs_equal(o1: list[RequestOutput], o2: list[RequestOutput]):
    assert [o.outputs for o in o1] == [o.outputs for o in o2]


@pytest.mark.parametrize("prompt_token_ids", TOKEN_IDS)
def test_consistency_single_prompt_tokens(llm: LLM, prompt_token_ids):
    sampling_params = SamplingParams(temperature=1.0, top_p=0.0)

    output1 = llm.generate(prompts=prompt_token_ids, sampling_params=sampling_params)

    output2 = llm.generate({"prompt": "", "prompt_token_ids": prompt_token_ids}, sampling_params=sampling_params)
    assert_outputs_equal(output1, output2)


def test_api_consistency_multi_prompt_tokens(llm: LLM):
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=0.0,
    )

    output1 = llm.generate(prompts=TOKEN_IDS, sampling_params=sampling_params)

    output2 = llm.generate(
        [{"prompt": "", "prompt_token_ids": p} for p in TOKEN_IDS],
        sampling_params=sampling_params,
    )

    assert_outputs_equal(output1, output2)


def test_multiple_sampling_params(llm: LLM):
    sampling_params = [
        SamplingParams(temperature=0.01, top_p=0.95),
        SamplingParams(temperature=0.3, top_p=0.95),
        SamplingParams(temperature=0.7, top_p=0.95),
        SamplingParams(temperature=0.99, top_p=0.95),
    ]

    # Multiple SamplingParams should be matched with each prompt
    outputs = llm.generate(prompts=PROMPTS, sampling_params=sampling_params)
    assert len(PROMPTS) == len(outputs)

    # Exception raised, if the size of params does not match the size of prompts
    with pytest.raises(ValueError):
        outputs = llm.generate(prompts=PROMPTS, sampling_params=sampling_params[:3])

    # Single SamplingParams should be applied to every prompt
    single_sampling_params = SamplingParams(temperature=0.3, top_p=0.95)
    outputs = llm.generate(prompts=PROMPTS, sampling_params=single_sampling_params)
    assert len(PROMPTS) == len(outputs)

    # sampling_params is None, default params should be applied
    outputs = llm.generate(prompts=PROMPTS, sampling_params=None)
    assert len(PROMPTS) == len(outputs)


if __name__ == "__main__":
    pytest.main([__file__])
