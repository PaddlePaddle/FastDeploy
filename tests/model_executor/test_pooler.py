# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import paddle
import pytest
from paddleformers.transformers.configuration_utils import PretrainedConfig

from fastdeploy.config import ModelConfig, PoolerConfig
from fastdeploy.engine.pooling_params import PoolingParams
from fastdeploy.model_executor.layers.pool.metadata import PoolingMetadata
from fastdeploy.model_executor.layers.pooler import (
    AllPool,
    CLSPool,
    DispatchPooler,
    EmbeddingPoolerHead,
    LastPool,
    MeanPool,
    Pooler,
    PoolerActivation,
    PoolerClassify,
    PoolerIdentity,
    PoolingType,
    ResolvedPoolingConfig,
    RewardPoolerHead,
    SimplePooler,
    StepPooler,
)


@pytest.fixture(autouse=True)
def _stub_pretrained_config(monkeypatch):
    """Avoid remote downloads when constructing :class:`ModelConfig`."""

    dummy_config = {
        "hidden_size": 8,
        "num_attention_heads": 2,
        "vocab_size": 10,
        "architectures": ["TestModel"],
    }

    def _fake_get_config_dict(cls, _name, **_kwargs):
        return dummy_config.copy(), None

    monkeypatch.setattr(PretrainedConfig, "get_config_dict", classmethod(_fake_get_config_dict))
    monkeypatch.setattr("fastdeploy.config.get_pooling_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("fastdeploy.config.ModelConfig._post_init", lambda self: None)

    original_cumsum = paddle.cumsum

    def _compatible_cumsum(x, axis=0, out=None):
        result = original_cumsum(x, axis=axis)
        if out is not None:
            out[:] = result
            return out
        return result

    monkeypatch.setattr(paddle, "cumsum", _compatible_cumsum)

    def _fake_fdconfig_init(self, model_config=None, cache_config=None, **kwargs):
        self.model_config = model_config or SimpleNamespace(num_labels=0)
        self.cache_config = cache_config
        self.scheduler_config = SimpleNamespace(max_num_seqs=0)
        self.parallel_config = kwargs.get("parallel_config")
        self.speculative_config = kwargs.get("speculative_config")
        self.eplb_config = kwargs.get("eplb_config")
        self.device_config = kwargs.get("device_config")
        self.load_config = kwargs.get("load_config")
        self.quant_config = kwargs.get("quant_config")
        self.graph_opt_config = kwargs.get("graph_opt_config")
        self.early_stop_config = kwargs.get("early_stop_config")
        self.plas_attention_config = kwargs.get("plas_attention_config")
        self.structured_outputs_config = kwargs.get("structured_outputs_config")
        self.router_config = kwargs.get("router_config")
        self.routing_replay_config = kwargs.get("routing_replay_config")

    monkeypatch.setattr("fastdeploy.config.FDConfig.__init__", _fake_fdconfig_init)
    yield


def build_metadata(
    prompt_lens: list[int],
    pooling_params: list[PoolingParams],
    *,
    token_ids: paddle.Tensor | None = None,
    num_tokens: list[int] | None = None,
):
    prompt_tensor = paddle.to_tensor(prompt_lens, dtype="int64")
    metadata = PoolingMetadata(
        prompt_lens=prompt_tensor,
        prompt_token_ids=token_ids,
        pooling_params=pooling_params,
    )
    metadata.build_pooling_cursor(num_tokens or prompt_lens, paddle.CPUPlace())
    return metadata


def make_model_config() -> ModelConfig:
    return ModelConfig({"model": "This is Model~"})


class TestResolvedConfigAndFactories:
    def test_resolved_config_and_pooler_factories(self):
        pooler_cfg = PoolerConfig()
        pooler_cfg.pooling_type = "MEAN"
        resolved = ResolvedPoolingConfig.from_config("embed", pooler_cfg)
        assert resolved.pooling_type is PoolingType.MEAN

        embed_pooler = Pooler.for_embed(pooler_cfg, make_model_config())
        assert isinstance(embed_pooler.head, EmbeddingPoolerHead)

        encode_cfg = PoolerConfig()
        encode_cfg.pooling_type = "STEP"
        encode_pooler = Pooler.for_encode(encode_cfg, make_model_config())
        assert isinstance(encode_pooler, StepPooler)

        reward_cfg = PoolerConfig()
        reward_cfg.pooling_type = "LAST"
        reward_pooler = Pooler.for_reward(reward_cfg, make_model_config())
        assert isinstance(reward_pooler.head, RewardPoolerHead)

    def test_pooler_activation_wrappers_and_classify_paths(self):
        assert isinstance(PoolerActivation.wraps(paddle.nn.Identity()), PoolerIdentity)
        assert isinstance(PoolerActivation.wraps(paddle.nn.Sigmoid()), PoolerClassify)

        custom = PoolerActivation.wraps(paddle.nn.Layer())
        assert custom.__class__.__name__ == "LambdaPoolerActivation"

        classify = PoolerClassify(static_num_labels=False)
        sigmoid_out = classify.forward_chunk(paddle.to_tensor([0.0], dtype="float32"))
        np.testing.assert_allclose(sigmoid_out.numpy(), 0.5, rtol=1e-3)

        softmax_out = classify.forward_chunk(paddle.to_tensor([0.0, 1.0, 2.0], dtype="float32"))
        np.testing.assert_allclose(softmax_out.numpy().sum(), 1.0, rtol=1e-6)
        assert softmax_out.shape[0] == 3


class TestPoolerHeads:
    def test_embedding_head_dimensions_and_normalization(self):
        pooling_params = [
            PoolingParams(task="embed", dimensions=2, normalize=True),
            PoolingParams(task="embed", dimensions=None, normalize=False),
        ]
        metadata = build_metadata([1, 1], pooling_params)
        pooled_data = [
            paddle.to_tensor([1.0, 2.0, 3.0], dtype="float32"),
            paddle.to_tensor([3.0, 4.0, 0.0], dtype="float32"),
        ]
        head = EmbeddingPoolerHead()
        output = head.forward(pooled_data, metadata)

        assert isinstance(output, list)
        assert output[0].shape[0] == 2
        np.testing.assert_allclose(np.linalg.norm(output[0].numpy()), 1.0, rtol=1e-5)
        assert output[1].shape[0] == 3
        np.testing.assert_allclose(output[1].numpy(), pooled_data[1].numpy())

    def test_reward_head_softmax_flags(self):
        pooling_params = [PoolingParams(task="encode", softmax=True), PoolingParams(task="encode", softmax=False)]
        metadata = build_metadata([1, 1], pooling_params)
        logits = [
            paddle.to_tensor([[0.0, 1.0]], dtype="float32"),
            paddle.to_tensor([[0.5, 0.5]], dtype="float32"),
        ]
        head = RewardPoolerHead()
        outputs = head.forward(logits, metadata)

        assert isinstance(outputs, list)
        np.testing.assert_allclose(outputs[0].numpy().sum(axis=-1), 1.0, rtol=1e-6)
        np.testing.assert_allclose(outputs[1].numpy(), logits[1].numpy())


class TestPoolingMethods:
    def setup_method(self):
        self.pooling_params = [PoolingParams(task="encode"), PoolingParams(task="encode")]
        self.metadata = build_metadata([2, 2], self.pooling_params)
        self.hidden_states = paddle.to_tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype="float32")

    def test_last_cls_all_and_mean_pool(self):
        cursor = self.metadata.pooling_cursor
        last_pool = LastPool()
        last_out = last_pool.forward(self.hidden_states, self.metadata)
        np.testing.assert_array_equal(
            last_out.numpy(), self.hidden_states.numpy()[cursor.last_token_indices_gpu.numpy()]
        )

        cls_pool = CLSPool()
        cls_out = cls_pool.forward(self.hidden_states, self.metadata)
        np.testing.assert_array_equal(
            cls_out.numpy(), self.hidden_states.numpy()[cursor.first_token_indices_gpu.numpy()]
        )

        all_pool = AllPool()
        all_out = all_pool.forward(self.hidden_states, self.metadata)
        assert len(all_out) == len(self.pooling_params)
        assert all(list(t.shape) == [2, 2] for t in all_out)

        mean_pool = MeanPool()
        self.metadata.pooling_cursor.prompt_lens_cpu = self.metadata.pooling_cursor.prompt_lens_cpu.astype("float32")
        self.metadata.pooling_cursor.num_scheduled_tokens_cpu = (
            self.metadata.pooling_cursor.num_scheduled_tokens_cpu.astype("float32")
        )
        mean_out = mean_pool.forward(self.hidden_states, self.metadata)
        expected = np.array([[0.5, 0.5], [2.5, 2.5]], dtype="float32")
        np.testing.assert_allclose(mean_out.numpy(), expected, rtol=1e-6)

    def test_partial_prefill_rejections(self):
        partial_metadata = build_metadata([2, 2], self.pooling_params, num_tokens=[1, 2])
        hidden_states = self.hidden_states

        with pytest.raises(AssertionError):
            AllPool().forward(hidden_states, partial_metadata)
        with pytest.raises(AssertionError):
            CLSPool().forward(hidden_states, partial_metadata)
        with pytest.raises(AssertionError):
            MeanPool().forward(hidden_states, partial_metadata)


class TestStepAndSimplePooler:
    def test_step_pooler_filters_tokens_and_ids(self):
        pooling_params = [
            PoolingParams(task="encode", step_tag_id=2, returned_token_ids=[0, 1]),
            PoolingParams(task="encode", step_tag_id=2, returned_token_ids=None),
        ]
        token_ids = paddle.to_tensor([[0, 1, 2], [2, 2, 1]], dtype="int64")
        metadata = build_metadata([3, 3], pooling_params, token_ids=token_ids)
        hidden_states = paddle.to_tensor(
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0], [6.0, 6.0]], dtype="float32"
        )

        pooler = StepPooler(make_model_config())
        output = pooler.forward(hidden_states, metadata)

        assert isinstance(output, list)
        assert list(output[0].shape) == [1, 2]
        np.testing.assert_allclose(output[0].numpy(), [[3.0, 3.0]], rtol=1e-6)
        assert output[1].shape[1] == 2
        assert output[1].shape[0] == 2

    def test_simple_pooler_embed_and_encode(self):
        resolved = ResolvedPoolingConfig(task="embed", pooling_type=PoolingType.ALL)
        simple = SimplePooler.from_config(resolved, make_model_config())
        assert isinstance(simple.head, EmbeddingPoolerHead)

        pooling_params = [PoolingParams(task="embed", normalize=True)]
        metadata = build_metadata([2], pooling_params)
        hidden = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
        result = simple.forward(hidden, metadata)
        np.testing.assert_allclose(np.linalg.norm(result.numpy(), axis=-1), 1.0, rtol=1e-6)

        encode_pooling = SimplePooler(LastPool(), RewardPoolerHead())
        encode_metadata = build_metadata([1], [PoolingParams(task="encode", softmax=True)])
        hidden_states = paddle.to_tensor([[0.1, 0.9]], dtype="float32")
        encode_result = encode_pooling.forward(hidden_states, encode_metadata)
        np.testing.assert_allclose(encode_result.numpy().sum(axis=-1), 1.0, rtol=1e-6)


class TestDispatchPooler:
    def test_dispatch_forward_and_error_path(self):
        pooling_params = [PoolingParams(task="encode"), PoolingParams(task="encode")]
        metadata = build_metadata([1, 1], pooling_params)
        hidden = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")

        pooler = SimplePooler(LastPool(), RewardPoolerHead())
        dispatch = DispatchPooler({"encode": pooler})
        output = dispatch.forward(hidden, metadata)

        assert len(output) == 2
        assert all(isinstance(vec, paddle.Tensor) for vec in output)

        bad_metadata = build_metadata([1], [PoolingParams(task="score")])
        with pytest.raises(ValueError):
            dispatch.forward(hidden[:1], bad_metadata)
