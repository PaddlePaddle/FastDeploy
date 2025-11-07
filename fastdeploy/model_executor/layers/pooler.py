"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Set
from dataclasses import dataclass
from enum import IntEnum
from itertools import groupby
from typing import Callable, Optional, TypeVar, Union, cast

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from fastdeploy.config import FDConfig, ModelConfig, PoolerConfig
from fastdeploy.engine.pooling_params import PoolingParams
from fastdeploy.engine.tasks import PoolingTask
from fastdeploy.model_executor.layers.pool.metadata import (
    PoolingCursor,
    PoolingMetadata,
)
from fastdeploy.output.pooler import PoolerOutput, PoolingSequenceGroupOutput
from fastdeploy.utils import get_logger

logger = get_logger("pooler", "pooler.log")

PoolingFn = Callable[
    [Union[paddle.Tensor, list[paddle.Tensor]], PoolingMetadata], Union[paddle.Tensor, list[paddle.Tensor]]
]
ClassifierFn = Callable[[paddle.Tensor], paddle.Tensor]


class PoolingType(IntEnum):
    """Enumeration for different types of pooling methods."""

    LAST = 0
    ALL = 1
    CLS = 2
    STEP = 3
    MEAN = 4


_T = TypeVar("_T", paddle.Tensor, list[paddle.Tensor])


@dataclass(frozen=True)
class ResolvedPoolingConfig:
    pooling_type: PoolingType
    task: PoolingTask

    @classmethod
    def from_config(
        cls,
        task: PoolingTask,
        pooler_config: PoolerConfig,
    ) -> "ResolvedPoolingConfig":
        assert pooler_config.pooling_type is not None
        return cls(task=task, pooling_type=PoolingType[pooler_config.pooling_type])


def get_pooling_params(pooling_metadata: PoolingMetadata) -> list[PoolingParams]:
    pooling_params = pooling_metadata.pooling_params
    return pooling_params


def get_tasks(pooling_metadata: PoolingMetadata) -> list[PoolingTask]:
    pooling_params = get_pooling_params(pooling_metadata)

    tasks: list[PoolingTask] = [task for pooling_param in pooling_params if (task := pooling_param.task) is not None]
    assert len(pooling_params) == len(tasks)

    return tasks


def get_prompt_token_ids(pooling_metadata: PoolingMetadata) -> list[paddle.Tensor]:
    assert (
        pooling_metadata.prompt_token_ids is not None
    ), "Please set `requires_token_ids=True` in `get_pooling_updates`"

    return [pooling_metadata.prompt_token_ids[i, :num] for i, num in enumerate(pooling_metadata.prompt_lens)]


@dataclass(frozen=True)
class PoolingParamsUpdate:
    requires_token_ids: bool = False
    """Set this flag to enable `get_prompt_token_ids` for your pooler."""

    def apply(self, params: PoolingParams) -> None:
        params.requires_token_ids = self.requires_token_ids


class Pooler(nn.Layer, ABC):
    """The interface required for all poolers used in pooling models in FastDeploy."""

    @staticmethod
    def for_encode(pooler_config: PoolerConfig, model_config: Optional["ModelConfig"] = None):
        if pooler_config.pooling_type == "STEP":
            return StepPooler()

        resolved_config = ResolvedPoolingConfig(task="encode", pooling_type=PoolingType.ALL)
        return SimplePooler.from_config(resolved_config, model_config)

    @staticmethod
    def for_embed(pooler_config: PoolerConfig, model_config: Optional["ModelConfig"] = None):
        resolved_config = ResolvedPoolingConfig.from_config(
            task="embed",
            pooler_config=pooler_config,
        )
        return SimplePooler.from_config(resolved_config, model_config)

    @staticmethod
    def for_classify(
        pooler_config: PoolerConfig,
        classify: Optional[ClassifierFn],
    ):
        pass

    @abstractmethod
    def get_supported_tasks(self) -> Set[PoolingTask]:
        """Determine which pooling tasks are supported."""
        raise NotImplementedError

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        """
        Construct the updated pooling parameters to use for a supported task.
        """
        return PoolingParamsUpdate()

    @abstractmethod
    def forward(
        self,
        hidden_states: Union[list[paddle.Tensor], paddle.Tensor],
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        raise NotImplementedError


class BasePoolerActication(nn.Layer, ABC):

    @abstractmethod
    def forward(self, pooled_data: _T) -> _T:
        # shape:
        # classify (& score) -> (batch_size, num_classes)
        # embed -> (batch_size, embedding_dim) or list(embedding_dim)
        #          (batch_size, dimensions) or list(dimensions) if using MRL
        raise NotImplementedError


class PoolerActivation(BasePoolerActication):

    @staticmethod
    def wraps(module: nn.Layer):
        if isinstance(module, nn.Identity):
            return PoolerIdentity()
        if isinstance(module, (nn.Sigmoid, nn.Softmax)):
            return PoolerClassify()

        return LambdaPoolerActivation(module)

    @abstractmethod
    def forward_chunk(self, pooled_data: paddle.Tensor) -> paddle.Tensor:
        raise NotImplementedError

    def forward(self, pooled_data: _T) -> _T:
        if isinstance(pooled_data, list):
            return [self.forward_chunk(data) for data in pooled_data]

        return self.forward_chunk(pooled_data)


class PoolerIdentity(PoolerActivation):

    def forward_chunk(self, pooled_data: paddle.Tensor) -> paddle.Tensor:
        return pooled_data


class PoolerClassify(PoolerActivation):

    def __init__(self, *, static_num_labels: bool = True) -> None:
        super().__init__()

        if static_num_labels:
            fd_config = FDConfig()
            self.num_labels = getattr(fd_config.model_config, "num_labels", 0)
            if self.num_labels == 0:
                logger.warning(
                    "num_labels should be > 0 for classification"
                    "models, falling back to softmax. "
                    "Please check if the configuration is correct."
                )
        else:
            self.num_labels = None

    def forward_chunk(self, pooled_data: paddle.Tensor) -> paddle.Tensor:
        num_labels = self.num_labels if self.num_labels is not None else pooled_data.shape[-1]
        if num_labels < 2:
            return F.sigmoid(pooled_data.astype("float32")).astype(pooled_data.dtype)

        return F.softmax(pooled_data.astype("float32"), axis=-1).astype(pooled_data.dtype)


class LambdaPoolerActivation(PoolerActivation):

    def __init__(self, fn: Callable[[paddle.Tensor], paddle.Tensor]):
        super().__init__()

        self.fn = fn

    def forward_chunk(self, pooled_data: paddle.Tensor) -> paddle.Tensor:
        return self.fn(pooled_data)


class PoolerHead(nn.Layer):

    def __init__(self, activation: PoolerActivation) -> None:
        super().__init__()
        self.activation = activation

    def forward(self, pooled_data: Union[list[paddle.Tensor], paddle.Tensor], pooling_metadata: PoolingMetadata):

        return self.activation(pooled_data)


class EmbeddingPoolerHead(PoolerHead):

    def __init__(self, model_config: Optional["ModelConfig"] = None) -> None:
        super().__init__(activation=PoolerNormalize())

        self.projector = None

    def forward(self, pooled_data: Union[list[paddle.Tensor], paddle.Tensor], pooling_metadata: PoolingMetadata):

        if isinstance(pooled_data, list):
            pooled_data = paddle.stack(pooled_data)
        # pooled_data shape: [batchsize, hidden_dimension]

        # Apply ST projector
        if self.projector is not None:
            projector = cast(nn.Layer, self.projector)

            def _proj(x: paddle.Tensor) -> paddle.Tensor:
                orig_dtype = x.dtype
                y = projector(x.astype("float32"))
                return y.astype(orig_dtype)

            pooled_data = _proj(pooled_data)
        # pooled_data shape: [batchsize, embedding_dimension]

        pooling_params = get_pooling_params(pooling_metadata)

        # for matryoshka representation
        dimensions_list = [pooling_param.dimensions for pooling_param in pooling_params]
        if any(d is not None for d in dimensions_list):
            # change the output dimension
            assert len(pooled_data) == len(dimensions_list)
            if len(set(dimensions_list)) == 1 and not isinstance(pooled_data, list):
                # if all dimensions are the same
                d = dimensions_list[0]
                pooled_data = pooled_data[..., :d]
            else:
                pooled_data = [vecs if d is None else vecs[..., :d] for vecs, d in zip(pooled_data, dimensions_list)]
        # for normalize
        flags = [p.normalize for p in pooling_params]
        if len(set(flags)) == 1:
            if flags[0]:
                pooled_data = self.activation(pooled_data)
        else:
            pooled_data = [self.activation(vecs) if f else vecs for vecs, f in zip(pooled_data, flags)]

        # pooled_data shape: [batchsize, embedding_dimension]
        return pooled_data


class RewardPoolerHead(PoolerHead):

    def __init__(self, model_config: Optional["ModelConfig"] = None) -> None:
        super().__init__(activation=PoolerClassify(static_num_labels=False))
        self.model_config = model_config

    def forward(self, pooled_data: Union[list[paddle.Tensor], paddle.Tensor], pooling_metadata: PoolingMetadata):
        pooling_params = get_pooling_params(pooling_metadata)

        # for softmax
        flags = [p.softmax for p in pooling_params]
        if len(set(flags)) == 1:
            if flags[0]:
                pooled_data = self.activation(pooled_data)
        else:
            pooled_data = [self.activation(vecs) if f else vecs for vecs, f in zip(pooled_data, flags)]

        return pooled_data


class PoolingMethod(nn.Layer, ABC):

    @staticmethod
    def from_pooling_type(pooling_type: PoolingType) -> "PoolingMethod":
        if pooling_type == PoolingType.LAST:
            return LastPool()
        if pooling_type == PoolingType.ALL:
            return AllPool()
        if pooling_type == PoolingType.CLS:
            return CLSPool()
        if pooling_type == PoolingType.MEAN:
            return MeanPool()
        raise NotImplementedError(f"Unsupported method: {pooling_type}")

    @abstractmethod
    def get_supported_tasks(self) -> Set[PoolingTask]:
        raise NotImplementedError

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return PoolingParamsUpdate()

    @abstractmethod
    def forward_all(
        self,
        hidden_states: paddle.Tensor,
        pooling_cursor: PoolingCursor,
    ) -> Union[list[paddle.Tensor], paddle.Tensor]:
        raise NotImplementedError

    def forward(
        self,
        hidden_states: paddle.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> Union[list[paddle.Tensor], paddle.Tensor]:
        pooling_cursor = pooling_metadata.pooling_cursor
        return self.forward_all(hidden_states, pooling_cursor)


class LastPool(PoolingMethod):

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"encode", "embed", "classify", "score"}

    def forward_all(
        self,
        hidden_states: paddle.Tensor,
        pooling_cursor: PoolingCursor,
    ) -> Union[list[paddle.Tensor], paddle.Tensor]:
        return hidden_states[pooling_cursor.last_token_indices_gpu]


class AllPool(PoolingMethod):
    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"encode"}

    def forward_all(
        self,
        hidden_states: paddle.Tensor,
        pooling_cursor: PoolingCursor,
    ) -> Union[list[paddle.Tensor], paddle.Tensor]:

        assert not pooling_cursor.is_partial_prefill(), "partial prefill not supported with ALL pooling"

        hidden_states_lst = list(hidden_states.split(pooling_cursor.num_scheduled_tokens_cpu.tolist()))
        return [hidden_states_lst[i] for i in pooling_cursor.index]


class MeanPool(PoolingMethod):

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"encode", "embed", "classify", "score"}

    def forward_all(
        self,
        hidden_states: paddle.Tensor,
        pooling_cursor: PoolingCursor,
    ) -> Union[list[paddle.Tensor], paddle.Tensor]:

        assert not pooling_cursor.is_partial_prefill(), "partial prefill not supported with MEAN pooling"

        if hidden_states.place.is_gpu_place():
            prompt_lens = pooling_cursor.prompt_lens_cpu.cuda()
        else:
            prompt_lens = pooling_cursor.prompt_lens_cpu

        # Use float32 for paddle.cumsum in MeanPool,
        # otherwise precision will be lost significantly.
        cumsum = paddle.cumsum(hidden_states.astype("float32"), axis=0)

        start_indices = pooling_cursor.first_token_indices_gpu
        end_indices = pooling_cursor.last_token_indices_gpu
        return (cumsum[end_indices] - cumsum[start_indices] + hidden_states[start_indices]) / prompt_lens.unsqueeze(1)


class CLSPool(PoolingMethod):

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"encode", "embed", "classify", "score"}

    def forward_all(
        self,
        hidden_states: paddle.Tensor,
        pooling_cursor: PoolingCursor,
    ) -> Union[list[paddle.Tensor], paddle.Tensor]:
        assert not pooling_cursor.is_partial_prefill(), "partial prefill not supported with CLS pooling"

        return hidden_states[pooling_cursor.first_token_indices_gpu]


class StepPooler(Pooler):
    def __init__(
        self,
    ) -> None:
        super().__init__()

        self.pooling = AllPool()
        self.head = RewardPoolerHead()

    def extract_states(
        self,
        hidden_states: Union[paddle.Tensor, list[paddle.Tensor]],
        pooling_metadata: PoolingMetadata,
    ) -> Union[list[paddle.Tensor], paddle.Tensor]:
        pooled_data_lst = self.pooling(hidden_states, pooling_metadata)
        prompt_token_ids = get_prompt_token_ids(pooling_metadata)

        pooled_data = list[paddle.Tensor]()

        pooling_params = get_pooling_params(pooling_metadata)

        for data, token_id, pooling_param in zip(pooled_data_lst, prompt_token_ids, pooling_params):
            step_tag_id = pooling_param.step_tag_id
            returned_token_ids = pooling_param.returned_token_ids

            if returned_token_ids is not None and len(returned_token_ids) > 0:
                data = data[:, returned_token_ids]

            if step_tag_id is not None:
                data = data[token_id == step_tag_id]
            pooled_data.append(data)

        return pooled_data

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"encode"}

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return PoolingParamsUpdate(requires_token_ids=True)

    def forward(
        self,
        hidden_states: Union[paddle.Tensor, list[paddle.Tensor]],
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        pooled_data = self.extract_states(hidden_states, pooling_metadata)
        pooling_params = get_pooling_params(pooling_metadata)
        assert len(pooled_data) == len(pooling_params)

        pooled_data = [self.head(d, p) for d, p in zip(pooled_data, pooling_params)]
        return pooled_data


class SimplePooler(Pooler):
    """A layer that pools specific information from hidden states.

    This layer does the following:
    1. Extracts specific tokens or aggregates data based on pooling method.
    2. Normalizes output if specified.
    3. Returns structured results as `PoolerOutput`.
    """

    @classmethod
    def from_config(
        cls,
        pooler_config: ResolvedPoolingConfig,
        model_config: Optional["ModelConfig"] = None,
    ) -> "SimplePooler":
        pooling = PoolingMethod.from_pooling_type(pooler_config.pooling_type)
        if pooler_config.task == "embed":
            head = EmbeddingPoolerHead(model_config)
        elif pooler_config.task == "encode":
            head = RewardPoolerHead(model_config)
        else:
            raise NotImplementedError(f"Unknown task: {pooler_config.task}")
        return cls(pooling, head)

    def __init__(self, pooling: PoolingMethod, head: PoolerHead) -> None:
        super().__init__()

        self.pooling = pooling
        self.head = head

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return self.pooling.get_supported_tasks()

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return self.pooling.get_pooling_updates(task)

    def forward(
        self,
        hidden_states: Union[paddle.Tensor, list[paddle.Tensor]],
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        pooled_data = self.pooling(hidden_states, pooling_metadata)
        pooled_data = self.head(pooled_data, pooling_metadata)
        return pooled_data


class PoolerNormalize(PoolerActivation):
    def forward_chunk(self, pooled_data: paddle.Tensor) -> paddle.Tensor:
        x = F.normalize(pooled_data.astype("float32"), p=2, axis=-1)
        return x.astype(pooled_data.dtype)


class DispatchPooler(Pooler):
    """Dispatches calls to a sub-pooler based on the pooling task."""

    def __init__(self, poolers_by_task: Mapping[PoolingTask, Pooler]) -> None:
        super().__init__()

        for task, pooler in poolers_by_task.items():
            if task not in pooler.get_supported_tasks():
                raise ValueError(
                    f"{pooler=} does not support {task=}. " f"Supported tasks: {pooler.get_supported_tasks()}"
                )

        self.poolers_by_task = poolers_by_task

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return set(self.poolers_by_task)

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return self.poolers_by_task[task].get_pooling_updates(task)

    def forward(
        self,
        hidden_states: Union[paddle.Tensor, list[paddle.Tensor]],
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        poolers_by_task = self.poolers_by_task

        outputs = list[PoolingSequenceGroupOutput]()
        offset = 0
        for task, group in groupby(get_tasks(pooling_metadata)):
            if not (pooler := poolers_by_task.get(task)):
                raise ValueError(f"Unsupported task: {task} " f"Supported tasks: {self.get_supported_tasks()}")

            num_items = len(list(group))
            group_output: PoolerOutput = pooler(
                hidden_states,
                pooling_metadata[offset : offset + num_items],
            )
            outputs.extend(group_output)
            offset += num_items

        return PoolerOutput(outputs)
