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

from typing import ClassVar, Literal, Protocol, Type

import paddle
from paddle import nn
from typing_extensions import TypeVar, runtime_checkable

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.forward_meta import ForwardMeta
from fastdeploy.model_executor.layers.pooler import Pooler

T = TypeVar("T", default=paddle.Tensor)
T_co = TypeVar("T_co", default=paddle.Tensor, covariant=True)


def is_pooling_model(model_cls: Type[nn.Layer]) -> bool:
    return getattr(model_cls, "is_pooling_model", False)


def get_default_pooling_type(model_cls: Type[nn.Layer] = None) -> str:
    if model_cls is not None:
        return getattr(model_cls, "default_pooling_type", "LAST")
    return "LAST"


@runtime_checkable
class FdModel(Protocol[T_co]):
    """The interface required for all models in FastDeploy."""

    def __init__(
        self,
        fd_config: FDConfig,
        prefix: str = "",
    ) -> None:
        pass

    def forward(
        self,
        ids_remove_padding: paddle.Tensor,
        forward_metadata: ForwardMeta,
    ) -> T_co:
        pass


class FdModelForPooling(FdModel[T_co], Protocol[T_co]):
    """The interface required for all pooling models in FastDeploy."""

    is_pooling_model: ClassVar[Literal[True]] = True
    """
    A flag that indicates this model supports pooling.

    Note:
        There is no need to redefine this flag if this class is in the
        MRO of your model class.
    """

    default_pooling_type: ClassVar[str] = "LAST"
    """
    Indicates the
    [fastdeploy.config.PoolerConfig.pooling_type][]
    to use by default.

    You can use the
    [fastdeploy.model_executor.models.interfaces_base.default_pooling_type][]
    decorator to conveniently set this field.
    """
    pooler: Pooler
    """The pooler is only called on TP rank 0."""


def default_pooling_type(pooling_type: str):
    def func(model):
        model.default_pooling_type = pooling_type  # type: ignore
        return model

    return func
