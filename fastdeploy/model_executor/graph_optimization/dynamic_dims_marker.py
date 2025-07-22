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

from __future__ import annotations

import dataclasses
import typing
from abc import abstractmethod
from collections.abc import Callable
from functools import partial
from typing import Annotated, Any, TypeVar, Union, get_origin, get_type_hints

import paddle
from paddle import Tensor
from paddleformers.utils.log import logger
from typing_extensions import TypeAlias

T = TypeVar("T")
U = TypeVar("U")

Accessor: TypeAlias = Callable[[T], U]


class DynamicDims:
    def __init__(self, dims: int | tuple[int]):
        self.dims = dims if isinstance(dims, tuple) else (dims,)

    def __repr__(self):
        return f"DynamicDims({self.dims})"


class DynamicDimTypeResolver:
    """
    Base class for dynamic dimension type resolvers.
    This class provides a mechanism to register and resolve dynamic dimensions
    based on type annotations. It uses a registry pattern to allow multiple
    resolvers to be registered and used in a flexible manner.
    """

    ALL_DYNAMIC_DIM_TYPE_RESOLVERS = []

    @classmethod
    def register_resolver(cls, resolver_cls: type[DynamicDimTypeResolver]):
        cls.ALL_DYNAMIC_DIM_TYPE_RESOLVERS.append(resolver_cls())
        return resolver_cls

    @abstractmethod
    def type_match(self, tp: type[Any]) -> bool:
        raise NotImplementedError

    @abstractmethod
    def extract_inner_types(
        self, data: Any, data_name: str, tp: type[Any]
    ) -> list[tuple[Accessor[Any, Any], str, type[Any]]]:
        raise NotImplementedError

    def resolve(self, data: Any, data_name: str, tp: type[Any]) -> None:
        inner_types = self.extract_inner_types(data, data_name, tp)
        for accessor, inner_data_name, inner_type in inner_types:
            self.generic_resolve(accessor(data), inner_data_name, inner_type)

    def generic_resolve(self, data: Any, data_name: str, tp: type[Any]) -> None:
        for resolver in self.ALL_DYNAMIC_DIM_TYPE_RESOLVERS:
            if resolver.type_match(tp):
                return resolver.resolve(data, data_name, tp)
            runtime_tp = type(data)
            if runtime_tp is not tp and resolver.type_match(runtime_tp):
                return resolver.resolve(data, data_name, runtime_tp)
        else:
            logger.debug(f"No resolver found for type {tp} and data {data_name}")


@DynamicDimTypeResolver.register_resolver
class DataClassDynamicDimTypeResolver(DynamicDimTypeResolver):
    def type_match(self, tp: type[Any]) -> bool:
        return dataclasses.is_dataclass(tp) and isinstance(tp, type)

    def extract_inner_types(
        self, data: Any, data_name: str, tp: type[Any]
    ) -> list[tuple[Accessor[Any, Any], str, type[Any]]]:
        type_hints = get_type_hints(tp, include_extras=True)
        return [  # type: ignore
            (
                # bind name by partial to avoid capture wrong free vars
                partial(lambda name, dt: getattr(dt, name), field.name),
                f"{data_name}.{field.name}",
                type_hints[field.name],
            )
            for field in dataclasses.fields(tp)
        ]


@DynamicDimTypeResolver.register_resolver
class OptionalDynamicDimTypeResolver(DynamicDimTypeResolver):
    def type_match(self, tp) -> bool:
        return get_origin(tp) is Union and len(tp.__args__) == 2 and tp.__args__[1] is type(None)  # noqa: E721

    def extract_inner_types(
        self, data: Any, data_name: str, tp: type[Any]
    ) -> list[tuple[Accessor[Any, Any], str, type[Any]]]:
        if data is None:
            return []
        inner_type = tp.__args__[0]
        return [(lambda x: x, data_name, inner_type)]  # No accessor needed for Optional


@DynamicDimTypeResolver.register_resolver
class ListDynamicDimTypeResolver(DynamicDimTypeResolver):
    def type_match(self, tp: type[Any]) -> bool:
        return get_origin(tp) is list

    def extract_inner_types(
        self, data: Any, data_name: str, tp: type[Any]
    ) -> list[tuple[Accessor[Any, Any], str, type[Any]]]:
        if not data:
            return []
        inner_type = typing.get_args(tp)[0] if tp.__args__ else Any
        return [(partial(lambda i, x: x[i], i), f"{data_name}[{i}]", inner_type) for i in range(len(data))]  # type: ignore


@DynamicDimTypeResolver.register_resolver
class ManualMarkedInnerFieldsDynamicDimTypeResolver(DynamicDimTypeResolver):
    INFER_DYNAMIC_DIMS_FIELDS_ATTR_NAME = "__infer_dynamic_dims_fields__"

    def type_match(self, tp: type[Any]) -> bool:
        return hasattr(tp, ManualMarkedInnerFieldsDynamicDimTypeResolver.INFER_DYNAMIC_DIMS_FIELDS_ATTR_NAME)

    def extract_inner_types(
        self, data: Any, data_name: str, tp: type[Any]
    ) -> list[tuple[Accessor[Any, Any], str, type[Any]]]:
        fields = getattr(tp, ManualMarkedInnerFieldsDynamicDimTypeResolver.INFER_DYNAMIC_DIMS_FIELDS_ATTR_NAME)
        if isinstance(fields, str):
            raise TypeError(
                f"{ManualMarkedInnerFieldsDynamicDimTypeResolver.INFER_DYNAMIC_DIMS_FIELDS_ATTR_NAME} should be tuple, but got {type(fields)}"
            )
        inner_types_dict = typing.get_type_hints(tp)
        return [
            (partial(lambda name, x: getattr(x, name), field_name), f"{data_name}.{field_name}", inner_type)
            for field_name, inner_type in inner_types_dict.items()
        ]


@DynamicDimTypeResolver.register_resolver
class AnnotatedTensorDynamicDimTypeResolver(DynamicDimTypeResolver):
    def type_match(self, tp: type[Any]) -> bool:
        return get_origin(tp) is Annotated and typing.get_args(tp)[0] is Tensor

    def resolve(self, data: Any, data_name: str, tp: type[Any]) -> None:
        base_type, *metadata = typing.get_args(tp)
        # Filter out DynamicDims instances
        dynamic_dims = [m for m in metadata if isinstance(m, DynamicDims)]
        if not dynamic_dims:
            return
        if len(dynamic_dims) > 1:
            raise ValueError("Multiple DynamicDims annotations found. Only one is allowed.")
        dynamic_dims = dynamic_dims[0].dims
        if not isinstance(data, Tensor):
            raise TypeError(f"data {data_name} has type annotation Tensor but got type {type(data)}")
        logger.debug(f"data {data_name} has dynamic dims {dynamic_dims} for type {tp}")
        paddle.jit.marker.dynamic_dims(data, dynamic_dims)


@DynamicDimTypeResolver.register_resolver
class TensorImplicitFirstDimOnlyDynamicDimTypeResolver(DynamicDimTypeResolver):
    def type_match(self, tp: type[Any]) -> bool:
        return tp is Tensor

    def resolve(self, data: Any, data_name: str, tp: type[Any]) -> None:
        # Tensor annotation has implicit dynamic_dims=(0, )
        dynamic_dims = (0,)
        if not isinstance(data, Tensor):
            raise TypeError(f"data {data_name} has type annotation Tensor but got type {type(data)}")
        logger.debug(f"data {data_name} has dynamic dims {dynamic_dims} for type {tp}")
        paddle.jit.marker.dynamic_dims(data, dynamic_dims)


def resolve_dynamic_dims(arg: Any, arg_name: str, annotation: type[Any]) -> None:
    DynamicDimTypeResolver().generic_resolve(arg, arg_name, annotation)
