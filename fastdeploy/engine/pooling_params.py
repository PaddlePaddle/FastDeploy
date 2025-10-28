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

from copy import deepcopy
from typing import TYPE_CHECKING, Annotated, Any, Dict, Optional

import msgspec

from fastdeploy.engine.sampling_params import RequestOutputKind
from fastdeploy.engine.tasks import PoolingTask

if TYPE_CHECKING:
    from fastdeploy.config import ModelConfig


class PoolingParams(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    array_like=True,
):  # type: ignore[call-arg]:
    """API parameters for pooling models.

    Attributes:
        normalize: Whether to normalize the embeddings outputs.
        dimensions: Reduce the dimensions of embeddings
                    if model support matryoshka representation.
        softmax: Whether to apply softmax to the reward outputs.
        step_tag_id: Step tag ID for process reward models to identify
                    specific steps in multi-step reasoning tasks.
        returned_token_ids: List of token IDs to return rewards for,
                           used for fine-grained reward calculation.
        task: Internal use only. Specifies the pooling task type
              ("embed" for embeddings, "encode" for reward models).
        requires_token_ids: Internal use only. Whether token ID information
                           is required for processing.
        extra_kwargs: Internal use only. Dictionary for storing additional
                     custom parameters for extended functionality.
        output_kind: Output type specification, fixed to FINAL_ONLY
                    (only final outputs are returned).
    """

    truncate_prompt_tokens: Optional[Annotated[int, msgspec.Meta(ge=-1)]] = None
    """If set to -1, will use the truncation size supported by the model. If
    set to an integer k, will use only the last k tokens from the prompt
    (i.e., left truncation). If set to `None`, truncation is disabled."""

    # for embeddings models
    dimensions: Optional[int] = None
    normalize: Optional[bool] = None

    # for reward models
    softmax: Optional[bool] = None
    step_tag_id: Optional[int] = None
    returned_token_ids: Optional[list[int]] = None

    task: Optional[PoolingTask] = None
    """Internal use only."""

    requires_token_ids: bool = False
    """Internal use only."""

    extra_kwargs: Optional[dict[str, Any]] = None
    """Internal use only."""

    output_kind: RequestOutputKind = RequestOutputKind.FINAL_ONLY

    @property
    def _all_parameters(self) -> list[str]:
        return ["dimensions", "normalize", "softmax", "step_tag_id", "returned_token_ids"]

    @property
    def valid_parameters(self):
        return {
            "embed": ["dimensions", "normalize"],
            "encode": ["softmax", "step_tag_id", "returned_token_ids"],
            "reward": ["dimensions", "normalize"],
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert instance to dictionary including only non-None attributes."""
        result = {}
        for field_name in self.__annotations__:
            if field_name == "output_kind":
                continue
            value = getattr(self, field_name, None)
            if isinstance(value, PoolingParams):
                result[field_name] = value.to_dict()
            else:
                result[field_name] = value
        if self.extra_kwargs:
            result.update(self.extra_kwargs)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PoolingParams":
        """Create instance from dictionary, separating known fields and extra_kwargs."""
        known_fields = set(cls.__annotations__.keys())
        init_kwargs = {k: v for k, v in data.items() if k in known_fields}
        extra_kwargs = {k: v for k, v in data.items() if k not in known_fields}

        if extra_kwargs:
            init_kwargs["extra_kwargs"] = extra_kwargs

        return cls(**init_kwargs)

    def clone(self) -> "PoolingParams":
        """Returns a deep copy of the PoolingParams instance."""
        return deepcopy(self)

    def verify(self, task: PoolingTask, model_config: Optional["ModelConfig"] = None) -> None:

        if self.task is None:
            self.task = task
        elif self.task != task:
            msg = f"You cannot overwrite {self.task=!r} with {task=!r}!"
            raise ValueError(msg)

        # NOTE: Task validation needs to done against the model instance,
        # which is not available in model config. So, it's not included
        # in this method

        self._merge_default_parameters(model_config)
        self._set_default_parameters(model_config)
        self._verify_valid_parameters()

    def _merge_default_parameters(self, model_config: Optional["ModelConfig"] = None) -> None:

        if model_config is None:
            return

        pooler_config = model_config.pooler_config
        if pooler_config is None:
            return

        assert self.task is not None, "task must be set"
        valid_parameters = self.valid_parameters[self.task]

        for k in valid_parameters:
            if getattr(pooler_config, k, None) is None:
                continue

            if getattr(self, k, None) is None:
                setattr(self, k, getattr(pooler_config, k))

    def _set_default_parameters(self, model_config: Optional["ModelConfig"]):
        if self.task == "embed":
            if self.normalize is None:
                self.normalize = True
        elif self.task == "encode":
            if self.softmax is None:
                self.softmax = True
        elif self.task == "reward":
            if self.normalize is None:
                self.normalize = True
        else:
            raise ValueError(f"Unknown pooling task: {self.task}")

    def _verify_valid_parameters(self):
        assert self.task is not None, "task must be set"
        valid_parameters = self.valid_parameters[self.task]
        invalid_parameters = []
        for k in self._all_parameters:
            if k in valid_parameters:
                continue

            if getattr(self, k, None) is not None:
                invalid_parameters.append(k)

        if invalid_parameters:
            raise ValueError(
                f"Task {self.task} only supports {valid_parameters} "
                f"parameters, does not support "
                f"{invalid_parameters} parameters"
            )

    def __repr__(self) -> str:
        return (
            f"PoolingParams("
            f"task={self.task}, "
            f"normalize={self.normalize}, "
            f"dimensions={self.dimensions}, "
            f"softmax={self.softmax}, "
            f"step_tag_id={self.step_tag_id}, "
            f"returned_token_ids={self.returned_token_ids}, "
            f"requires_token_ids={self.requires_token_ids}, "
            f"extra_kwargs={self.extra_kwargs})"
        )

    def __post_init__(self) -> None:
        assert self.output_kind == RequestOutputKind.FINAL_ONLY, "For pooling output_kind has to be FINAL_ONLY"
