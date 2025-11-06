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

from collections.abc import Iterable
from typing import TypeVar

import paddle
import paddle.nn as nn

from fastdeploy.config import ModelConfig
from fastdeploy.transformer_utils.config import get_hf_file_to_dict

_T = TypeVar("_T", bound=type[nn.Layer])

_GENERATE_SUFFIXES = [
    "ForCausalLM",
    "ForConditionalGeneration",
    "ChatModel",
    "LMHeadModel",
]


def _load_dense_weights(linear: nn.Linear, folder: str, model_config: "ModelConfig") -> bool:
    """Load weights using vLLM's weight_loader pattern."""

    from fastdeploy.model_executor.utils import default_weight_loader

    filename = "model.safetensors"
    file_path = f"{folder}/{filename}" if folder else filename

    try:
        file_bytes = get_hf_file_to_dict(file_path, model_config.model, model_config.revision)
        if not file_bytes:
            return False

        state_dict = {}
        if filename.endswith(".safetensors"):
            import io

            from safetensors.numpy import load as load_safetensors

            numpy_tensors = load_safetensors(io.BytesIO(file_bytes))
            for key, numpy_array in numpy_tensors.items():
                state_dict[key] = paddle.to_tensor(numpy_array)
        else:
            import io

            state_dict = paddle.load(io.BytesIO(file_bytes))

        weight_keys = ["weight", "linear.weight", "dense.weight"]

        for weight_key in weight_keys:
            if weight_key in state_dict:
                weight_loader = getattr(linear.weight, "weight_loader", default_weight_loader)
                weight_loader(linear.weight, state_dict[weight_key].astype(paddle.float32))
                bias_key = weight_key.replace("weight", "bias")
                if linear.bias is not None and bias_key in state_dict:
                    bias_loader = getattr(linear.bias, "weight_loader", default_weight_loader)
                    bias_loader(linear.bias, state_dict[bias_key].astype(paddle.float32))
                return True
    except Exception as e:
        print(f"Failed to load :{e}")
        return False
    return False


def _create_pooling_model_cls(orig_cls: _T) -> _T:

    class ModelForPooling(orig_cls):

        def __init__(self, fd_config, *args, **kwargs):
            super().__init__(fd_config, *args, **kwargs)
            self.fd_config = fd_config
            self.is_pooling_model = True

            # These are not used in pooling models
            for attr in ("lm_head", "logits_processor"):
                if hasattr(self, attr):
                    delattr(self, attr)

            # If the model already defines a pooler instance, don't overwrite it
            if not getattr(self, "pooler", None):
                self._init_pooler(fd_config)

        def _init_pooler(self, fd_config):
            raise NotImplementedError

        def load_weights(self, weights: Iterable[tuple[str, paddle.Tensor]]):
            # TODO: Support uninitialized params tracking

            # We have deleted this attribute, so don't load it
            weights = ((name, data) for name, data in weights if not name.startswith("lm_head."))

            # If `*ForCausalLM` defines `load_weights` on the inner model
            # and there are no other inner modules with parameters,
            # we support loading from both `*Model` and `*ForCausalLM`

            if hasattr(self, "model") and hasattr(self.model, "load_weights"):
                # Whether only `self.model` contains parameters
                model_is_only_param = all(
                    name == "model" or not any(child.parameters()) for name, child in self.named_children()
                )
                if model_is_only_param:
                    weights = ((name[6:], data) for name, data in weights if name.startswith("model."))
                    loaded_params = self.model.load_weights(weights)
                    loaded_params = {f"model.{name}" for name in loaded_params}
                    return loaded_params

            # For most other models
            if hasattr(orig_cls, "load_weights"):
                return orig_cls.load_weights(self, weights)  # type: ignore
            # Fallback
            else:
                raise ValueError("No load_weights method found in the model.")

    return ModelForPooling


def _get_pooling_model_name(orig_model_name: str, pooling_suffix: str) -> str:
    model_name = orig_model_name

    for generate_suffix in _GENERATE_SUFFIXES:
        model_name = model_name.removesuffix(generate_suffix)
    return model_name + pooling_suffix


def as_embedding_model(cls: _T) -> _T:
    """
    Subclass an existing FastDeploy model to support embeddings.

    By default, the embeddings of the whole prompt are extracted from the
    normalized hidden state corresponding to the last token.

    Note:
        We assume that no extra layers are added to the original model;
        please implement your own model if this is not the case.
    """
    # Avoid modifying existing embedding models
    from fastdeploy.model_executor.models.interfaces_base import is_pooling_model

    if is_pooling_model(cls):
        return cls

    from fastdeploy.model_executor.layers.pooler import DispatchPooler, Pooler

    class ModelForEmbedding(_create_pooling_model_cls(cls)):

        def _init_pooler(self, fd_config, prefix: str = ""):
            pooler_config = fd_config.model_config.pooler_config
            assert pooler_config is not None

            self.pooler = DispatchPooler(
                {
                    "encode": Pooler.for_encode(pooler_config, fd_config.model_config),
                    "embed": Pooler.for_embed(pooler_config, fd_config.model_config),
                },
            )

    ModelForEmbedding.__name__ = _get_pooling_model_name(cls.__name__, "ForEmbedding")

    return ModelForEmbedding
