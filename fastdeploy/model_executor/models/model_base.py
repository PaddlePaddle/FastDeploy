"""
# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
"""

from abc import ABC, abstractmethod
from typing import Dict, Union

import numpy as np
import paddle
from paddle import nn
from paddleformers.transformers import PretrainedModel


class ModelRegistry:
    """
    Used to register and retrieve model classes.
    """

    _arch_to_model_cls = {}
    _arch_to_pretrained_model_cls = {}

    @classmethod
    def register_model_class(cls, model_class):
        """register model class"""
        if issubclass(model_class, ModelForCasualLM) and model_class is not ModelForCasualLM:
            cls._arch_to_model_cls[model_class.name()] = model_class
        return model_class

    @classmethod
    def register_pretrained_model(cls, pretrained_model):
        """register pretrained model class"""
        if (
            issubclass(pretrained_model, PretrainedModel)
            and pretrained_model is not PretrainedModel
            and hasattr(pretrained_model, "arch_name")
        ):
            cls._arch_to_pretrained_model_cls[pretrained_model.arch_name()] = pretrained_model

        return pretrained_model

    @classmethod
    def get_pretrain_cls(cls, architectures: str):
        """get_pretrain_cls"""
        return cls._arch_to_pretrained_model_cls[architectures]

    @classmethod
    def get_class(cls, name):
        """get model class"""
        if name not in cls._arch_to_model_cls:
            raise ValueError(f"Model '{name}' is not registered!")
        return cls._arch_to_model_cls[name]

    @classmethod
    def get_supported_archs(cls):
        assert len(cls._arch_to_model_cls) >= len(
            cls._arch_to_pretrained_model_cls
        ), "model class num is more than pretrained model registry num"
        return [key for key in cls._arch_to_model_cls.keys()]


class ModelForCasualLM(nn.Layer, ABC):
    """
    Base class for LM
    """

    def __init__(self, configs):
        """
        Args:
            configs (dict): Configurations including parameters such as max_dec_len, min_dec_len, decode_strategy,
                vocab_size, use_topp_sampling, etc.
        """
        super(ModelForCasualLM, self).__init__()
        self.fd_config = configs

    @abstractmethod
    def set_state_dict(self, state_dict: Dict[str, Union[np.ndarray, paddle.Tensor]]):
        """
        Load model parameters from a given state dictionary.

        Args:
            state_dict (dict[str, np.ndarray | paddle.Tensor]):
                A dictionary containing model parameters, where keys are parameter names
                and values are NumPy arrays or PaddlePaddle tensors.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        input_ids=None,
        pos_emb=None,
        **model_kwargs,
    ):
        """
        Defines the forward pass of the model for generating text.

        Args:
            input_ids (Tensor, optional): The input token ids to the model.
            pos_emb (Tensor, optional): position Embeddings for model.
            **model_kwargs: Additional keyword arguments for the model.

        Returns:
            Tensor or list of Tensors: Generated tokens or decoded outputs.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_logits(self, hidden_state, **logits_prosessor_kwargs):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def name(self):
        raise NotImplementedError
