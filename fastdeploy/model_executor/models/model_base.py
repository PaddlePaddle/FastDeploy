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


class ModelForCasualLM(nn.Layer, ABC):
    """
    LM基类
    """

    def __init__(self, configs):
        super(ModelForCasualLM, self).__init__()
        self.fd_config = configs

    @abstractmethod
    def set_state_dict(self, state_dict: Dict[str, Union[np.ndarray, paddle.Tensor]]):
        """加载模型参数"""
        raise NotImplementedError

    @abstractmethod
    def forward(self, input_ids=None, pos_emb=None, **model_kwargs):
        """前向传播"""
        raise NotImplementedError

    @abstractmethod
    def compute_logits(self, hidden_state, **logits_prosessor_kwargs):
        """计算logits"""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def name(cls):
        """返回模型名称"""
        raise NotImplementedError
