# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import os
from collections import OrderedDict

import numpy as np
import paddle
import paddle.distributed.fleet as fleet
from fastdeploy_llm.utils.logging_util import logger


def is_exist(model_prompt_path):
    """判断用户指定的prefix soft prompt是否存在
    """
    if model_prompt_path is None or model_prompt_path == "" or not os.path.exists(
            model_prompt_path):
        return False
    return True


def get_prefix_caches(model_prompt_dir_path, model_id, num_layers,
                      num_attention_heads, hidden_size, prompt_num,
                      model_prompt_dict):
    """
    获取用户请求中prefix_key对应的soft prompts
    """
    float_type = 'float32'
    if (model_id is None) or (model_id == ""):
        model_prompt_path = None
    else:
        model_prompt_path = os.path.join(model_prompt_dir_path,
                                         '8-{}'.format(model_id), '1',
                                         'task_prompt_embeddings.npy')
    if is_exist(model_prompt_path):
        model_id_index = model_id + "-" + str(fleet.worker_index())
        if model_id_index in model_prompt_dict:
            prefix_prompt_split = model_prompt_dict[model_id_index]
        else:
            try:
                prefix_prompt = np.load(model_prompt_path)
                # for all_gather prompt
                worker_range = prefix_prompt.shape[3] // fleet.worker_num()
                start_index = fleet.worker_index() * worker_range
                end_index = (fleet.worker_index() + 1) * worker_range
                prefix_prompt_split = paddle.to_tensor(
                    prefix_prompt[:, :, :, start_index:end_index, :])
                model_prompt_dict[model_id_index] = prefix_prompt_split
            except:
                logger.error("Exception happend while load prefix cache: {}, will fallback to situation without prefix.".format(model_prompt_path)) 
                return np.zeros([num_layers, 2, num_attention_heads, prompt_num // fleet.worker_num(), hidden_size // num_attention_heads], dtype=float_type)
        return prefix_prompt_split.astype(float_type).numpy()
    else:
        # Construct an embedding with no prefix caches
        return np.zeros(
            [
                num_layers, 2, num_attention_heads, prompt_num //
                fleet.worker_num(), hidden_size // num_attention_heads
            ],
            dtype=float_type)


def pad_prefix_caches(prefix_caches, max_len, pad_style="right"):
    """ Pad the instances to the max sequence length in batch, and return corresponding masks
        Args:
            prefix_caches(list[np.array]): A list stored all prefix caches in a batch.
            pad_style: Left or right padding
        Returns:
            new_prefix_caches(list[np.array]): A list stored all prefix caches after padding.
    """
    new_prefix_caches = []
    if pad_style == "left":
        for cache in prefix_caches:
            if cache.shape[3] != max_len:
                pad_data = np.zeros([
                    cache.shape[0], cache.shape[1], cache.shape[2],
                    max_len - cache.shape[3], cache.shape[4]
                ], cache.dtype)
                new_cache = np.concatenate((pad_data, cache), axis=3)
            else:
                new_cache = cache
            new_prefix_caches.append(new_cache)
    else:
        for cache in prefix_caches:
            if cache.shape[3] != max_len:
                pad_data = np.zeros([
                    cache.shape[0], cache.shape[1], cache.shape[2],
                    max_len - cache.shape[3], cache.shape[4]
                ], cache.dtype)
                new_cache = np.concatenate((cache, pad_data), axis=3)
            else:
                new_cache = cache
            new_prefix_caches.append(new_cache)
    return new_prefix_caches


class LimitedSizeDict(OrderedDict):
    """
    限制长度的dict
    """

    def __init__(self, *args, **kwargs):
        """init"""
        self.size_limit = kwargs.pop("size_limit", None)
        OrderedDict.__init__(self, *args, **kwargs)
        self._check_size_limit()

    def __setitem__(self, key, value):
        """setitem"""
        OrderedDict.__setitem__(self, key, value)
        self._check_size_limit()

    def _check_size_limit(self):
        """检查dict长度是否超过限制"""
        if self.size_limit is not None:
            while len(self) > self.size_limit:
                self.popitem(last=False)
