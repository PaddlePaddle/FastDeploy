# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import paddle
import paddle.nn as nn


def disabled_train(mode="train"):
    return


def freeze_parameters(params, enable_eval=False):
    if (not isinstance(params, nn.Layer)) and (not isinstance(params, paddle.Tensor)):
        raise TypeError(
            "An instance of Paddle.nn.Layer or paddle.Tensor expected, but received: {}".format(type(params))
        )

    if isinstance(params, paddle.Tensor):
        params.stop_gradient = True
    else:
        if enable_eval:
            params.eval()
            params.train = disabled_train

        for name, param in params.named_parameters():
            param.stop_gradient = True


def transfer_param(p, is_bias=False, dtype="float16", restore_data=False):
    param_shape = p.shape
    # Allow CPU/GPU and float16/float32 transfer
    # NOTE: str(p.place) differs between paddle develop and 2.2
    if str(p.dtype)[-len(dtype) :] == dtype and ("gpu" in str(p.place).lower() or "cuda" in str(p.place).lower()):
        return p
    if restore_data:
        if paddle.in_dynamic_mode():
            param_data = p.numpy()
            # Creating parameters with Assign initializer is too slow. Maybe we
            # can cast to fp16 directly and get a tensor, while we do it more
            # elaborately to get a ParamBase. Also note `VarBase.set_value`
            # enforce the same dtype and can not be used directly.
            new_p = type(p)(shape=param_shape, dtype=dtype, is_bias=is_bias)
            new_p.value().get_tensor().set(param_data.astype(dtype), paddle.framework._current_expected_place())
            return new_p
        else:
            param_data = np.array(paddle.static.global_scope().find_var(p.name).get_tensor())
    return paddle.create_parameter(
        shape=param_shape,
        dtype=dtype,
        is_bias=is_bias,
        default_initializer=paddle.nn.initializer.Assign(param_data) if restore_data else None,
    )
