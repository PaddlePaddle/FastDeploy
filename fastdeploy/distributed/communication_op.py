"""
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
"""

import paddle
import paddle.distributed as dist
from paddle.distributed import fleet

from fastdeploy.distributed.parallel_state import get_tensor_model_parallel_world_size

_TP_AR = None


def use_custom_allreduce(custom_all_reduce_max_bytes: int = 8192 * 1024):
    hcg = fleet.get_hybrid_communicate_group()
    model_parallel_group = hcg.get_model_parallel_group()
    global _TP_AR
    if get_tensor_model_parallel_world_size() > 1 and paddle.is_compiled_with_cuda():
        from fastdeploy.distributed.custom_all_reduce import CustomAllreduce

        _TP_AR = CustomAllreduce(model_parallel_group, custom_all_reduce_max_bytes)


try:

    @paddle.jit.marker.unified
    def tensor_model_parallel_all_reduce(
        input_: paddle.Tensor,
    ) -> paddle.Tensor:
        """All-reduce the input tensor across model parallel group."""
        global _TP_AR
        if _TP_AR is not None and _TP_AR.should_custom_ar(input_):
            _TP_AR.all_reduce(input_, input_)
        elif paddle.in_dynamic_mode():
            hcg = fleet.get_hybrid_communicate_group()
            mp_group = hcg.get_model_parallel_group()
            dist.all_reduce(input_, group=mp_group)
        else:
            dist.all_reduce(input_)

except:
    tensor_model_parallel_all_reduce = None
