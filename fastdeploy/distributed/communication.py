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

from contextlib import contextmanager, nullcontext

import paddle
import paddle.distributed as dist
from paddle.distributed import fleet

_TP_AR = None


@contextmanager
def capture_custom_allreduce():
    global _TP_AR
    ar_context = nullcontext()
    if _TP_AR is not None:
        ar_context = _TP_AR.capture()
    with ar_context:
        yield


def use_custom_allreduce(custom_all_reduce_max_bytes: int = 8192 * 1024):
    hcg = fleet.get_hybrid_communicate_group()
    model_parallel_group = hcg.get_model_parallel_group()
    global _TP_AR
    from fastdeploy.distributed.custom_all_reduce import CustomAllreduce

    _TP_AR = CustomAllreduce(model_parallel_group, custom_all_reduce_max_bytes)


def custom_ar_clear_ipc_handles():
    global _TP_AR
    if _TP_AR is not None:
        _TP_AR.clear_ipc_handles()


try:

    @paddle.jit.marker.unified
    def tensor_model_parallel_all_reduce(
        input_: paddle.Tensor,
        group_: paddle.distributed.communication.group.Group = None,
    ) -> paddle.Tensor:
        """All-reduce the input tensor across model parallel group."""
        global _TP_AR
        if _TP_AR is not None and _TP_AR.should_custom_ar(input_):
            # TODO: supports different_group custom allreduce
            input_ = _TP_AR.custom_all_reduce(input_)
        elif paddle.in_dynamic_mode():
            if group_ is not None:
                dist.all_reduce(input_, group=group_)
            else:
                hcg = fleet.get_hybrid_communicate_group()
                mp_group = hcg.get_model_parallel_group()
                dist.all_reduce(input_, group=mp_group)
        else:
            dist.all_reduce(input_)
        return input_

except:
    tensor_model_parallel_all_reduce = None

from paddle.distributed.communication import stream
from paddle.distributed.communication.reduce import ReduceOp

try:

    def all_reduce(
        tensor,
        op,
        group,
        sync_op: bool = True,
    ):
        return stream.all_reduce(tensor, op=op, group=group, sync_op=sync_op, use_calc_stream=True)

    @paddle.jit.marker.unified
    def tensor_model_parallel_all_reduce_custom(input_: paddle.Tensor) -> paddle.Tensor:
        """All-reduce the input tensor across model parallel group on calc stream."""
        if paddle.in_dynamic_mode():
            hcg = dist.fleet.get_hybrid_communicate_group()
            mp_group = hcg.get_model_parallel_group()
            all_reduce(input_, op=ReduceOp.SUM, group=mp_group)
        else:
            dist.all_reduce(input_)

except:
    tensor_model_parallel_all_reduce_custom = None
