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
from paddle import distributed as dist
from paddle.distributed import fleet
from paddle.distributed.fleet.utils.sequence_parallel_utils import (
    RowSequenceParallelLinear,
)

__all__ = [
    "scatter_axis",
    "all_gather_group",
    "reduce_scatter_group",
    "RowSequenceParallelLinear",
]


def scatter_axis(input, group=None, axis=0):
    """
    在MP 间按照第 0 维对`input`进行均匀切分。
    这个API 跟`distributed.scatter`并没有什么关系
    """
    if group is None:
        hcg = fleet.get_hybrid_communicate_group()
        group = hcg.get_model_parallel_group()
    parallelism = group.nranks
    if parallelism == 1:
        return input.clone()
    rank = group.rank
    seq_len = input.shape[axis]
    assert seq_len % parallelism == 0, (
        f"Input sequence length {seq_len} can't be divided exactly" f" by sequence parallelism {parallelism}"
    )
    interval = seq_len // parallelism
    input = paddle.slice(
        input,
        axes=[axis],
        starts=[interval * rank],
        ends=[interval * (rank + 1)],
    )
    # slice use stride, so we maintain the memory of whole input, use assign to free the whole input
    # which can avoid OOM.
    input = paddle.assign(input)
    return input


def all_gather_group(input, group=None, axis=0):
    """Perform collective all-gather operation across a process group with axis control.

    Functional Behavior:
      - Aggregates input tensors from all processes in the specified group
      - Supports concatenation along arbitrary dimensions (axis parameter)
      - Optimizes for axis=0 via direct shape expansion to avoid concatenation overhead

    Args:
        input (Tensor):        Local tensor to be gathered (shape: [..., D, ...])
        group (ProcessGroup):  Communication group (defaults to model parallel group)
        axis (int):            Concatenation dimension (default=0)

    Returns:
        Tensor: Concatenated tensor combining inputs from all processes:
                - When axis=0: shape [D*N, ...] (N = group size)
                - Otherwise:   shape [..., D*N, ...] along specified axis
    """
    if group is None:
        hcg = fleet.get_hybrid_communicate_group()
        group = hcg.get_model_parallel_group()
    parallelism = group.nranks
    if parallelism == 1:
        return input.clone()
    output_shape = input.shape
    if axis == 0:
        output_shape[axis] = output_shape[axis] * parallelism
        output = paddle.empty(shape=output_shape, dtype=input.dtype)
        dist.stream.all_gather(output, input, group=group, use_calc_stream=True)
        return output
    outputs = [paddle.empty(output_shape, dtype=input.dtype) for _ in range(parallelism)]
    dist.stream.all_gather(outputs, input, group=group, use_calc_stream=True)
    output = paddle.concat(outputs, axis=axis)
    return output


def reduce_scatter_group(input, group=None):
    """Perform reduce-scatter collective operation across a process group.

    Functional Behavior:
      - Aggregates (sums) input tensors across all processes in the group
      - Scatters the reduced result equally to all participants
      - Operates along the first dimension (axis=0) of the input tensor

    Args:
        input (Tensor):        Local tensor to reduce (shape: [N*K, ...] where N=group_size)
        group (ProcessGroup): Communication group (defaults to model parallel group)

    Returns:
        Tensor: Scattered portion of reduced tensor with shape [K, ...]
    """
    if group is None:
        hcg = fleet.get_hybrid_communicate_group()
        group = hcg.get_model_parallel_group()
    parallelism = group.nranks
    if parallelism == 1:
        return input.clone()
    output_shape = input.shape
    assert (
        input.shape[0] % parallelism == 0
    ), f"Input sequence length {input.shape[0]} can't be divided exactly by sequence parallelism {parallelism}"
    output_shape[0] = output_shape[0] // parallelism
    output = paddle.empty(shape=output_shape, dtype=input.dtype)
    dist.stream.reduce_scatter(output, input, op=dist.ReduceOp.SUM, group=group, use_calc_stream=True)
    return output
