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


from contextlib import contextmanager, nullcontext

import paddle
import paddle.distributed as dist
from paddle.distributed import fleet

import fastdeploy.envs as envs
from fastdeploy.utils import register_custom_python_op

# Constants
DEFAULT_CUSTOM_ALL_REDUCE_MAX_BYTES = 8192 * 1024
SUPPORTED_DTYPES = (paddle.float32, paddle.float16, paddle.bfloat16)

# Global custom all-reduce instance
_TP_AR = None


@contextmanager
def capture_custom_allreduce():
    global _TP_AR
    ar_context = nullcontext()
    if _TP_AR is not None:
        ar_context = _TP_AR.capture()
    with ar_context:
        yield


def use_custom_allreduce(
    tp_group: paddle.distributed.communication.group.Group = None,
    custom_all_reduce_max_bytes: int = DEFAULT_CUSTOM_ALL_REDUCE_MAX_BYTES,
) -> None:
    if tp_group is None:
        hcg = fleet.get_hybrid_communicate_group()
        tp_group = hcg.get_model_parallel_group()
    global _TP_AR
    from fastdeploy.distributed.custom_all_reduce import CustomAllreduce

    _TP_AR = CustomAllreduce(tp_group, custom_all_reduce_max_bytes)


def custom_ar_clear_ipc_handles():
    global _TP_AR
    if _TP_AR is not None:
        _TP_AR.clear_ipc_handles()


try:

    def tensor_model_parallel_all_reduce_infer_meta(
        x: "paddle.static.MetaTensor", group_: paddle.distributed.communication.group.Group
    ) -> paddle.static.MetaTensor:
        return paddle.static.MetaTensor(shape=x.shape, dtype=x.dtype)

    @register_custom_python_op(
        name="tensor_model_parallel_all_reduce",
        infer_meta=tensor_model_parallel_all_reduce_infer_meta,
        input_names=["input_"],
        output_names=["out"],
        inplace_map={},
    )
    def tensor_model_parallel_all_reduce(
        input_: paddle.Tensor,
        group_: paddle.distributed.communication.group.Group = None,
    ) -> paddle.Tensor:
        global _TP_AR
        if input_.shape[0] == 0:
            return input_

        if envs.FD_DETERMINISTIC_MODE:
            # Lazy initialization of custom all-reduce for deterministic mode
            if _TP_AR is None:
                try:
                    hcg = fleet.get_hybrid_communicate_group()
                    tp_group = hcg.get_model_parallel_group()
                    if tp_group is not None and tp_group.nranks > 1:
                        use_custom_allreduce(tp_group)
                except Exception as e:
                    raise RuntimeError(
                        "DETERMINISTIC_MODE is enabled but cannot auto-initialize custom all-reduce. "
                        "TP all-reduce would use NCCL which may produce non-deterministic results "
                        "due to floating-point accumulation order. "
                        "Ensure fleet is initialized before any TP operations, "
                        "or explicitly call use_custom_allreduce() beforehand."
                    ) from e
            if input_.dtype not in SUPPORTED_DTYPES:
                raise AssertionError(
                    f"DETERMINISTIC_MODE is enabled but input tensor dtype={input_.dtype} is not supported. "
                    f"Custom all-reduce only supports: {', '.join(str(d) for d in SUPPORTED_DTYPES)}. "
                    f"Input tensor shape: {input_.shape}, dtype: {input_.dtype}."
                )
            # Compute size from .shape to avoid numel() which triggers
            # cudaErrorStreamCaptureImplicit during CUDA Graph capture
            inp_size = 1
            for s in input_.shape:
                inp_size *= s
            inp_size *= input_.element_size()
            # Check 16-byte alignment requirement for deterministic mode
            if inp_size % 16 != 0:
                raise RuntimeError(
                    f"DETERMINISTIC_MODE is enabled but input tensor size ({inp_size} bytes) "
                    f"is not a multiple of 16. Custom all-reduce requires 16-byte aligned tensors. "
                    f"Input tensor shape: {input_.shape}, element_size: {input_.element_size()} bytes, "
                    f"total size: {inp_size} bytes."
                )

        # Use custom all-reduce if available and applicable
        if _TP_AR is not None and _TP_AR.should_custom_ar(input_):
            # TODO: supports different_group custom allreduce
            input_ = _TP_AR.custom_all_reduce(input_)
        elif paddle.in_dynamic_mode():
            if envs.FD_DETERMINISTIC_MODE:
                raise RuntimeError(
                    "DETERMINISTIC_MODE is enabled but falling back to NCCL all-reduce in dynamic mode. "
                    "This may produce non-deterministic results due to floating-point "
                    "accumulation order. "
                    "Ensure custom all-reduce is properly initialized via use_custom_allreduce()."
                )
            if group_ is not None:
                dist.all_reduce(input_, group=group_)
            else:
                hcg = fleet.get_hybrid_communicate_group()
                mp_group = hcg.get_model_parallel_group()
                dist.all_reduce(input_, group=mp_group)
        else:
            if envs.FD_DETERMINISTIC_MODE:
                raise RuntimeError(
                    "DETERMINISTIC_MODE is enabled but using NCCL all-reduce in static mode. "
                    "This may produce non-deterministic results due to floating-point "
                    "accumulation order. "
                    "Use dynamic mode with custom all-reduce enabled for deterministic results."
                )
            dist.all_reduce(input_)
        return input_

    @paddle.jit.marker.unified
    def decode_alltoall_transpose(
        input_: paddle.Tensor,
        out: paddle.Tensor = None,
    ) -> paddle.Tensor:
        """alltoall and transpose in decode."""
        if input_.shape[0] == 0:
            return input_
        global _TP_AR
        input_ = _TP_AR.decode_alltoall_transpose(input_, out)
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
        if input_.shape[0] == 0:
            return input_
        if paddle.in_dynamic_mode():
            hcg = dist.fleet.get_hybrid_communicate_group()
            mp_group = hcg.get_model_parallel_group()
            all_reduce(input_, op=ReduceOp.SUM, group=mp_group)
        else:
            dist.all_reduce(input_)

except:
    tensor_model_parallel_all_reduce_custom = None
