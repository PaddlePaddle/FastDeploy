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

"""Communication module for distributed tensor parallel operations."""

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
    """Context manager for capturing custom all-reduce operations.

    Yields a null context unless custom all-reduce is initialized.
    """
    ar_context = nullcontext()
    if _TP_AR is not None:
        ar_context = _TP_AR.capture()
    with ar_context:
        yield


def use_custom_allreduce(
    tp_group: paddle.distributed.communication.group.Group = None,
    custom_all_reduce_max_bytes: int = DEFAULT_CUSTOM_ALL_REDUCE_MAX_BYTES,
) -> None:
    """Initialize custom all-reduce for tensor parallel operations.

    Args:
        tp_group: The tensor parallel group. If None, uses the model parallel group
            from the fleet's hybrid communicate group.
        custom_all_reduce_max_bytes: Maximum tensor size in bytes for which custom
            all-reduce will be used.
    """
    if tp_group is None:
        hcg = fleet.get_hybrid_communicate_group()
        tp_group = hcg.get_model_parallel_group()
    global _TP_AR
    from fastdeploy.distributed.custom_all_reduce import CustomAllreduce

    _TP_AR = CustomAllreduce(tp_group, custom_all_reduce_max_bytes)


def custom_ar_clear_ipc_handles() -> None:
    """Clear IPC handles for custom all-reduce.

    Should be called when shutting down or reinitializing custom all-reduce.
    """
    if _TP_AR is not None:
        _TP_AR.clear_ipc_handles()


try:

    def tensor_model_parallel_all_reduce_infer_meta(
        x: "paddle.static.MetaTensor", group_: paddle.distributed.communication.group.Group
    ) -> paddle.static.MetaTensor:
        """Infer meta tensor shape and dtype for tensor_model_parallel_all_reduce."""
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
        """All-reduce the input tensor across model parallel group.

        Args:
            input_: Input tensor to all-reduce. Expected shape [seq_len, hidden_size].
            group_: Communication group. If None, uses the model parallel group.

        Returns:
            All-reduced tensor with same shape and dtype as input.

        Raises:
            RuntimeError: In deterministic mode when custom all-reduce is not initialized
                or when input does not meet custom all-reduce requirements.
            AssertionError: In deterministic mode when input dtype is not supported.
        """
        inp_size = input_.shape[0] * input_.shape[1] * input_.element_size()
        if inp_size == 0:
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

        # Use custom all-reduce if available and applicable
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
            # Static mode - fail fast if deterministic mode is enabled
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
        """Perform alltoall and transpose operations for decoding.

        Args:
            input_: Input tensor.
            out: Optional output tensor for in-place operation.

        Returns:
            Transposed alltoall result.

        Raises:
            RuntimeError: If custom all-reduce is not initialized.
        """
        if input_.shape[0] == 0:
            return input_
        if _TP_AR is None:
            raise RuntimeError("decode_alltoall_transpose requires custom all-reduce to be initialized.")
        return _TP_AR.decode_alltoall_transpose(input_, out)

except Exception:  # pylint: disable=broad-except
    # Registration may fail in certain environments; set functions to None
    tensor_model_parallel_all_reduce = None


# Import stream and reduce operations for custom all-reduce
from paddle.distributed.communication import stream
from paddle.distributed.communication.reduce import ReduceOp


def _get_model_parallel_group():
    """Get the model parallel group from fleet.

    Returns:
        The model parallel communication group.

    Raises:
        RuntimeError: If fleet is not initialized.
    """
    hcg = fleet.get_hybrid_communicate_group()
    return hcg.get_model_parallel_group()


def _stream_all_reduce(
    tensor: paddle.Tensor,
    op,
    group: paddle.distributed.communication.group.Group,
    sync_op: bool = True,
) -> paddle.Tensor:
    """Perform all-reduce using stream API.

    Args:
        tensor: Input tensor to all-reduce.
        op: Reduce operation (e.g., ReduceOp.SUM).
        group: Communication group.
        sync_op: Whether to synchronize operation.

    Returns:
        All-reduced tensor.
    """
    return stream.all_reduce(tensor, op=op, group=group, sync_op=sync_op, use_calc_stream=True)


try:

    @paddle.jit.marker.unified
    def tensor_model_parallel_all_reduce_custom(input_: paddle.Tensor) -> paddle.Tensor:
        """All-reduce the input tensor across model parallel group on calc stream.

        Args:
            input_: Input tensor to all-reduce.

        Returns:
            All-reduced tensor.
        """
        if input_.shape[0] == 0:
            return input_
        if paddle.in_dynamic_mode():
            mp_group = _get_model_parallel_group()
            _stream_all_reduce(input_, op=ReduceOp.SUM, group=mp_group)
        else:
            dist.all_reduce(input_)

except Exception:  # pylint: disable=broad-except
    # Registration may fail in certain environments; set function to None
    tensor_model_parallel_all_reduce_custom = None
