"""
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

from typing import Optional, Tuple

import paddle
import paddle.distributed as dist

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.utils import has_flashinfer
from fastdeploy.utils import get_logger

logger = get_logger("flashinfer", "flashinfer.log")

_flashinfer_comm = None
_workspace_manager = None


def _get_flashinfer_comm():
    """Lazily import flashinfer.comm to avoid side effects at module load time."""
    global _flashinfer_comm
    if _flashinfer_comm is not None:
        return _flashinfer_comm
    if has_flashinfer():
        try:
            with paddle.use_compat_guard(enable=True, scope={"flashinfer"}):
                import flashinfer.comm as comm

                _flashinfer_comm = comm
        except ImportError:
            logger.warning("flashinfer.comm is not available, falling back to standard " "implementation")
    return _flashinfer_comm


class FlashInferWorkspaceManager:
    def __init__(self):
        self.workspace_tensor = None
        self.ipc_handles = None
        self.world_size = None
        self.rank = None
        self.initialized = False

    def initialize(
        self,
        world_size: int,
        rank: int,
        max_token_num: int,
        hidden_dim: int,
        group=None,
        use_fp32_lamport: bool = False,
    ):
        """Initialize workspace"""
        if self.initialized and self.world_size == world_size:
            return

        comm = _get_flashinfer_comm()
        if comm is None:
            logger.warning("FlashInfer comm not available, skipping workspace " "initialization")
            return

        self.cleanup()

        self.ipc_handles, self.workspace_tensor = comm.trtllm_create_ipc_workspace_for_all_reduce_fusion(
            rank,
            world_size,
            max_token_num,
            hidden_dim,
            group=group,
            use_fp32_lamport=use_fp32_lamport,
        )

        self.world_size = world_size
        self.rank = rank
        self.initialized = True

        logger.info(f"FlashInfer workspace initialized for rank {rank}, " f"world_size {world_size}")

    def cleanup(self):
        """Clean up workspace"""
        if self.initialized and self.ipc_handles is not None:
            try:
                comm = _get_flashinfer_comm()
                if comm is not None:
                    comm.trtllm_destroy_ipc_workspace_for_all_reduce(self.ipc_handles, group=dist.get_group())
            except Exception as e:
                logger.warning(f"Failed to cleanup FlashInfer workspace: {e}")
            finally:
                self.workspace_tensor = None
                self.ipc_handles = None
                self.initialized = False


_workspace_manager = FlashInferWorkspaceManager()


def ensure_workspace_initialized(
    fd_config: FDConfig, max_token_num: int = 2048, hidden_dim: int = 4096, use_fp32_lamport: bool = False
):
    """Ensure workspace is initialized"""
    comm = _get_flashinfer_comm()
    if not has_flashinfer() or comm is None:
        return False

    assert fd_config is not None
    world_size = fd_config.parallel_config.tensor_parallel_size
    if world_size <= 1:
        return False

    rank = dist.get_rank()

    if not _workspace_manager.initialized or _workspace_manager.world_size != world_size:
        _workspace_manager.initialize(
            world_size=world_size,
            rank=rank,
            max_token_num=max_token_num,
            hidden_dim=hidden_dim,
            use_fp32_lamport=use_fp32_lamport,
        )

    return _workspace_manager.initialized


def flashinfer_allreduce_residual_rmsnorm(
    fd_config: FDConfig,
    input_tensor: paddle.Tensor,
    residual: paddle.Tensor,
    weight: paddle.Tensor,
    eps: float = 1e-6,
    max_token_num: int = 2048,
    use_oneshot: Optional[bool] = None,
    trigger_completion_at_end: bool = False,
    fp32_acc: bool = False,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    """
    Use FlashInfer's fused allreduce + residual + RMS norm operation
    """
    comm = _get_flashinfer_comm()
    if not has_flashinfer() or comm is None:
        logger.debug("FlashInfer not available, falling back to standard " "implementation")
        return None, None

    assert fd_config is not None
    world_size = fd_config.parallel_config.tensor_parallel_size
    if world_size <= 1:
        logger.debug("Single GPU, no need for allreduce fusion")
        return None, None

    assert input_tensor.shape[0] <= max_token_num

    if not ensure_workspace_initialized(
        fd_config=fd_config,
        max_token_num=max_token_num,
        hidden_dim=input_tensor.shape[-1],
        use_fp32_lamport=(input_tensor.dtype == paddle.float32),
    ):
        logger.debug("FlashInfer workspace not available")
        return None, None

    token_num, hidden_dim = input_tensor.shape

    residual_out = paddle.empty_like(residual)
    norm_out = paddle.empty_like(input_tensor)
    # support empty tensor
    if input_tensor.shape[0] == 0:
        return norm_out, residual_out
    comm.trtllm_allreduce_fusion(
        allreduce_in=input_tensor,
        world_size=world_size,
        world_rank=dist.get_rank(),
        token_num=token_num,
        hidden_dim=hidden_dim,
        workspace_ptrs=_workspace_manager.workspace_tensor,
        launch_with_pdl=True,
        use_oneshot=use_oneshot,
        trigger_completion_at_end=trigger_completion_at_end,
        fp32_acc=fp32_acc,
        pattern_code=(comm.AllReduceFusionPattern.kARResidualRMSNorm),
        allreduce_out=None,
        residual_in=residual,
        residual_out=residual_out,
        norm_out=norm_out,
        quant_out=None,
        scale_out=None,
        rms_gamma=weight,
        rms_eps=eps,
        scale_factor=None,
        layout_code=None,
    )

    return norm_out, residual_out


def cleanup_flashinfer_workspace():
    global _workspace_manager
    if _workspace_manager is not None:
        _workspace_manager.cleanup()
