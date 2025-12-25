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

import os
import traceback
from abc import abstractmethod

import paddle
from paddle import nn
from paddle.distributed.communication import deep_ep

import fastdeploy
from fastdeploy.config import MoEPhase
from fastdeploy.utils import singleton


class DeepEPEngineBase:
    """
    Base class for DeepEP engine implementations.
    """

    def __init__(
        self,
        num_max_dispatch_tokens_per_rank: int,
        hidden_size: int,
        num_experts: int,
        ep_size: int,
        ep_rank: int,
        splitwise_role: str,
        moe_phase: MoEPhase,
        async_finish: bool = False,
        group=None,
    ):
        """
        Initialize the DeepEP engine base.
        Args:
            group: The MPI group object.
            ep_size: The number of ranks.
            rank_id: The rank id.
            num_max_dispatch_tokens_per_rank: The maximum number of tokens per rank to dispatch.
            hidden_size: The hidden_size dimension of the model.
            num_experts: The number of experts.
        """
        self.num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.ep_size = ep_size
        self.rank_id = ep_rank
        self.splitwise_role = splitwise_role
        self.moe_phase = moe_phase
        self.async_finish = async_finish
        # TODO(@wufeisheng): Support configurable EP size​
        if group is None:
            group = paddle.distributed.new_group(range(ep_size))
        self.group = group
        self.num_local_experts = num_experts // ep_size
        self.deepep_engine = None

    def barrier_all(self):
        """
        barrier_all
        """
        if self.deepep_engine is not None:
            self.deepep_engine.barrier_all()
        else:
            raise RuntimeError("The deepep engine has not been initialized yet.")


@singleton
class DeepEPEngineHighThroughput(DeepEPEngineBase):
    """
    High throughput version of DeepEP engine for prefill phase.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deepep_engine = deep_ep.Buffer(
            self.group,
            int(1e9),
            0,
            low_latency_mode=False,
            num_qps_per_rank=1,
        )


@singleton
class DeepEPEngineLowLatency(DeepEPEngineBase):
    """
    Low latency version of DeepEP engine for decode phase.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.get_low_latency_buffer()

    def get_low_latency_buffer(self):
        """
        Initialize XPU-compatible low latency buffer for decode phase.
        Args:
            group: The MPI group object.
            num_max_dispatch_tokens_per_rank: The maximum number of tokens per rank to dispatch.
            hidden_size: The hidden_size dimension of the model.
        """
        try:
            print("[DeepEP] Initializing low latency buffer for XPU...")
            print(
                f"[DeepEP] hidden_size={self.hidden_size}, ep_size={self.ep_size}, "
                f"num_experts={self.num_experts}, num_local_experts={self.num_local_experts}"
            )

            # Check XPU environment
            device_info = paddle.get_device()
            print(f"[DeepEP] Current device: {device_info}")

            if not device_info.startswith("xpu"):
                raise RuntimeError(f"Expected XPU device, but got {device_info}")

            # Check critical environment variables
            bkcl_comm_size = os.environ.get("BKCL_COMM_SIZE")
            bkcl_rank = os.environ.get("BKCL_RANK_ID")
            if bkcl_comm_size is None or bkcl_rank is None:
                print("[DeepEP] Warning: BKCL_COMM_SIZE or BKCL_RANK_ID not set")
            else:
                print(f"[DeepEP] BKCL environment: COMM_SIZE={bkcl_comm_size}, RANK_ID={bkcl_rank}")

            # Get buffer size hint
            print("[DeepEP] Getting RDMA size hint...")
            num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
                self.num_max_dispatch_tokens_per_rank,
                self.hidden_size,
                self.ep_size,
                self.num_experts,
            )
            print(f"[DeepEP] RDMA size hint: {num_rdma_bytes} bytes")

            # NOTES: for best performance, the QP number **must** be equal to the number of the local experts
            if self.num_experts % self.ep_size != 0:
                raise ValueError(f"num_experts({self.num_experts}) must be divisible by ep_size({self.ep_size})")

            print("[DeepEP] Creating DeepEP buffer with following parameters:")
            print(f"  - group: {self.group}")
            print("  - num_nvl_bytes: 0 (XPU doesn't support NVLink)")
            print(f"  - num_rdma_bytes: {num_rdma_bytes}")
            print("  - low_latency_mode: True")
            print(f"  - num_qps_per_rank: {self.num_experts // self.ep_size}")

            self.deepep_engine = deep_ep.Buffer(
                self.group,
                0,  # num_nvl_bytes=0 for XPU
                num_rdma_bytes,
                low_latency_mode=True,
                num_qps_per_rank=self.num_experts // self.ep_size,
            )
            print("[DeepEP] Low latency buffer initialized successfully!")

        except Exception as e:
            print("[DeepEP] ERROR: Failed to initialize low latency buffer!")
            print(f"[DeepEP] Error type: {type(e).__name__}")
            print(f"[DeepEP] Error message: {str(e)}")
            print("[DeepEP] Traceback:")
            traceback.print_exc()
            raise e

    def low_latency_dispatch(
        self,
        hidden_states: paddle.Tensor,
        topk_idx: paddle.Tensor,
        expertwise_scale,
        use_fp8: bool = False,
        quant_group_size: int = 128,
    ):
        """
        Args:
            hidden_states: [token_num, hidden_size] 'bfloat16/int8'
            topk_idx: [token_num, num_topk] 'int64'

        Returns:
            recv_hidden_states: [num_local_experts,
                                 num_max_dispatch_tokens_per_rank * ep_size, hidden_size]
                                 ep_size * num_local_experts = num_experts
            recv_count: [num_local_experts]
            recv_count: a tensor shaped `[num_local_experts]` with type `torch.int`, indicating how many tokens each
                expert receive. As mentioned before, all not tokens are valid in `recv_x`.
            handle: the communication handle to be used in the `low_latency_combine` function.
            event: the event after executing the kernel (valid only if `async_finish` is set).
            hook: the receiving hook function (valid only if `return_recv_hook` is set).
        """
        try:
            # 调试信息输入
            print("[DeepEP] Starting low_latency_dispatch...")
            print(f"[DeepEP] hidden_states: shape={hidden_states.shape}, dtype={hidden_states.dtype}")
            print(f"[DeepEP] hidden_states: device={hidden_states.place}")
            print(f"[DeepEP] topk_idx: shape={topk_idx.shape}, dtype={topk_idx.dtype}")

            if expertwise_scale is not None:
                print(f"[DeepEP] expertwise_scale: shape={expertwise_scale.shape}, dtype={expertwise_scale.dtype}")
                print(f"[DeepEP] Using FP8 mode: {use_fp8}, quant_group_size: {quant_group_size}")
            else:
                print("[DeepEP] Not using FP8 mode")

            if self.deepep_engine is None:
                raise RuntimeError("DeepEP engine not initialized. Please call get_low_latency_buffer() first.")

            print("==========before engine low_latency_dispatch")

            (
                packed_recv_x,
                recv_expert_count,
                handle,
                _,
                dispatch_hook,
            ) = self.deepep_engine.low_latency_dispatch(
                hidden_states,
                topk_idx,
                expertwise_scale,
                self.num_max_dispatch_tokens_per_rank,
                self.num_experts,
                use_fp8=use_fp8,
                async_finish=False,
                return_recv_hook=True,
                num_per_channel=quant_group_size,
            )

            print("==========after engine low_latency_dispatch")

            print("[DeepEP] low_latency_dispatch completed successfully")
            print(f"[DeepEP] packed_recv_x: shape={packed_recv_x.shape}, dtype={packed_recv_x.dtype}")
            print(f"[DeepEP] recv_expert_count: shape={recv_expert_count.shape}, dtype={recv_expert_count.dtype}")
            # print(f"[DeepEP] recv_expert_count values: {recv_expert_count.numpy()}")
            print(f"[DeepEP] dispatch_hook type: {type(dispatch_hook)}")
            print(f"[DeepEP] handle type: {type(handle)}")

            return packed_recv_x, recv_expert_count, handle, dispatch_hook

        except Exception as e:
            print("[DeepEP] ERROR in low_latency_dispatch!")
            print(f"[DeepEP] Error type: {type(e).__name__}")
            print(f"[DeepEP] Error message: {str(e)}")
            print("[DeepEP] Full traceback:")
            traceback.print_exc()

            # 尝试获取更多调试信息
            if hasattr(e, "__cause__"):
                print(f"[DeepEP] Error cause: {e.__cause__}")
            if hasattr(e, "args"):
                print(f"[DeepEP] Error args: {e.args}")

            raise e

    def low_latency_combine(
        self,
        hidden_states: paddle.Tensor,
        topk_idx: paddle.Tensor,
        topk_weights: paddle.Tensor,
        handle,
    ):
        """
        Return:
            combined_hidden_states: [num_tokens, hidden_size]
        """
        print("============enter low_latency_combine")
        if paddle.__version__ != "0.0.0" and paddle.__version__ <= "3.1.0":
            # TODO(@wanglongzhi): Delete them when deepep in PaddlePaddle is fixed
            # and when the default recommended version of PaddlePaddle is greater than 3.1.0
            src_info, layout_range, num_max_dispatch_tokens_per_rank, num_experts = handle
            handle = (src_info, layout_range, num_max_dispatch_tokens_per_rank, None, num_experts)

        if self.deepep_engine is None:
            raise RuntimeError("DeepEP buffer not initialized!")
        combined_hidden_states, _, combine_hook = self.deepep_engine.low_latency_combine(
            hidden_states,
            topk_idx,
            topk_weights,
            handle,
            async_finish=False,
            return_recv_hook=True,
        )
        print("============after low_latency_combine")
        return combined_hidden_states, combine_hook

    def clean_low_latency_buffer(self):
        """
        clean_low_latency_buffer
        """
        pass


class XPUEPRunner:
    """
    EPRunnerBase
    """

    def __init__(
        self,
        top_k: int,
        hidden_size: int,
        num_experts: int,
        splitwise_role: str,
        moe_phase: MoEPhase,
        num_max_dispatch_tokens_per_rank: int = 1,
        ep_size: int = 1,
        ep_rank: int = 0,
        redundant_experts_num: int = 0,
        ep_group=None,
    ):
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.splitwise_role = splitwise_role
        self.moe_phase = moe_phase
        self.num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.redundant_experts_num = redundant_experts_num
        self.ep_group = ep_group
        self.ep_engine = None
        self.init_ep_engine()

    def init_ep_engine(self):
        """Initialize the EP engine with default implementation"""
        self._init_ep_engine(self._get_engine_class())

    def _init_ep_engine(self, engine_class):
        self.ep_engine = engine_class(
            num_max_dispatch_tokens_per_rank=self.num_max_dispatch_tokens_per_rank,
            hidden_size=self.hidden_size,
            num_experts=self.num_experts + self.redundant_experts_num,
            ep_size=self.ep_size,
            ep_rank=self.ep_rank,
            splitwise_role=self.splitwise_role,
            moe_phase=self.moe_phase,
            group=self.ep_group,
        )

    @abstractmethod
    def _get_engine_class(self):
        """Get the engine class to be initialized"""
        raise NotImplementedError("Subclasses must implement this method")

    def moe_select(self, layer: nn.Layer, gate_out: paddle.Tensor):
        """
        moe_select
        """
        if layer.redundant_table_manger is not None:
            (
                ep_rank_to_expert_id_list,
                expert_id_to_ep_rank_array,
                expert_in_rank_num_list,
                tokens_per_expert_stats_list,
            ) = layer.redundant_table_manger.get_ep_rank_to_expert_id_list_by_layer(layer.layer_idx)

            topk_idx, topk_weights = fastdeploy.model_executor.ops.xpu.moe_redundant_topk_select(
                gating_logits=gate_out,
                expert_id_to_ep_rank_array=expert_id_to_ep_rank_array,
                expert_in_rank_num_list=expert_in_rank_num_list,
                tokens_per_expert_stats_list=tokens_per_expert_stats_list,
                bias=layer.gate_correction_bias,
                moe_topk=self.top_k,
                apply_norm_weight=True,  # apply_norm_weight
                enable_softmax_top_k_fused=False,
                redundant_ep_rank_num_plus_one=layer.fd_config.eplb_config.redundant_experts_num + 1,
            )
        else:
            topk_idx, topk_weights = fastdeploy.model_executor.ops.xpu.moe_topk_select(
                gate_out,
                layer.gate_correction_bias,
                self.top_k,
                True,  # apply_norm_weight,
            )
        return topk_idx, topk_weights

    @abstractmethod
    def dispatch(self, *args, **kwargs):
        """
        dispatch
        """
        raise NotImplementedError

    @abstractmethod
    def combine(self, *args, **kwargs):
        """
        combine
        """
        raise NotImplementedError

    def clean_low_latency_buffer(self):
        self.ep_engine.clean_low_latency_buffer()

    def barrier_all(self):
        self.ep_engine.barrier_all()


class XPUEPPrefillRunner(XPUEPRunner):
    """
    EPPrefillRunner
    """

    def __init__(
        self,
        top_k: int,
        hidden_size: int,
        num_experts: int,
        splitwise_role: str,
        num_max_dispatch_tokens_per_rank: int,
        ep_size: int = 1,
        ep_rank: int = 0,
        redundant_experts_num: int = 0,
        ep_group=None,
        moe_phase: MoEPhase = MoEPhase("prefill"),
    ):
        super().__init__(
            top_k,
            hidden_size,
            num_experts,
            splitwise_role,
            moe_phase,
            num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank,
            ep_size=ep_size,
            ep_rank=ep_rank,
            redundant_experts_num=redundant_experts_num,
            ep_group=ep_group,
        )

    def _get_engine_class(self):
        return DeepEPEngineHighThroughput

    def dispatch(
        self,
        x: paddle.Tensor,
        topk_idx: paddle.Tensor,
        topk_weights: paddle.Tensor,
        expert_alignment: int = 1,
        *args,
        **kwargs,
    ):
        buffer = self.ep_engine.deepep_engine
        if buffer is None:
            raise RuntimeError("DeepEP buffer not initialized!")

        # 获取详细的分发布局信息，与GPU版本对齐
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            event,
        ) = buffer.get_dispatch_layout(
            topk_idx,
            self.ep_engine.num_experts,
            previous_event=kwargs.get("previous_event", None),
            allocate_on_comm_stream=False,  # XPU暂时不支持流分配
            async_finish=self.ep_engine.async_finish,
        )

        x_scale_tensor = kwargs.get("x_scale_tensor", None)
        dispatch_args = {
            "x": (x, x_scale_tensor) if x_scale_tensor is not None else x,
            "num_tokens_per_rank": num_tokens_per_rank,
            "num_tokens_per_rdma_rank": num_tokens_per_rdma_rank,
            "is_token_in_rank": is_token_in_rank,
            "num_tokens_per_expert": num_tokens_per_expert,
            "async_finish": self.ep_engine.async_finish,
            "topk_idx": topk_idx,
            "topk_weights": topk_weights,
            "expert_alignment": expert_alignment,
            "previous_event": event,
        }
        return buffer.dispatch(**dispatch_args)

    def combine(
        self,
        tmp_ffn_out: paddle.Tensor,
        handle: tuple,
        recv_topk_weights: paddle.Tensor,
        event=None,
    ):
        buffer = self.ep_engine.deepep_engine
        if buffer is None:
            raise RuntimeError("DeepEP buffer not initialized!")

        combine_args = {
            "x": tmp_ffn_out,
            "handle": handle,
            "async_finish": self.ep_engine.async_finish,
            "topk_weights": recv_topk_weights,
            "previous_event": event,
        }
        fused_moe_out, _, event = buffer.combine(**combine_args)
        return fused_moe_out, event


class XPUEPDecoderRunner(XPUEPRunner):
    """
    EPDecoderRunner
    """

    def __init__(
        self,
        top_k: int,
        hidden_size: int,
        num_experts: int,
        splitwise_role: str,
        num_max_dispatch_tokens_per_rank: int,
        ep_size: int = 1,
        ep_rank: int = 0,
        redundant_experts_num: int = 0,
        ep_group=None,
        moe_phase: MoEPhase = MoEPhase("decode"),
    ):
        super().__init__(
            top_k,
            hidden_size,
            num_experts,
            splitwise_role,
            moe_phase,
            num_max_dispatch_tokens_per_rank,
            ep_size=ep_size,
            ep_rank=ep_rank,
            redundant_experts_num=redundant_experts_num,
            ep_group=ep_group,
        )

    def _get_engine_class(self):
        return DeepEPEngineLowLatency

    def dispatch(
        self,
        x: paddle.Tensor,
        topk_idx: paddle.Tensor,
        topk_weights: paddle.Tensor,
        *args,
        **kwargs,
    ):
        expertwise_scale = kwargs.get("expertwise_scale", None)
        use_fp8 = expertwise_scale is not None
        quant_group_size = kwargs.get("quant_group_size", 128)

        (
            recv_hidden_states,
            recv_expert_count,
            handle,
            dispatch_hook,
        ) = self.ep_engine.low_latency_dispatch(x, topk_idx, expertwise_scale, use_fp8, quant_group_size)
        # valid_token_num is optional:
        # - if valid_token_num is None, it means that we CANNOT accurately know
        #   the size of the tensor, but the advantage is that it can reduce
        #   the overhead of kernel launch.
        # - if valid_token_num is NOT None, it means that we CAN accurately know
        #   the size of the tensor, but the disadvantage is that it will interrupt
        #   the process of kernel launch.
        valid_token_num = int(paddle.sum(recv_expert_count).numpy())
        if valid_token_num is None and dispatch_hook is not None:
            dispatch_hook()

        if valid_token_num is None:
            valid_token_num = -1

        if isinstance(recv_hidden_states, tuple):
            recv_x = recv_hidden_states[0]
            recv_x_scale = recv_hidden_states[1]
        else:
            recv_x = recv_hidden_states
            recv_x_scale = None

        return recv_x, recv_x_scale, recv_expert_count, handle, valid_token_num

    def combine(self, ffn_out, topk_idx, topk_weights, handle):
        combined_hidden_states, combine_hook = self.ep_engine.low_latency_combine(
            ffn_out, topk_idx, topk_weights, handle
        )
        if combine_hook is not None:
            combine_hook()
        print("===============================> low_latency_combine")
        return combined_hidden_states
