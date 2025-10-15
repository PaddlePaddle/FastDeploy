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

from abc import abstractmethod

import deep_ep
import paddle
from paddle import nn
from paddleformers.utils.log import logger

import fastdeploy
from fastdeploy.config import MoEPhase
from fastdeploy.utils import singleton


@singleton
class DeepEPEngine:
    """
    A wrapper class for DeepEP engine.
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
        Initialize the DeepEP engine.
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
        self.init_deepep_engine()

    def init_deepep_engine(self):
        if self.splitwise_role == "mixed" or self.moe_phase.phase == "prefill":
            self.deepep_engine = deep_ep.Buffer(
                self.group,
                int(1e9),
                0,
                num_experts=self.num_experts,
                low_latency_mode=False,
                num_qps_per_rank=1,
            )
        elif self.moe_phase.phase == "decode":
            logger.info("Initializing Low Latency Buffer")
            self.get_low_latency_buffer()
        else:
            raise ValueError(f"Unknown generation phase {self.moe_phase}")

    def get_low_latency_buffer(self):
        """
        Get the DeepEP buffer.
        Args:
            group: The MPI group object.
            num_max_dispatch_tokens_per_rank: The maximum number of tokens per rank to dispatch.
            hidden_size: The hidden_size dimension of the model.
        """
        # NOTES: the low-latency mode will consume much more space than the normal mode
        # So we recommend that `num_max_dispatch_tokens_per_rank`
        #   (the actual batch size in the decoding engine) should be less than 256
        num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
            self.num_max_dispatch_tokens_per_rank,
            self.hidden_size,
            self.ep_size,
            self.num_experts,
        )
        # Allocate a buffer if not existed or not enough buffer size
        if (
            self.deepep_engine is None
            or self.deepep_engine.group != self.group
            or not self.deepep_engine.low_latency_mode
            or self.deepep_engine.num_rdma_bytes < num_rdma_bytes
        ):
            # NOTES: for best performance, the QP number **must** be equal to the number of the local experts
            assert self.num_experts % self.ep_size == 0
            self.deepep_engine = deep_ep.Buffer(
                self.group,
                0,
                num_rdma_bytes,
                self.num_experts,
                low_latency_mode=True,
                num_qps_per_rank=self.num_experts // self.num_ranks,
            )

    def low_latency_dispatch(
        self,
        hidden_states: paddle.Tensor,
        topk_idx: paddle.Tensor,
        expertwise_scale,
        use_fp8: bool = False,
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
        moe_in_w4a8_scale = None
        (
            packed_recv_x,
            recv_expert_count,
            handle,
            dispatch_hook,
            valid_token_num,
        ) = self.deepep_engine.low_latency_dispatch(
            hidden_states,
            moe_in_w4a8_scale,
            topk_idx,
            self.num_max_dispatch_tokens_per_rank,
            self.num_experts,
            use_fp8=use_fp8,
            async_finish=False,
            return_recv_hook=True,
        )

        return packed_recv_x, recv_expert_count, handle, dispatch_hook, valid_token_num

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
        combined_hidden_states, combine_hook = self.deepep_engine.low_latency_combine(
            hidden_states,
            topk_idx,
            topk_weights,
            handle,
            async_finish=False,
            return_recv_hook=True,
        )
        return combined_hidden_states, combine_hook

    def clean_low_latency_buffer(self):
        """
        clean_low_latency_buffer
        """
        pass

    def barrier_all(self):
        """
        barrier_all
        """
        self.deepep_engine.barrier_all()


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
        self.init_ep_engine()

    def init_ep_engine(self):
        self.ep_engine = DeepEPEngine(
            num_max_dispatch_tokens_per_rank=self.num_max_dispatch_tokens_per_rank,
            hidden_size=self.hidden_size,
            num_experts=self.num_experts + self.redundant_experts_num,
            ep_size=self.ep_size,
            ep_rank=self.ep_rank,
            splitwise_role=self.splitwise_role,
            moe_phase=self.moe_phase,
            group=self.ep_group,
        )

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
                redundant_ep_rank_num_plus_one=layer.fd_config.model_config.redundant_experts_num + 1,
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

    def dispatch(
        self,
        x: paddle.Tensor,
        topk_idx: paddle.Tensor,
        topk_weights: paddle.Tensor,
        *args,
        **kwargs,
    ):
        self.num_combined_tokens = x.shape[0]
        x_scale_tensor = kwargs.get("x_scale_tensor", None)
        dispatch_args = {
            "x": (x, x_scale_tensor) if x_scale_tensor is not None else x,
            "topk_idx": topk_idx,
            "topk_weights": topk_weights,
        }
        return self.ep_engine.deepep_engine.dispatch(**dispatch_args)

    def combine(
        self,
        tmp_ffn_out: paddle.Tensor,
        handle: tuple,
        recv_topk_weights: paddle.Tensor,
    ):
        combine_args = {
            "x": tmp_ffn_out,
            "topk_weights": recv_topk_weights,
            "num_combined_tokens": self.num_combined_tokens,
        }
        fused_moe_out, _, _ = self.ep_engine.deepep_engine.combine(**combine_args)

        return fused_moe_out


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

        (
            recv_hidden_states,
            recv_expert_count,
            handle,
            dispatch_hook,
            valid_token_num,
        ) = self.ep_engine.low_latency_dispatch(x, topk_idx, expertwise_scale, use_fp8)
        # no need to call dispatch_hook here, because it has already been done in xDeepEP
        # if dispatch_hook is not None:
        #     dispatch_hook()

        return recv_hidden_states, recv_expert_count, handle, valid_token_num

    def combine(self, ffn_out, topk_idx, topk_weights, handle):
        combined_hidden_states, combine_hook = self.ep_engine.low_latency_combine(
            ffn_out, topk_idx, topk_weights, handle
        )
        if combine_hook is not None:
            combine_hook()

        return combined_hidden_states
