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

import paddle
from paddle import nn
from paddleformers.utils.log import logger

try:
    from paddle.distributed.communication import deep_ep
except:
    logger.warning("import deep_ep Failed!")

from typing import Optional

import fastdeploy
from fastdeploy.config import MoEPhase
from fastdeploy.utils import singleton


class DeepEPBufferManager:
    _engine: Optional["DeepEPEngine"] = None

    @classmethod
    def set_engine(cls, engine: "DeepEPEngine"):
        cls._engine = engine

    @classmethod
    def clear_buffer(cls):
        if cls._engine:
            cls._engine.clear_deep_ep_buffer()

    @classmethod
    def recreate_buffer(cls):
        if cls._engine:
            cls._engine.create_deep_ep_buffer()


class DeepEPBuffer:
    """
    Encapsulates DeepEP buffer creation, management and cleanup.
    """

    def __init__(
        self,
        group,
        hidden_size: int,
        num_experts: int,
        ep_size: int,
        num_max_dispatch_tokens_per_rank: int,
        splitwise_role: str,
        moe_phase: MoEPhase,
        use_internode_ll_two_stage: bool = False,
        top_k: int = 8,
    ):
        self.group = group
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.ep_size = ep_size
        self.num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank
        self.splitwise_role = splitwise_role
        self.moe_phase = moe_phase
        self.use_internode_ll_two_stage = use_internode_ll_two_stage
        self.top_k = top_k

        self.deepep_buffer = None
        self.num_nvl_bytes = 0
        self.num_rdma_bytes = 0

        # Precompute buffer sizes
        self._compute_buffer_sizes()

    def _compute_buffer_sizes(self, param_bytes: int = 2):
        hidden_bytes = self.hidden_size * param_bytes  # bf16 or fp16

        for config in (
            deep_ep.Buffer.get_dispatch_config(self.group.world_size),
            deep_ep.Buffer.get_combine_config(self.group.world_size),
        ):
            self.num_nvl_bytes = max(
                config.get_nvl_buffer_size_hint(hidden_bytes, self.group.world_size), self.num_nvl_bytes
            )
            self.num_rdma_bytes = max(
                config.get_rdma_buffer_size_hint(hidden_bytes, self.group.world_size), self.num_rdma_bytes
            )

        if self.splitwise_role == "mixed" or self.moe_phase.phase == "decode":
            if not self.use_internode_ll_two_stage:
                num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
                    self.num_max_dispatch_tokens_per_rank,
                    self.hidden_size,
                    self.ep_size,
                    self.num_experts,
                )
            else:
                num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint_two_stage(
                    self.num_max_dispatch_tokens_per_rank, self.hidden_size, self.ep_size, self.num_experts, self.top_k
                )
                num_nvl_bytes = deep_ep.Buffer.get_low_latency_nvl_size_hint_two_stage(
                    self.num_max_dispatch_tokens_per_rank,
                    self.hidden_size,
                    self.ep_size,
                    self.num_experts,
                    self.top_k,
                    True,  # just supports dispatch_use_fp8 = True now!
                )
                self.num_nvl_bytes = max(self.num_nvl_bytes, num_nvl_bytes)
            self.num_rdma_bytes = max(self.num_rdma_bytes, num_rdma_bytes)

        logger.info(f"DeepEP num nvl bytes : {self.num_nvl_bytes}, num rdma bytes : {self.num_rdma_bytes}")

    def create_buffer(self):
        """Create or recreate buffer based on role and phase."""
        if self.deepep_buffer is not None:
            self.clear_buffer()

        if self.splitwise_role == "mixed":
            logger.info("Initializing mixed mode buffer (low latency).")
            self.deepep_buffer = deep_ep.Buffer(
                self.group,
                self.num_nvl_bytes,
                self.num_rdma_bytes,
                low_latency_mode=True,
                num_qps_per_rank=24,
            )
            self.deepep_buffer.set_num_sms(14)  # TODO: tune in future
        else:
            if self.moe_phase.phase == "decode":
                self._create_low_latency_buffer()
            elif self.moe_phase.phase == "prefill":
                logger.info("Initializing High Throughput Buffer for prefill phase.")
                self.deepep_buffer = deep_ep.Buffer(
                    self.group,
                    self.num_nvl_bytes,
                    self.num_rdma_bytes,
                    low_latency_mode=True,
                    num_qps_per_rank=24,
                )
            else:
                raise ValueError(f"Unknown generation phase: {self.moe_phase.phase}")

        logger.info("DeepEP buffer created successfully.")

    def _create_low_latency_buffer(self):
        if self.deepep_buffer is None:
            assert self.num_experts % self.ep_size == 0
            if self.ep_size // 8 > 1:
                num_qps_per_rank_now = self.ep_size // 8
            else:
                num_qps_per_rank_now = 1
            self.deepep_buffer = deep_ep.Buffer(
                self.group,
                self.num_nvl_bytes,
                self.num_rdma_bytes,
                low_latency_mode=True,
                num_qps_per_rank=num_qps_per_rank_now,
            )

    def clear_buffer(self):
        """Clear buffer and free memory."""
        if self.deepep_buffer is not None:
            del self.deepep_buffer
            self.deepep_buffer = None
            logger.info("DeepEP buffer cleared.")

    def get_buffer(self):
        return self.deepep_buffer

    def clean_low_latency_buffer(self):
        if self.deepep_buffer is not None:
            if not self.use_internode_ll_two_stage:
                self.deepep_buffer.clean_low_latency_buffer(
                    self.num_max_dispatch_tokens_per_rank,
                    self.hidden_size,
                    self.num_experts,
                )
            else:
                self.deepep_buffer.clean_low_latency_two_stage_buffer(
                    self.num_max_dispatch_tokens_per_rank,
                    self.hidden_size,
                    self.num_experts,
                    self.top_k,
                    self.ep_size,
                    True,  # just supports dispatch_use_fp8 = True now!
                )

    def barrier_all(self):
        if self.deepep_buffer is not None:
            self.deepep_buffer.barrier_all()


@singleton
class DeepEPEngine:
    """
    A wrapper class for DeepEP engine.
    Manages buffer lifecycle based on role and phase.
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
        async_finish: bool = True,
        group=None,
        use_internode_ll_two_stage: bool = False,
        top_k: int = 8,
    ):
        if group is None:
            group = paddle.distributed.new_group(range(ep_size))
        self.group = group
        self.ep_size = ep_size
        self.rank_id = ep_rank
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_local_experts = num_experts // ep_size
        self.top_k = top_k
        self.async_finish = async_finish

        self.ep_config = None

        # Store phase and role for buffer management
        self._splitwise_role = splitwise_role
        self._moe_phase = moe_phase

        # Initialize buffer manager
        self.buffer = DeepEPBuffer(
            group=self.group,
            hidden_size=hidden_size,
            num_experts=num_experts,
            ep_size=ep_size,
            num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank,
            splitwise_role=splitwise_role,
            moe_phase=moe_phase,
            use_internode_ll_two_stage=use_internode_ll_two_stage,
            top_k=self.top_k,
        )
        self.buffer.create_buffer()

        # Register for global buffer management
        DeepEPBufferManager.set_engine(self)

    @property
    def deepep_engine(self):
        """Backward compatibility alias."""
        return self.buffer.get_buffer()

    def clear_deep_ep_buffer(self):
        self.buffer.clear_buffer()

    def create_deep_ep_buffer(self):
        self.buffer.create_buffer()

    def low_latency_dispatch(
        self,
        hidden_states: paddle.Tensor,
        topk_idx: paddle.Tensor,
        expertwise_scale,
        use_fp8: bool = False,
        quant_group_size: int = 128,
    ):
        if self.deepep_engine is None:
            raise RuntimeError("DeepEP buffer not initialized!")

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
            self.buffer.num_max_dispatch_tokens_per_rank,
            self.num_experts,
            use_fp8=use_fp8,
            async_finish=False,
            return_recv_hook=True,
            # num_per_channel=quant_group_size,
        )

        return packed_recv_x, recv_expert_count, handle, dispatch_hook

    def low_latency_dispatch_two_stage(
        self,
        hidden_states: paddle.Tensor,
        topk_idx: paddle.Tensor,
        topk_weights: paddle.Tensor,
        expertwise_scale,
        use_fp8: bool = False,
    ):
        if self.deepep_engine is None:
            raise RuntimeError("DeepEP buffer not initialized!")

        (
            packed_recv_x,
            packed_recv_count,
            _,
            handle,
            _,
            dispatch_hook,
        ) = self.deepep_engine.low_latency_dispatch_two_stage(
            hidden_states,
            topk_idx,
            topk_weights,
            self.buffer.num_max_dispatch_tokens_per_rank,
            self.num_experts,
            use_fp8=use_fp8,
            async_finish=False,
            return_recv_hook=True,
        )

        return packed_recv_x, packed_recv_count, handle, dispatch_hook

    def low_latency_combine(
        self,
        hidden_states: paddle.Tensor,
        topk_idx: paddle.Tensor,
        topk_weights: paddle.Tensor,
        handle,
    ):
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
        return combined_hidden_states, combine_hook

    def low_latency_combine_two_stage(
        self,
        hidden_states: paddle.Tensor,
        topk_idx: paddle.Tensor,
        topk_weights: paddle.Tensor,
        dispatch_use_fp8: bool,
        handle,
    ):
        if self.deepep_engine is None:
            raise RuntimeError("DeepEP buffer not initialized!")

        combined_hidden_states, _, combine_hook = self.deepep_engine.low_latency_combine_two_stage(
            hidden_states,
            topk_idx,
            topk_weights,
            handle,
            async_finish=False,
            dispatch_use_fp8=dispatch_use_fp8,
            return_recv_hook=True,
        )
        return combined_hidden_states, combine_hook

    def clean_low_latency_buffer(self):
        self.buffer.clean_low_latency_buffer()

    def barrier_all(self):
        self.buffer.barrier_all()


class EPRunner:
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
        use_internode_ll_two_stage: bool = False,
    ):
        self.top_k = top_k
        self.num_experts = num_experts
        self.redundant_experts_num = redundant_experts_num
        self.use_internode_ll_two_stage = use_internode_ll_two_stage
        self.ep_engine = DeepEPEngine(
            num_max_dispatch_tokens_per_rank=num_max_dispatch_tokens_per_rank,
            hidden_size=hidden_size,
            num_experts=num_experts + redundant_experts_num,
            ep_size=ep_size,
            ep_rank=ep_rank,
            splitwise_role=splitwise_role,
            moe_phase=moe_phase,
            group=ep_group,
            use_internode_ll_two_stage=self.use_internode_ll_two_stage,
            top_k=self.top_k,
        )

    def moe_select(self, layer: nn.Layer, gate_out: paddle.Tensor):
        if layer.redundant_table_manger is not None:
            (
                ep_rank_to_expert_id_list,
                expert_id_to_ep_rank_array,
                expert_in_rank_num_list,
                tokens_per_expert_stats_list,
            ) = layer.redundant_table_manger.get_ep_rank_to_expert_id_list_by_layer(layer.layer_idx)

            if layer.topk_method == "noaux_tc":
                from .moe import get_moe_scores

                score, topk_weights, topk_idx = get_moe_scores(
                    gate_out,
                    layer.n_group,
                    layer.topk_group,
                    layer.top_k,
                    layer.routed_scaling_factor,
                    layer.gate_correction_bias,
                    getattr(layer, "renormalize", True),
                    expert_id_to_ep_rank_array=expert_id_to_ep_rank_array,
                    expert_in_rank_num_list=expert_in_rank_num_list,
                    tokens_per_expert_stats_list=tokens_per_expert_stats_list,
                    redundant_ep_rank_num_plus_one=layer.fd_config.model_config.redundant_experts_num + 1,
                )
            else:
                topk_idx, topk_weights = fastdeploy.model_executor.ops.gpu.moe_redundant_topk_select(
                    gating_logits=gate_out,
                    expert_id_to_ep_rank_array=expert_id_to_ep_rank_array,
                    expert_in_rank_num_list=expert_in_rank_num_list,
                    tokens_per_expert_stats_list=tokens_per_expert_stats_list,
                    bias=layer.gate_correction_bias,
                    moe_topk=self.top_k,
                    apply_norm_weight=True,
                    enable_softmax_top_k_fused=False,
                    redundant_ep_rank_num_plus_one=layer.fd_config.model_config.redundant_experts_num + 1,
                )
        else:
            if layer.topk_method == "noaux_tc":
                from fastdeploy.model_executor.layers.moe.moe import get_moe_scores

                score, topk_weights, topk_idx = get_moe_scores(
                    gate_out,
                    layer.n_group,
                    layer.topk_group,
                    layer.top_k,
                    layer.routed_scaling_factor,
                    layer.gate_correction_bias,
                    getattr(layer, "renormalize", True),
                )
            else:
                topk_idx, topk_weights = fastdeploy.model_executor.ops.gpu.moe_topk_select(
                    gate_out,
                    layer.gate_correction_bias,
                    self.top_k,
                    True,
                    False,
                )
        return topk_idx, topk_weights

    @abstractmethod
    def dispatch(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def combine(self, *args, **kwargs):
        raise NotImplementedError

    def clean_low_latency_buffer(self):
        self.ep_engine.clean_low_latency_buffer()

    def clear_deep_ep_buffer(self):
        self.ep_engine.clear_deep_ep_buffer()

    def create_deep_ep_buffer(self):
        self.ep_engine.create_deep_ep_buffer()


class EPPrefillRunner(EPRunner):
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
        moe_phase: MoEPhase = MoEPhase("prefill"),
        ep_group=None,
        use_internode_ll_two_stage: bool = False,
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
            use_internode_ll_two_stage=use_internode_ll_two_stage,
        )

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

        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            event,
        ) = buffer.get_dispatch_layout(topk_idx, self.num_experts, async_finish=self.ep_engine.async_finish)

        x_scale_tensor = kwargs.get("x_scale_tensor", None)
        dispatch_args = {
            "x": (x, x_scale_tensor) if x_scale_tensor is not None else x,
            "num_tokens_per_rank": num_tokens_per_rank,
            "num_tokens_per_rdma_rank": num_tokens_per_rdma_rank,
            "is_token_in_rank": is_token_in_rank,
            "num_tokens_per_expert": num_tokens_per_expert,
            "config": self.ep_engine.ep_config,  # assuming ep_config still in engine
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
    ):
        buffer = self.ep_engine.deepep_engine
        if buffer is None:
            raise RuntimeError("DeepEP buffer not initialized!")

        combine_args = {
            "x": tmp_ffn_out,
            "handle": handle,
            "config": self.ep_engine.ep_config,
            "async_finish": self.ep_engine.async_finish,
            "topk_weights": recv_topk_weights,
        }
        fused_moe_out, _, event = buffer.combine(**combine_args)
        return fused_moe_out, event


class EPDecoderRunner(EPRunner):
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
        use_internode_ll_two_stage: bool = False,
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
            use_internode_ll_two_stage=use_internode_ll_two_stage,
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
        use_fp8 = kwargs.get("use_fp8", False)

        if not self.use_internode_ll_two_stage:
            recv_hidden_states, recv_expert_count, handle, dispatch_hook = self.ep_engine.low_latency_dispatch(
                x, topk_idx, expertwise_scale, use_fp8
            )
        else:
            # just supports dispatch_use_fp8 = True now!
            assert use_fp8 is True
            recv_hidden_states, recv_expert_count, handle, dispatch_hook = (
                self.ep_engine.low_latency_dispatch_two_stage(x, topk_idx, topk_weights, expertwise_scale, use_fp8)
            )
        if dispatch_hook is not None:
            dispatch_hook()

        return recv_hidden_states, recv_expert_count, handle

    def combine(self, ffn_out, topk_idx, topk_weights, handle):
        if not self.use_internode_ll_two_stage:
            combined_hidden_states, combine_hook = self.ep_engine.low_latency_combine(
                ffn_out, topk_idx, topk_weights, handle
            )
        else:
            combined_hidden_states, combine_hook = self.ep_engine.low_latency_combine_two_stage(
                ffn_out, topk_idx, topk_weights, True, handle  # just supports dispatch_use_fp8 = True now!
            )
        if combine_hook is not None:
            combine_hook()

        return combined_hidden_states
