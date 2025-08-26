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

"""test for moe ops"""

import unittest

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.incubate.nn.functional import swiglu

from fastdeploy.model_executor.ops.gpu import (
    fused_expert_moe,
    moe_expert_dispatch,
    moe_expert_ffn,
    moe_expert_reduce,
)

# Set random seeds for reproducibility
paddle.seed(42)
np.random.seed(42)


class Expert(nn.Layer):
    """A single expert layer using SwiGLU activation."""

    def __init__(self, d_model, d_feedforward):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_feedforward * 2)  # *2 for SwiGLU
        self.fc2 = nn.Linear(d_feedforward, d_model)

    def forward(self, x):
        """forward"""
        x = self.fc1(x)
        x = swiglu(x)
        return self.fc2(x)


class TestFusedMoeConsistency(unittest.TestCase):
    """Test case for verifying consistency between baseline and fused MoE implementations."""

    @classmethod
    def setUpClass(cls):
        """Class-level setup that runs once before all tests."""
        cls.set_config()
        paddle.set_default_dtype(cls.dtype)

    @classmethod
    def set_config(cls):
        """Set the configuration parameters for the test."""
        cls.dtype = "bfloat16"
        cls.batch_size = 8
        cls.seq_len = 128
        cls.num_experts = 16
        cls.d_model = 8192
        cls.d_feedforward = 128
        cls.top_k = 4
        cls.rtol = 1e-2
        cls.atol = 1e-2

    def setUp(self):
        """Test-level setup that runs before each test."""
        self.init_experts()
        self.prepare_data()

    def init_experts(self):
        """Initialize expert layers and gate weights."""
        self.experts = nn.LayerList([Expert(self.d_model, self.d_feedforward) for _ in range(self.num_experts)])

        # Initialize gate weights
        self.gate = nn.Linear(self.d_model, self.num_experts)
        self.gate_weight = self.gate.weight.cast("float32")

    def prepare_data(self):
        """Prepare input data and expert parameters."""
        # Input tensor
        self.x = paddle.randn([self.batch_size, self.seq_len, self.d_model], dtype=self.dtype)

        # Stack expert parameters for fused operations
        self.w0 = paddle.stack([e.fc1.weight for e in self.experts]).astype(self.dtype)
        self.b0 = (
            paddle.stack([e.fc1.bias for e in self.experts]).reshape([self.num_experts, 1, -1]).astype(self.dtype)
        )
        self.w1 = paddle.stack([e.fc2.weight for e in self.experts]).astype(self.dtype)
        self.b1 = (
            paddle.stack([e.fc2.bias for e in self.experts]).reshape([self.num_experts, 1, -1]).astype(self.dtype)
        )

    def baseline_forward(self, hidden_states):
        """Baseline implementation processing experts sequentially."""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.reshape([-1, hidden_dim])

        # Routing computation
        logits = paddle.matmul(hidden_states.cast("float32"), self.gate_weight)
        weights = F.softmax(logits, axis=-1)
        routing_weights, selected_experts = paddle.topk(weights, self.top_k, axis=-1)

        # Initialize output
        final_hidden_states = paddle.zeros_like(hidden_states)
        expert_mask = paddle.transpose(F.one_hot(selected_experts, num_classes=self.num_experts), [2, 1, 0])

        # Process each expert
        for expert_id in range(self.num_experts):
            idx, top_x = paddle.where(expert_mask[expert_id])
            if top_x.size == 0:  # Skip if no tokens for this expert
                continue

            current_state = paddle.index_select(hidden_states, top_x, axis=0)
            expert_out = self.experts[expert_id](current_state)

            current_hidden_states = expert_out * routing_weights[top_x, idx].reshape([-1, 1])
            paddle.index_add_(
                x=final_hidden_states,
                index=top_x.squeeze(),
                axis=0,
                value=current_hidden_states.to(hidden_states.dtype),
            )

        return final_hidden_states.reshape([batch_size, seq_len, hidden_dim])

    def fused_forward(self, x):
        """Fused MoE implementation using a single kernel."""
        return fused_expert_moe(
            x,
            self.gate_weight,
            self.w0,
            self.w1,
            self.b0,
            None,  # No bias for second part of SwiGLU
            self.b1,
            None,  # No activation for second linear
            "None",  # No activation type
            self.top_k,
            False,  # Not renormalizing topk
            False,  # Not using expert capacity
        )

    def split_forward(self, hidden_states):
        """Split implementation using separate dispatch/ffn/reduce ops."""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.reshape([-1, hidden_dim])

        # Routing computation
        logits = paddle.matmul(hidden_states.cast("float32"), self.gate_weight)
        scores = F.softmax(logits, axis=-1)

        # Dispatch tokens to experts
        (
            permute_input,
            tokens_expert_prefix_sum,
            permute_indices_per_token,
            top_k_weights,
            top_k_indices,
            expert_idx_per_token,
        ) = moe_expert_dispatch(hidden_states, scores, None, None, self.top_k, False, topk_only_mode=True)

        # Process through experts
        ffn_out = moe_expert_ffn(
            permute_input,
            tokens_expert_prefix_sum,
            self.w0,
            self.w1,
            self.b0,
            None,
            None,
            None,
            None,
            "none",
            False,
        )

        # Combine results
        output = moe_expert_reduce(
            ffn_out,
            top_k_weights,
            permute_indices_per_token,
            top_k_indices,
            None,
            norm_topk_prob=False,
            routed_scaling_factor=1.0,
        )

        return output.reshape([batch_size, seq_len, hidden_dim])

    def test_consistency(self):
        """Test consistency between all three implementations."""
        # Compute outputs
        base_out = self.baseline_forward(self.x)
        fused_out = self.fused_forward(self.x)
        split_out = self.split_forward(self.x)

        # Convert to float32 for comparison
        base_out = base_out.cast("float32").numpy()
        fused_out = fused_out.cast("float32").numpy()
        split_out = split_out.cast("float32").numpy()

        # Compare baseline vs fused
        np.testing.assert_allclose(
            base_out,
            fused_out,
            rtol=self.rtol,
            atol=self.atol,
            err_msg="Baseline and fused outputs differ",
        )

        # Compare baseline vs split
        np.testing.assert_allclose(
            base_out,
            split_out,
            rtol=self.rtol,
            atol=self.atol,
            err_msg="Baseline and split outputs differ",
        )


if __name__ == "__main__":
    unittest.main()
