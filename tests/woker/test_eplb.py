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

import numpy as np
import pytest

from fastdeploy.worker import eplb


@pytest.mark.parametrize(
    "num_layers,num_experts,num_groups,num_nodes,num_gpus,num_replicas",
    [
        (3, 16, 4, 2, 4, 16),  # Hierarchical scenario
        (2, 8, 3, 2, 4, 8),
    ],
)
def test_rebalance_experts_shapes(num_layers, num_experts, num_groups, num_nodes, num_gpus, num_replicas):
    # Generate random weight array
    weight = np.random.randint(1, 10, size=(num_layers, num_experts)).astype(np.float32)
    phy2log, log2phy, logcnt = eplb.rebalance_experts(
        weight, num_replicas=num_replicas, num_groups=num_groups, num_nodes=num_nodes, num_gpus=num_gpus
    )

    # Check output shapes
    assert phy2log.shape == (num_layers, num_replicas)
    assert log2phy.shape[0] == num_layers
    assert log2phy.shape[1] == num_experts
    assert logcnt.shape == (num_layers, num_experts)

    # Check value validity
    assert (logcnt >= 1).all()
    assert phy2log.min() >= 0
    assert phy2log.max() < num_experts


def test_rebalance_experts_consistency_small():
    num_layers = 1
    num_experts = 4
    num_groups = 4
    num_nodes = 2
    num_gpus = 4
    num_replicas = 4

    weight = np.ones((num_layers, num_experts), dtype=np.float32)

    phy2log, log2phy, logcnt = eplb.rebalance_experts(
        weight, num_replicas=num_replicas, num_groups=num_groups, num_nodes=num_nodes, num_gpus=num_gpus
    )

    # Each physical replica maps to unique logical expert
    for layer in range(num_layers):
        for phy in range(num_replicas):
            log_id = phy2log[layer, phy]
            rank = np.where(log2phy[layer, log_id] == phy)[0]
            assert len(rank) == 1


def test_replicate_experts_edge():
    weight = np.ones((1, 4), dtype=np.float32)
    phy2log, rank, logcnt = eplb.replicate_experts(weight, num_phy=4)

    assert (logcnt == 1).all()
    assert phy2log.shape == (1, 4)
    assert rank.shape == (1, 4)


def test_balanced_packing_edge():
    weight = np.array([[5, 3, 2, 1]], dtype=np.float32)
    pack_index, rank_in_pack = eplb.balanced_packing(weight, num_packs=4)

    assert (rank_in_pack == 0).all()
    assert (pack_index == np.arange(4)).all()


if __name__ == "__main__":
    pytest.main([__file__])
