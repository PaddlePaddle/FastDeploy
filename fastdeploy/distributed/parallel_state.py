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

from paddle.distributed import fleet


def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    hcg = fleet.get_hybrid_communicate_group()
    mp_size = hcg.get_model_parallel_world_size()
    return mp_size


def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    hcg = fleet.get_hybrid_communicate_group()
    mp_rank = hcg.get_model_parallel_rank()
    return mp_rank
