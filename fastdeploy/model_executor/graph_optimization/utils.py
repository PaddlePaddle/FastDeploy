"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import contextlib
from typing import List


def get_input_length_list(num_tokens: int, batch_size: int) -> List[int]:
    """
    Generates a list of input sequence lengths for CUDA Graph capture.

    This function addresses a specific problem: in the pure prefill stage, variable
    input lengths (e.g., `prompt[160, 0]` vs. `prompt[80, 80]`) can lead to different
    CUDA Grid dimensions for kernels like `split_q_block`. This prevents CUDA Graph
    reuse.

    The **`split_q_block`** kernel calculates the total number of blocks, which directly
    determines the `griddim.x` launch parameter for the **`multi_query_append_attention_kernel`**.
    The blocks for a single sequence are determined by the formula:
    `num_blocks = ceil((sequence_length * group_size) / block_shape_q)`

    Due to the `ceil` (ceiling) function, distributing a total number of tokens across
    a batch of shorter sequences will result in a larger total block count. For example,
    with a `group_size` of 5 and `block_shape_q` of 64:
    - A single sequence of 160 tokens requires `ceil((160 * 5) / 64) = 13` blocks.
    - Two sequences of 80 tokens each require `ceil((80 * 5) / 64) * 2 = 7 * 2 = 14` blocks.

    To ensure graph replayability, this function creates a "dummy" list of sequence
    lengths that's designed to produce the theoretical maximum `encoder_num_blocks_x_cpu`
    for the given `num_tokens` and `batch_size`. This strategy ensures the captured
    CUDA Graph has the largest possible grid dimensions. At runtime, if the actual number
    of blocks is less than or equal to this maximum, the kernel can safely execute by
    using an early-exit mechanism.

    Args:
        num_tokens (int): The total number of tokens across all sequences.
        batch_size (int): The number of sequences (requests) in the batch.

    Returns:
        List[int]: A list of integers representing the sequence length for each request.
                   This list is crafted to maximize the total number of blocks.
    """
    input_length_list = []
    if num_tokens < batch_size:
        input_length_list = [1] * num_tokens
    else:
        input_length_list = [1] * (batch_size - 1)
        input_length_list.append(num_tokens - batch_size + 1)
    return input_length_list


def create_guard(default_value):
    _state = default_value

    @contextlib.contextmanager
    def state_guard(current_state):
        nonlocal _state
        old_state = _state
        _state = current_state
        try:
            yield
        finally:
            _state = old_state

    def get_state():
        return _state

    return state_guard, get_state


sot_warmup_guard, in_sot_warmup_mode = create_guard(False)
profile_run_guard, in_profile_run_mode = create_guard(False)
