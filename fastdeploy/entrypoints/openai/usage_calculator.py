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

import numpy as np


def count_tokens(tokens):
    """
    Count the number of tokens in a nested list or array structure.
    """
    count = 0
    stack = [tokens]
    while stack:
        current = stack.pop()
        if isinstance(current, (list, tuple, np.ndarray)):
            for item in reversed(current):
                stack.append(item)
        else:
            count += 1
    return count
