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

from dataclasses import dataclass
from enum import Enum


@dataclass
class ModelWeightsStatus:
    NORMAL = 0
    UPDATING = 1
    CLEARING = -1
    CLEARED = -2


@dataclass
class PrefixTreeStatus:
    NORMAL = 0
    UPDATING = 1
    CLEARING = -1
    CLEARED = -2


@dataclass
class KVCacheStatus:
    NORMAL = 0
    UPDATING = 1
    CLEARING = -1
    CLEARED = -2


@dataclass
class ExistTaskStatus:
    EMPTY = 0
    EXIST = 1
    REFUSE = 2


class RearrangeExpertStatus(Enum):
    FREE = 0
    DOING = 1
    LOAD_SUCC = 2  # load weight from disk success
    DONE = 3
