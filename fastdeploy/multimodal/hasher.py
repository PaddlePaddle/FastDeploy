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

import hashlib
import pickle

import numpy as np


class MultimodalHasher:

    @classmethod
    def hash_features(cls, obj: object) -> str:
        if isinstance(obj, np.ndarray):
            # Encode shape and dtype into the hash to avoid collisions between
            # arrays that share the same raw bytes but differ in layout, e.g.
            # a (6,4) vs (4,6) array, or float32 vs uint8 reinterpretation.
            header = f"{obj.shape}|{obj.dtype}|".encode()
            return hashlib.sha256(header + obj.tobytes()).hexdigest()
        return hashlib.sha256((pickle.dumps(obj))).hexdigest()
