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

from fastdeploy.utils import data_processor_logger


class MultimodalHasher:

    @classmethod
    def hash_features(cls, obj: object) -> str:
        if isinstance(obj, np.ndarray):
            return hashlib.sha256((obj.tobytes())).hexdigest()

        data_processor_logger.warning(
            f"Unsupported type for hashing features: {type(obj)}" + ", use pickle for serialization"
        )
        return hashlib.sha256((pickle.dumps(obj))).hexdigest()
