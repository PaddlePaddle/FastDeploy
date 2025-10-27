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

import hashlib
import pickle
import unittest

import numpy as np

from fastdeploy.multimodal.hasher import MultimodalHasher


class TestHashFeatures(unittest.TestCase):
    def test_hash_features_ndarray(self):
        """Test hash features with numpy ndarray"""
        arr = np.random.randint(low=0, high=255, size=(28, 28), dtype=np.uint8)
        arr_hash = MultimodalHasher.hash_features(arr)
        target_hash = hashlib.sha256((arr.tobytes())).hexdigest()
        assert arr_hash == target_hash, f"Ndarray hash mismatch: {arr_hash} != {target_hash}"

    def test_hash_features_object(self):
        """Test hash features with unsupported object type"""
        obj = {"key": "value"}
        obj_hash = MultimodalHasher.hash_features(obj)
        target_hash = hashlib.sha256((pickle.dumps(obj))).hexdigest()
        assert obj_hash == target_hash, f"Dict hash mismatch: {obj_hash} != {target_hash}"

        obj = "test hasher str"
        obj_hash = MultimodalHasher.hash_features(obj)
        target_hash = hashlib.sha256((pickle.dumps(obj))).hexdigest()
        assert obj_hash == target_hash, f"Str hash mismatch: {obj_hash} != {target_hash}"


if __name__ == "__main__":
    unittest.main()
