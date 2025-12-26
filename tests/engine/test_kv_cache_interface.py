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

import unittest

from fastdeploy.engine.kv_cache_interface import AttentionSpec, KVCacheSpec


class TestKVCacheSpec(unittest.TestCase):

    def test_merge_valid(self):
        # Create two valid KVCacheSpec objects with the same block_size and block_memory_used
        spec1 = KVCacheSpec(block_size=256, block_memory_used=1024)
        spec2 = KVCacheSpec(block_size=256, block_memory_used=1024)

        merged_spec = KVCacheSpec.merge([spec1, spec2])

        self.assertEqual(merged_spec.block_size, spec1.block_size)
        self.assertEqual(merged_spec.block_memory_used, spec1.block_memory_used)

    def test_merge_invalid(self):
        spec1 = KVCacheSpec(block_size=256, block_memory_used=1024)
        spec2 = KVCacheSpec(block_size=512, block_memory_used=1024)

        with self.assertRaises(AssertionError):
            KVCacheSpec.merge([spec1, spec2])

    def test_attention_spec_inheritance(self):
        # Create an AttentionSpec object
        attention_spec = AttentionSpec(
            block_size=256, block_memory_used=1024, num_kv_heads=12, head_size=64, dtype="float32"
        )

        self.assertEqual(attention_spec.block_size, 256)
        self.assertEqual(attention_spec.block_memory_used, 1024)
        self.assertEqual(attention_spec.num_kv_heads, 12)
        self.assertEqual(attention_spec.head_size, 64)
        self.assertEqual(attention_spec.dtype, "float32")

    def test_attention_spec_merge(self):
        # Create two AttentionSpec objects with the same attributes
        spec1 = AttentionSpec(block_size=256, block_memory_used=1024, num_kv_heads=12, head_size=64, dtype="float32")
        spec2 = AttentionSpec(block_size=256, block_memory_used=1024, num_kv_heads=12, head_size=64, dtype="float32")

        merged_spec = AttentionSpec.merge([spec1, spec2])

        self.assertEqual(merged_spec.block_size, spec1.block_size)
        self.assertEqual(merged_spec.block_memory_used, spec1.block_memory_used)
        self.assertEqual(merged_spec.num_kv_heads, spec1.num_kv_heads)
        self.assertEqual(merged_spec.head_size, spec1.head_size)
        self.assertEqual(merged_spec.dtype, spec1.dtype)

    def test_attention_spec_merge_invalid(self):
        # Create two AttentionSpec objects with different attributes
        spec1 = AttentionSpec(block_size=256, block_memory_used=1024, num_kv_heads=12, head_size=64, dtype="float32")
        spec2 = AttentionSpec(block_size=512, block_memory_used=1024, num_kv_heads=12, head_size=64, dtype="float32")

        with self.assertRaises(AssertionError):
            AttentionSpec.merge([spec1, spec2])


if __name__ == "__main__":
    unittest.main()
