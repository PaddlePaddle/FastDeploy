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

from fastdeploy.engine.request import ImagePosition
from fastdeploy.multimodal.encoder_cache_manager import EncoderCacheManager


def test_mm_encoder_cache():
    max_encoder_cache = 4096
    encoder_cache = EncoderCacheManager(max_encoder_cache=max_encoder_cache)

    mm_hashes = ["mm_hash1", "mm_hash2"]
    mm_positions = [ImagePosition(offset=120, length=400), ImagePosition(offset=620, length=800)]

    cache_length = mm_positions[0].length + mm_positions[1].length
    evict_hashes = encoder_cache.apply_cache(mm_hashes=mm_hashes, mm_positions=mm_positions)
    assert evict_hashes == [], "The evicted hashes should be empty."
    assert encoder_cache.encoder_cache.keys() == [
        "mm_hash1",
        "mm_hash2",
    ], "The cache should contain mm_hash1 and mm_hash2."
    assert (
        encoder_cache.encoder_cache_size == cache_length
    ), "The cache size should be the sum of the lengths of mm_hash1 and mm_hash2."
    assert (
        encoder_cache.encoder_cache_size <= max_encoder_cache
    ), "The cache size should be less than or equal to the max_encoder_cache."

    mm_hashes = ["mm_hash3", "mm_hash4"]
    mm_positions = [ImagePosition(offset=20, length=1204), ImagePosition(offset=1800, length=2048)]
    cache_length += mm_positions[0].length + mm_positions[1].length - 400
    evict_hashes = encoder_cache.apply_cache(mm_hashes=mm_hashes, mm_positions=mm_positions)
    assert evict_hashes == ["mm_hash1"], "The evicted hashes should be mm_hash1."
    assert encoder_cache.encoder_cache.keys() == [
        "mm_hash2",
        "mm_hash3",
        "mm_hash4",
    ], "The cache should contain mm_hash2, mm_hash3, and mm_hash4."
    assert (
        encoder_cache.encoder_cache_size == cache_length
    ), "The cache size should be the sum of the lengths of mm_hash2, mm_hash3, and mm_hash4."
    assert (
        encoder_cache.encoder_cache_size <= max_encoder_cache
    ), "The cache size should be less than or equal to the max_encoder_cache."

    cache_state = encoder_cache.has_mm_cache(["mm_hash1", "mm_hash2", "mm_hash3", "mm_hash4"])
    assert cache_state == [False, True, True, True], "The cache state should be [False, True, True, True]."

    evict_hashes = encoder_cache.apply_cache(
        mm_hashes=["mm_hash2"], mm_positions=[ImagePosition(offset=620, length=800)]
    )
    assert evict_hashes == [], "The evicted hashes should be empty."
    assert (
        encoder_cache.encoder_cache_size == cache_length
    ), "The cache size should be the sum of the lengths of mm_hash2, mm_hash3, and mm_hash4."
    assert (
        encoder_cache.encoder_cache_size <= max_encoder_cache
    ), "The cache size should be less than or equal to the max_encoder_cache."

    cache_length -= 1204
    evict_hashes = encoder_cache.evict_cache(token_nums=800)
    assert evict_hashes == ["mm_hash3"], "The evicted hashes should be mm_hash3."
    assert encoder_cache.encoder_cache.keys() == [
        "mm_hash4",
        "mm_hash2",
    ], "The cache should contain mm_hash2 and mm_hash4."
    assert (
        encoder_cache.encoder_cache_size == cache_length
    ), "The cache size should be the sum of the lengths of mm_hash2 and mm_hash4."
    assert (
        encoder_cache.encoder_cache_size <= max_encoder_cache
    ), "The cache size should be less than or equal to the max_encoder_cache."

    encoder_cache.clear_cache()
    assert encoder_cache.encoder_cache_size == 0, "The cache size should be 0."
    assert encoder_cache.encoder_cache.keys() == [], "The cache should be empty."
