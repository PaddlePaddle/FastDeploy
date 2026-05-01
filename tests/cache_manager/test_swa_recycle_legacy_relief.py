# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
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
"""T53 PR1 legacy-pool relief tests for per-head uniform SWA block recycle."""

from types import SimpleNamespace

import pytest

from fastdeploy.engine.sched.resource_manager_v1 import ResourceManagerV1


class _FakeCacheManager:
    def __init__(self, kv_num_heads=2):
        self.kv_num_heads = kv_num_heads
        self.head_recycled = []
        self.legacy_recycled = []

    def recycle_gpu_blocks_head_wise(self, cache_ids, req_id=None):
        self.head_recycled.append((req_id, list(cache_ids)))

    def recycle_gpu_blocks(self, block_ids, req_id=None):
        self.legacy_recycled.append((req_id, list(block_ids)))


def _build_manager():
    rm = object.__new__(ResourceManagerV1)
    rm.config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16, enable_prefix_caching=False),
        scheduler_config=SimpleNamespace(splitwise_role="mixed"),
        model_config=SimpleNamespace(
            window_size=64,
            sink_size=32,
            num_key_value_heads=2,
            head_wise_swa_ratio=1.0,
        ),
    )
    rm.cache_manager = _FakeCacheManager(kv_num_heads=2)
    rm.enable_cache_manager_v1 = False
    rm.swa_head_recycle_upto = {}
    rm.swa_head_block_tables = {}
    rm.swa_legacy_recycle_upto = {}
    rm.swa_legacy_recycled_blocks = {}
    rm.using_extend_tables_req_id = set()
    return rm


def test_uniform_swa_recycle_returns_legacy_blocks_without_shifting_block_tables(monkeypatch):
    """Uniform SWA frees legacy IDs once while preserving absolute block-table positions."""
    monkeypatch.setattr("fastdeploy.engine.sched.resource_manager_v1.envs.FD_HEAD_WISE_KV_CACHE", 1)
    rm = _build_manager()
    rm.swa_head_block_tables["req-uniform"] = [list(range(100, 132)), list(range(200, 232))]
    original_block_tables = list(range(1000, 1032))
    req = SimpleNamespace(
        request_id="req-uniform",
        num_total_tokens=512,
        num_computed_tokens=512,
        block_tables=list(original_block_tables),
        num_cached_blocks=0,
    )

    released = rm.recycle_request_swa_head_cache(req)

    assert released == 26 * 2
    assert req.block_tables == original_block_tables
    assert rm.cache_manager.legacy_recycled == [("req-uniform", original_block_tables[2:28])]

    rm.recycle_request_swa_head_cache(req)
    assert rm.cache_manager.legacy_recycled == [("req-uniform", original_block_tables[2:28])]

    rm._free_blocks(req)
    final_legacy_recycle = rm.cache_manager.legacy_recycled[-1][1]
    assert not set(final_legacy_recycle).intersection(original_block_tables[2:28])
    assert set(final_legacy_recycle) == set(original_block_tables[:2] + original_block_tables[28:])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
