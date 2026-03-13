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

from types import SimpleNamespace
from unittest.mock import patch

import pytest

# -- Stubs ------------------------------------------------------------------


class _StubCacheManager:
    """Minimal PrefixCacheManager surface for unit-testing ResourceManager."""

    def __init__(self, *args, num_blocks=100, **kwargs):
        self.num_gpu_blocks = num_blocks
        self.gpu_free_block_list = list(range(num_blocks))
        self._recycled = []
        self._released = []

    def allocate_gpu_blocks(self, n):
        out = self.gpu_free_block_list[:n]
        self.gpu_free_block_list = self.gpu_free_block_list[n:]
        return out

    def recycle_gpu_blocks(self, blocks):
        self._recycled.extend(blocks)
        self.gpu_free_block_list.extend(blocks)

    def release_block_ids_async(self, task):
        self._released.append(task)

    def free_block_ids_async(self, n):
        return n

    def update_cache_config(self, cfg):
        pass

    def request_block_ids(self, task, block_size, dec_token_num):
        total = (len(task.prompt_token_ids) + block_size - 1) // block_size
        common = list(range(total // 2))
        unique = list(range(100, 100 + total - total // 2))
        return common, unique, {"gpu_cache_blocks": len(common), "cpu_cache_blocks": 0}


class _Task:
    """Real task object with all fields ResourceManager touches."""

    def __init__(self, request_id="req-1", prompt_len=128, disaggregate_info=None):
        self.request_id = request_id
        self.prompt_token_ids = list(range(prompt_len))
        self.prompt_token_ids_len = prompt_len
        self.block_tables = []
        self.need_block_tables = []
        self.disaggregate_info = disaggregate_info
        self.seq_lens_decoder = 0
        self.inference_time_cost = -1.0
        self.tokens_all_num = 0
        self.idx = 0
        self.num_cached_tokens = 0
        self.gpu_cache_token_num = 0
        self.cpu_cache_token_num = 0
        self.cache_info = None
        self.cache_prepare_time = 0.0
        self._seed = None

    def get(self, k):
        return self._seed if k == "seed" else None

    def set(self, k, v):
        if k == "seed":
            self._seed = v


def _cache_cfg(block_size=64, dec_token_num=128, max_block_num_per_seq=16, enable_prefix_caching=False):
    return SimpleNamespace(
        block_size=block_size,
        dec_token_num=dec_token_num,
        max_block_num_per_seq=max_block_num_per_seq,
        enable_prefix_caching=enable_prefix_caching,
    )


def _config(cache_config=None):
    return SimpleNamespace(cache_config=cache_config or _cache_cfg())


def _noop_logger():
    return SimpleNamespace(
        info=lambda *a, **kw: None,
        debug=lambda *a, **kw: None,
        error=lambda *a, **kw: None,
        warning=lambda *a, **kw: None,
    )


def _stub_metrics():
    m = SimpleNamespace()
    for n in (
        "max_batch_size",
        "batch_size",
        "available_gpu_block_num",
        "gpu_cache_usage_perc",
        "prefix_cache_token_num",
        "prefix_gpu_cache_token_num",
        "prefix_cpu_cache_token_num",
    ):
        setattr(m, n, SimpleNamespace(set=lambda v: None, inc=lambda v: None))
    return m


@pytest.fixture()
def rm_factory():
    """Yield a factory that creates ResourceManagers with stubbed deps."""
    with (
        patch("fastdeploy.engine.resource_manager.PrefixCacheManager", _StubCacheManager),
        patch("fastdeploy.engine.resource_manager.main_process_metrics", _stub_metrics()),
        patch("fastdeploy.engine.resource_manager.llm_logger", _noop_logger()),
    ):
        from fastdeploy.engine.resource_manager import ResourceManager

        def make(max_seqs=4, block_size=64, dec_token=128, enable_prefix=False, num_free=100):
            cc = _cache_cfg(block_size, dec_token, 16, enable_prefix)
            rm = ResourceManager(max_seqs, _config(cc), 1, "mixed")
            rm.cache_manager = _StubCacheManager(num_blocks=num_free)
            return rm

        yield make


# -- Tests ------------------------------------------------------------------


def test_init_block_math_and_config(rm_factory):
    """Constructor fields, block calculations, reset_cache_config."""
    rm = rm_factory(max_seqs=8, block_size=64, dec_token=128)
    assert rm.max_num_seqs == 8
    assert rm.stop_flags == [True] * 8
    assert rm.get_required_block_number(100) == 4
    assert rm.get_encoder_block_number(100) == 2
    assert rm.get_decoder_block_number() == 2
    assert rm.total_block_number() == 100
    rm.reset_cache_config(_cache_cfg(block_size=128))
    assert rm.cfg.block_size == 128


def test_availability_and_sufficiency(rm_factory):
    """available_batch, available_block_num, is_resource_sufficient."""
    rm = rm_factory(max_seqs=4, dec_token=0, num_free=100)
    assert rm.available_batch() == 4
    assert rm.available_block_num() == 100
    assert rm.is_resource_sufficient(64)
    rm.stop_flags = [False] * 4
    assert not rm.is_resource_sufficient(1)
    rm2 = rm_factory(max_seqs=4, num_free=0)
    assert not rm2.is_resource_sufficient(64)


def test_allocate_no_prefix(rm_factory):
    """Main allocation path without prefix caching (happy + empty-blocks)."""
    rm = rm_factory(max_seqs=4, enable_prefix=False, dec_token=0, num_free=100)
    tasks = [_Task(request_id=f"r{i}") for i in range(3)]
    result = rm.allocate_resources_for_new_tasks(tasks)
    assert len(result) == 3
    assert rm.stop_flags == [False, False, False, True]
    assert rm.real_bsz == 3
    assert all(t.get("seed") is not None for t in result)
    assert all(len(t.block_tables) > 0 for t in result)


def test_allocate_with_prefix(rm_factory):
    """Allocation with prefix cache (exercises _record_request_cache_info)."""
    rm = rm_factory(max_seqs=4, enable_prefix=True, dec_token=0, block_size=64, num_free=100)
    t = _Task(prompt_len=256)
    result = rm.allocate_resources_for_new_tasks([t])
    assert len(result) == 1
    assert len(t.block_tables) > 0
    assert t.num_cached_tokens >= 0
    assert t.cache_info is not None


def test_allocate_disaggregate(rm_factory):
    """Disaggregate prefill/decode paths (prefix + no-prefix)."""
    rm = rm_factory(max_seqs=4, enable_prefix=True, dec_token=0, block_size=64, num_free=100)
    t = _Task(prompt_len=256, disaggregate_info={"role": "prefill"})
    rm.allocate_resources_for_new_tasks([t])
    assert "block_tables" in t.disaggregate_info
    assert t.request_id in rm.req_dict
    # No-prefix + decode
    rm2 = rm_factory(max_seqs=4, enable_prefix=False, dec_token=0, num_free=100)
    t2 = _Task(prompt_len=128, disaggregate_info={"role": "decode"})
    rm2.allocate_resources_for_new_tasks([t2])
    assert t2.request_id in rm2.req_dict


def test_recycle_free_and_check(rm_factory):
    """_recycle_block_tables, free_block_tables, check_and_free_block_tables."""
    rm = rm_factory(enable_prefix=False, num_free=100)
    t = _Task()
    t.block_tables = [0, 1, 2]
    rm._recycle_block_tables(t)
    assert 0 in rm.cache_manager._recycled
    # Prefix recycle delegates to release_block_ids_async
    rm2 = rm_factory(enable_prefix=True, num_free=100)
    t2 = _Task()
    t2.block_tables = [0, 1]
    rm2._recycle_block_tables(t2)
    assert t2 in rm2.cache_manager._released
    # free + check paths
    assert rm.free_block_tables(10) == 10
    rm.check_and_free_block_tables()
    rm3 = rm_factory(enable_prefix=True, num_free=5)
    rm3.check_and_free_block_tables()


def test_info_and_cache_usage(rm_factory):
    """info() string and get_gpu_cache_usage_perc."""
    rm = rm_factory(num_free=100)
    assert "ResourceManager info" in rm.info()
    rm.cache_manager.num_gpu_blocks = 100
    rm.cache_manager.gpu_free_block_list = list(range(80))
    assert abs(rm.get_gpu_cache_usage_perc() - 0.2) < 1e-9
    rm2 = rm_factory(num_free=0)
    rm2.cache_manager.num_gpu_blocks = 0
    assert rm2.get_gpu_cache_usage_perc() == 0.0


def test_delete_cached_data(rm_factory):
    """_delete_cached_data: full and partial cache hits."""
    rm = rm_factory(block_size=64)
    t = _Task(prompt_len=128)
    rm._delete_cached_data(t, 128)
    assert t.prompt_token_ids_len == 64
    assert t.seq_lens_decoder == 64
    t2 = _Task(prompt_len=256)
    rm._delete_cached_data(t2, 64)
    assert t2.prompt_token_ids_len == 192
    assert t2.seq_lens_decoder == 64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
