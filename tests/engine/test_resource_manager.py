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

from unittest.mock import MagicMock, patch

import pytest


def _cache_cfg(block_size=64, dec_token_num=128, max_block_num_per_seq=16, enable_prefix_caching=False):
    c = MagicMock()
    c.block_size = block_size
    c.dec_token_num = dec_token_num
    c.max_block_num_per_seq = max_block_num_per_seq
    c.enable_prefix_caching = enable_prefix_caching
    return c


def _config(cache_config=None):
    cfg = MagicMock()
    cfg.cache_config = cache_config or _cache_cfg()
    return cfg


def _task(**kw):
    t = MagicMock()
    t.request_id = kw.get("request_id", "req-001")
    t.prompt_token_ids = kw.get("prompt_token_ids", list(range(128)))
    t.prompt_token_ids_len = kw.get("prompt_token_ids_len", len(t.prompt_token_ids))
    t.block_tables = kw.get("block_tables", [])
    t.need_block_tables = kw.get("need_block_tables", [])
    t.disaggregate_info = kw.get("disaggregate_info", None)
    t.seq_lens_decoder = 0
    t.inference_time_cost = -1.0
    t.tokens_all_num = 0
    t.idx = 0
    t.num_cached_tokens = 0
    t.gpu_cache_token_num = 0
    t.cpu_cache_token_num = 0
    t.cache_info = None
    t.cache_prepare_time = 0.0
    _seed = [kw.get("seed", None)]
    t.get = lambda k: _seed[0] if k == "seed" else None

    def _set(k, v):
        if k == "seed":
            _seed[0] = v

    t.set = _set
    return t


@pytest.fixture()
def rm_env():
    """Patch heavy deps and yield a factory for ResourceManager."""
    with (
        patch("fastdeploy.engine.resource_manager.PrefixCacheManager") as pcm,
        patch("fastdeploy.engine.resource_manager.main_process_metrics", new_callable=MagicMock) as met,
        patch("fastdeploy.engine.resource_manager.llm_logger", new_callable=MagicMock) as log,
    ):
        inst = MagicMock()
        inst.num_gpu_blocks = 100
        inst.gpu_free_block_list = list(range(100))
        pcm.return_value = inst

        def factory(max_seqs=4, block_size=64, dec_token=128, enable_prefix=False, num_free=100, max_per_seq=16):
            from fastdeploy.engine.resource_manager import ResourceManager

            cc = _cache_cfg(block_size, dec_token, max_per_seq, enable_prefix)
            rm = ResourceManager(max_seqs, _config(cc), 1, "mixed")
            rm.cache_manager.gpu_free_block_list = list(range(num_free))
            rm.cache_manager.num_gpu_blocks = num_free
            rm.cache_manager.allocate_gpu_blocks = MagicMock(side_effect=lambda n: list(range(n)))
            return rm

        class Env:
            make = staticmethod(factory)
            pcm_cls = pcm
            metrics = met
            logger = log

        yield Env


# ── Setup, configuration, and block calculations ───────────────────────────
class TestResourceManagerConfig:
    """Constructor, reset_cache_config, block math, availability checks."""

    def test_init_fields(self, rm_env):
        rm = rm_env.make(max_seqs=8)
        assert rm.max_num_seqs == 8
        assert rm.stop_flags == [True] * 8
        assert rm.tasks_list == [None] * 8
        assert rm.req_dict == {}
        assert rm.real_bsz == 0

    def test_init_prefix_flag_and_pcm(self, rm_env):
        rm = rm_env.make(enable_prefix=True)
        assert rm.enable_prefix_cache is True
        rm_env.pcm_cls.assert_called()

    def test_init_max_batch_metric(self, rm_env):
        rm_env.make(max_seqs=16)
        rm_env.metrics.max_batch_size.set.assert_called_with(16)

    def test_reset_cache_config(self, rm_env):
        rm = rm_env.make()
        new = _cache_cfg(block_size=128)
        rm.reset_cache_config(new)
        assert rm.cfg.block_size == 128
        rm.cache_manager.update_cache_config.assert_called_once_with(new)

    def test_block_required(self, rm_env):
        rm = rm_env.make(block_size=64, dec_token=128)
        assert rm.get_required_block_number(100) == 4  # (100+63+128)//64

    def test_block_required_exact(self, rm_env):
        rm = rm_env.make(block_size=64, dec_token=0)
        assert rm.get_required_block_number(64) == 1

    def test_block_encoder_decoder(self, rm_env):
        rm = rm_env.make(block_size=64, dec_token=128)
        assert rm.get_encoder_block_number(100) == 2
        assert rm.get_decoder_block_number() == 2

    def test_total_block_delegates(self, rm_env):
        rm = rm_env.make()
        rm.cache_manager.num_gpu_blocks = 1024
        assert rm.total_block_number() == 1024

    def test_available_batch(self, rm_env):
        rm = rm_env.make(max_seqs=4)
        assert rm.available_batch() == 4
        rm.stop_flags[0] = rm.stop_flags[2] = False
        assert rm.available_batch() == 2

    def test_available_blocks(self, rm_env):
        rm = rm_env.make(num_free=5)
        rm.cache_manager.gpu_free_block_list = [0, 1, 2, 3, 4]
        assert rm.available_block_num() == 5

    def test_is_resource_sufficient(self, rm_env):
        rm = rm_env.make(block_size=64, dec_token=0, num_free=100)
        assert rm.is_resource_sufficient(64) is True

    def test_insufficient_no_batch(self, rm_env):
        rm = rm_env.make(max_seqs=2)
        rm.stop_flags = [False, False]
        assert rm.is_resource_sufficient(1) is False

    def test_insufficient_no_blocks(self, rm_env):
        rm = rm_env.make(num_free=0)
        assert rm.is_resource_sufficient(64) is False


# ── Allocation, block tables, recycling ─────────────────────────────────────
class TestResourceManagerAllocate:
    """_get_block_tables, allocate_resources_for_new_tasks, _recycle, free,
    check_and_free, _delete_cached_data, _record_request_cache_info."""

    # _get_block_tables
    def test_get_blocks_all(self, rm_env):
        rm = rm_env.make(block_size=64, dec_token=0)
        assert len(rm._get_block_tables(64)) == 1

    def test_get_blocks_encoder_decoder(self, rm_env):
        rm = rm_env.make(block_size=64, dec_token=128)
        assert len(rm._get_block_tables(100, "encoder")) == 2
        assert len(rm._get_block_tables(0, "decoder")) == 2

    def test_get_blocks_unknown_raises(self, rm_env):
        rm = rm_env.make()
        with pytest.raises(ValueError):
            rm._get_block_tables(64, "invalid")

    def test_get_blocks_insufficient(self, rm_env):
        rm = rm_env.make(block_size=64, dec_token=0, num_free=0)
        assert rm._get_block_tables(64) == []

    # allocate — no prefix
    def test_allocate_single(self, rm_env):
        rm = rm_env.make(enable_prefix=False, dec_token=0)
        res = rm.allocate_resources_for_new_tasks([_task()])
        assert len(res) == 1
        assert rm.stop_flags[0] is False

    def test_allocate_multiple_and_bsz(self, rm_env):
        rm = rm_env.make(max_seqs=4, enable_prefix=False, dec_token=0)
        ts = [_task(request_id=f"r{i}") for i in range(3)]
        res = rm.allocate_resources_for_new_tasks(ts)
        assert len(res) == 3
        assert rm.stop_flags == [False, False, False, True]
        assert rm.real_bsz == 3

    def test_allocate_skips_occupied(self, rm_env):
        rm = rm_env.make(max_seqs=4, enable_prefix=False, dec_token=0)
        rm.stop_flags[0] = False
        t = _task()
        rm.allocate_resources_for_new_tasks([t])
        assert t.idx == 1

    def test_allocate_sets_seed(self, rm_env):
        rm = rm_env.make(enable_prefix=False, dec_token=0)
        t = _task(seed=None)
        rm.allocate_resources_for_new_tasks([t])
        assert t.get("seed") is not None

    def test_allocate_empty(self, rm_env):
        rm = rm_env.make(enable_prefix=False)
        assert rm.allocate_resources_for_new_tasks([]) == []

    def test_allocate_disaggregate(self, rm_env):
        rm = rm_env.make(enable_prefix=False, dec_token=0)
        for role in ("prefill", "decode"):
            t = _task(request_id=f"r-{role}", disaggregate_info={"role": role})
            rm.allocate_resources_for_new_tasks([t])
            assert t.request_id in rm.req_dict

    def test_allocate_retry_on_empty(self, rm_env):
        rm = rm_env.make(enable_prefix=False, dec_token=0)
        t = _task()
        call_count = [0]
        orig = rm._get_block_tables

        def _mock(n, typ="all"):
            call_count[0] += 1
            if call_count[0] == 1:
                return []
            rm.cache_manager.gpu_free_block_list = list(range(10))
            return orig(n, typ)

        rm._get_block_tables = _mock
        rm.cache_manager.gpu_free_block_list = []
        assert len(rm.allocate_resources_for_new_tasks([t])) == 1

    # allocate — with prefix
    def test_prefix_allocates_and_records(self, rm_env):
        rm = rm_env.make(enable_prefix=True, dec_token=0, block_size=64)
        rm.cache_manager.request_block_ids = MagicMock(
            return_value=([10, 11], [20, 21], {"gpu_cache_blocks": 2, "cpu_cache_blocks": 0})
        )
        t = _task(prompt_token_ids=list(range(256)))
        res = rm.allocate_resources_for_new_tasks([t])
        assert len(res) == 1
        assert t.block_tables == [10, 11, 20, 21]

    def test_prefix_insufficient(self, rm_env):
        rm = rm_env.make(enable_prefix=True, dec_token=0)
        rm.cache_manager.request_block_ids = MagicMock(
            return_value=([10], None, {"gpu_cache_blocks": 1, "cpu_cache_blocks": 0})
        )
        assert rm.allocate_resources_for_new_tasks([_task()]) is None

    def test_prefix_disaggregate(self, rm_env):
        rm = rm_env.make(enable_prefix=True, dec_token=0, block_size=64)
        rm.cache_manager.request_block_ids = MagicMock(
            return_value=([10, 11], [20, 21], {"gpu_cache_blocks": 2, "cpu_cache_blocks": 0})
        )
        for role in ("prefill", "decode"):
            t = _task(request_id=f"r-{role}", prompt_token_ids=list(range(256)), disaggregate_info={"role": role})
            rm.allocate_resources_for_new_tasks([t])
            assert t.request_id in rm.req_dict

    # recycle / free / check_and_free
    def test_recycle_prefix_releases(self, rm_env):
        rm = rm_env.make(enable_prefix=True)
        t = _task()
        rm._recycle_block_tables(t)
        rm.cache_manager.release_block_ids_async.assert_called_once_with(t)

    def test_recycle_normal(self, rm_env):
        rm = rm_env.make(enable_prefix=False)
        t = _task(block_tables=[1, 2, 3])
        rm._recycle_block_tables(t)
        rm.cache_manager.recycle_gpu_blocks.assert_called_once_with([1, 2, 3])

    def test_free_delegates(self, rm_env):
        rm = rm_env.make()
        rm.free_block_tables(32)
        rm.cache_manager.free_block_ids_async.assert_called_once_with(32)

    def test_check_and_free_prefix(self, rm_env):
        rm = rm_env.make(enable_prefix=True, num_free=5, max_per_seq=16)
        rm.check_and_free_block_tables()
        rm.cache_manager.free_block_ids_async.assert_called_once_with(16)

    def test_check_and_free_above_threshold(self, rm_env):
        rm = rm_env.make(enable_prefix=True, num_free=20, max_per_seq=16)
        rm.check_and_free_block_tables()
        rm.cache_manager.free_block_ids_async.assert_not_called()

    def test_check_and_free_no_prefix(self, rm_env):
        rm = rm_env.make(enable_prefix=False)
        rm.check_and_free_block_tables()
        rm.cache_manager.free_block_ids_async.assert_not_called()

    # cache helpers
    def test_delete_cached_data(self, rm_env):
        rm = rm_env.make(block_size=64)
        t = _task(prompt_token_ids=list(range(128)))
        rm._delete_cached_data(t, 128)
        assert t.prompt_token_ids_len == 64
        assert t.seq_lens_decoder == 64
        t2 = _task(prompt_token_ids=list(range(256)))
        rm._delete_cached_data(t2, 64)
        assert t2.prompt_token_ids_len == 192

    def test_record_cache_info(self, rm_env):
        rm = rm_env.make(block_size=64)
        t = _task(prompt_token_ids=list(range(256)))
        hit = {"gpu_cache_blocks": 2, "cpu_cache_blocks": 1}
        cached = rm._record_request_cache_info(t, [10, 11], [20, 21, 22], hit)
        assert cached == 128
        assert t.block_tables == [10, 11, 20, 21, 22]
        assert t.num_cached_tokens == 128
        assert t.cache_info == (2, 2)


# ── Info & metrics ──────────────────────────────────────────────────────────
class TestResourceManagerInfo:
    """info string and gpu_cache_usage_perc."""

    def test_info_string(self, rm_env):
        rm = rm_env.make()
        rm.cache_manager.num_gpu_blocks = 100
        rm.cache_manager.gpu_free_block_list = list(range(80))
        info = rm.info()
        assert "ResourceManager info" in info
        assert "total_block_number: 100" in info

    def test_usage_calc(self, rm_env):
        rm = rm_env.make()
        rm.cache_manager.num_gpu_blocks = 100
        rm.cache_manager.gpu_free_block_list = list(range(80))
        assert abs(rm.get_gpu_cache_usage_perc() - 0.2) < 1e-9

    def test_usage_full(self, rm_env):
        rm = rm_env.make(num_free=0)
        rm.cache_manager.num_gpu_blocks = 100
        rm.cache_manager.gpu_free_block_list = []
        assert abs(rm.get_gpu_cache_usage_perc() - 1.0) < 1e-9

    def test_usage_zero_total(self, rm_env):
        rm = rm_env.make(num_free=0)
        rm.cache_manager.num_gpu_blocks = 0
        rm.cache_manager.gpu_free_block_list = []
        assert rm.get_gpu_cache_usage_perc() == 0.0
