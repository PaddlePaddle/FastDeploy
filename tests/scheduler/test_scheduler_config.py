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

import unittest

from fastdeploy.scheduler.config import (
    DPLocalSchedulerConfig,
    GlobalSchedulerConfig,
    LocalSchedulerConfig,
    SchedulerConfig,
)


class TestLocalSchedulerConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = LocalSchedulerConfig()
        self.assertEqual(cfg.max_size, -1)
        self.assertEqual(cfg.ttl, 900)
        self.assertEqual(cfg.max_model_len, 8192)
        self.assertFalse(cfg.enable_chunked_prefill)

    def test_auto_threshold(self):
        """long_prefill_token_threshold should be 4% of max_model_len when set to 0."""
        cfg = LocalSchedulerConfig(max_model_len=10000, long_prefill_token_threshold=0)
        self.assertEqual(cfg.long_prefill_token_threshold, 400)

    def test_explicit_threshold(self):
        cfg = LocalSchedulerConfig(long_prefill_token_threshold=512)
        self.assertEqual(cfg.long_prefill_token_threshold, 512)

    def test_custom_values(self):
        cfg = LocalSchedulerConfig(max_size=100, ttl=300, max_model_len=4096)
        self.assertEqual(cfg.max_size, 100)
        self.assertEqual(cfg.ttl, 300)
        self.assertEqual(cfg.max_model_len, 4096)

    def test_kwargs_ignored(self):
        """Extra kwargs should not raise."""
        cfg = LocalSchedulerConfig(unknown_key="value")
        self.assertFalse(hasattr(cfg, "unknown_key"))


class TestDPLocalSchedulerConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = DPLocalSchedulerConfig()
        self.assertEqual(cfg.splitwise_role, "prefill")

    def test_custom_role(self):
        cfg = DPLocalSchedulerConfig(splitwise_role="decode")
        self.assertEqual(cfg.splitwise_role, "decode")


class TestGlobalSchedulerConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = GlobalSchedulerConfig()
        self.assertEqual(cfg.host, "127.0.0.1")
        self.assertEqual(cfg.port, 6379)
        self.assertEqual(cfg.db, 0)
        self.assertIsNone(cfg.password)
        self.assertEqual(cfg.topic, "default")

    def test_check_invalid_ttl(self):
        cfg = GlobalSchedulerConfig(ttl=-1)
        with self.assertRaises(ValueError):
            cfg.check()

    def test_check_invalid_min_load_score(self):
        cfg = GlobalSchedulerConfig(min_load_score=0)
        with self.assertRaises(ValueError):
            cfg.check()

    def test_check_invalid_load_shards_num(self):
        cfg = GlobalSchedulerConfig(load_shards_num=0)
        with self.assertRaises(ValueError):
            cfg.check()

    def test_auto_threshold(self):
        cfg = GlobalSchedulerConfig(max_model_len=20000, long_prefill_token_threshold=0)
        self.assertEqual(cfg.long_prefill_token_threshold, 800)


class TestSchedulerConfig(unittest.TestCase):
    def test_local_scheduler(self):
        cfg = SchedulerConfig({"name": "local", "max_size": 50, "ttl": 600})
        self.assertEqual(cfg.name, "local")
        self.assertIsInstance(cfg.config, LocalSchedulerConfig)
        self.assertEqual(cfg.config.max_size, 50)

    def test_dp_scheduler(self):
        cfg = SchedulerConfig({"name": "dp", "splitwise_role": "decode"})
        self.assertEqual(cfg.name, "dp")
        self.assertIsInstance(cfg.config, DPLocalSchedulerConfig)

    def test_global_scheduler(self):
        cfg = SchedulerConfig({"name": "global", "host": "redis.local"})
        self.assertEqual(cfg.name, "global")
        self.assertIsInstance(cfg.config, GlobalSchedulerConfig)
        self.assertEqual(cfg.config.host, "redis.local")

    def test_check_unknown_name_raises(self):
        cfg = SchedulerConfig({"name": "unknown"})
        with self.assertRaises(Exception):
            cfg.check()

    def test_default_attrs(self):
        cfg = SchedulerConfig({"name": "local"})
        self.assertEqual(cfg.max_num_batched_tokens, 2048)
        self.assertEqual(cfg.max_extra_num_batched_tokens, 16384)
        self.assertEqual(cfg.max_num_seqs, 34)
        self.assertEqual(cfg.splitwise_role, "mixed")
        self.assertFalse(cfg.enable_overlap_schedule)

    def test_attrs_override(self):
        cfg = SchedulerConfig({"name": "local", "max_num_seqs": 64, "max_num_batched_tokens": 4096})
        self.assertEqual(cfg.max_num_seqs, 64)
        self.assertEqual(cfg.max_num_batched_tokens, 4096)


if __name__ == "__main__":
    unittest.main()
