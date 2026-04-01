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
Tests for fastdeploy/engine/engine.py (LLMEngine)
"""

import os
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

from fastdeploy.engine.engine import LLMEngine


class TestLLMEngineStopProfile(unittest.TestCase):
    """测试 LLMEngine._stop_profile 方法"""

    def test_stop_profile_logs_worker_traceback_and_returns_false(self):
        """测试 worker 进程失败时，_stop_profile 打印 traceback 并返回 False"""
        eng = object.__new__(LLMEngine)
        eng.do_profile = 1
        eng.get_profile_block_num_signal = type("Sig", (), {"value": np.array([0])})()
        eng.worker_proc = Mock(poll=lambda: 1)

        with tempfile.TemporaryDirectory() as temp_dir:
            worker_log = os.path.join(temp_dir, "workerlog.0")
            with open(worker_log, "w", encoding="utf-8") as fp:
                fp.write(
                    "Traceback (most recent call last):\n"
                    "ValueError: The total number of blocks cannot be less than zero.\n"
                )

            with (
                patch("fastdeploy.engine.engine.time.sleep", lambda *_: None),
                patch("fastdeploy.engine.engine.envs.FD_LOG_DIR", temp_dir),
                patch("fastdeploy.engine.engine.console_logger.error") as mock_error,
            ):
                result = eng._stop_profile()

        self.assertFalse(result)
        error_messages = [call.args[0] for call in mock_error.call_args_list]
        self.assertTrue(any("Traceback (most recent call last):" in msg for msg in error_messages))
        self.assertTrue(any("The total number of blocks cannot be less than zero" in msg for msg in error_messages))

    def test_stop_profile_returns_true_on_success(self):
        """测试 _stop_profile 正常完成时返回 True"""
        eng = object.__new__(LLMEngine)
        eng.do_profile = 1
        eng.get_profile_block_num_signal = type("Sig", (), {"value": np.array([100])})()
        eng.worker_proc = Mock(poll=lambda: None)
        eng.ipc_signal_suffix = "_test"
        eng.cfg = types.SimpleNamespace(
            parallel_config=types.SimpleNamespace(device_ids="0"),
            scheduler_config=types.SimpleNamespace(splitwise_role="decode"),
            cache_config=Mock(enable_prefix_caching=False, reset=Mock()),
        )
        eng.engine = types.SimpleNamespace(
            start_cache_service=lambda *_: None,
            resource_manager=Mock(reset_cache_config=Mock()),
        )
        eng.cache_manager_processes = None

        result = eng._stop_profile()

        self.assertTrue(result)


class TestLLMEngineStart(unittest.TestCase):
    """测试 LLMEngine.start 方法中的错误处理"""

    class _Sig:
        def __init__(self, val):
            self.value = np.array([val])

    def test_start_returns_false_when_profile_worker_dies(self):
        """测试当 profile worker 失败时，start 返回 False"""
        eng = object.__new__(LLMEngine)
        eng.is_started = False
        eng.api_server_pid = None
        eng.do_profile = 1
        port = int(os.getenv("FD_ENGINE_QUEUE_PORT", "6778"))
        eng.cfg = types.SimpleNamespace(
            parallel_config=types.SimpleNamespace(engine_worker_queue_port=[port], device_ids="0"),
            scheduler_config=types.SimpleNamespace(splitwise_role="mixed", max_num_seqs=8),
            cache_config=types.SimpleNamespace(
                enable_prefix_caching=True,
                block_size=64,
                num_gpu_blocks_override=None,
                total_block_num=0,
                num_cpu_blocks=0,
            ),
            model_config=types.SimpleNamespace(max_model_len=128),
        )
        eng._init_worker_signals = lambda: setattr(eng, "loaded_model_signal", self._Sig(1))
        eng.launch_components = lambda: None
        eng.worker_proc = None
        eng.engine = types.SimpleNamespace(
            start=lambda: None,
            create_data_processor=lambda: setattr(eng.engine, "data_processor", object()),
            data_processor=object(),
        )
        eng._start_worker_service = lambda: Mock(stdout=Mock(), poll=lambda: 1)
        eng.check_worker_initialize_status = lambda: False
        eng._stop_profile = lambda: False

        with patch("fastdeploy.engine.engine.time.sleep", lambda *_: None):
            result = eng.start(api_server_pid=None)

        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
