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

import os
import unittest

from fastdeploy import envs


class TestEnvsGetattr(unittest.TestCase):
    """Test the module-level __getattr__ lazy evaluation."""

    def test_default_values(self):
        with _clean_env("FD_DEBUG"):
            self.assertEqual(envs.FD_DEBUG, 0)

        with _clean_env("FD_LOG_DIR"):
            self.assertEqual(envs.FD_LOG_DIR, "log")

        with _clean_env("FD_MAX_STOP_SEQS_NUM"):
            self.assertEqual(envs.FD_MAX_STOP_SEQS_NUM, 5)

    def test_env_override(self):
        with _set_env("FD_DEBUG", "1"):
            self.assertEqual(envs.FD_DEBUG, 1)

        with _set_env("FD_LOG_DIR", "/tmp/mylog"):
            self.assertEqual(envs.FD_LOG_DIR, "/tmp/mylog")

    def test_bool_env(self):
        with _set_env("FD_USE_HF_TOKENIZER", "1"):
            self.assertTrue(envs.FD_USE_HF_TOKENIZER)

        with _set_env("FD_USE_HF_TOKENIZER", "0"):
            self.assertFalse(envs.FD_USE_HF_TOKENIZER)

    def test_unknown_attr_raises(self):
        with self.assertRaises(AttributeError):
            _ = envs.THIS_DOES_NOT_EXIST

    def test_list_env_fd_plugins(self):
        with _clean_env("FD_PLUGINS"):
            self.assertIsNone(envs.FD_PLUGINS)

        with _set_env("FD_PLUGINS", "a,b,c"):
            self.assertEqual(envs.FD_PLUGINS, ["a", "b", "c"])

    def test_list_env_fd_api_key(self):
        with _clean_env("FD_API_KEY"):
            self.assertEqual(envs.FD_API_KEY, [])

        with _set_env("FD_API_KEY", "key1,key2"):
            self.assertEqual(envs.FD_API_KEY, ["key1", "key2"])


class TestEnvsSetattr(unittest.TestCase):
    """Test module-level __setattr__."""

    def test_setattr_known_var(self):
        original = envs.FD_DEBUG
        try:
            envs.FD_DEBUG = 42
            self.assertEqual(envs.FD_DEBUG, 42)
        finally:
            envs.FD_DEBUG = original

    def test_setattr_unknown_var_raises(self):
        with self.assertRaises(AttributeError):
            envs.UNKNOWN_VAR_XYZ = 1


class TestValidateSplitKvSize(unittest.TestCase):
    """Test _validate_split_kv_size via FD_DETERMINISTIC_SPLIT_KV_SIZE."""

    def test_valid_power_of_two(self):
        with _set_env("FD_DETERMINISTIC_SPLIT_KV_SIZE", "16"):
            self.assertEqual(envs.FD_DETERMINISTIC_SPLIT_KV_SIZE, 16)

        with _set_env("FD_DETERMINISTIC_SPLIT_KV_SIZE", "1"):
            self.assertEqual(envs.FD_DETERMINISTIC_SPLIT_KV_SIZE, 1)

    def test_invalid_not_power_of_two(self):
        with _set_env("FD_DETERMINISTIC_SPLIT_KV_SIZE", "3"):
            with self.assertRaises(ValueError):
                _ = envs.FD_DETERMINISTIC_SPLIT_KV_SIZE

    def test_invalid_zero(self):
        with _set_env("FD_DETERMINISTIC_SPLIT_KV_SIZE", "0"):
            with self.assertRaises(ValueError):
                _ = envs.FD_DETERMINISTIC_SPLIT_KV_SIZE

    def test_invalid_negative(self):
        with _set_env("FD_DETERMINISTIC_SPLIT_KV_SIZE", "-4"):
            with self.assertRaises(ValueError):
                _ = envs.FD_DETERMINISTIC_SPLIT_KV_SIZE


class TestEnvsDir(unittest.TestCase):
    def test_dir_returns_keys(self):
        result = dir(envs)
        self.assertIn("FD_DEBUG", result)
        self.assertIn("FD_LOG_DIR", result)


class TestGetUniqueName(unittest.TestCase):
    def test_with_shm_uuid(self):
        with _set_env("SHM_UUID", "abc123"):
            result = envs.get_unique_name(None, "prefix")
            self.assertEqual(result, "prefix_abc123")

    def test_without_shm_uuid(self):
        with _clean_env("SHM_UUID"):
            result = envs.get_unique_name(None, "prefix")
            self.assertEqual(result, "prefix_")


# ---- helpers ----


class _clean_env:
    """Context manager to temporarily remove an env var."""

    def __init__(self, key):
        self.key = key

    def __enter__(self):
        self.old = os.environ.pop(self.key, None)
        return self

    def __exit__(self, *exc):
        if self.old is not None:
            os.environ[self.key] = self.old
        else:
            os.environ.pop(self.key, None)


class _set_env:
    """Context manager to temporarily set an env var."""

    def __init__(self, key, value):
        self.key = key
        self.value = value

    def __enter__(self):
        self.old = os.environ.get(self.key)
        os.environ[self.key] = self.value
        return self

    def __exit__(self, *exc):
        if self.old is not None:
            os.environ[self.key] = self.old
        else:
            os.environ.pop(self.key, None)


if __name__ == "__main__":
    unittest.main()
