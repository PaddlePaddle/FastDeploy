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
Tests that force reimport of modules to exercise module-level try/except
fallback paths under coverage.

Covers:
- flash_attn_backend.py lines 29-30, 35-36
- mla_attention_backend.py lines 30-31
- moba_attention_backend.py lines 27-28
- cache_manager/ops.py lines 124-125
- custom_all_reduce/custom_all_reduce.py lines 43-44
"""

import importlib
import sys
import unittest
from unittest.mock import MagicMock, patch


def _try_import(module_path):
    """Try to import a module, return (module, True) or (None, False)."""
    try:
        return importlib.import_module(module_path), True
    except Exception:
        return None, False


class TestFlashAttnBackendFallbacks(unittest.TestCase):
    """Exercise except blocks in flash_attn_backend.py lines 29-30, 35-36."""

    MODULE = "fastdeploy.model_executor.layers.attention.flash_attn_backend"
    PARENT = "paddle.nn.functional.flash_attention"

    @classmethod
    def setUpClass(cls):
        cls.mod, cls.can_test = _try_import(cls.MODULE)
        cls.parent, _ = _try_import(cls.PARENT)

    def test_flash_attention_v3_varlen_fallback(self):
        """Removing flash_attention_v3_varlen forces the except branch (L29-30)."""
        if not self.can_test:
            self.skipTest(f"Cannot import {self.MODULE}")

        attr = "flash_attention_v3_varlen"
        had = hasattr(self.parent, attr)
        saved = getattr(self.parent, attr, None)

        if had:
            delattr(self.parent, attr)

        try:
            importlib.reload(self.mod)
            self.assertIsNone(self.mod.flash_attention_v3_varlen)
        finally:
            if had:
                setattr(self.parent, attr, saved)
            importlib.reload(self.mod)

    def test_flashmask_attention_fallback(self):
        """Removing flashmask_attention forces the except branch (L35-36)."""
        if not self.can_test:
            self.skipTest(f"Cannot import {self.MODULE}")

        attr = "flashmask_attention"
        had = hasattr(self.parent, attr)
        saved = getattr(self.parent, attr, None)

        if had:
            delattr(self.parent, attr)

        try:
            importlib.reload(self.mod)
            self.assertIsNone(self.mod.flashmask_attention)
        finally:
            if had:
                setattr(self.parent, attr, saved)
            importlib.reload(self.mod)


class TestMLAAttentionBackendFallback(unittest.TestCase):
    """Exercise except block in mla_attention_backend.py lines 30-31."""

    MODULE = "fastdeploy.model_executor.layers.attention.mla_attention_backend"
    PARENT = "paddle.nn.functional.flash_attention"

    @classmethod
    def setUpClass(cls):
        cls.mod, cls.can_test = _try_import(cls.MODULE)
        cls.parent, _ = _try_import(cls.PARENT)

    def test_flash_attention_v3_varlen_fallback(self):
        """Removing flash_attention_v3_varlen forces the except branch (L30-31)."""
        if not self.can_test:
            self.skipTest(f"Cannot import {self.MODULE}")

        attr = "flash_attention_v3_varlen"
        had = hasattr(self.parent, attr)
        saved = getattr(self.parent, attr, None)

        if had:
            delattr(self.parent, attr)

        try:
            importlib.reload(self.mod)
            self.assertIsNone(self.mod.flash_attention_v3_varlen)
        finally:
            if had:
                setattr(self.parent, attr, saved)
            importlib.reload(self.mod)


class TestMobaAttentionBackendFallback(unittest.TestCase):
    """Exercise except block in moba_attention_backend.py lines 27-28."""

    MODULE = "fastdeploy.model_executor.layers.attention.moba_attention_backend"

    @classmethod
    def setUpClass(cls):
        cls.mod, cls.can_test = _try_import(cls.MODULE)

    def test_moba_attention_fallback(self):
        """When moba_attention ops are unavailable, fallback sets them to None (L27-28)."""
        if not self.can_test:
            self.skipTest(f"Cannot import {self.MODULE}")

        # Mock the gpu ops module to not have the required functions
        fake_gpu = MagicMock(spec=[])
        with patch.dict(sys.modules, {"fastdeploy.model_executor.ops.gpu": fake_gpu}):
            importlib.reload(self.mod)
            self.assertIsNone(self.mod.moba_attention)
            self.assertIsNone(self.mod.get_cur_cu_seq_len_k)

        # Restore
        importlib.reload(self.mod)


class TestCacheManagerOpsFallback(unittest.TestCase):
    """Exercise except block in cache_manager/ops.py lines 124-125."""

    MODULE = "fastdeploy.cache_manager.ops"

    @classmethod
    def setUpClass(cls):
        cls.mod, cls.can_test = _try_import(cls.MODULE)

    def test_import_fallback(self):
        """When cache manager ops fail to import, fallback sets symbols to None (L124-125)."""
        if not self.can_test:
            self.skipTest(f"Cannot import {self.MODULE}")

        # Force the platform check to fail by mocking current_platform
        mock_platform = MagicMock()
        mock_platform.is_cuda.return_value = False
        mock_platform.is_maca.return_value = False
        mock_platform.is_xpu.return_value = False

        with patch.object(self.mod, "current_platform", mock_platform):
            importlib.reload(self.mod)
            self.assertIsNone(self.mod.cuda_host_alloc)
            self.assertIsNone(self.mod.cuda_host_free)

        # Restore
        importlib.reload(self.mod)


class TestCustomAllReduceFallback(unittest.TestCase):
    """Exercise except block in custom_all_reduce.py lines 43-44."""

    MODULE = "fastdeploy.distributed.custom_all_reduce.custom_all_reduce"

    @classmethod
    def setUpClass(cls):
        cls.mod, cls.can_test = _try_import(cls.MODULE)

    def test_meta_size_fallback(self):
        """When meta_size() raises, custom_ar is set to False (L43-44)."""
        if not self.can_test:
            self.skipTest(f"Cannot import {self.MODULE}")

        with patch.object(self.mod, "meta_size", side_effect=RuntimeError("mocked")):
            importlib.reload(self.mod)
            self.assertFalse(self.mod.custom_ar)

        # Restore
        importlib.reload(self.mod)


if __name__ == "__main__":
    unittest.main()
