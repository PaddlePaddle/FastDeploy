"""
# Copyright (c) 2026  PaddlePaddle Authors. All Rights Reserved.
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
from unittest.mock import MagicMock, patch

from fastdeploy.model_executor.model_loader.default_loader import DefaultModelLoader


class TestDefaultModelLoaderInit(unittest.TestCase):
    """Test DefaultModelLoader.__init__."""

    def test_init(self):
        """__init__ stores load_config and logs info."""
        load_config = MagicMock()
        loader = DefaultModelLoader(load_config)
        self.assertIs(loader.load_config, load_config)


class TestDefaultModelLoaderDownloadModel(unittest.TestCase):
    """Test DefaultModelLoader.download_model."""

    def test_download_model_is_noop(self):
        """download_model does nothing (pass)."""
        loader = DefaultModelLoader(MagicMock())
        # Should not raise
        result = loader.download_model(MagicMock())
        self.assertIsNone(result)


class TestDefaultModelLoaderCleanMemoryFragments(unittest.TestCase):
    """Test DefaultModelLoader.clean_memory_fragments."""

    @patch("fastdeploy.model_executor.model_loader.default_loader.current_platform")
    def test_clean_memory_on_cuda(self, mock_platform):
        """clean_memory_fragments clears tensors and empties cache on CUDA."""
        mock_platform.is_cuda.return_value = True
        mock_platform.is_maca.return_value = False
        mock_platform.is_iluvatar.return_value = False

        loader = DefaultModelLoader(MagicMock())

        import paddle

        tensor_mock = MagicMock(spec=paddle.Tensor)
        state_dict = {"layer.weight": tensor_mock}

        with patch("paddle.device.empty_cache") as mock_empty, patch("paddle.device.synchronize") as mock_sync:
            loader.clean_memory_fragments(state_dict)

        tensor_mock.value.return_value.get_tensor.return_value._clear.assert_called_once()
        mock_empty.assert_called_once()
        mock_sync.assert_called_once()

    @patch("fastdeploy.model_executor.model_loader.default_loader.current_platform")
    def test_clean_memory_on_maca(self, mock_platform):
        """clean_memory_fragments works on MACA platform."""
        mock_platform.is_cuda.return_value = False
        mock_platform.is_maca.return_value = True
        mock_platform.is_iluvatar.return_value = False

        loader = DefaultModelLoader(MagicMock())

        with patch("paddle.device.empty_cache") as mock_empty, patch("paddle.device.synchronize") as mock_sync:
            loader.clean_memory_fragments({"key": MagicMock()})

        mock_empty.assert_called_once()
        mock_sync.assert_called_once()

    @patch("fastdeploy.model_executor.model_loader.default_loader.current_platform")
    def test_clean_memory_on_iluvatar(self, mock_platform):
        """clean_memory_fragments works on Iluvatar platform."""
        mock_platform.is_cuda.return_value = False
        mock_platform.is_maca.return_value = False
        mock_platform.is_iluvatar.return_value = True

        loader = DefaultModelLoader(MagicMock())

        with patch("paddle.device.empty_cache") as mock_empty, patch("paddle.device.synchronize") as mock_sync:
            loader.clean_memory_fragments({"key": MagicMock()})

        mock_empty.assert_called_once()
        mock_sync.assert_called_once()

    @patch("fastdeploy.model_executor.model_loader.default_loader.current_platform")
    def test_clean_memory_skips_on_unsupported_platform(self, mock_platform):
        """clean_memory_fragments does nothing on unsupported platform."""
        mock_platform.is_cuda.return_value = False
        mock_platform.is_maca.return_value = False
        mock_platform.is_iluvatar.return_value = False

        loader = DefaultModelLoader(MagicMock())

        with patch("paddle.device.empty_cache") as mock_empty, patch("paddle.device.synchronize") as mock_sync:
            loader.clean_memory_fragments({"key": MagicMock()})

        mock_empty.assert_not_called()
        mock_sync.assert_not_called()

    @patch("fastdeploy.model_executor.model_loader.default_loader.current_platform")
    def test_clean_memory_empty_state_dict(self, mock_platform):
        """clean_memory_fragments still empties cache even with empty state_dict."""
        mock_platform.is_cuda.return_value = True
        mock_platform.is_maca.return_value = False
        mock_platform.is_iluvatar.return_value = False

        loader = DefaultModelLoader(MagicMock())

        with patch("paddle.device.empty_cache") as mock_empty, patch("paddle.device.synchronize") as mock_sync:
            loader.clean_memory_fragments({})

        # empty_cache and synchronize still called (they're outside the `if state_dict:` block)
        mock_empty.assert_called_once()
        mock_sync.assert_called_once()


class TestDefaultModelLoaderLoadWeights(unittest.TestCase):
    """Test DefaultModelLoader.load_weights."""

    @patch("fastdeploy.model_executor.model_loader.default_loader.load_composite_checkpoint")
    @patch("fastdeploy.model_executor.model_loader.default_loader.ModelRegistry")
    def test_load_weights(self, mock_registry, mock_load_checkpoint):
        """load_weights loads checkpoint, sets state dict, and cleans memory."""
        loader = DefaultModelLoader(MagicMock())
        loader.clean_memory_fragments = MagicMock()

        mock_registry.get_pretrain_cls.return_value = "pretrain_cls"
        mock_load_checkpoint.return_value = {"layer.weight": "tensor"}

        model = MagicMock()
        fd_config = MagicMock()
        fd_config.model_config.model = "/path/to/model"

        loader.load_weights(model, fd_config, "MyArchitecture")

        mock_registry.get_pretrain_cls.assert_called_once_with("MyArchitecture")
        mock_load_checkpoint.assert_called_once_with("/path/to/model", "pretrain_cls", fd_config, return_numpy=True)
        model.set_state_dict.assert_called_once_with({"layer.weight": "tensor"})
        loader.clean_memory_fragments.assert_called_once_with({"layer.weight": "tensor"})


class TestDefaultModelLoaderLoadModel(unittest.TestCase):
    """Test DefaultModelLoader.load_model."""

    @patch("fastdeploy.model_executor.model_loader.default_loader.ModelRegistry")
    def test_load_model_normal(self, mock_registry):
        """load_model creates model, loads weights, returns model."""
        loader = DefaultModelLoader(MagicMock())
        loader.load_weights = MagicMock()

        mock_model = MagicMock()
        mock_registry.get_class.return_value = MagicMock(return_value=mock_model)

        fd_config = MagicMock()
        fd_config.model_config.architectures = ["TestModel"]
        fd_config.load_config.dynamic_load_weight = False

        result = loader.load_model(fd_config)

        mock_registry.get_class.assert_called_once_with("TestModel")
        mock_model.eval.assert_called_once()
        loader.load_weights.assert_called_once_with(mock_model, fd_config, "TestModel")
        self.assertIs(result, mock_model)

    @patch("fastdeploy.model_executor.model_loader.default_loader.ModelRegistry")
    @patch("paddle.LazyGuard")
    def test_load_model_dynamic_load_non_mtp(self, mock_lazy_guard, mock_registry):
        """load_model with dynamic_load_weight renames arch and skips load_weights."""
        loader = DefaultModelLoader(MagicMock())
        loader.load_weights = MagicMock()

        mock_model = MagicMock()
        mock_registry.get_class.return_value = MagicMock(return_value=mock_model)
        mock_lazy_guard.return_value.__enter__ = MagicMock()
        mock_lazy_guard.return_value.__exit__ = MagicMock(return_value=False)

        fd_config = MagicMock()
        fd_config.model_config.architectures = ["Ernie5ForCausalLM"]
        fd_config.load_config.dynamic_load_weight = True
        fd_config.speculative_config.model_type = "eagle"  # not mtp

        with patch("fastdeploy.rl", create=True):
            result = loader.load_model(fd_config)

        # Ernie5ForCausalLM -> Ernie5MoeForCausalLM + RL
        mock_registry.get_class.assert_called_once_with("Ernie5MoeForCausalLMRL")
        mock_model.eval.assert_called_once()
        loader.load_weights.assert_not_called()
        self.assertIs(result, mock_model)

    @patch("fastdeploy.model_executor.model_loader.default_loader.ModelRegistry")
    @patch("paddle.LazyGuard")
    def test_load_model_dynamic_load_mtp(self, mock_lazy_guard, mock_registry):
        """load_model with dynamic_load_weight and mtp renames to MTP arch."""
        loader = DefaultModelLoader(MagicMock())
        loader.load_weights = MagicMock()

        mock_model = MagicMock()
        mock_registry.get_class.return_value = MagicMock(return_value=mock_model)
        mock_lazy_guard.return_value.__enter__ = MagicMock()
        mock_lazy_guard.return_value.__exit__ = MagicMock(return_value=False)

        fd_config = MagicMock()
        fd_config.model_config.architectures = ["Ernie5ForCausalLM"]
        fd_config.load_config.dynamic_load_weight = True
        fd_config.speculative_config.model_type = "mtp"

        with patch("fastdeploy.rl", create=True):
            result = loader.load_model(fd_config)

        # Ernie5ForCausalLM -> Ernie5MTPForCausalLM + RL
        mock_registry.get_class.assert_called_once_with("Ernie5MTPForCausalLMRL")
        mock_model.eval.assert_called_once()
        loader.load_weights.assert_not_called()
        self.assertIs(result, mock_model)


if __name__ == "__main__":
    unittest.main()
