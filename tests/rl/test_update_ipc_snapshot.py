"""
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

import unittest
from unittest.mock import MagicMock, patch


def _make_manager(model_dir="/fake/model", rank=0, meta_src_id=0, load_strategy="ipc_snapshot"):
    """Build a DynamicWeightManager instance bypassing __init__, for testing _update_ipc_snapshot only."""
    from fastdeploy.rl.dynamic_weight_manager import DynamicWeightManager

    obj = object.__new__(DynamicWeightManager)

    # fd_config mock
    fd_config = MagicMock()
    fd_config.model_config.model = model_dir
    fd_config.load_config.load_strategy = load_strategy
    obj.fd_config = fd_config

    obj.meta_src_id = meta_src_id
    obj.rank = rank
    # Two mock parameters to verify _update_model_from_state is called correctly
    obj.state_dict = {
        "layer.weight": MagicMock(name="layer.weight"),
        "layer.bias": MagicMock(name="layer.bias"),
    }
    return obj


class TestUpdateIpcSnapshot(unittest.TestCase):
    """Unit tests for DynamicWeightManager._update_ipc_snapshot.

    Covers all 4 loading priority branches inside the function:
      Priority 1 - chunked part files
      Priority 2 - single full pdparams file
      Priority 3 - legacy format
      Priority 4 - shared directory fallback
    Plus the error path when no snapshot is found anywhere.
    """

    def setUp(self):
        # Pre-import the module so that fastdeploy.rl is set in fastdeploy.__dict__
        # before @patch decorators resolve their targets via _importer.
        # Without this, fastdeploy.__getattr__ prints a warning and returns None for "rl",
        # causing _importer to later fail with AttributeError when the test runs first
        # (e.g., alphabetical order in unittest).
        import fastdeploy.rl.dynamic_weight_manager  # noqa: F401

    # ------------------------------------------------------------------
    # Priority 1: chunked part files
    # ------------------------------------------------------------------
    @patch("fastdeploy.rl.dynamic_weight_manager.gc.collect")
    @patch("fastdeploy.rl.dynamic_weight_manager.paddle.load")
    @patch("fastdeploy.rl.dynamic_weight_manager.os.path.exists")
    @patch("fastdeploy.rl.dynamic_weight_manager.glob.glob")
    @patch("fastdeploy.rl.dynamic_weight_manager.paddle.distributed.get_rank", return_value=0)
    def test_priority1_chunked_part_files(self, _mock_rank, mock_glob, mock_exists, mock_load, mock_gc):
        """When part files are found, load each part in order and do not fall through to other priorities."""
        model_dir = "/fake/model"
        part_files = [
            f"{model_dir}/model_state.tp0.0.part1.pdparams",
            f"{model_dir}/model_state.tp0.0.part2.pdparams",
        ]
        mock_glob.return_value = part_files

        fake_state_dict_1 = {"layer.weight": MagicMock()}
        fake_state_dict_2 = {"layer.bias": MagicMock()}
        mock_load.side_effect = [fake_state_dict_1, fake_state_dict_2]

        mgr = _make_manager(model_dir=model_dir)
        mgr._update_model_from_state = MagicMock()

        mgr._update_ipc_snapshot()

        # glob should be called to search for part files
        mock_glob.assert_called_once()
        # os.path.exists must NOT be called: priority 1 should return early
        mock_exists.assert_not_called()
        # paddle.load should be called once per part file
        self.assertEqual(mock_load.call_count, 2)
        mock_load.assert_any_call(part_files[0], safetensors=True)
        mock_load.assert_any_call(part_files[1], safetensors=True)
        # _update_model_from_state should be called once per part with correct args
        self.assertEqual(mgr._update_model_from_state.call_count, 2)
        mgr._update_model_from_state.assert_any_call(fake_state_dict_1, "snapshot-part1")
        mgr._update_model_from_state.assert_any_call(fake_state_dict_2, "snapshot-part2")
        # gc.collect should be called after each part to free memory
        self.assertEqual(mock_gc.call_count, 2)

    # ------------------------------------------------------------------
    # Priority 2: single full pdparams file
    # ------------------------------------------------------------------
    @patch("fastdeploy.rl.dynamic_weight_manager.paddle.load")
    @patch("fastdeploy.rl.dynamic_weight_manager.os.path.exists")
    @patch("fastdeploy.rl.dynamic_weight_manager.glob.glob", return_value=[])
    @patch("fastdeploy.rl.dynamic_weight_manager.paddle.distributed.get_rank", return_value=0)
    def test_priority2_single_full_file(self, _mock_rank, _mock_glob, mock_exists, mock_load):
        """When no part files exist, load from model_state.tp{rank}.{id}.pdparams and return."""
        model_dir = "/fake/model"
        full_path = f"{model_dir}/model_state.tp0.0.pdparams"

        # Only full_path exists
        mock_exists.side_effect = lambda p: p == full_path

        fake_state_dict = {"layer.weight": MagicMock()}
        mock_load.return_value = fake_state_dict

        mgr = _make_manager(model_dir=model_dir)
        mgr._update_model_from_state = MagicMock()

        mgr._update_ipc_snapshot()

        mock_load.assert_called_once_with(full_path, safetensors=True)
        mgr._update_model_from_state.assert_called_once_with(fake_state_dict, "snapshot")

    # ------------------------------------------------------------------
    # Priority 3: legacy format
    # ------------------------------------------------------------------
    @patch("fastdeploy.rl.dynamic_weight_manager.paddle.load")
    @patch("fastdeploy.rl.dynamic_weight_manager.os.path.exists")
    @patch("fastdeploy.rl.dynamic_weight_manager.glob.glob", return_value=[])
    @patch("fastdeploy.rl.dynamic_weight_manager.paddle.distributed.get_rank", return_value=0)
    def test_priority3_legacy_format(self, _mock_rank, _mock_glob, mock_exists, mock_load):
        """When the full path does not exist, fall back to legacy model_state.tp0{id}.pdparams."""
        model_dir = "/fake/model"
        legacy_path = f"{model_dir}/model_state.tp00.pdparams"

        # full_path absent, legacy_path present
        mock_exists.side_effect = lambda p: p == legacy_path

        fake_state_dict = {"layer.weight": MagicMock()}
        mock_load.return_value = fake_state_dict

        mgr = _make_manager(model_dir=model_dir)
        mgr._update_model_from_state = MagicMock()

        mgr._update_ipc_snapshot()

        mock_load.assert_called_once_with(legacy_path, safetensors=True)
        mgr._update_model_from_state.assert_called_once_with(fake_state_dict, "snapshot")

    # ------------------------------------------------------------------
    # Priority 4: shared directory fallback
    # ------------------------------------------------------------------
    @patch("fastdeploy.rl.dynamic_weight_manager.paddle.load")
    @patch("fastdeploy.rl.dynamic_weight_manager.os.path.exists")
    @patch("fastdeploy.rl.dynamic_weight_manager.glob.glob", return_value=[])
    @patch("fastdeploy.rl.dynamic_weight_manager.paddle.distributed.get_rank", return_value=0)
    def test_priority4_shared_fallback(self, _mock_rank, _mock_glob, mock_exists, mock_load):
        """When all three local paths are absent, load from /shared_ipc_meta/."""
        model_dir = "/fake/model"
        fallback_path = "/shared_ipc_meta/model_state.tp0.0.pdparams"

        # Only fallback_path exists
        mock_exists.side_effect = lambda p: p == fallback_path

        fake_state_dict = {"layer.weight": MagicMock()}
        mock_load.return_value = fake_state_dict

        mgr = _make_manager(model_dir=model_dir)
        mgr._update_model_from_state = MagicMock()

        mgr._update_ipc_snapshot()

        mock_load.assert_called_once_with(fallback_path)
        mgr._update_model_from_state.assert_called_once_with(fake_state_dict, "snapshot")

    # ------------------------------------------------------------------
    # Error path: no snapshot found anywhere
    # ------------------------------------------------------------------
    @patch("fastdeploy.rl.dynamic_weight_manager.os.path.exists", return_value=False)
    @patch("fastdeploy.rl.dynamic_weight_manager.glob.glob", return_value=[])
    @patch("fastdeploy.rl.dynamic_weight_manager.paddle.distributed.get_rank", return_value=0)
    def test_no_snapshot_raises_file_not_found(self, _mock_rank, _mock_glob, _mock_exists):
        """Should raise FileNotFoundError when none of the candidate paths exist."""
        mgr = _make_manager()
        mgr._update_model_from_state = MagicMock()

        with self.assertRaises(FileNotFoundError) as ctx:
            mgr._update_ipc_snapshot()
        self.assertIn("No snapshot found", str(ctx.exception))

    # ------------------------------------------------------------------
    # Priority 1 sort correctness: part files loaded in numeric order
    # ------------------------------------------------------------------
    @patch("fastdeploy.rl.dynamic_weight_manager.gc.collect")
    @patch("fastdeploy.rl.dynamic_weight_manager.paddle.load")
    @patch("fastdeploy.rl.dynamic_weight_manager.os.path.exists")
    @patch("fastdeploy.rl.dynamic_weight_manager.glob.glob")
    @patch("fastdeploy.rl.dynamic_weight_manager.paddle.distributed.get_rank", return_value=0)
    def test_priority1_part_files_sorted_by_number(self, _mock_rank, mock_glob, mock_exists, mock_load, _mock_gc):
        """When glob returns part files out of order, they should be loaded in ascending numeric order."""
        model_dir = "/fake/model"
        # Intentionally return files in wrong order
        part_files_unordered = [
            f"{model_dir}/model_state.tp0.0.part3.pdparams",
            f"{model_dir}/model_state.tp0.0.part1.pdparams",
            f"{model_dir}/model_state.tp0.0.part2.pdparams",
        ]
        mock_glob.return_value = part_files_unordered
        mock_load.return_value = {}

        mgr = _make_manager(model_dir=model_dir)
        mgr._update_model_from_state = MagicMock()

        # Capture the actual load order
        loaded_paths = []
        mock_load.side_effect = lambda p, **_kw: loaded_paths.append(p) or {}

        mgr._update_ipc_snapshot()

        # os.path.exists must NOT be called: priority 1 should return early
        mock_exists.assert_not_called()

        expected_order = [
            f"{model_dir}/model_state.tp0.0.part1.pdparams",
            f"{model_dir}/model_state.tp0.0.part2.pdparams",
            f"{model_dir}/model_state.tp0.0.part3.pdparams",
        ]
        self.assertEqual(loaded_paths, expected_order)


if __name__ == "__main__":
    unittest.main()
