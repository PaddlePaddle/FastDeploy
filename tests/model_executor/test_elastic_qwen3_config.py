# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
"""Unit tests for ``qwen3_elastic.config_elastic.populate_elastic_fields``.

The function lifts elastic fields from ``pretrained_config`` (the raw ckpt
config.json) onto ``model_config``, with the documented defaults from
ELASTIC_CONFIG_FIELDS. This test guards:

1. Default values when ckpt has no elastic fields.
2. Override is honored when ckpt provides a value.
3. Idempotence (calling twice does not overwrite already-populated fields).
4. ``block_size`` MUST come from ``pretrained_config``, never from
   ``model_config`` directly (FD's ``cache_config.block_size = 64`` would
   otherwise corrupt the BSA block grid).
"""

import importlib.util
import os
import unittest

# See test_elastic_qwen3_utils.py for why we file-load this module instead of
# importing ``fastdeploy.model_executor.models.qwen3_elastic.config_elastic``
# the regular way (parent ``models/__init__`` -> attention.ops chain pulls
# in compiled custom-op symbols that may be missing in some builds).
_HERE = os.path.dirname(os.path.abspath(__file__))
_CFG_PATH = os.path.normpath(
    os.path.join(_HERE, "..", "..", "fastdeploy", "model_executor", "models", "qwen3_elastic", "config_elastic.py")
)
_spec = importlib.util.spec_from_file_location("qwen3_elastic_config_under_test", _CFG_PATH)
_cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cfg)
ELASTIC_CONFIG_FIELDS = _cfg.ELASTIC_CONFIG_FIELDS
populate_elastic_fields = _cfg.populate_elastic_fields


class _NS:
    """Lightweight namespace mimicking the model_config / pretrained_config object."""

    pass


class TestPopulateElasticFields(unittest.TestCase):
    def test_defaults_when_ckpt_empty(self):
        mc = _NS()
        mc.pretrained_config = _NS()
        populate_elastic_fields(mc)
        for attr, (_, default) in ELASTIC_CONFIG_FIELDS.items():
            self.assertEqual(getattr(mc, attr), default, msg=f"attr={attr}")

    def test_ckpt_override(self):
        mc = _NS()
        pc = _NS()
        pc.local_window_size = 4096
        pc.sink_size = 256
        pc.toggle_type = "streaming"
        pc.retrieval_mode = "xattn"
        pc.xattn_threshold = 0.5
        pc.block_size = 64
        mc.pretrained_config = pc

        populate_elastic_fields(mc)

        self.assertEqual(mc.local_window_size, 4096)
        self.assertEqual(mc.sink_size, 256)
        self.assertEqual(mc.toggle_type, "streaming")
        self.assertEqual(mc.retrieval_mode, "xattn")
        self.assertAlmostEqual(mc.xattn_threshold, 0.5)
        self.assertEqual(mc.block_size, 64)
        # Other fields fall back to defaults.
        self.assertTrue(mc.enable_ada_sparsity)
        self.assertEqual(mc.pooling_mode, "ctx_q")

    def test_idempotent(self):
        mc = _NS()
        mc.pretrained_config = _NS()
        populate_elastic_fields(mc)
        # Pretend user has set a custom value AFTER the first population.
        mc.toggle_type = "custom"
        # Second call must NOT overwrite it.
        populate_elastic_fields(mc)
        self.assertEqual(mc.toggle_type, "custom")

    def test_block_size_from_pretrained_config_not_model_config(self):
        """FD's ``cache_config.block_size`` (64) often leaks onto model_config
        via attribute proxying. ``populate_elastic_fields`` must read from
        ``pretrained_config`` so the BSA block grid uses the elastic 128.
        """
        mc = _NS()
        pc = _NS()
        # ckpt elastic block_size (granularity used by xattn / BSA)
        pc.block_size = 128
        mc.pretrained_config = pc
        # model_config also has a (different) block_size leaking from
        # cache_config -- we ensure populate_elastic_fields reads from `pc`.
        populate_elastic_fields(mc)
        self.assertEqual(mc.block_size, 128)

    def test_no_pretrained_config_falls_back_to_model_config(self):
        """When pretrained_config is missing, populate from model_config itself."""
        mc = _NS()
        mc.pretrained_config = None
        mc.toggle_type = "xattn"
        populate_elastic_fields(mc)
        self.assertEqual(mc.toggle_type, "xattn")
        # defaults still fill remaining attrs
        self.assertEqual(mc.sink_size, 128)


if __name__ == "__main__":
    import unittest as _u

    _u.main()
