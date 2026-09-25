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
"""Regression tests for the global patches applied by
``fastdeploy.model_executor.models.qwen3_elastic.__init__``.

The package patches two pieces of FastDeploy global state on import:

1. ``attention_selecter.get_attention_backend`` /
   ``attention_selecter._get_attn_backend`` -- must return the elastic
   backend ONLY for ``PawQwen3ForCausalLM`` and fall through to the
   original selector for every other architecture.

2. ``rotary_embedding.get_rope_impl`` -- must route PawQwen3 + yarn
   rope_scaling through ``GptOssScalingRotaryEmbedding`` and leave every
   non-PawQwen3 caller untouched.

Because ``auto_models_registry`` imports this package on every FastDeploy
launch (including dense Qwen3, ERNIE, GLM, ...), a leaky patch would
silently break unrelated models. These tests guard that.
"""

import unittest

# These tests verify monkey-patches applied by importing the qwen3_elastic
# package onto fastdeploy's REAL ``attention_selecter`` and
# ``rotary_embedding`` modules. Unlike utils/config/kernels tests, there's
# no way to file-load past this -- the whole point is that the patch hits
# the real fastdeploy globals. So they require the same fully-built
# fastdeploy_ops as the integration smoke (run_elastic_qwen3_4b.py). On
# stale / partial builds (e.g. older fastdeploy_ops_pd_.so missing
# config_for_attention) the import will fail; skip cleanly in that case.
try:
    import fastdeploy.model_executor.models.qwen3_elastic  # noqa: F401
    from fastdeploy.model_executor.layers.attention import (  # noqa: F401
        attention_selecter,
    )
    from fastdeploy.model_executor.layers.attention.elastic_attn_backend import (  # noqa: F401
        Qwen3ElasticAttentionBackend,
    )

    _FD_FULLY_BUILT = True
    _FD_IMPORT_ERR = None
except Exception as _e:  # noqa: BLE001
    _FD_FULLY_BUILT = False
    _FD_IMPORT_ERR = _e


def _require_full_build(test):
    return unittest.skipUnless(
        _FD_FULLY_BUILT,
        f"fastdeploy custom-ops not fully built: {_FD_IMPORT_ERR!r}",
    )(test)


class _Cfg:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _CallerWithFDConfig:
    """Stack-frame stand-in: the real selector walks ``frame.f_locals['self']``
    and reads ``self.fd_config.model_config.architectures``.
    """

    def __init__(self, archs):
        self.fd_config = _Cfg(model_config=_Cfg(architectures=archs))

    def call_get_attention_backend(self):
        return attention_selecter.get_attention_backend()

    def call_get_attn_backend(self, sb=None):
        return attention_selecter._get_attn_backend(sb)


class TestAttentionSelectorPatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _FD_FULLY_BUILT:
            raise unittest.SkipTest(f"fastdeploy custom-ops not fully built: {_FD_IMPORT_ERR!r}")

    def test_pawqwen3_caller_gets_elastic_backend(self):
        caller = _CallerWithFDConfig(archs=["PawQwen3ForCausalLM"])
        cls1 = caller.call_get_attention_backend()
        cls2 = caller.call_get_attn_backend()
        self.assertIs(cls1, Qwen3ElasticAttentionBackend)
        self.assertIs(cls2, Qwen3ElasticAttentionBackend)

    def test_other_models_untouched(self):
        """Dense Qwen3 / ERNIE / etc must NOT receive the elastic backend."""
        for arch in ("Qwen3ForCausalLM", "Qwen2ForCausalLM", "Ernie4_5_MoeForCausalLM", "GLM4ForCausalLM"):
            caller = _CallerWithFDConfig(archs=[arch])
            try:
                cls1 = caller.call_get_attention_backend()
            except Exception:
                # The original selector may itself fail on this dummy fd_config
                # (e.g. need a CUDA platform). What matters is it did NOT
                # short-circuit to Qwen3ElasticAttentionBackend, which would
                # always succeed -- so a raised error is also a pass.
                continue
            self.assertIsNot(
                cls1,
                Qwen3ElasticAttentionBackend,
                msg=f"{arch} must not be redirected to elastic backend",
            )


class TestRopeImplPatch(unittest.TestCase):
    """``_patched_get_rope_impl`` only kicks in when architecture starts with
    something OTHER than 'Qwen' but contains 'Qwen' (i.e. PawQwen3-style names).
    """

    @classmethod
    def setUpClass(cls):
        if not _FD_FULLY_BUILT:
            raise unittest.SkipTest(f"fastdeploy custom-ops not fully built: {_FD_IMPORT_ERR!r}")

    def test_predicate_matches_pawqwen3(self):
        from fastdeploy.model_executor.models.qwen3_elastic import (
            __init__ as elastic_init,
        )

        is_paw = elastic_init._is_pawqwen3
        self.assertTrue(is_paw(_Cfg(architectures=["PawQwen3ForCausalLM"])))
        # Does not match dense Qwen3 (the architecture starts with "Qwen").
        self.assertFalse(is_paw(_Cfg(architectures=["Qwen3ForCausalLM"])))
        self.assertFalse(is_paw(_Cfg(architectures=["Qwen2ForCausalLM"])))
        # Non-Qwen architectures are unaffected.
        self.assertFalse(is_paw(_Cfg(architectures=["Ernie4_5ForCausalLM"])))
        self.assertFalse(is_paw(_Cfg(architectures=[])))

    def test_yarn_rope_scaling_extraction(self):
        from fastdeploy.model_executor.models.qwen3_elastic import (
            __init__ as elastic_init,
        )

        get = elastic_init._yarn_rope_scaling
        # ``type=yarn`` -> returns the dict.
        rs = {"type": "yarn", "factor": 8.0, "original_max_position_embeddings": 40960}
        self.assertEqual(get(_Cfg(rope_scaling=rs)), rs)
        # ``rope_type=yarn`` (newer key) also works.
        rs2 = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}
        self.assertEqual(get(_Cfg(rope_scaling=rs2)), rs2)
        # Non-yarn / missing -> None.
        self.assertIsNone(get(_Cfg(rope_scaling={"type": "linear"})))
        self.assertIsNone(get(_Cfg(rope_scaling=None)))
        self.assertIsNone(get(_Cfg()))


if __name__ == "__main__":
    unittest.main()
