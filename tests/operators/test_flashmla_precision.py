"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
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

import paddle

paddle.set_default_dtype("bfloat16")

from fastdeploy.model_executor.layers.attention.mla_attention_backend import (
    MLAAttentionBackend,
)


class TestFlashMLA(unittest.TestCase):
    def setUp(self):
        pass

    def test_flashmla(self):
        # import paddle.profiler as profiler
        # p = profiler.Profiler(
        #     targets=[profiler.ProfilerTarget.CPU, profiler.ProfilerTarget.GPU],
        #     on_trace_ready=profiler.export_chrome_tracing("./profile_log"),
        # )
        # p.start()
        # p.step()

        bsz = 128
        kv_len = 1024 * 128
        page_size = 64
        decoder_q = paddle.randn([bsz, 1, 128, 576], dtype="bfloat16")
        cache_seqlens = paddle.zeros([bsz], dtype="int32") + kv_len
        block_tables = paddle.arange((kv_len // page_size + 1) * bsz, dtype="int32").reshape([bsz, -1])
        latent_cache = paddle.randn([bsz * block_tables.shape[1], 1, page_size, 576], dtype="bfloat16")
        # copy from dsv3
        attn_softmax_scale = 0.1352337788608801

        baseline_out = MLAAttentionBackend.flashmla_baseline(
            decoder_q, latent_cache, block_tables, cache_seqlens, attn_softmax_scale
        )

        prop = paddle.device.cuda.get_device_properties()
        if prop.major == 10:

            for i in range(10):
                decoder_res = MLAAttentionBackend.mla_blackwell(
                    decoder_q, latent_cache, block_tables, cache_seqlens, attn_softmax_scale
                )
        elif prop.major == 9:
            paddle.enable_compat(scope={"flash_mla"})  # Enable paddle.enable_compat before importing flash_mla
            try:
                import flash_mla
            except ImportError:
                print(100 * "Please install flash_mla first")
                return

            tile_scheduler_metadata, num_splits = flash_mla.get_mla_metadata()

            new_cache_shape = latent_cache.shape
            assert new_cache_shape[1] == 1
            new_cache_shape[1], new_cache_shape[2] = new_cache_shape[2], new_cache_shape[1]

            decoder_res, _ = flash_mla.flash_mla_with_kvcache(
                decoder_q,
                # 外面的开源仓库的kv cache存储格式和FD的不同
                # 幸好这里缓存的头是1，直接view即可，否则上上下下要改很多！
                latent_cache.view(new_cache_shape),
                block_tables,
                cache_seqlens,
                512,  # t.dv,
                tile_scheduler_metadata,
                num_splits,
                softmax_scale=attn_softmax_scale,
                causal=True,
            )
        # p.stop()
        max_diff = (decoder_res - baseline_out).abs().max().item()
        print(decoder_res - baseline_out)
        self.assertLessEqual(max_diff, 0.1)


if __name__ == "__main__":
    unittest.main()
