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

# Adapt from
# https://github.com/sgl-project/sglang/blob/main/python/sglang/test/attention/test_flashattn_backend.py

import unittest

import paddle

from fastdeploy.model_executor.layers.attention import PaddleNativeAttnBackend, Attention
from fastdeploy.model_executor.model_runner import ReqToTokenPool, KVCache, MHATokenToKVPool
from fastdeploy.model_executor.model_runner.model_runner_minimal_os import MinimalModelRunner
from fastdeploy.model_executor.model_runner import ForwardMeta, ForwardMode


class MockModelRunner:
    def __init__(
        self,
        page_size=1,
        num_heads=2,
        head_dim=8,
    ):
        self.device = "cuda"
        self.dtype = paddle.float16
        # Max batch size for the test.
        max_batch_size = 160
        # Total tokens(prefix + extend + decode) in the test should not exceed this length.
        max_context_len = 2048
        self.model_config = type(
            "ModelConfig",
            (),
            {
                "context_len": max_context_len,
            },
        )
        self.sliding_window_size = None
        self.device = self.device
        # Create a large enough req_to_token_pool to fit the test usage.
        self.req_to_token_pool = type(
            "TokenPool",
            (),
            {
                # A typical max_bs * max_context_len for cuda graph decode
                "size": max_batch_size,
                # Add req_to_token attribute
                "req_to_token": paddle.zeros(
                    [max_batch_size, max_context_len],
                    dtype=paddle.int32
                ),
            },
        )
        self.page_size = page_size
        max_total_num_tokens = max_batch_size * max_context_len
        self.token_to_kv_pool = MHATokenToKVPool(
            size=max_total_num_tokens,
            page_size=page_size,
            dtype=self.dtype,
            head_num=num_heads,
            head_dim=head_dim,
            layer_num=1,  # only consider layer=1 for unit test
            device=self.device
        )


class TestNativePaddleAttentionBackend(unittest.TestCase):
    def setUp(self):
        # Test parameters
        self.batch_size = 2
        self.seq_len = 256
        self.num_heads = 2
        self.head_dim = 128
        self.device = "gpu"
        self.dtype = paddle.float16

    def _init_model_runner(self, page_size=1):
        self.model_runner = MockModelRunner(
            page_size=page_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )
        self.backend = PaddleNativeAttnBackend(self.model_runner)
        self.ref_backend = PaddleNativeAttnBackend(self.model_runner)
        self.model_runner.model_config.num_attention_heads = self.num_heads

    def _mock_write_to_req_to_token_pool(self, batch_size, seq_len, page_size):
        # if page_size > 1, the token pool stores the index to the page.
        # so we need to multiply the index by page_size.
        self.req_to_token = (
            paddle.arange(0, batch_size, dtype=paddle.int32)[:, None]
            * seq_len
            + paddle.arange(0, seq_len, dtype=paddle.int32)[None, :]
            + page_size
        )
        self.model_runner.req_to_token_pool.req_to_token[:batch_size, :seq_len] = (
            self.req_to_token
        )

    def _create_attention_layer(self):
        """Create attention layer for testing."""
        return Attention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            num_kv_heads=self.num_heads,
            layer_id=0,
        )

    def _create_qkv_tensors(self, tokens_len):
        """Create q, k, v tensors for testing."""
        shape = (tokens_len, self.num_heads, self.head_dim)
        return (
            paddle.randn(shape, dtype=self.dtype),
            paddle.randn(shape, dtype=self.dtype),
            paddle.randn(shape, dtype=self.dtype),
        )

    def _run_reference_forward(
        self, mode, q, k, v, layer, forward_batch, expected_shape
    ):
        """Run reference forward pass using native backend."""
        if mode == ForwardMode.EXTEND:
            output = self.ref_backend.forward_extend(
                q, k, v, layer, forward_batch)
        else:  # ForwardMode.DECODE
            output = self.ref_backend.forward_decode(
                q, k, v, layer, forward_batch)
        return output.view(expected_shape)

    def _verify_output(self, output, expected_shape, output_ref=None):
        """Verify output tensor shape, dtype, and values."""
        self.assertEqual(
            output.shape,
            expected_shape,
            f"Expected shape {expected_shape}, got {output.shape}",
        )
        self.assertEqual(output.dtype, self.dtype)
        self.assertEqual(
            paddle.isnan(output).sum().item(), 0, "Output contains NaN values"
        )

        if output_ref is not None:
            if not paddle.allclose(output, output_ref, atol=1e-1, rtol=0.0):
                # Check where the values differ beyond the given tolerances
                diff_mask = ~paddle.isclose(
                    output, output_ref, atol=1e-1, rtol=0.0)

                # Find the first index where the difference occurs
                if diff_mask.any():
                    first_mismatch_idx = diff_mask.nonzero()[0]
                    print(
                        "First mismatch at index:", tuple(
                            first_mismatch_idx.tolist())
                    )
                    print("output:", output[tuple(
                        first_mismatch_idx.tolist())])
                    print("output_ref:", output_ref[tuple(
                        first_mismatch_idx.tolist())])
                raise AssertionError(
                    "Attention output is not close to the torch native backend output"
                )

    def _create_forward_batch(self, mode, q_len=None, prefix_len=0, page_size=1):
        """Create a forward batch for testing based on mode and lengths."""
        self._init_model_runner(page_size=page_size)

        # Default to self.seq_len if not specified
        q_len = q_len or self.seq_len

        if mode == ForwardMode.EXTEND:
            total_len = prefix_len + q_len
            out_cache_start = prefix_len * self.batch_size
            out_cache_end = total_len * self.batch_size

            forward_batch = ForwardMeta(
                batch_size=self.batch_size,
                input_ids=paddle.randint(
                    0, 100, (self.batch_size, q_len)
                ),
                out_cache_loc=paddle.arange(
                    out_cache_start, out_cache_end
                ),
                seq_lens_sum=self.batch_size * total_len,  # need to be real
                forward_mode=mode,
                req_pool_indices=paddle.arange(self.batch_size),
                seq_lens=paddle.to_tensor(
                    [total_len] * self.batch_size
                ),
                extend_prefix_lens=paddle.to_tensor(
                    [prefix_len] * self.batch_size
                ),
                extend_seq_lens=paddle.to_tensor(
                    [q_len] * self.batch_size
                ),
                seq_lens_cpu=paddle.to_tensor(
                    [total_len] * self.batch_size, place="cpu"),
                extend_prefix_lens_cpu=paddle.to_tensor(
                    [prefix_len] * self.batch_size, place="cpu"
                ),
                extend_seq_lens_cpu=paddle.to_tensor(
                    [q_len] * self.batch_size, place="cpu"
                ),
                attn_backend=self.backend,
            )
        else:  # ForwardMode.DECODE
            decode_len = q_len  # Assuming 1 for decode testing
            total_len = self.seq_len + decode_len
            if mode == ForwardMode.DECODE and page_size > 1:
                # Get next page_size multiple of self.seq_len
                out_cache_start = (
                    self.batch_size * self.seq_len // page_size + 1
                ) * page_size
                # out_cache_end is the start of the next block
                out_cache_end = out_cache_start + decode_len * page_size
            else:
                out_cache_start = self.batch_size * self.seq_len
                out_cache_end = self.batch_size * total_len

            forward_batch = ForwardMeta(
                batch_size=self.batch_size,
                input_ids=paddle.randint(
                    0, 100, (self.batch_size, decode_len)
                ),
                out_cache_loc=paddle.to_tensor(
                    [out_cache_start, out_cache_end]
                ),
                seq_lens_sum=self.batch_size * total_len,
                forward_mode=mode,
                req_pool_indices=paddle.arange(self.batch_size),
                seq_lens=paddle.to_tensor(
                    [total_len] * self.batch_size
                ),
                seq_lens_cpu=paddle.to_tensor(
                    [total_len] * self.batch_size, place="cpu"),
                attn_backend=self.backend,
            )

        # Add token pool
        forward_batch.req_to_token_pool = self.model_runner.req_to_token_pool

        # Write current batch's req_to_token to req_to_token_pool
        self._mock_write_to_req_to_token_pool(
            self.batch_size, total_len, page_size)
        # Add kv pool for this forward batch
        forward_batch.token_to_kv_pool = self.model_runner.token_to_kv_pool

        return forward_batch

    def _setup_kv_cache(self, forward_batch, layer, cache_len):
        # Create constant values for the prefix cache for easy debugging
        cache_k = paddle.ones(
            [self.batch_size * cache_len,
             self.num_heads,
             self.head_dim],
            dtype=self.dtype,
        )
        cache_v = (
            paddle.ones(
                [self.batch_size * cache_len,
                 self.num_heads,
                 self.head_dim],
                dtype=self.dtype,
            )
            * 2
        )

        # Set the prefix KV cache
        forward_batch.token_to_kv_pool.set_kv_buffer(
            layer,
            paddle.arange(self.batch_size * cache_len),
            cache_k,
            cache_v,
            layer.k_scale,
            layer.v_scale,
        )

    def _run_attention_test(self, mode, q_len, prefix_len=0, page_size=1):
        """
            Run an attention test with the specified parameters.
        Args:
            mode: ForwardMode.EXTEND or ForwardMode.DECODE
            q_len: Length of the query sequence. For decode mode, q_len is 1.
            prefix_len: Length of the prefix sequence for extend mode
            page_size: Page size for the KV cache
        """
        layer = self._create_attention_layer()

        # Create forward batch and set up
        forward_batch = self._create_forward_batch(
            mode, q_len, prefix_len, page_size)

        # Create QKV tensors for the input
        q, k, v = self._create_qkv_tensors(self.batch_size * q_len)

        # KV cache for prefixed extend is prefix_len
        # KV cache for decode is same as seq_len
        # No KV cache for extend without prefix
        if mode == ForwardMode.EXTEND:
            if prefix_len > 0:
                self._setup_kv_cache(forward_batch, layer, prefix_len)
        else:
            self._setup_kv_cache(forward_batch, layer, self.seq_len)

        self.backend.init_attention_metadata(forward_batch)

        if mode == ForwardMode.EXTEND:
            expected_shape = [
                self.batch_size * q_len,
                self.num_heads,  self.head_dim,
            ]
            output = self.backend.forward_extend(q, k, v, layer, forward_batch)
        else:
            expected_shape = [self.batch_size, self.num_heads * self.head_dim]
            output = self.backend.forward_decode(q, k, v, layer, forward_batch)

        output_ref = self._run_reference_forward(
            mode, q, k, v, layer, forward_batch, expected_shape
        )

        self._verify_output(output, expected_shape, output_ref)

        return output

    def test_forward_extend(self):
        """Test the standard extend operation."""
        self._run_attention_test(ForwardMode.EXTEND, q_len=self.seq_len)

    def test_forward_decode(self):
        """Test the decode operation with cached tokens."""
        self._run_attention_test(ForwardMode.DECODE, q_len=1)

    def test_forward_extend_with_prefix(self):
        """Test extending from cached prefix tokens."""
        prefix_len = self.seq_len // 2
        extend_len = self.seq_len - prefix_len
        self._run_attention_test(
            ForwardMode.EXTEND, q_len=extend_len, prefix_len=prefix_len
        )

    def test_forward_extend_with_page_size_greater_than_1(self):
        """Test extending from cached prefix tokens with page size greater than 1."""
        self._run_attention_test(
            ForwardMode.EXTEND, q_len=self.seq_len, page_size=64)

    def test_forward_decode_with_page_size_greater_than_1(self):
        """Test decode operation with page size greater than 1."""
        self._run_attention_test(ForwardMode.DECODE, q_len=1, page_size=64)


if __name__ == "__main__":
    unittest.main()
