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

import threading
import unittest
from unittest.mock import MagicMock, patch

import paddle

from fastdeploy.model_executor.forward_meta import ForwardMeta
from fastdeploy.worker.tbo import (
    GLOBAL_ATTN_BUFFERS,
    GLOBAL_THREAD_INFO,
    creat_empty_forward_meta,
    is_last_thread,
    let_another_thread_run,
    split_batch_decoder_layers,
)


class TestIsLastThread(unittest.TestCase):
    """Test is_last_thread function."""

    @patch("threading.current_thread")
    def test_thread1_is_last(self, mock_current_thread):
        """is_last_thread returns True when thread name is 'thread1'."""
        mock_current_thread.return_value.name = "thread1"
        self.assertTrue(is_last_thread())

    @patch("threading.current_thread")
    def test_thread0_is_not_last(self, mock_current_thread):
        """is_last_thread returns False when thread name is 'thread0'."""
        mock_current_thread.return_value.name = "thread0"
        self.assertFalse(is_last_thread())

    @patch("threading.current_thread")
    def test_unknown_thread_is_not_last(self, mock_current_thread):
        """is_last_thread returns False for unknown thread names."""
        mock_current_thread.return_value.name = "MainThread"
        self.assertFalse(is_last_thread())


class TestLetAnotherThreadRun(unittest.TestCase):
    """Test let_another_thread_run function."""

    @patch("threading.current_thread")
    def test_thread0_sets_event1_waits_event0(self, mock_current_thread):
        """thread0 sets event1 and waits on event0."""
        mock_current_thread.return_value.name = "thread0"

        mock_event0 = MagicMock()
        mock_event1 = MagicMock()
        original_info = GLOBAL_THREAD_INFO.copy()
        GLOBAL_THREAD_INFO["thread0"] = [mock_event0, mock_event1]

        try:
            let_another_thread_run()
            mock_event1.set.assert_called_once()
            mock_event0.wait.assert_called_once()
            mock_event0.clear.assert_called_once()
        finally:
            GLOBAL_THREAD_INFO.update(original_info)

    @patch("threading.current_thread")
    def test_thread1_sets_event0_waits_event1(self, mock_current_thread):
        """thread1 sets event0 and waits on event1."""
        mock_current_thread.return_value.name = "thread1"

        mock_event0 = MagicMock()
        mock_event1 = MagicMock()
        original_info = GLOBAL_THREAD_INFO.copy()
        GLOBAL_THREAD_INFO["thread1"] = [mock_event1, mock_event0]

        try:
            let_another_thread_run()
            mock_event0.set.assert_called_once()
            mock_event1.wait.assert_called_once()
            mock_event1.clear.assert_called_once()
        finally:
            GLOBAL_THREAD_INFO.update(original_info)

    @patch("threading.current_thread")
    def test_unknown_thread_does_nothing(self, mock_current_thread):
        """Unknown thread name skips event operations."""
        mock_current_thread.return_value.name = "unknown_thread"
        # Should not raise
        let_another_thread_run()


class TestCreatEmptyForwardMeta(unittest.TestCase):
    """Test creat_empty_forward_meta function."""

    def _make_forward_meta(self):
        """Create a minimal ForwardMeta for testing."""
        ids = paddle.to_tensor([1, 2, 3, 4, 5], dtype="int64")
        rotary = paddle.randn([5, 64])
        attn_backend = MagicMock()
        caches = [paddle.randn([2, 128])]
        fm = ForwardMeta(
            ids_remove_padding=ids,
            rotary_embs=rotary,
            attn_backend=attn_backend,
            caches=caches,
        )
        fm.hidden_states = paddle.randn([5, 256])
        fm.decode_states = paddle.randn([5, 128])
        return fm

    def test_returns_forward_meta(self):
        """creat_empty_forward_meta returns a ForwardMeta instance."""
        fm = self._make_forward_meta()
        result = creat_empty_forward_meta(fm)
        self.assertIsInstance(result, ForwardMeta)

    def test_ids_remove_padding_is_empty(self):
        """Result has zero-length ids_remove_padding."""
        fm = self._make_forward_meta()
        result = creat_empty_forward_meta(fm)
        self.assertEqual(result.ids_remove_padding.shape[0], 0)

    def test_hidden_states_is_empty(self):
        """Result has zero-length hidden_states."""
        fm = self._make_forward_meta()
        result = creat_empty_forward_meta(fm)
        self.assertEqual(result.hidden_states.shape[0], 0)

    def test_decode_states_is_empty(self):
        """Result has zero-length decode_states."""
        fm = self._make_forward_meta()
        result = creat_empty_forward_meta(fm)
        self.assertEqual(result.decode_states.shape[0], 0)

    def test_shared_rotary_embs(self):
        """Result shares rotary_embs with input."""
        fm = self._make_forward_meta()
        result = creat_empty_forward_meta(fm)
        self.assertEqual(result.rotary_embs.data_ptr(), fm.rotary_embs.data_ptr())

    def test_shared_attn_backend(self):
        """Result shares attn_backend with input."""
        fm = self._make_forward_meta()
        result = creat_empty_forward_meta(fm)
        self.assertIs(result.attn_backend, fm.attn_backend)

    def test_shared_caches(self):
        """Result shares caches with input."""
        fm = self._make_forward_meta()
        result = creat_empty_forward_meta(fm)
        self.assertIs(result.caches, fm.caches)


class TestSplitBatchDecoderLayersSmallBatch(unittest.TestCase):
    """Test split_batch_decoder_layers with small token count (< 1024)."""

    def _make_forward_meta(self, num_tokens):
        """Create ForwardMeta with given token count."""
        ids = paddle.arange(num_tokens, dtype="int64")
        rotary = paddle.randn([num_tokens, 64])
        attn_backend = MagicMock()
        caches = [paddle.randn([2, 128])]
        fm = ForwardMeta(
            ids_remove_padding=ids,
            rotary_embs=rotary,
            attn_backend=attn_backend,
            caches=caches,
        )
        fm.hidden_states = paddle.randn([num_tokens, 256])
        fm.decode_states = paddle.randn([num_tokens, 128])
        return fm

    def test_small_batch_returns_early(self):
        """Tokens < 1024 returns [empty_meta, original_meta]."""
        fm = self._make_forward_meta(512)
        fd_config = MagicMock()

        result = split_batch_decoder_layers(fm, fd_config)

        self.assertEqual(len(result), 2)
        # First element should be empty
        self.assertEqual(result[0].ids_remove_padding.shape[0], 0)
        # Second element is the original
        self.assertIs(result[1], fm)

    def test_tbo_microbatch_id_set(self):
        """tbo_microbatch_id is set to 0 and 1."""
        fm = self._make_forward_meta(100)
        fd_config = MagicMock()

        result = split_batch_decoder_layers(fm, fd_config)

        self.assertEqual(result[0].tbo_microbatch_id, 0)
        self.assertEqual(result[1].tbo_microbatch_id, 1)

    def test_exactly_1023_tokens(self):
        """1023 tokens returns early (less than 1024)."""
        fm = self._make_forward_meta(1023)
        fd_config = MagicMock()

        result = split_batch_decoder_layers(fm, fd_config)

        self.assertEqual(result[0].ids_remove_padding.shape[0], 0)
        self.assertIs(result[1], fm)


class TestSplitBatchDecoderLayersLargeBatch(unittest.TestCase):
    """Test split_batch_decoder_layers with large token count (>= 1024)."""

    def _make_large_forward_meta(self, num_tokens, num_batches):
        """Create a ForwardMeta for large batch splitting tests."""
        ids = paddle.arange(num_tokens, dtype="int64")
        rotary = paddle.randn([num_batches, 64])
        attn_backend = MagicMock()
        caches = [paddle.randn([2, 128])]

        # Create batch_id_per_token: assign tokens evenly across batches
        tokens_per_batch = num_tokens // num_batches
        batch_ids = []
        for b in range(num_batches):
            batch_ids.extend([b] * tokens_per_batch)
        # Handle remainder
        remaining = num_tokens - len(batch_ids)
        batch_ids.extend([num_batches - 1] * remaining)
        batch_id_per_token = paddle.to_tensor(batch_ids, dtype="int32")

        # Create cu_seqlens_q: cumulative sequence lengths
        seq_lens = paddle.full([num_batches], tokens_per_batch, dtype="int32")
        if remaining > 0:
            seq_lens[-1] = tokens_per_batch + remaining
        cumsum = paddle.cumsum(seq_lens).cast("int32")
        cu_seqlens = paddle.concat([paddle.zeros([1], dtype="int32"), cumsum])

        # Create seq_lens_this_time, seq_lens_encoder, seq_lens_decoder
        seq_lens_this_time = seq_lens.clone()
        seq_lens_encoder = paddle.zeros([num_batches], dtype="int32")
        seq_lens_decoder = seq_lens.clone()

        block_tables = paddle.zeros([num_batches, 16], dtype="int32")

        fm = ForwardMeta(
            ids_remove_padding=ids,
            rotary_embs=rotary,
            attn_backend=attn_backend,
            caches=caches,
        )
        fm.batch_id_per_token = batch_id_per_token
        fm.cu_seqlens_q = cu_seqlens
        fm.seq_lens_this_time = seq_lens_this_time
        fm.seq_lens_encoder = seq_lens_encoder
        fm.seq_lens_decoder = seq_lens_decoder
        fm.block_tables = block_tables
        fm.hidden_states = paddle.randn([num_tokens, 256])
        fm.decode_states = paddle.randn([num_batches, 128])
        fm.attn_mask_offsets = None
        return fm

    def setUp(self):
        """Set up GLOBAL_ATTN_BUFFERS for tests."""
        GLOBAL_ATTN_BUFFERS[0] = {}
        GLOBAL_ATTN_BUFFERS[1] = {}

    def tearDown(self):
        """Clean up GLOBAL_ATTN_BUFFERS."""
        GLOBAL_ATTN_BUFFERS.pop(0, None)
        GLOBAL_ATTN_BUFFERS.pop(1, None)

    def test_split_produces_two_results(self):
        """Large batch split produces exactly 2 ForwardMeta results."""
        num_tokens = 2048
        num_batches = 4
        fm = self._make_large_forward_meta(num_tokens, num_batches)
        fd_config = MagicMock()
        fd_config.model_config.image_patch_id = -999  # Non-existent token

        result = split_batch_decoder_layers(fm, fd_config)

        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], ForwardMeta)
        self.assertIsInstance(result[1], ForwardMeta)

    def test_split_covers_all_tokens(self):
        """Split result covers all tokens from original."""
        num_tokens = 2048
        num_batches = 4
        fm = self._make_large_forward_meta(num_tokens, num_batches)
        fd_config = MagicMock()
        fd_config.model_config.image_patch_id = -999

        result = split_batch_decoder_layers(fm, fd_config)

        total = result[0].ids_remove_padding.shape[0] + result[1].ids_remove_padding.shape[0]
        self.assertEqual(total, num_tokens)

    def test_tbo_microbatch_ids_set_correctly(self):
        """Both chunks have correct tbo_microbatch_id."""
        num_tokens = 2048
        num_batches = 4
        fm = self._make_large_forward_meta(num_tokens, num_batches)
        fd_config = MagicMock()
        fd_config.model_config.image_patch_id = -999

        result = split_batch_decoder_layers(fm, fd_config)

        self.assertEqual(result[0].tbo_microbatch_id, 0)
        self.assertEqual(result[1].tbo_microbatch_id, 1)

    def test_split_with_special_token_at_boundary(self):
        """Special tokens at split boundary cause offset adjustment."""
        num_tokens = 2048
        num_batches = 4
        fm = self._make_large_forward_meta(num_tokens, num_batches)

        # Put special token exactly at the split point
        split_point = (num_tokens + 1) // 2  # chunk_token_num
        special_token_id = 12345
        ids_np = fm.ids_remove_padding.numpy()
        ids_np[split_point] = special_token_id
        ids_np[split_point + 1] = special_token_id
        fm.ids_remove_padding = paddle.to_tensor(ids_np, dtype="int64")

        fd_config = MagicMock()
        fd_config.model_config.image_patch_id = special_token_id

        result = split_batch_decoder_layers(fm, fd_config)

        # First chunk should be larger than half (shifted past special tokens)
        self.assertGreater(result[0].ids_remove_padding.shape[0], split_point)

    def test_split_with_all_special_tokens_returns_early(self):
        """If all remaining tokens are special, returns early with empty split."""
        num_tokens = 2048
        num_batches = 4
        fm = self._make_large_forward_meta(num_tokens, num_batches)

        special_token_id = 99999
        # Fill everything from split_point to end with special tokens
        split_point = (num_tokens + 1) // 2
        ids_np = fm.ids_remove_padding.numpy()
        ids_np[split_point:] = special_token_id
        fm.ids_remove_padding = paddle.to_tensor(ids_np, dtype="int64")

        fd_config = MagicMock()
        fd_config.model_config.image_patch_id = special_token_id

        result = split_batch_decoder_layers(fm, fd_config)

        # Should return early: [empty, original]
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].ids_remove_padding.shape[0], 0)

    def test_split_with_attn_mask_offsets_double(self):
        """attn_mask_offsets with shape[0] == 2*total_token_num is sliced."""
        num_tokens = 2048
        num_batches = 4
        fm = self._make_large_forward_meta(num_tokens, num_batches)
        fm.attn_mask_offsets = paddle.arange(num_tokens * 2, dtype="int32")

        fd_config = MagicMock()
        fd_config.model_config.image_patch_id = -999

        result = split_batch_decoder_layers(fm, fd_config)

        # Both chunks should have attn_mask_offsets
        self.assertIsNotNone(result[0].attn_mask_offsets)
        self.assertIsNotNone(result[1].attn_mask_offsets)

    def test_split_with_attn_mask_offsets_single(self):
        """attn_mask_offsets with shape[0] == total_token_num is sliced."""
        num_tokens = 2048
        num_batches = 4
        fm = self._make_large_forward_meta(num_tokens, num_batches)
        fm.attn_mask_offsets = paddle.arange(num_tokens, dtype="int32")

        fd_config = MagicMock()
        fd_config.model_config.image_patch_id = -999

        result = split_batch_decoder_layers(fm, fd_config)

        self.assertIsNotNone(result[0].attn_mask_offsets)
        self.assertIsNotNone(result[1].attn_mask_offsets)

    def test_split_with_attn_mask_offsets_invalid_raises(self):
        """attn_mask_offsets with invalid shape raises AssertionError."""
        num_tokens = 2048
        num_batches = 4
        fm = self._make_large_forward_meta(num_tokens, num_batches)
        # Invalid size: neither total_token_num nor 2*total_token_num
        fm.attn_mask_offsets = paddle.arange(100, dtype="int32")

        fd_config = MagicMock()
        fd_config.model_config.image_patch_id = -999

        with self.assertRaises(AssertionError) as ctx:
            split_batch_decoder_layers(fm, fd_config)
        self.assertIn("Invalid attn_mask_offsets shape", str(ctx.exception))

    def test_split_with_6d_rotary_embs(self):
        """6D rotary_embs are sliced per batch for each chunk."""
        num_tokens = 2048
        num_batches = 4
        fm = self._make_large_forward_meta(num_tokens, num_batches)
        # Shape: [num_batches, 2, 1, dim, 1, head_dim]
        fm.rotary_embs = paddle.randn([num_batches, 2, 1, 64, 1, 32])

        fd_config = MagicMock()
        fd_config.model_config.image_patch_id = -999

        result = split_batch_decoder_layers(fm, fd_config)

        # Both should have rotary_embs with first dim < num_batches
        self.assertEqual(len(result[0].rotary_embs.shape), 6)
        self.assertEqual(len(result[1].rotary_embs.shape), 6)
        total_batches = result[0].rotary_embs.shape[0] + result[1].rotary_embs.shape[0]
        self.assertEqual(total_batches, num_batches)

    def test_global_attn_buffers_applied(self):
        """GLOBAL_ATTN_BUFFERS attributes are set on result chunks."""
        num_tokens = 2048
        num_batches = 4
        fm = self._make_large_forward_meta(num_tokens, num_batches)

        custom_val_0 = paddle.to_tensor([42])
        custom_val_1 = paddle.to_tensor([99])
        GLOBAL_ATTN_BUFFERS[0] = {"custom_attr": custom_val_0}
        GLOBAL_ATTN_BUFFERS[1] = {"custom_attr": custom_val_1}

        fd_config = MagicMock()
        fd_config.model_config.image_patch_id = -999

        result = split_batch_decoder_layers(fm, fd_config)

        self.assertEqual(result[0].custom_attr.item(), 42)
        self.assertEqual(result[1].custom_attr.item(), 99)

    def test_split_hidden_states_coverage(self):
        """hidden_states is split by token range for each chunk."""
        num_tokens = 2048
        num_batches = 4
        fm = self._make_large_forward_meta(num_tokens, num_batches)

        fd_config = MagicMock()
        fd_config.model_config.image_patch_id = -999

        result = split_batch_decoder_layers(fm, fd_config)

        total_hidden = result[0].hidden_states.shape[0] + result[1].hidden_states.shape[0]
        self.assertEqual(total_hidden, num_tokens)

    def test_split_seq_lens_encoder_with_prefill(self):
        """seq_lens_encoder adjustment when encoder tokens are present."""
        num_tokens = 2048
        num_batches = 4
        fm = self._make_large_forward_meta(num_tokens, num_batches)
        # Set first and last batch to have encoder tokens
        tokens_per_batch = num_tokens // num_batches
        encoder_lens = paddle.zeros([num_batches], dtype="int32")
        encoder_lens[0] = tokens_per_batch
        encoder_lens[-1] = tokens_per_batch
        fm.seq_lens_encoder = encoder_lens

        fd_config = MagicMock()
        fd_config.model_config.image_patch_id = -999

        result = split_batch_decoder_layers(fm, fd_config)

        # Should not raise and produce valid results
        self.assertEqual(len(result), 2)
        self.assertIsNotNone(result[0].seq_lens_encoder)
        self.assertIsNotNone(result[1].seq_lens_encoder)


class TestGlobalThreadInfoStructure(unittest.TestCase):
    """Test GLOBAL_THREAD_INFO module-level structure."""

    def test_thread0_has_two_events(self):
        """GLOBAL_THREAD_INFO['thread0'] contains two events."""
        self.assertEqual(len(GLOBAL_THREAD_INFO["thread0"]), 2)

    def test_thread1_has_two_events(self):
        """GLOBAL_THREAD_INFO['thread1'] contains two events."""
        self.assertEqual(len(GLOBAL_THREAD_INFO["thread1"]), 2)

    def test_events_are_threading_events(self):
        """Events in GLOBAL_THREAD_INFO are threading.Event instances."""
        for events in GLOBAL_THREAD_INFO.values():
            for event in events:
                self.assertIsInstance(event, threading.Event)

    def test_thread0_and_thread1_share_events_cross(self):
        """thread0's events are thread1's events in reverse order."""
        t0_events = GLOBAL_THREAD_INFO["thread0"]
        t1_events = GLOBAL_THREAD_INFO["thread1"]
        # thread0 = [event0, event1], thread1 = [event1, event0]
        self.assertIs(t0_events[0], t1_events[1])
        self.assertIs(t0_events[1], t1_events[0])


if __name__ == "__main__":
    unittest.main()
