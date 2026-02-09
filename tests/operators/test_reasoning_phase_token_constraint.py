import unittest

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import (
    get_padding_offset,
    reasoning_phase_token_constraint,
)


class TestReasoningPhaseTokenConstraint(unittest.TestCase):

    def setUp(self):
        paddle.set_device("gpu")

        # ------------------------
        # Basic config
        # ------------------------
        self.bs = 2
        self.max_seq_len = 8
        self.vocab_size = 16

        self.think_end_id = 9
        self.line_break_id = 10

        # ------------------------
        # seq / step
        # ------------------------
        self.step_idx = paddle.to_tensor([4, 4], dtype="int64")

        self.seq_lens_this_time = paddle.to_tensor([2, 2], dtype="int32")
        self.seq_lens_encoder = paddle.to_tensor([0, 0], dtype="int32")
        self.stop_flags = paddle.to_tensor([False, False], dtype="bool")

        # ------------------------
        # pre_ids
        #
        # batch 0:
        #   ... \n <think_end> \n \n   → status 1 -> 2
        #
        # batch 1:
        #   contains think_end, but pattern not complete → status 0 -> 1
        # ------------------------
        pre_ids = np.zeros((self.bs, self.max_seq_len), dtype=np.int64)

        # batch 0
        pre_ids[0, 1] = self.line_break_id
        pre_ids[0, 2] = self.think_end_id
        pre_ids[0, 3] = self.line_break_id
        pre_ids[0, 4] = self.line_break_id

        # batch 1
        pre_ids[1, 3] = self.think_end_id

        self.pre_ids = paddle.to_tensor(pre_ids, dtype="int64")

        # ------------------------
        # reasoning_status (init)
        # ------------------------
        self.reasoning_status = paddle.to_tensor([1, 0], dtype="int32")

        # ------------------------
        # allowed tokens
        # ------------------------
        self.allowed_tokens = paddle.to_tensor([2, 5, 7], dtype="int64")

        # ------------------------
        # speculative layout
        #
        # each batch has exactly 1 token this step
        # token_idx == bs_idx
        # ------------------------

        self.token_num = paddle.sum(self.seq_lens_this_time)

        seq_lens_output = paddle.to_tensor([2, 2], dtype="int32")
        output_token_num = paddle.sum(seq_lens_output)

        useless_inputs = paddle.zeros([self.bs, self.max_seq_len], dtype="int64")
        _, self.output_padding_offset, self.output_cum_offsets, _ = get_padding_offset(
            useless_inputs,
            seq_lens_output,
            None,
            None,
            output_token_num.item(),
        )

        # self.output_padding_offset = paddle.zeros([self.token_num], dtype="int32")
        # self.output_cum_offsets = paddle.zeros([self.bs], dtype="int32")

        # ------------------------
        # logits
        # ------------------------
        np.random.seed(2024)
        logits = np.random.randn(self.token_num, self.vocab_size).astype("float32")
        self.logits = paddle.to_tensor(logits, dtype="float32")

        self.enable_thinking = paddle.to_tensor([True, True], dtype="bool")

    def test_reasoning_status_and_logits_enforce(self):
        logits_before = self.logits.numpy().copy()

        # ------------------------
        # call custom op
        # ------------------------
        reasoning_phase_token_constraint(
            self.logits,
            self.pre_ids,
            self.stop_flags,
            self.seq_lens_this_time,
            self.seq_lens_encoder,
            self.step_idx,
            self.allowed_tokens,
            self.reasoning_status,
            self.output_padding_offset,
            self.output_cum_offsets,
            self.enable_thinking,
            self.think_end_id,
            self.line_break_id,
        )

        logits_after = self.logits.numpy()
        status_after = self.reasoning_status.numpy()

        # ============================================================
        # 1. reasoning_status check
        # ============================================================
        # batch 0: 1 -> 2
        self.assertEqual(status_after[0], 2)

        # batch 1: 0 -> 1
        self.assertEqual(status_after[1], 1)

        # ============================================================
        # 2. logits enforce check
        # ============================================================
        # batch 0 should be enforced (status == 2)
        for vid in range(self.vocab_size):
            if vid in self.allowed_tokens.numpy():
                self.assertAlmostEqual(
                    logits_after[0, vid],
                    logits_before[0, vid],
                    places=5,
                )
            else:
                self.assertLess(logits_after[0, vid], -1e9)

        # batch 1 should be untouched
        np.testing.assert_allclose(
            logits_after[1],
            logits_before[1],
            rtol=1e-5,
            atol=1e-6,
        )

    def test_status_0_to_1_only(self):
        """
        status == 0
        recent tokens contain <think_end>
        => status: 0 -> 1
        logits should NOT be enforced
        """

        # ------------------------
        # setup: only think_end appears
        # ------------------------
        pre_ids = np.zeros((self.bs, self.max_seq_len), dtype=np.int64)

        # batch 0: think_end at cur_step - 1
        pre_ids[0, 3] = self.think_end_id

        # batch 1: no think_end
        pre_ids[1, :] = 0

        self.pre_ids = paddle.to_tensor(pre_ids, dtype="int64")

        self.reasoning_status = paddle.to_tensor([0, 0], dtype="int32")

        logits_before = self.logits.numpy().copy()

        # ------------------------
        # call op
        # ------------------------
        reasoning_phase_token_constraint(
            self.logits,
            self.pre_ids,
            self.stop_flags,
            self.seq_lens_this_time,
            self.seq_lens_encoder,
            self.step_idx,
            self.allowed_tokens,
            self.reasoning_status,
            self.output_padding_offset,
            self.output_cum_offsets,
            self.enable_thinking,
            self.think_end_id,
            self.line_break_id,
        )

        status_after = self.reasoning_status.numpy()
        logits_after = self.logits.numpy()

        # ============================================================
        # 1. reasoning_status
        # ============================================================
        # batch 0: 0 -> 1
        self.assertEqual(status_after[0], 1)

        # batch 1: stays 0
        self.assertEqual(status_after[1], 0)

        # ============================================================
        # 2. logits must be untouched
        # ============================================================
        np.testing.assert_allclose(
            logits_after,
            logits_before,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_status_2_to_3_only(self):
        # Force initial status = 2
        self.reasoning_status = paddle.to_tensor([2, 2], dtype="int32")

        logits_before = self.logits.numpy().copy()

        reasoning_phase_token_constraint(
            self.logits,
            self.pre_ids,
            self.stop_flags,
            self.seq_lens_this_time,
            self.seq_lens_encoder,
            self.step_idx,
            self.allowed_tokens,
            self.reasoning_status,
            self.output_padding_offset,
            self.output_cum_offsets,
            self.enable_thinking,
            self.think_end_id,
            self.line_break_id,
        )

        status_after = self.reasoning_status.numpy()
        logits_after = self.logits.numpy()

        # status: 2 -> 3
        self.assertTrue(np.all(status_after == 3))

        # logits should NOT be changed
        np.testing.assert_allclose(
            logits_after,
            logits_before,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_status_1_to_2(self):
        # batch 0 enforce，batch 1 not enforce
        self.reasoning_status = paddle.to_tensor([1, 2], dtype="int32")

        logits_before = self.logits.numpy().copy()

        reasoning_phase_token_constraint(
            self.logits,
            self.pre_ids,
            self.stop_flags,
            self.seq_lens_this_time,
            self.seq_lens_encoder,
            self.step_idx,
            self.allowed_tokens,
            self.reasoning_status,
            self.output_padding_offset,
            self.output_cum_offsets,
            self.enable_thinking,
            self.think_end_id,
            self.line_break_id,
        )

        logits_after = self.logits.numpy()
        # Find batch 0's token_idx
        token_idx_batch0 = 0  # speculate_get_output_padding_offset 下，第一个 token 一定是 batch 0

        # batch 0 first token should be enforced
        for vid in range(self.vocab_size):
            if vid in self.allowed_tokens.numpy():
                self.assertAlmostEqual(
                    logits_after[token_idx_batch0, vid],
                    logits_before[token_idx_batch0, vid],
                    places=5,
                )
            else:
                self.assertLess(logits_after[token_idx_batch0, vid], -1e9)

        # batch 0 second token（如果存在）必须 untouched
        if self.token_num > 1:
            np.testing.assert_allclose(
                logits_after[token_idx_batch0 + 1],
                logits_before[token_idx_batch0 + 1],
                rtol=1e-5,
                atol=1e-6,
            )
        np.testing.assert_equal(self.reasoning_status.numpy(), [2, 3])

    def test_status_0_to_2(self):
        # batch 0 enforce，batch 1 not enforce
        self.reasoning_status = paddle.to_tensor([0, 0], dtype="int32")
        self.enable_thinking = paddle.to_tensor([False, False], dtype="bool")

        self.step_idx = paddle.to_tensor([0, 0], dtype="int64")

        self.seq_lens_this_time = paddle.to_tensor([15, 15], dtype="int32")
        self.seq_lens_encoder = paddle.to_tensor([15, 15], dtype="int32")

        logits_before = self.logits.numpy().copy()

        reasoning_phase_token_constraint(
            self.logits,
            self.pre_ids,
            self.stop_flags,
            self.seq_lens_this_time,
            self.seq_lens_encoder,
            self.step_idx,
            self.allowed_tokens,
            self.reasoning_status,
            self.output_padding_offset,
            self.output_cum_offsets,
            self.enable_thinking,
            self.think_end_id,
            self.line_break_id,
        )

        logits_after = self.logits.numpy()
        # Find batch 0's token_idx
        token_idx_batch0 = 0

        # batch 0 first token should be enforced
        for vid in range(self.vocab_size):
            if vid in self.allowed_tokens.numpy():
                self.assertAlmostEqual(
                    logits_after[token_idx_batch0, vid],
                    logits_before[token_idx_batch0, vid],
                    places=5,
                )
            else:
                self.assertLess(logits_after[token_idx_batch0, vid], -1e9)

        if self.token_num > 1:
            np.testing.assert_allclose(
                logits_after[token_idx_batch0 + 1],
                logits_before[token_idx_batch0 + 1],
                rtol=1e-5,
                atol=1e-6,
            )
        np.testing.assert_equal(self.reasoning_status.numpy(), [2, 2])

    def test_empty_allowed_tokens(self):
        empty_allowed = paddle.empty([0], dtype="int64")

        logits_before = self.logits.numpy().copy()

        reasoning_phase_token_constraint(
            self.logits,
            self.pre_ids,
            self.stop_flags,
            self.seq_lens_this_time,
            self.seq_lens_encoder,
            self.step_idx,
            empty_allowed,
            self.reasoning_status,
            self.output_padding_offset,
            self.output_cum_offsets,
            self.enable_thinking,
            self.think_end_id,
            self.line_break_id,
        )

        logits_after = self.logits.numpy()

        np.testing.assert_allclose(
            logits_after,
            logits_before,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_perf_bsz128_vocab100k_status2(self):
        """
        Performance benchmark:
        bsz = 128
        vocab = 100k
        all status == 2
        all tokens are batch-first tokens
        """

        paddle.set_device("gpu")

        # ------------------------
        # config
        # ------------------------
        bs = 256
        vocab_size = 100000
        max_seq_len = 1024

        think_end_id = 9
        line_break_id = 10

        # ------------------------
        # seq / step
        # ------------------------
        step_idx = paddle.full([bs], 4, dtype="int64")
        seq_lens_this_time = paddle.full([bs], 1, dtype="int32")
        seq_lens_encoder = paddle.zeros([bs], dtype="int32")
        stop_flags = paddle.zeros([bs], dtype="bool")

        # ------------------------
        # pre_ids: force 1 -> 2 pattern
        # ------------------------
        pre_ids = np.zeros((bs, max_seq_len), dtype=np.int64)
        for i in range(bs):
            pre_ids[i, 1] = line_break_id
            pre_ids[i, 2] = think_end_id
            pre_ids[i, 3] = line_break_id
            pre_ids[i, 4] = line_break_id

        pre_ids = paddle.to_tensor(pre_ids, dtype="int64")

        # ------------------------
        # reasoning_status: start from 1
        # ------------------------
        reasoning_status = paddle.ones([bs], dtype="int32")

        # ------------------------
        # allowed tokens (small set)
        # ------------------------
        allowed_tokens = paddle.to_tensor([1, 5, 42, 999], dtype="int64")

        # ------------------------
        # speculative layout
        # each batch exactly 1 token
        # token_idx == bs_idx
        # ------------------------

        token_num = paddle.sum(seq_lens_this_time)

        seq_lens_output = paddle.full(bs, 2, dtype="int32")
        output_token_num = paddle.sum(seq_lens_output)

        useless_inputs = paddle.zeros([self.bs, self.max_seq_len], dtype="int64")
        _, output_padding_offset, output_cum_offsets, _ = get_padding_offset(
            useless_inputs,
            seq_lens_output,
            None,
            None,
            output_token_num.item(),
        )

        # ------------------------
        # logits
        # ------------------------
        logits = paddle.randn([token_num, vocab_size], dtype="float32")

        enable_thinking = paddle.ones(shape=[bs, 1], dtype="int32").astype("bool")

        # ------------------------
        # warmup
        # ------------------------
        for _ in range(5):
            reasoning_phase_token_constraint(
                logits,
                pre_ids,
                stop_flags,
                seq_lens_this_time,
                seq_lens_encoder,
                step_idx,
                allowed_tokens,
                reasoning_status,
                output_padding_offset,
                output_cum_offsets,
                enable_thinking,
                think_end_id,
                line_break_id,
            )

        paddle.device.cuda.synchronize()

        # ------------------------
        # timing
        # ------------------------
        iters = 20
        start = paddle.device.cuda.Event(enable_timing=True)
        end = paddle.device.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(iters):
            reasoning_phase_token_constraint(
                logits,
                pre_ids,
                stop_flags,
                seq_lens_this_time,
                seq_lens_encoder,
                step_idx,
                allowed_tokens,
                reasoning_status,
                output_padding_offset,
                output_cum_offsets,
                enable_thinking,
                think_end_id,
                line_break_id,
            )
        end.record()

        paddle.device.cuda.synchronize()
        elapsed_ms = paddle.device.cuda.Event.elapsed_time(start, end)
        avg_ms = elapsed_ms / iters

        print(f"[PERF] bsz={bs}, vocab={vocab_size}, " f"avg latency = {avg_ms:.3f} ms")

        # ------------------------
        # correctness spot check
        # ------------------------
        logits_np = logits.numpy()
        print(logits)
        for b in [0, 100, 200]:  # sample few batches
            for vid in range(vocab_size):
                if vid in allowed_tokens.numpy():
                    continue
                # print(f"b: {b}, vid: {vid}")
                self.assertLess(logits_np[b, vid], -1e9)


if __name__ == "__main__":
    unittest.main()
