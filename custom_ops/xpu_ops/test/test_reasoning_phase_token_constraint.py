import time
import unittest

import numpy as np
import paddle

from fastdeploy.model_executor.ops.xpu import (
    get_padding_offset,
    reasoning_phase_token_constraint,
)


def _print_input(test_name, **kwargs):
    print(f"\n{'=' * 60}")
    print(f"[{test_name}] INPUT STATE")
    print(f"{'=' * 60}")
    for k, v in kwargs.items():
        if hasattr(v, "numpy"):
            val = v.numpy()
        else:
            val = v
        print(f"  {k:25s} = {val}")


def _print_status(test_name, status_before, status_after, expect_pairs):
    """expect_pairs: list of (batch_idx, expected_before, expected_after)"""
    print(f"\n{'=' * 60}")
    print(f"[{test_name}] OUTPUT STATE")
    print(f"{'=' * 60}")
    print(f"  reasoning_status (before) = {status_before}")
    print(f"  reasoning_status (after)  = {status_after}")
    for bi, eb, ea in expect_pairs:
        ok = status_after[bi] == ea
        print(
            f"    batch {bi}: {status_before[bi]} -> {status_after[bi]}  "
            f"(expect {eb} -> {ea})  {'OK' if ok else 'FAIL'}"
        )


def _print_logits_enforce(token_idx, batch_label, logits_before, logits_after, vocab_size, allowed_set):
    print(f"\n  --- token {token_idx} ({batch_label}, should be enforced) ---")
    print(f"  {'vid':>4s}  {'before':>12s}  {'after':>12s}  " f"{'allowed':>7s}  {'expect':>12s}  {'pass':>4s}")
    all_ok = True
    for vid in range(vocab_size):
        is_allowed = vid in allowed_set
        bval = logits_before[token_idx, vid]
        aval = logits_after[token_idx, vid]
        if is_allowed:
            expect_str = f"{bval:>12.5f}"
            ok = abs(aval - bval) < 1e-3
        else:
            expect_str = "    < -1e9"
            ok = aval < -1e9
        flag = "OK" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(
            f"  {vid:>4d}  {bval:>12.5f}  {aval:>12.5f}  "
            f"{'YES' if is_allowed else ' NO':>7s}  "
            f"{expect_str}  {flag:>4s}"
        )
    print(f"  token {token_idx} overall: {'PASS' if all_ok else 'FAIL'}")


def _print_logits_untouched(token_idx, batch_label, logits_before, logits_after):
    diff = np.max(np.abs(logits_after[token_idx] - logits_before[token_idx]))
    ok = diff < 1e-5
    print(f"\n  --- token {token_idx} ({batch_label}, should be untouched) ---")
    print(f"  max |after - before| = {diff:.2e}  (expect ~0)  " f"{'PASS' if ok else 'FAIL'}")


class TestReasoningPhaseTokenConstraint(unittest.TestCase):

    def setUp(self):
        paddle.set_device("xpu")

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
        # token_ids_all
        #
        # batch 0:
        #   step_idx=4, pre_ids_now[0..3]
        #   pattern: \n <think_end> \n \n  -> status 1 -> 2
        #   t3=pre_ids_now[0]=\n, t2=pre_ids_now[1]=<think_end>, t1=pre_ids_now[2]=\n, t0=pre_ids_now[3]=\n
        #
        # batch 1:
        #   contains think_end at pre_ids_now[2], but pattern not complete -> status 0 -> 1
        # ------------------------
        token_ids_all = np.zeros((self.bs, self.max_seq_len), dtype=np.int64)

        # batch 0: pattern \n <think_end> \n \n at pre_ids_now[0..3]
        token_ids_all[0, 0] = self.line_break_id
        token_ids_all[0, 1] = self.think_end_id
        token_ids_all[0, 2] = self.line_break_id
        token_ids_all[0, 3] = self.line_break_id

        # batch 1: think_end at pre_ids_now[2]
        token_ids_all[1, 2] = self.think_end_id

        self.token_ids_all = paddle.to_tensor(token_ids_all, dtype="int64")
        self.prompt_lens = paddle.zeros([self.bs], dtype="int64")

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
        _, self.output_batch_id_per_token, self.output_cu_seqlens_q, _ = get_padding_offset(
            useless_inputs,
            seq_lens_output,
            None,
            None,
            output_token_num.item(),
        )

        # ------------------------
        # logits
        # ------------------------
        np.random.seed(2024)
        logits = np.random.randn(self.token_num, self.vocab_size).astype("float32")
        self.logits = paddle.to_tensor(logits, dtype="float32")

        self.enable_thinking = paddle.to_tensor([True, True], dtype="bool")

    def test_reasoning_status_and_logits_enforce(self):
        logits_before = self.logits.numpy().copy()
        status_before = self.reasoning_status.numpy().copy()
        allowed_set = set(self.allowed_tokens.numpy().tolist())
        test_name = "test_reasoning_status_and_logits_enforce"

        _print_input(
            test_name,
            bs=self.bs,
            vocab_size=self.vocab_size,
            token_num=self.token_num,
            step_idx=self.step_idx,
            seq_lens_this_time=self.seq_lens_this_time,
            seq_lens_encoder=self.seq_lens_encoder,
            stop_flags=self.stop_flags,
            enable_thinking=self.enable_thinking,
            reasoning_status=self.reasoning_status,
            allowed_tokens=self.allowed_tokens,
            batch_id_per_token=self.output_batch_id_per_token,
            cu_seqlens_q=self.output_cu_seqlens_q,
        )

        # ------------------------
        # call custom op
        # ------------------------
        reasoning_phase_token_constraint(
            self.logits,
            self.token_ids_all,
            self.prompt_lens,
            self.stop_flags,
            self.seq_lens_this_time,
            self.seq_lens_encoder,
            self.step_idx,
            self.allowed_tokens,
            self.reasoning_status,
            self.output_batch_id_per_token,
            self.output_cu_seqlens_q,
            self.enable_thinking,
            self.think_end_id,
            self.line_break_id,
        )

        logits_after = self.logits.numpy()
        status_after = self.reasoning_status.numpy()

        _print_status(test_name, status_before, status_after, [(0, 1, 2), (1, 0, 1)])
        _print_logits_enforce(0, "batch 0", logits_before, logits_after, self.vocab_size, allowed_set)
        _print_logits_untouched(1, "batch 1", logits_before, logits_after)
        print("=" * 60)

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
        # step_idx=4, pre_ids_now[0..3]
        # think_end at pre_ids_now[2] (cur_step - 2 = 4 - 2 = 2)
        # ------------------------
        token_ids_all = np.zeros((self.bs, self.max_seq_len), dtype=np.int64)

        # batch 0: think_end at pre_ids_now[2]
        token_ids_all[0, 2] = self.think_end_id

        # batch 1: no think_end
        token_ids_all[1, :] = 0

        self.token_ids_all = paddle.to_tensor(token_ids_all, dtype="int64")

        self.reasoning_status = paddle.to_tensor([0, 0], dtype="int32")

        logits_before = self.logits.numpy().copy()
        status_before = self.reasoning_status.numpy().copy()
        test_name = "test_status_0_to_1_only"

        _print_input(
            test_name,
            reasoning_status=self.reasoning_status,
            step_idx=self.step_idx,
            token_ids_all=self.token_ids_all,
            enable_thinking=self.enable_thinking,
        )

        # ------------------------
        # call op
        # ------------------------
        reasoning_phase_token_constraint(
            self.logits,
            self.token_ids_all,
            self.prompt_lens,
            self.stop_flags,
            self.seq_lens_this_time,
            self.seq_lens_encoder,
            self.step_idx,
            self.allowed_tokens,
            self.reasoning_status,
            self.output_batch_id_per_token,
            self.output_cu_seqlens_q,
            self.enable_thinking,
            self.think_end_id,
            self.line_break_id,
        )

        status_after = self.reasoning_status.numpy()
        logits_after = self.logits.numpy()

        _print_status(test_name, status_before, status_after, [(0, 0, 1), (1, 0, 0)])

        for ti in range(logits_after.shape[0]):
            _print_logits_untouched(ti, f"batch token {ti}", logits_before, logits_after)
        print("=" * 60)

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
        status_before = self.reasoning_status.numpy().copy()
        test_name = "test_status_2_to_3_only"

        _print_input(test_name, reasoning_status=self.reasoning_status, step_idx=self.step_idx)

        reasoning_phase_token_constraint(
            self.logits,
            self.token_ids_all,
            self.prompt_lens,
            self.stop_flags,
            self.seq_lens_this_time,
            self.seq_lens_encoder,
            self.step_idx,
            self.allowed_tokens,
            self.reasoning_status,
            self.output_batch_id_per_token,
            self.output_cu_seqlens_q,
            self.enable_thinking,
            self.think_end_id,
            self.line_break_id,
        )

        status_after = self.reasoning_status.numpy()
        logits_after = self.logits.numpy()

        _print_status(test_name, status_before, status_after, [(0, 2, 3), (1, 2, 3)])

        for ti in range(logits_after.shape[0]):
            _print_logits_untouched(ti, f"batch token {ti}", logits_before, logits_after)
        print("=" * 60)

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
        # batch 0 enforce, batch 1 not enforce
        self.reasoning_status = paddle.to_tensor([1, 2], dtype="int32")

        logits_before = self.logits.numpy().copy()
        status_before = self.reasoning_status.numpy().copy()
        allowed_set = set(self.allowed_tokens.numpy().tolist())
        test_name = "test_status_1_to_2"

        _print_input(
            test_name,
            reasoning_status=self.reasoning_status,
            step_idx=self.step_idx,
            allowed_tokens=self.allowed_tokens,
            batch_id_per_token=self.output_batch_id_per_token,
            cu_seqlens_q=self.output_cu_seqlens_q,
        )

        reasoning_phase_token_constraint(
            self.logits,
            self.token_ids_all,
            self.prompt_lens,
            self.stop_flags,
            self.seq_lens_this_time,
            self.seq_lens_encoder,
            self.step_idx,
            self.allowed_tokens,
            self.reasoning_status,
            self.output_batch_id_per_token,
            self.output_cu_seqlens_q,
            self.enable_thinking,
            self.think_end_id,
            self.line_break_id,
        )

        logits_after = self.logits.numpy()
        status_after = self.reasoning_status.numpy()

        _print_status(test_name, status_before, status_after, [(0, 1, 2), (1, 2, 3)])

        token_idx_batch0 = 0
        _print_logits_enforce(
            token_idx_batch0, "batch 0 first token", logits_before, logits_after, self.vocab_size, allowed_set
        )
        _print_logits_untouched(token_idx_batch0 + 1, "batch 0 second token", logits_before, logits_after)

        print(
            f"\n  reasoning_status final = {status_after}  "
            f"(expect [2, 3])  "
            f"{'PASS' if np.array_equal(status_after, [2, 3]) else 'FAIL'}"
        )
        print("=" * 60)

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

        # batch 0 second token (if exists) must be untouched
        if self.token_num > 1:
            np.testing.assert_allclose(
                logits_after[token_idx_batch0 + 1],
                logits_before[token_idx_batch0 + 1],
                rtol=1e-5,
                atol=1e-6,
            )
        np.testing.assert_equal(self.reasoning_status.numpy(), [2, 3])

    def test_status_0_to_2(self):
        # batch 0 enforce, batch 1 not enforce
        self.reasoning_status = paddle.to_tensor([0, 0], dtype="int32")
        self.enable_thinking = paddle.to_tensor([False, False], dtype="bool")

        self.step_idx = paddle.to_tensor([0, 0], dtype="int64")

        self.seq_lens_this_time = paddle.to_tensor([15, 15], dtype="int32")
        self.seq_lens_encoder = paddle.to_tensor([15, 15], dtype="int32")

        logits_before = self.logits.numpy().copy()
        status_before = self.reasoning_status.numpy().copy()
        allowed_set = set(self.allowed_tokens.numpy().tolist())
        test_name = "test_status_0_to_2"

        _print_input(
            test_name,
            reasoning_status=self.reasoning_status,
            enable_thinking=self.enable_thinking,
            step_idx=self.step_idx,
            seq_lens_this_time=self.seq_lens_this_time,
            seq_lens_encoder=self.seq_lens_encoder,
            allowed_tokens=self.allowed_tokens,
            batch_id_per_token=self.output_batch_id_per_token,
            cu_seqlens_q=self.output_cu_seqlens_q,
        )

        reasoning_phase_token_constraint(
            self.logits,
            self.token_ids_all,
            self.prompt_lens,
            self.stop_flags,
            self.seq_lens_this_time,
            self.seq_lens_encoder,
            self.step_idx,
            self.allowed_tokens,
            self.reasoning_status,
            self.output_batch_id_per_token,
            self.output_cu_seqlens_q,
            self.enable_thinking,
            self.think_end_id,
            self.line_break_id,
        )

        logits_after = self.logits.numpy()
        status_after = self.reasoning_status.numpy()

        _print_status(test_name, status_before, status_after, [(0, 0, 2), (1, 0, 2)])

        token_idx_batch0 = 0
        _print_logits_enforce(
            token_idx_batch0, "batch 0 first token", logits_before, logits_after, self.vocab_size, allowed_set
        )
        if self.token_num > 1:
            _print_logits_untouched(token_idx_batch0 + 1, "batch 0 second token", logits_before, logits_after)

        print(
            f"\n  reasoning_status final = {status_after}  "
            f"(expect [2, 2])  "
            f"{'PASS' if np.array_equal(status_after, [2, 2]) else 'FAIL'}"
        )
        print("=" * 60)

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
        status_before = self.reasoning_status.numpy().copy()
        test_name = "test_empty_allowed_tokens"

        _print_input(test_name, reasoning_status=self.reasoning_status, allowed_tokens_len=0)

        reasoning_phase_token_constraint(
            self.logits,
            self.token_ids_all,
            self.prompt_lens,
            self.stop_flags,
            self.seq_lens_this_time,
            self.seq_lens_encoder,
            self.step_idx,
            empty_allowed,
            self.reasoning_status,
            self.output_batch_id_per_token,
            self.output_cu_seqlens_q,
            self.enable_thinking,
            self.think_end_id,
            self.line_break_id,
        )

        logits_after = self.logits.numpy()
        status_after = self.reasoning_status.numpy()

        _print_status(test_name, status_before, status_after, [(0, 1, 2), (1, 0, 1)])

        for ti in range(logits_after.shape[0]):
            _print_logits_untouched(ti, f"token {ti}", logits_before, logits_after)
        print("=" * 60)

        np.testing.assert_allclose(
            logits_after,
            logits_before,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_perf_bsz128_vocab100k_status2(self):
        """
        Performance benchmark:
        bsz = 256
        vocab = 100k
        all status == 1 -> 2
        all tokens are batch-first tokens
        """

        paddle.set_device("xpu")

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
        # token_ids_all: force 1 -> 2 pattern
        # step_idx=4, pre_ids_now[0..3]
        # pattern: t3=\n, t2=<think_end>, t1=\n, t0=\n
        # ------------------------
        token_ids_all_np = np.zeros((bs, max_seq_len), dtype=np.int64)
        for i in range(bs):
            token_ids_all_np[i, 0] = line_break_id
            token_ids_all_np[i, 1] = think_end_id
            token_ids_all_np[i, 2] = line_break_id
            token_ids_all_np[i, 3] = line_break_id

        token_ids_all = paddle.to_tensor(token_ids_all_np, dtype="int64")
        prompt_lens = paddle.zeros([bs], dtype="int64")

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
        # ------------------------
        token_num = paddle.sum(seq_lens_this_time)

        seq_lens_output = paddle.full([bs], 2, dtype="int32")
        output_token_num = paddle.sum(seq_lens_output)

        useless_inputs = paddle.zeros([bs, max_seq_len], dtype="int64")
        _, batch_id_per_token, cu_seqlens_q, _ = get_padding_offset(
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

        enable_thinking = paddle.ones([bs], dtype="bool")

        # ------------------------
        # warmup
        # ------------------------
        for _ in range(5):
            reasoning_phase_token_constraint(
                logits,
                token_ids_all,
                prompt_lens,
                stop_flags,
                seq_lens_this_time,
                seq_lens_encoder,
                step_idx,
                allowed_tokens,
                reasoning_status,
                batch_id_per_token,
                cu_seqlens_q,
                enable_thinking,
                think_end_id,
                line_break_id,
            )

        paddle.device.xpu.synchronize()

        # ------------------------
        # timing
        # ------------------------
        iters = 20
        start = time.time()
        for _ in range(iters):
            reasoning_phase_token_constraint(
                logits,
                token_ids_all,
                prompt_lens,
                stop_flags,
                seq_lens_this_time,
                seq_lens_encoder,
                step_idx,
                allowed_tokens,
                reasoning_status,
                batch_id_per_token,
                cu_seqlens_q,
                enable_thinking,
                think_end_id,
                line_break_id,
            )

        paddle.device.xpu.synchronize()
        elapsed_ms = (time.time() - start) * 1000
        avg_ms = elapsed_ms / iters

        print(f"[PERF] bsz={bs}, vocab={vocab_size}, " f"avg latency = {avg_ms:.3f} ms")

        # ------------------------
        # correctness spot check
        # ------------------------
        logits_np = logits.numpy()
        for b in [0, 100, 200]:  # sample few batches
            for vid in range(vocab_size):
                if vid in allowed_tokens.numpy():
                    continue
                self.assertLess(logits_np[b, vid], -1e9)


if __name__ == "__main__":
    unittest.main()
