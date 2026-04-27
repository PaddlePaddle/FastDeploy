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

import os
import unittest

import numpy as np
import paddle

LIMIT_THINKING_TEST_DEBUG = os.environ.get("LIMIT_THINKING_TEST_DEBUG", "0") == "1"

try:
    from fastdeploy.model_executor.ops.xpu import (
        limit_thinking_content_length as xpu_op,
    )

    HAS_XPU = True
except ImportError:
    HAS_XPU = False


def ref_impl(
    next_tokens,  # [bs] int64
    max_think_lens,  # [bs] int32
    max_reply_lens,  # [bs] int32
    step_idx,  # [bs] int64
    eos_token_ids,  # [eos_len] int64
    limit_status,  # [bs] int32
    stop_flags,  # [bs] bool
    think_end_id,  # int
    inject_token_ids,  # [inject_len] int64, may be empty
    splitwise_role_is_decode,
):
    next_tokens = next_tokens.copy()
    max_reply_lens = max_reply_lens.copy()
    limit_status = limit_status.copy()

    bs = len(next_tokens)
    inject_len = len(inject_token_ids)
    eos_token_id_len = len(eos_token_ids)

    for bid in range(bs):
        if stop_flags[bid]:
            continue

        max_think_len = int(max_think_lens[bid])
        max_reply_len = int(max_reply_lens[bid])
        if max_think_len < 0 and max_reply_len < 0:
            continue

        done_status = (inject_len + 1) if inject_len > 0 else 1
        reply_base = done_status + 1

        status = int(limit_status[bid])
        if status < 0:
            status = 0
        prev_status = status

        next_token = int(next_tokens[bid])
        step = int(step_idx[bid])

        # 1) 思考阶段：永远监听 think_end_id
        if status == 0 and next_token == think_end_id:
            status = done_status
            if max_reply_len >= 0:
                max_reply_len += 2

        # 2) 仅当启用"思考截断"(max_think_len >= 0)时触发注入
        if max_think_len >= 0 and status < reply_base:
            if max_think_len > 0:
                if status == 0 and step == max_think_len:
                    status = 1 if inject_len > 0 else done_status
            elif max_think_len == 0:
                if status == 0 and not splitwise_role_is_decode:
                    status = 1 if inject_len > 0 else done_status
                elif status == 0 and splitwise_role_is_decode:
                    status = 2 if inject_len > 0 else done_status + 1

            # eos 提前触发注入
            if status == 0 and inject_len > 0:
                for i in range(eos_token_id_len):
                    if eos_token_ids[i] == next_token:
                        status = 1
                        break

            # 注入序列
            if inject_len > 0 and 1 <= status <= inject_len:
                next_token = int(inject_token_ids[status - 1])
                status += 1
                if status > done_status:
                    status = done_status

        became_done_this_step = status == done_status and prev_status != done_status and prev_status < reply_base

        # 3) 回复长度限制
        if max_reply_len >= 0:
            if not became_done_this_step:
                if status == done_status:
                    status = reply_base
                if status >= reply_base:
                    reply_len = status - reply_base
                    if reply_len >= max_reply_len:
                        if eos_token_id_len > 0:
                            next_token = int(eos_token_ids[0])
                        status = reply_base + max_reply_len
                    else:
                        status = reply_base + (reply_len + 1)

        next_tokens[bid] = next_token
        limit_status[bid] = status
        max_reply_lens[bid] = max_reply_len

    return {
        "next_tokens": next_tokens,
        "max_reply_lens": max_reply_lens,
        "limit_status": limit_status,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────
def run_op(np_inputs, think_end_id, splitwise_role_is_decode, device, op_fn):
    paddle.set_device(device)
    next_tokens = paddle.to_tensor(np_inputs["next_tokens"].copy())
    max_think_lens = paddle.to_tensor(np_inputs["max_think_lens"].copy())
    max_reply_lens = paddle.to_tensor(np_inputs["max_reply_lens"].copy())
    step_idx = paddle.to_tensor(np_inputs["step_idx"].copy())
    limit_status = paddle.to_tensor(np_inputs["limit_status"].copy())
    stop_flags = paddle.to_tensor(np_inputs["stop_flags"].copy())
    eos_token_ids = paddle.to_tensor(np_inputs["eos_token_ids"].copy())
    inject_token_ids = paddle.to_tensor(np_inputs["inject_token_ids"].copy())

    op_fn(
        next_tokens,
        max_think_lens,
        max_reply_lens,
        step_idx,
        limit_status,
        stop_flags,
        eos_token_ids,
        inject_token_ids,
        think_end_id,
        splitwise_role_is_decode,
    )
    return {
        "next_tokens": next_tokens.numpy(),
        "max_reply_lens": max_reply_lens.numpy(),
        "limit_status": limit_status.numpy(),
    }


def run_ref(np_inputs, think_end_id, splitwise_role_is_decode):
    return ref_impl(
        np_inputs["next_tokens"].copy(),
        np_inputs["max_think_lens"].copy(),
        np_inputs["max_reply_lens"].copy(),
        np_inputs["step_idx"].copy(),
        np_inputs["eos_token_ids"].copy(),
        np_inputs["limit_status"].copy(),
        np_inputs["stop_flags"].copy(),
        think_end_id,
        np_inputs["inject_token_ids"].copy(),
        splitwise_role_is_decode,
    )


def assert_equal(expected, actual, label):
    for key in expected:
        np.testing.assert_array_equal(
            expected[key],
            actual[key],
            err_msg=f"[{label}] field='{key}' mismatch",
        )


def make_inputs(
    bs,
    next_tokens,
    max_think_lens,
    max_reply_lens,
    step_idx,
    limit_status,
    stop_flags,
    eos_token_ids,
    inject_token_ids,
):
    return {
        "next_tokens": np.array(next_tokens, dtype=np.int64),
        "max_think_lens": np.array(max_think_lens, dtype=np.int32),
        "max_reply_lens": np.array(max_reply_lens, dtype=np.int32),
        "step_idx": np.array(step_idx, dtype=np.int64),
        "limit_status": np.array(limit_status, dtype=np.int32),
        "stop_flags": np.array(stop_flags, dtype=bool),
        "eos_token_ids": np.array(eos_token_ids, dtype=np.int64),
        "inject_token_ids": np.array(inject_token_ids, dtype=np.int64),
    }


def run_all_and_compare(test_case, np_inputs, think_end_id, splitwise_role_is_decode=False):
    if LIMIT_THINKING_TEST_DEBUG:
        print("\n========== [INPUT] ==========")
        print(f"  think_end_id            : {think_end_id}")
        print(f"  splitwise_role_is_decode: {splitwise_role_is_decode}")
        for k, v in np_inputs.items():
            print(f"  {k:25s}: {v}")

    ref_out = run_ref(np_inputs, think_end_id, splitwise_role_is_decode)

    if LIMIT_THINKING_TEST_DEBUG:
        print("---------- [REF OUTPUT] ----------")
        for k, v in ref_out.items():
            print(f"  {k:25s}: {v}")

    if HAS_XPU:
        xpu_out = run_op(np_inputs, think_end_id, splitwise_role_is_decode, "xpu:0", xpu_op)

        if LIMIT_THINKING_TEST_DEBUG:
            print("---------- [XPU OUTPUT] ----------")
            for k, v in xpu_out.items():
                print(f"  {k:25s}: {v}")

        assert_equal(ref_out, xpu_out, "XPU vs ref")
    else:
        test_case.skipTest("XPU is not available; only ref logic verified.")


# ─────────────────────────────────────────────────────────────────────────────
# 测试用例
# ─────────────────────────────────────────────────────────────────────────────
THINK_END_ID = 100
EOS_ID = 2


class TestLimitThinkingContentLength(unittest.TestCase):
    def test_think_end_natural(self):
        """模型自然输出 think_end_id：status 0 → done_status，max_reply_len += 2。
        inject_len=0 → done_status=1, reply_base=2。"""
        np_inputs = make_inputs(
            bs=1,
            next_tokens=[THINK_END_ID],
            max_think_lens=[-1],
            max_reply_lens=[5],
            step_idx=[3],
            limit_status=[0],
            stop_flags=[False],
            eos_token_ids=[EOS_ID],
            inject_token_ids=[],
        )
        run_all_and_compare(self, np_inputs, THINK_END_ID)

    def test_inject_truncation(self):
        """超过 max_think_len 触发注入序列。
        inject_len=2 → done_status=3, reply_base=4。
        step=5=max_think_len → 触发注入，token 被替换为 inject[0]=200，status→1→2。"""
        np_inputs = make_inputs(
            bs=1,
            next_tokens=[999],
            max_think_lens=[5],
            max_reply_lens=[-1],
            step_idx=[5],  # step == max_think_len
            limit_status=[0],
            stop_flags=[False],
            eos_token_ids=[EOS_ID],
            inject_token_ids=[200, 201],
        )
        run_all_and_compare(self, np_inputs, THINK_END_ID)

    def test_reply_len_limit(self):
        """回复计数达到 max_reply_len 上限，强制写入 EOS。
        inject_len=0 → done_status=1, reply_base=2。
        status=4（reply_len=2=max_reply_len）→ 强制 EOS。"""
        np_inputs = make_inputs(
            bs=1,
            next_tokens=[999],
            max_think_lens=[-1],
            max_reply_lens=[2],
            step_idx=[10],
            limit_status=[4],  # reply_base(2) + reply_len(2)，已到上限
            stop_flags=[False],
            eos_token_ids=[EOS_ID],
            inject_token_ids=[],
        )
        run_all_and_compare(self, np_inputs, THINK_END_ID)

    def test_no_limit(self):
        """max_think_len<0 且 max_reply_len<0，整个 batch 直接跳过，输出不变。"""
        np_inputs = make_inputs(
            bs=2,
            next_tokens=[111, 222],
            max_think_lens=[-1, -1],
            max_reply_lens=[-1, -1],
            step_idx=[5, 8],
            limit_status=[0, 0],
            stop_flags=[False, False],
            eos_token_ids=[EOS_ID],
            inject_token_ids=[],
        )
        run_all_and_compare(self, np_inputs, THINK_END_ID)

    def test_inject_len_zero_behavior(self):
        """inject_len=0：超时直接进入 done_status=1，token 不替换。
        done_status=1, reply_base=2。"""
        np_inputs = make_inputs(
            bs=1,
            next_tokens=[999],
            max_think_lens=[3],
            max_reply_lens=[-1],
            step_idx=[3],  # step == max_think_len
            limit_status=[0],
            stop_flags=[False],
            eos_token_ids=[EOS_ID],
            inject_token_ids=[],
        )
        run_all_and_compare(self, np_inputs, THINK_END_ID)

    def test_already_stopped(self):
        """stop_flags=True 的 batch，直接跳过，输出不变。"""
        np_inputs = make_inputs(
            bs=2,
            next_tokens=[111, 222],
            max_think_lens=[5, 5],
            max_reply_lens=[10, 10],
            step_idx=[6, 6],
            limit_status=[0, 0],
            stop_flags=[True, False],
            eos_token_ids=[EOS_ID],
            inject_token_ids=[],
        )
        run_all_and_compare(self, np_inputs, THINK_END_ID)

    def test_splitwise_decode_node(self):
        """splitwise_role_is_decode=True 且 max_think_len=0：
        D 节点从 inject_token_ids[1] 开始注入（status 直接跳到 2）。
        inject_len=3 → done_status=4, reply_base=5。"""
        np_inputs = make_inputs(
            bs=1,
            next_tokens=[999],
            max_think_lens=[0],
            max_reply_lens=[-1],
            step_idx=[1],
            limit_status=[0],
            stop_flags=[False],
            eos_token_ids=[EOS_ID],
            inject_token_ids=[200, 201, 202],
        )
        run_all_and_compare(self, np_inputs, THINK_END_ID, splitwise_role_is_decode=True)

    def test_eos_early_in_thinking(self):
        """思考阶段模型提前输出 EOS，触发注入序列覆盖 eos。
        inject_len=3 → done_status=4, reply_base=5。
        next_token=EOS(2) → status=1 → inject[0]=200。"""
        np_inputs = make_inputs(
            bs=1,
            next_tokens=[EOS_ID],  # 模型提前输出 EOS
            max_think_lens=[10],
            max_reply_lens=[-1],
            step_idx=[3],
            limit_status=[0],
            stop_flags=[False],
            eos_token_ids=[EOS_ID],
            inject_token_ids=[200, 201, 202],
        )
        run_all_and_compare(self, np_inputs, THINK_END_ID)

    def test_inject_full_sequence(self):
        """连续 3 步完成注入序列，验证每步状态推进。
        inject_len=3 → done_status=4, reply_base=5。
        step=5: status 0→1, token→inject[0]
        step=6: status 1→2, token→inject[1]
        step=7: status 2→3, token→inject[2]"""
        inject_ids = [200, 201, 202]
        for step, expected_status_in, expected_status_out, expected_token in [
            (5, 0, 2, 200),  # status 0→1→2 (注入 inject[0] 后 status+1)
            (6, 2, 3, 201),  # status 2→3 (注入 inject[1])
            (7, 3, 4, 202),  # status 3→4=done_status (注入 inject[2])
        ]:
            np_inputs = make_inputs(
                bs=1,
                next_tokens=[999],
                max_think_lens=[5],
                max_reply_lens=[-1],
                step_idx=[step],
                limit_status=[expected_status_in],
                stop_flags=[False],
                eos_token_ids=[EOS_ID],
                inject_token_ids=inject_ids,
            )
            ref_out = run_ref(np_inputs, THINK_END_ID, False)
            self.assertEqual(
                ref_out["next_tokens"][0], expected_token, f"step={step}: expected token {expected_token}"
            )
            self.assertEqual(
                ref_out["limit_status"][0], expected_status_out, f"step={step}: expected status {expected_status_out}"
            )

    def test_became_done_not_count_reply(self):
        """刚进入 done_status 的这一步不计入回复。
        inject_len=0 → done_status=1, reply_base=2。
        status=0, next_token=think_end_id → became_done_this_step=True，
        不切换到 reply_base，回复计数从下一步才开始。"""
        np_inputs = make_inputs(
            bs=1,
            next_tokens=[THINK_END_ID],
            max_think_lens=[-1],
            max_reply_lens=[3],
            step_idx=[8],
            limit_status=[0],
            stop_flags=[False],
            eos_token_ids=[EOS_ID],
            inject_token_ids=[],
        )
        run_all_and_compare(self, np_inputs, THINK_END_ID)

    def test_multi_batch_mixed(self):
        """多 batch 混合场景：不同状态、不同限制。
        batch0: 思考中，超时触发注入
        batch1: 已在回复阶段，回复超限强制 EOS
        batch2: 无限制，直接跳过"""
        np_inputs = make_inputs(
            bs=3,
            next_tokens=[999, 888, 777],
            max_think_lens=[5, -1, -1],
            max_reply_lens=[-1, 2, -1],
            step_idx=[5, 12, 6],
            limit_status=[0, 4, 0],  # batch1: reply_base(2)+reply_len(2)=4
            stop_flags=[False, False, False],
            eos_token_ids=[EOS_ID],
            inject_token_ids=[200, 201],
        )
        run_all_and_compare(self, np_inputs, THINK_END_ID)

    def test_negative_status_reset(self):
        """limit_status < 0 时重置为 0 再处理。"""
        np_inputs = make_inputs(
            bs=1,
            next_tokens=[999],
            max_think_lens=[5],
            max_reply_lens=[-1],
            step_idx=[5],
            limit_status=[-1],  # 负值，应重置为 0
            stop_flags=[False],
            eos_token_ids=[EOS_ID],
            inject_token_ids=[200, 201],
        )
        run_all_and_compare(self, np_inputs, THINK_END_ID)

    def test_reply_len_progressive_counting(self):
        """逐步推进回复计数，验证 status 递增直到超限。
        inject_len=0 → done_status=1, reply_base=2。
        max_reply_len=2:
          step1: status=done_status(1) → reply_base(2), reply_len=0<2 → status=3
          step2: status=3, reply_len=1<2 → status=4
          step3: status=4, reply_len=2>=2 → 强制EOS, status=4"""
        for step, status_in, expected_status, expected_token in [
            (10, 1, 3, 999),  # done→reply_base(2), reply_len=0→status=3
            (11, 3, 4, 999),  # reply_len=1→status=4
            (12, 4, 4, EOS_ID),  # reply_len=2>=2, force EOS
        ]:
            np_inputs = make_inputs(
                bs=1,
                next_tokens=[999],
                max_think_lens=[-1],
                max_reply_lens=[2],
                step_idx=[step],
                limit_status=[status_in],
                stop_flags=[False],
                eos_token_ids=[EOS_ID],
                inject_token_ids=[],
            )
            ref_out = run_ref(np_inputs, THINK_END_ID, False)
            self.assertEqual(
                ref_out["next_tokens"][0], expected_token, f"step={step}: expected token {expected_token}"
            )
            self.assertEqual(
                ref_out["limit_status"][0], expected_status, f"step={step}: expected status {expected_status}"
            )


if __name__ == "__main__":
    unittest.main()
