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
        speculate_limit_thinking_content_length as xpu_op,
    )

    HAS_XPU = True
except ImportError:
    HAS_XPU = False


def ref_impl(
    next_tokens,  # [bs, tokens_per_step] int64
    max_think_lens,  # [bs] int32
    max_reply_lens,  # [bs] int32
    step_idx,  # [bs] int64
    eos_token_ids,  # [eos_len] int64
    limit_status,  # [bs] int32
    accept_num,  # [bs] int32
    stop_flags,  # [bs] bool
    think_end_id,  # int
    inject_token_ids,  # [inject_len] int64, may be empty
    splitwise_role_is_decode,
):
    next_tokens = next_tokens.copy()
    max_reply_lens = max_reply_lens.copy()
    step_idx = step_idx.copy()
    limit_status = limit_status.copy()
    accept_num = accept_num.copy()

    bs = len(accept_num)
    inject_len = len(inject_token_ids)
    eos_token_id_len = len(eos_token_ids)

    for bid in range(bs):
        original_accept_num = int(accept_num[bid])
        if original_accept_num <= 0:
            continue
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

        new_accept_num = original_accept_num
        current_base_step = int(step_idx[bid]) - original_accept_num + 1

        for token_offset in range(original_accept_num):
            next_token = int(next_tokens[bid, token_offset])
            current_step = current_base_step + token_offset

            prev_status = status
            condition_triggered = False

            # 1) 思考阶段监听 think_end_id
            if status == 0 and next_token == think_end_id:
                status = done_status
                if max_reply_len >= 0:
                    max_reply_len += 2

            # 2) 注入触发（仅 max_think_len >= 0 时）
            if max_think_len >= 0 and status < reply_base:
                if max_think_len > 0:
                    if status == 0 and (current_step - 1) == max_think_len:
                        status = 1 if inject_len > 0 else done_status
                elif max_think_len == 0:
                    if status == 0 and not splitwise_role_is_decode:
                        status = 1 if inject_len > 0 else done_status
                    elif status == 0 and splitwise_role_is_decode:
                        status = 2 if inject_len > 0 else done_status + 1

                # eos 触发注入
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
                    condition_triggered = True

            became_done_this_token = status == done_status and prev_status != done_status and prev_status < reply_base

            # 3) 回复长度限制
            if max_reply_len >= 0:
                if not became_done_this_token:
                    if status == done_status:
                        status = reply_base
                    if status >= reply_base:
                        reply_len = status - reply_base
                        if reply_len >= max_reply_len:
                            if eos_token_id_len > 0:
                                next_token = int(eos_token_ids[0])
                            status = reply_base + max_reply_len
                            condition_triggered = True
                        else:
                            status = reply_base + (reply_len + 1)

            next_tokens[bid, token_offset] = next_token

            if condition_triggered:
                new_accept_num = token_offset + 1
                break

        discarded = original_accept_num - new_accept_num
        if discarded > 0:
            step_idx[bid] -= discarded

        accept_num[bid] = new_accept_num
        limit_status[bid] = status
        max_reply_lens[bid] = max_reply_len

    return {
        "next_tokens": next_tokens,
        "max_reply_lens": max_reply_lens,
        "step_idx": step_idx,
        "limit_status": limit_status,
        "accept_num": accept_num,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────
def run_op(np_inputs, think_end_id, splitwise_role_is_decode, device, op_fn):
    """在指定 device 上运行算子，返回输出字段的 numpy dict。"""
    paddle.set_device(device)
    next_tokens = paddle.to_tensor(np_inputs["next_tokens"].copy())
    max_think_lens = paddle.to_tensor(np_inputs["max_think_lens"].copy())
    max_reply_lens = paddle.to_tensor(np_inputs["max_reply_lens"].copy())
    step_idx = paddle.to_tensor(np_inputs["step_idx"].copy())
    limit_status = paddle.to_tensor(np_inputs["limit_status"].copy())
    accept_num = paddle.to_tensor(np_inputs["accept_num"].copy())
    stop_flags = paddle.to_tensor(np_inputs["stop_flags"].copy())
    eos_token_ids = paddle.to_tensor(np_inputs["eos_token_ids"].copy())
    inject_token_ids = paddle.to_tensor(np_inputs["inject_token_ids"].copy())

    op_fn(
        next_tokens,
        max_think_lens,
        max_reply_lens,
        step_idx,
        limit_status,
        accept_num,
        stop_flags,
        eos_token_ids,
        inject_token_ids,
        think_end_id,
        splitwise_role_is_decode,
    )
    return {
        "next_tokens": next_tokens.numpy(),
        "max_reply_lens": max_reply_lens.numpy(),
        "step_idx": step_idx.numpy(),
        "limit_status": limit_status.numpy(),
        "accept_num": accept_num.numpy(),
    }


def run_ref(np_inputs, think_end_id, splitwise_role_is_decode):
    return ref_impl(
        np_inputs["next_tokens"].copy(),
        np_inputs["max_think_lens"].copy(),
        np_inputs["max_reply_lens"].copy(),
        np_inputs["step_idx"].copy(),
        np_inputs["eos_token_ids"].copy(),
        np_inputs["limit_status"].copy(),
        np_inputs["accept_num"].copy(),
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
    tokens_per_step,
    next_tokens,
    max_think_lens,
    max_reply_lens,
    step_idx,
    limit_status,
    accept_num,
    stop_flags,
    eos_token_ids,
    inject_token_ids,
):
    return {
        "next_tokens": np.array(next_tokens, dtype=np.int64).reshape(bs, tokens_per_step),
        "max_think_lens": np.array(max_think_lens, dtype=np.int32),
        "max_reply_lens": np.array(max_reply_lens, dtype=np.int32),
        "step_idx": np.array(step_idx, dtype=np.int64),
        "limit_status": np.array(limit_status, dtype=np.int32),
        "accept_num": np.array(accept_num, dtype=np.int32),
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


class TestSpeculateLimitThinkingContentLength(unittest.TestCase):
    def test_think_end_natural(self):
        """模型自然输出 think_end_id：status 0 → done_status，max_reply_len += 2。
        inject_len=0 → done_status=1, reply_base=2。"""
        np_inputs = make_inputs(
            bs=1,
            tokens_per_step=1,
            next_tokens=[THINK_END_ID],  # 模型正好输出 </think>
            max_think_lens=[-1],  # 不强制截断
            max_reply_lens=[5],
            step_idx=[3],
            limit_status=[0],
            accept_num=[1],
            stop_flags=[False],
            eos_token_ids=[EOS_ID],
            inject_token_ids=[],
        )
        run_all_and_compare(self, np_inputs, THINK_END_ID)

    def test_inject_truncation(self):
        """超过 max_think_len 触发注入序列，accept_num 截断至当前 token。
        inject_len=2 → done_status=3, reply_base=4。
        step_idx=6, current_step-1=5=max_think_len → 触发注入，token 被替换为 inject[0]=200。"""
        np_inputs = make_inputs(
            bs=1,
            tokens_per_step=1,
            next_tokens=[999],
            max_think_lens=[5],
            max_reply_lens=[-1],
            step_idx=[6],  # current_base_step=6, current_step-1=5 == max_think_len
            limit_status=[0],
            accept_num=[1],
            stop_flags=[False],
            eos_token_ids=[EOS_ID],
            inject_token_ids=[200, 201],
        )
        run_all_and_compare(self, np_inputs, THINK_END_ID)

    def test_reply_len_limit(self):
        """回复计数达到 max_reply_len 上限，强制写入 EOS，截断 accept_num。
        inject_len=0 → done_status=1, reply_base=2。
        status=4（reply_len=2=max_reply_len）→ 强制 EOS。"""
        np_inputs = make_inputs(
            bs=1,
            tokens_per_step=1,
            next_tokens=[999],
            max_think_lens=[-1],
            max_reply_lens=[2],
            step_idx=[10],
            limit_status=[4],  # reply_base(2) + reply_len(2)，已到上限
            accept_num=[1],
            stop_flags=[False],
            eos_token_ids=[EOS_ID],
            inject_token_ids=[],
        )
        run_all_and_compare(self, np_inputs, THINK_END_ID)

    def test_no_limit(self):
        """max_think_len<0 且 max_reply_len<0，整个 batch 直接跳过，输出不变。"""
        np_inputs = make_inputs(
            bs=2,
            tokens_per_step=1,
            next_tokens=[111, 222],
            max_think_lens=[-1, -1],
            max_reply_lens=[-1, -1],
            step_idx=[5, 8],
            limit_status=[0, 0],
            accept_num=[1, 1],
            stop_flags=[False, False],
            eos_token_ids=[EOS_ID],
            inject_token_ids=[],
        )
        run_all_and_compare(self, np_inputs, THINK_END_ID)

    def test_inject_len_zero_v1_behavior(self):
        """inject_len=0 退化为 v1 行为：超时直接进入 done_status=1，token 不替换。"""
        np_inputs = make_inputs(
            bs=1,
            tokens_per_step=1,
            next_tokens=[999],
            max_think_lens=[3],
            max_reply_lens=[-1],
            step_idx=[4],  # current_step-1=3 == max_think_len=3
            limit_status=[0],
            accept_num=[1],
            stop_flags=[False],
            eos_token_ids=[EOS_ID],
            inject_token_ids=[],  # inject_len=0
        )
        run_all_and_compare(self, np_inputs, THINK_END_ID)

    def test_multi_token_per_step(self):
        """tokens_per_step=3，第 2 个 token（offset=1）触发回复超限，
        前 1 个 token 保留，accept_num 截断为 2，step_idx 回退 1。
        inject_len=0 → done_status=1, reply_base=2。
        status=2（reply_len=0），max_reply_len=1：
          offset=0: reply_len=0 < 1 → 正常输出，status→3
          offset=1: reply_len=1 >= 1 → 强制 EOS，截断"""
        np_inputs = make_inputs(
            bs=1,
            tokens_per_step=3,
            next_tokens=[501, 502, 503],
            max_think_lens=[-1],
            max_reply_lens=[1],
            step_idx=[12],
            limit_status=[2],  # = reply_base, reply_len=0
            accept_num=[3],
            stop_flags=[False],
            eos_token_ids=[EOS_ID],
            inject_token_ids=[],
        )
        run_all_and_compare(self, np_inputs, THINK_END_ID)

    def test_already_stopped(self):
        """stop_flags=True 的 batch，直接跳过，输出不变。"""
        np_inputs = make_inputs(
            bs=2,
            tokens_per_step=1,
            next_tokens=[111, 222],
            max_think_lens=[5, 5],
            max_reply_lens=[10, 10],
            step_idx=[6, 6],
            limit_status=[0, 0],
            accept_num=[1, 1],
            stop_flags=[True, False],  # batch0 已停止
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
            tokens_per_step=1,
            next_tokens=[999],
            max_think_lens=[0],  # 立即触发
            max_reply_lens=[-1],
            step_idx=[1],
            limit_status=[0],
            accept_num=[1],
            stop_flags=[False],
            eos_token_ids=[EOS_ID],
            inject_token_ids=[200, 201, 202],
        )
        run_all_and_compare(self, np_inputs, THINK_END_ID, splitwise_role_is_decode=True)


if __name__ == "__main__":
    unittest.main()
