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

import time

import numpy as np
import paddle

from fastdeploy.config import EarlyStopConfig
from fastdeploy.model_executor.layers.sample.early_stopper import RepetitionEarlyStopper

paddle.set_device("gpu")
np.random.seed(2025)
paddle.seed(2025)


def simulate_step_probs(
    batch_size, early_stop_batch_id, fixed_token_id, vocab_size, step_i, trigger_flags, high_prob=0.99
):
    """
    Generate a probability distribution for the specified batch of samples,
    some samples start to have "high confidence" after some step_i,
    high_prob is the confidence of the target token (such as 0.95).
    """
    probs = np.random.rand(batch_size, vocab_size).astype("float32")
    probs /= probs.sum(axis=1, keepdims=True)

    for i in range(batch_size):
        if step_i >= trigger_flags[i]:
            low_prob = (1.0 - high_prob) / (vocab_size - 1)
            probs[i].fill(low_prob)
            if i == early_stop_batch_id:
                probs[i, fixed_token_id] = high_prob
    return probs


def remove_min_max(lst):
    """
    remove the min and max value
    """
    if len(lst) < 2:
        return lst
    min_val = min(lst)
    max_val = max(lst)
    return [x for x in lst if x != min_val and x != max_val]


def test_repetition_early_stopper():
    # This test only for 1 batch to trigger early stop
    batch_size = 20
    vocab_size = 16
    window_size = 4
    threshold = 0.9
    eos_token_id = vocab_size
    max_steps = 10

    # Select a token as final token
    fixed_token_id = np.random.randint(0, vocab_size)
    # Set a batch to trigger early stop
    early_stop_batch_id = np.random.randint(0, batch_size)
    print(f"{fixed_token_id=}\n{early_stop_batch_id=}\n{eos_token_id=}")

    # Determine the first step in each batch where the high probability starts to appear
    trigger_step_flags = [[i, np.random.randint(0, max_steps + 1)] for i in range(batch_size)]
    trigger_step_flags = dict(trigger_step_flags)
    cfg = EarlyStopConfig({"enable_early_stop": True, "window_size": window_size, "threshold": threshold})
    stopper = RepetitionEarlyStopper()
    stopper.initialize(batch_size, cfg)

    next_tokens = paddle.randint(0, vocab_size, shape=[batch_size, 1], dtype="int64")
    next_tokens[early_stop_batch_id, 0] = fixed_token_id

    print(f"{next_tokens=}\ntrigger_start={trigger_step_flags[early_stop_batch_id]}")

    triggered_step = [None] * batch_size
    stop_flags = paddle.zeros_like(next_tokens)
    for step in range(max_steps):
        print(f"\n===== Step {step} =====")
        flags = [trigger_step_flags[i] for i in range(batch_size)]
        probs_np = simulate_step_probs(batch_size, early_stop_batch_id, fixed_token_id, vocab_size, step, flags)
        probs = paddle.to_tensor(probs_np)
        print("Before process:")
        print("tokens:\n", stop_flags.numpy().T)

        stopper.process(probs, next_tokens, stop_flags)

        print("After process:")
        print("tokens:\n", stop_flags.numpy().T)

        out_np = stop_flags.numpy()
        for i in range(batch_size):
            if out_np[i, 0] and triggered_step[i] is None:
                triggered_step[i] = step

    # Show which step trigger the early stop in batch i
    print("trigger_step: ", triggered_step)
    assert (
        triggered_step[early_stop_batch_id] == trigger_step_flags[early_stop_batch_id] + window_size - 1
    ), "not expected trigger step"


def test_consistency():
    batch_size = 20
    vocab_size = 103424
    window_size = 3000
    threshold = 0.9
    eos_token_id = vocab_size
    max_steps = 10

    fixed_token_id = np.random.randint(0, vocab_size)
    early_stop_batch_id = np.random.randint(0, batch_size)

    trigger_step_flags = [[i, np.random.randint(0, max_steps + 1)] for i in range(batch_size)]
    trigger_step_flags = dict(trigger_step_flags)
    cfg = EarlyStopConfig({"enable_early_stop": True, "window_size": window_size, "threshold": threshold})
    stopper_normal = RepetitionEarlyStopper()
    stopper_normal.initialize(batch_size, cfg)
    stopper_triton = RepetitionEarlyStopper()
    stopper_triton.initialize(batch_size, cfg)

    next_tokens_normal = paddle.randint(0, vocab_size, shape=[batch_size, 1], dtype="int64")
    next_tokens_triton = next_tokens_normal.clone()

    next_tokens_normal[early_stop_batch_id, 0] = fixed_token_id
    next_tokens_triton[early_stop_batch_id, 0] = fixed_token_id

    stop_flags_normal = paddle.zeros_like(next_tokens_normal)
    stop_flags_triton = stop_flags_normal.clone()

    triggered_step_normal = [None] * batch_size
    triggered_step_triton = [None] * batch_size

    for step in range(max_steps):

        flags = [trigger_step_flags[i] for i in range(batch_size)]
        probs_np = simulate_step_probs(batch_size, early_stop_batch_id, fixed_token_id, vocab_size, step, flags)
        probs = paddle.to_tensor(probs_np)

        stopper_normal.process_normal(probs, next_tokens_normal, stop_flags_normal)
        stopper_triton.process_triton(probs, next_tokens_triton, stop_flags_triton)

        assert np.allclose(stop_flags_normal.numpy(), stop_flags_triton.numpy()), f"stop flags mismatch at step {step}"

        trunc_scores_diff = paddle.abs(stopper_normal.trunc_scores - stopper_triton.trunc_scores)
        assert paddle.all(trunc_scores_diff < 1e-5), f"trunc_scores mismatch at step {step}"

        out_normal = stop_flags_normal.numpy()
        out_triton = stop_flags_triton.numpy()
        for i in range(batch_size):
            if out_normal[i, 0] == eos_token_id and triggered_step_normal[i] is None:
                triggered_step_normal[i] = step
            if out_triton[i, 0] == eos_token_id and triggered_step_triton[i] is None:
                triggered_step_triton[i] = step

    for i in range(batch_size):
        expected = triggered_step_normal[i]
        actual = triggered_step_triton[i]
        assert expected == actual, f"Sample {i} triggered at different steps: {expected} vs {actual}"

    print("[consistency]Triton vs Normal: All tokens, states, and trigger timings match.")


def test_consistency_with_real_batch_size():
    batch_size = 20
    real_batch_size = 15
    vocab_size = 103424
    window_size = 3000
    threshold = 0.9
    eos_token_id = vocab_size
    max_steps = 10

    fixed_token_id = np.random.randint(0, vocab_size)
    early_stop_batch_id = np.random.randint(0, real_batch_size)

    trigger_step_flags = [[i, np.random.randint(0, max_steps + 1)] for i in range(batch_size)]
    trigger_step_flags = dict(trigger_step_flags)
    cfg = EarlyStopConfig({"enable_early_stop": True, "window_size": window_size, "threshold": threshold})
    stopper_normal = RepetitionEarlyStopper()
    stopper_normal.initialize(batch_size, cfg)
    stopper_triton = RepetitionEarlyStopper()
    stopper_triton.initialize(batch_size, cfg)

    next_tokens_normal = paddle.randint(0, vocab_size, shape=[real_batch_size, 1], dtype="int64")
    next_tokens_triton = next_tokens_normal.clone()

    next_tokens_normal[early_stop_batch_id, 0] = fixed_token_id
    next_tokens_triton[early_stop_batch_id, 0] = fixed_token_id

    stop_flags_normal = paddle.zeros_like(next_tokens_normal)
    stop_flags_triton = stop_flags_normal.clone()

    triggered_step_normal = [None] * batch_size
    triggered_step_triton = [None] * batch_size

    for step in range(max_steps):

        flags = [trigger_step_flags[i] for i in range(real_batch_size)]
        probs_np = simulate_step_probs(real_batch_size, early_stop_batch_id, fixed_token_id, vocab_size, step, flags)
        probs = paddle.to_tensor(probs_np)

        stopper_normal.process_normal(probs, next_tokens_normal, stop_flags_normal)
        stopper_triton.process_triton(probs, next_tokens_triton, stop_flags_triton)

        assert np.allclose(stop_flags_normal.numpy(), stop_flags_triton.numpy()), f"stop flags mismatch at step {step}"

        trunc_scores_diff = paddle.abs(stopper_normal.trunc_scores - stopper_triton.trunc_scores)
        assert paddle.all(trunc_scores_diff < 1e-5), f"trunc_scores mismatch at step {step}"

        out_normal = stop_flags_normal.numpy()
        out_triton = stop_flags_triton.numpy()
        for i in range(real_batch_size):
            if out_normal[i, 0] == eos_token_id and triggered_step_normal[i] is None:
                triggered_step_normal[i] = step
            if out_triton[i, 0] == eos_token_id and triggered_step_triton[i] is None:
                triggered_step_triton[i] = step

    for i in range(batch_size):
        expected = triggered_step_normal[i]
        actual = triggered_step_triton[i]
        assert expected == actual, f"Sample {i} triggered at different steps: {expected} vs {actual}"

    print("[consistency_with_real_batch_size]Triton vs Normal: All tokens, states, and trigger timings match.")


def test_performance():
    batch_size = 256
    vocab_size = 103424
    window_size = 3000
    threshold = 0.9
    eos_token_id = vocab_size
    max_steps = 50

    fixed_token_id = np.random.randint(0, vocab_size)
    early_stop_batch_id = np.random.randint(0, batch_size)
    print(f"{fixed_token_id=}\n{early_stop_batch_id=}")

    trigger_step_flags = [[i, np.random.randint(0, max_steps + 1)] for i in range(batch_size)]
    trigger_step_flags = dict(trigger_step_flags)

    next_tokens = paddle.randint(0, vocab_size, shape=[batch_size, 1], dtype="int64")
    next_tokens[early_stop_batch_id, 0] = fixed_token_id
    cfg = EarlyStopConfig({"enable_early_stop": True, "window_size": window_size, "threshold": threshold})
    print("Testing performance triton...")
    seconds = []
    stopper = RepetitionEarlyStopper()
    stopper.initialize(batch_size, cfg)
    stop_flags = paddle.zeros_like(next_tokens)
    for step in range(max_steps):
        flags = [trigger_step_flags[i] for i in range(batch_size)]
        probs_np = simulate_step_probs(batch_size, early_stop_batch_id, fixed_token_id, vocab_size, step, flags)
        probs = paddle.to_tensor(probs_np)
        s = time.perf_counter()
        stopper.process_triton(probs, next_tokens, stop_flags)
        e = time.perf_counter()
        seconds.append(e - s)
    print(
        f"triton:\nexecute times: {max_steps}\ntotal execution time: {np.sum(seconds)*1000} ms \navg every step execution time: {np.mean(remove_min_max(seconds))*1000} ms"
    )

    print("Testing performance normal...")
    seconds = []
    stopper = RepetitionEarlyStopper()
    stopper.initialize(batch_size, cfg)
    stop_flags = paddle.zeros_like(next_tokens)
    for step in range(max_steps):
        flags = [trigger_step_flags[i] for i in range(batch_size)]
        probs_np = simulate_step_probs(batch_size, early_stop_batch_id, fixed_token_id, vocab_size, step, flags)
        probs = paddle.to_tensor(probs_np)
        s = time.perf_counter()
        stopper.process_normal(probs, next_tokens, stop_flags)
        e = time.perf_counter()
        seconds.append(e - s)
    print(
        f"normal:\nexecute times: {max_steps}\ntotal execution time: {np.sum(seconds)*1000} ms \navg every step execution time: {np.mean(remove_min_max(seconds))*1000} ms"
    )

    print("Config:")
    print(f"{batch_size=}, {window_size=}, {threshold=}, {eos_token_id=}, {vocab_size=}, {max_steps=}")


if __name__ == "__main__":
    test_repetition_early_stopper()
    test_consistency()
    test_consistency_with_real_batch_size()
    test_performance()
