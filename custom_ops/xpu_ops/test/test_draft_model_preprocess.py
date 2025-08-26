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

import numpy as np
import paddle

from fastdeploy.model_executor.ops.xpu import draft_model_preprocess


def run_test(device="xpu"):
    paddle.seed(2022)

    # Define parameters
    bsz = 10
    draft_tokens_len = 4
    input_ids_len = 8
    max_draft_token = 10

    truncate_first_token = True
    splitwise_prefill = False
    # Create input tensors
    if device == "cpu":
        paddle.set_device(device)

    draft_tokens = paddle.randint(0, 100, [bsz, draft_tokens_len], dtype="int64")
    input_ids = paddle.randint(0, 100, [bsz, input_ids_len], dtype="int64")
    stop_flags = paddle.randint(0, 1, [bsz], dtype="int").cast("bool")
    seq_lens_this_time = paddle.randint(0, 100, [bsz], dtype="int32")
    seq_lens_encoder = paddle.randint(0, 100, [bsz], dtype="int32")
    seq_lens_decoder = paddle.randint(0, 100, [bsz], dtype="int32")
    step_idx = paddle.randint(0, 100, [bsz], dtype="int64")
    seq_lens_encoder_record = paddle.randint(0, 100, [bsz], dtype="int32")
    seq_lens_decoder_record = paddle.randint(0, 100, [bsz], dtype="int32")
    not_need_stop = paddle.zeros([1], dtype="bool").cpu()
    batch_drop = paddle.zeros([bsz], dtype="bool")

    # Output tensors
    accept_tokens = paddle.randint(0, 100, [bsz, 100], dtype="int64")
    accept_num = paddle.randint(1, max_draft_token + 5, [bsz], dtype="int32")
    base_model_seq_lens_encoder = paddle.randint(0, 100, [bsz], dtype="int32")
    base_model_seq_lens_decoder = paddle.randint(0, 100, [bsz], dtype="int32")
    base_model_step_idx = paddle.randint(0, 100, [bsz], dtype="int64")
    base_model_stop_flags = paddle.zeros([bsz], dtype="bool")
    base_model_is_block_step = paddle.zeros([bsz], dtype="bool")
    base_model_draft_tokens = paddle.zeros([bsz, max_draft_token], dtype="int64")
    # Run the op
    outputs = draft_model_preprocess(
        draft_tokens,
        input_ids,
        stop_flags,
        seq_lens_this_time,
        seq_lens_encoder,
        seq_lens_decoder,
        step_idx,
        seq_lens_encoder_record,
        seq_lens_decoder_record,
        not_need_stop,
        batch_drop,
        accept_tokens,
        accept_num,
        base_model_seq_lens_encoder,
        base_model_seq_lens_decoder,
        base_model_step_idx,
        base_model_stop_flags,
        base_model_is_block_step,
        base_model_draft_tokens,
        max_draft_token=max_draft_token,
        truncate_first_token=truncate_first_token,
        splitwise_prefill=splitwise_prefill,
    )

    # Return results for comparison
    results = {
        "draft_tokens": draft_tokens.numpy(),
        "input_ids": input_ids.numpy(),
        "stop_flags": stop_flags.numpy(),
        "seq_lens_this_time": seq_lens_this_time.numpy(),
        "accept_tokens": accept_tokens.numpy(),
        "accept_num": accept_num.numpy(),
        "not_need_stop": not_need_stop.numpy(),
        "outputs": [x.numpy() for x in outputs],
    }
    return results


def compare_results(cpu_results, xpu_results):
    # Compare all outputs
    for key in cpu_results:
        if key == "outputs":
            for i, (cpu_out, xpu_out) in enumerate(zip(cpu_results[key], xpu_results[key])):
                np.testing.assert_allclose(
                    cpu_out,
                    xpu_out,
                    rtol=1e-5,
                    atol=1e-8,
                    err_msg=f"Output {i} mismatch between CPU and GPU",
                )
        else:
            np.testing.assert_allclose(
                cpu_results[key],
                xpu_results[key],
                rtol=1e-5,
                atol=1e-8,
                err_msg=f"{key} mismatch between CPU and GPU",
            )
    print("CPU and GPU results match!")


def test_draft_model_preprocess():

    print("Running XPU test...")
    xpu_results = run_test("xpu")

    print("Running CPU test...")
    cpu_results = run_test("cpu")

    print("Comparing results...")
    compare_results(cpu_results, xpu_results)

    print("Test passed!")


if __name__ == "__main__":
    test_draft_model_preprocess()
