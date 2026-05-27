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

import os

import paddle

from fastdeploy.utils import data_processor_logger

# Determine which runner is in use
_USE_FD_RUNNER = os.environ.get("EB5_ENABLE_FD_RUNNER", "0") == "1"


def get_entropy(logits):
    if paddle.any(paddle.isinf(logits) & (logits < 0)):
        logits = paddle.clip(logits, min=1e-9)

    a0 = logits - paddle.max(logits, axis=-1, keepdim=True)
    ea0 = paddle.exp(a0)
    z0 = paddle.sum(ea0, axis=-1, keepdim=True)
    p0 = ea0 / z0
    return paddle.sum(p0 * (paddle.log(z0) - a0), axis=-1)


def calculate_logits_entropy_fd(logits, share_inputs, temperature):
    real_bsz = share_inputs["seq_lens_this_time"].shape[0]
    seq_lens_encoder = share_inputs["seq_lens_encoder"][:real_bsz]
    seq_lens_this_time = share_inputs["seq_lens_this_time"]
    # GPU runner uses 1D [N], HPU/GCU/ernie5_runner uses 2D [N,1]; flatten to 1D
    if seq_lens_encoder.ndim == 2:
        seq_lens_encoder = seq_lens_encoder.squeeze(1)
    if seq_lens_this_time.ndim == 2:
        seq_lens_this_time = seq_lens_this_time.squeeze(1)
    real_seq_lens = paddle.where(
        seq_lens_encoder != 0,
        paddle.ones([1], dtype="int32"),
        seq_lens_this_time,
    )

    # gpu_model_runner path: logits[i] is the single logit for slot i
    for i in range(real_bsz):
        if int(real_seq_lens[i]) == 0:
            continue
        t = temperature[i]
        if t > 0 and t != 1.0:
            logits[i] = logits[i].scale_(1 / t)

    entropy_tensor = get_entropy(logits[:real_bsz])

    for i in range(real_bsz):
        if int(real_seq_lens[i]) == 0:
            continue
        entropy_val = float(entropy_tensor[i])
        share_inputs["entropy_list"][i].append(entropy_val)
        if (
            share_inputs["stop_flags"][i]
            and share_inputs["seq_lens_decoder"][i] != 0
            and len(share_inputs["entropy_list"][i]) != 0
        ):
            data_processor_logger.info(
                f"req_id: {share_inputs['req_ids'][i]}, entropy: {sum(share_inputs['entropy_list'][i])/len(share_inputs['entropy_list'][i])}"
            )
            share_inputs["entropy_list"][i] = []


def calculate_logits_entropy(logits, share_inputs, temperature):
    """
    Calculate entropy for each token's logits.

    Two paths:
    - gpu_model_runner (EB5_ENABLE_FD_RUNNER=1): logits shape [max_bsz, vocab],
      one logit per slot, direct indexing.
    - ernie5_model_runner (EB5_ENABLE_FD_RUNNER=0): logits shape [total_tokens, vocab],
      flattened across batch, uses repeat_interleave to map back.
    """
    if _USE_FD_RUNNER:
        calculate_logits_entropy_fd(logits, share_inputs, temperature)
        return
    else:
        real_bsz = share_inputs["seq_lens_this_time"].shape[0]
        real_seq_lens = paddle.where(
            share_inputs["seq_lens_encoder"][:real_bsz].squeeze(1) != 0,
            paddle.ones([1], dtype="int32"),
            share_inputs["seq_lens_this_time"].squeeze(1),
        )

        batch_indices = paddle.arange(real_bsz, dtype="int32")
        batch_id_per_token = paddle.repeat_interleave(batch_indices, real_seq_lens)
        for i in range(logits.shape[0]):
            if temperature[batch_id_per_token[i]] > 0 and temperature[batch_id_per_token[i]] != 1.0:
                logits[i] = logits[i].scale_(1 / temperature[batch_id_per_token[i]])

        entropy_tensor = get_entropy(logits)
        entropy = entropy_tensor.tolist()

        for i in range(real_bsz):
            for _ in range(real_seq_lens[i]):
                share_inputs["entropy_list"][i].append(entropy.pop(0))
            if (
                share_inputs["stop_flags"][i]
                and share_inputs["seq_lens_decoder"][i] != 0
                and len(share_inputs["entropy_list"][i]) != 0
            ):
                data_processor_logger.info(
                    f"req_id: {share_inputs['req_ids'][i]}, entropy: {sum(share_inputs['entropy_list'][i])/len(share_inputs['entropy_list'][i])}"
                )
                share_inputs["entropy_list"][i] = []


def speculate_calculate_logits_entropy_fd(logits, share_inputs, temperature):
    real_bsz = share_inputs["seq_lens_this_time"].shape[0]
    total_accepted_num = int(paddle.sum(share_inputs["accept_num"][:real_bsz]))

    if total_accepted_num == 0:
        # Still check stop_flags for ENTROPY-DONE even when no tokens accepted
        for i in range(real_bsz):
            if (
                share_inputs["stop_flags"][i]
                and share_inputs["seq_lens_decoder"][i] != 0
                and len(share_inputs["entropy_list"][i]) != 0
            ):
                data_processor_logger.info(
                    f"req_id: {share_inputs['req_ids'][i]}, entropy: {sum(share_inputs['entropy_list'][i])/len(share_inputs['entropy_list'][i])}"
                )
                share_inputs["entropy_list"][i] = []
        return

    # fd_runner: logits shape is [sum(seq_lens_this_time), vocab], need to index accepted rows
    seq_lens_encoder = share_inputs["seq_lens_encoder"][:real_bsz]
    seq_lens_this_time = share_inputs["seq_lens_this_time"]
    if seq_lens_encoder.ndim == 2:
        seq_lens_encoder = seq_lens_encoder.squeeze(1)
    if seq_lens_this_time.ndim == 2:
        seq_lens_this_time = seq_lens_this_time.squeeze(1)
    real_seq_lens = paddle.where(
        seq_lens_encoder != 0,
        paddle.ones([1], dtype="int32"),
        seq_lens_this_time,
    )
    seq_start_idx = paddle.concat([paddle.zeros([1], dtype="int32"), paddle.cumsum(real_seq_lens, dtype="int32")])
    repeated_starts = paddle.repeat_interleave(seq_start_idx[:-1], share_inputs["accept_num"][:real_bsz])
    offsets = paddle.concat([paddle.arange(share_inputs["accept_num"][i].item()) for i in range(real_bsz)]).astype(
        "int32"
    )
    accepted_idx = repeated_starts + offsets

    accepted_logits = paddle.empty([total_accepted_num, logits.shape[1]], dtype=logits.dtype)
    for i in range(total_accepted_num):
        accepted_logits[i] = logits[accepted_idx[i]]

    batch_indices = paddle.arange(share_inputs["accept_num"].shape[0], dtype="int32")
    batch_id_per_token = paddle.repeat_interleave(batch_indices, share_inputs["accept_num"])
    for i in range(accepted_logits.shape[0]):
        if temperature[batch_id_per_token[i]] > 0 and temperature[batch_id_per_token[i]] != 1.0:
            accepted_logits[i] = accepted_logits[i].scale_(1 / temperature[batch_id_per_token[i]])

    entropy_tensor = get_entropy(accepted_logits)
    entropy = entropy_tensor.tolist()

    for i in range(real_bsz):
        for _ in range(share_inputs["accept_num"][i]):
            share_inputs["entropy_list"][i].append(entropy.pop(0))
        if (
            share_inputs["stop_flags"][i]
            and share_inputs["seq_lens_decoder"][i] != 0
            and len(share_inputs["entropy_list"][i]) != 0
        ):
            data_processor_logger.info(
                f"req_id: {share_inputs['req_ids'][i]}, entropy: {sum(share_inputs['entropy_list'][i])/len(share_inputs['entropy_list'][i])}"
            )
            share_inputs["entropy_list"][i] = []


def speculate_calculate_logits_entropy(logits, share_inputs, temperature):
    """
    Calculate entropy for speculative decoding (MTP) accepted tokens.

    Two paths:
    - fd_runner (EB5_ENABLE_FD_RUNNER=1): logits shape [sum(seq_lens_this_time), vocab],
      contains all positions including rejected; must use accepted_idx to extract.
    - ernie5_runner (EB5_ENABLE_FD_RUNNER=0): logits shape [total_accepted_num, vocab],
      already contains only accepted tokens ordered by slot.
    """
    if _USE_FD_RUNNER:
        speculate_calculate_logits_entropy_fd(logits, share_inputs, temperature)
        return
    else:
        real_bsz = share_inputs["seq_lens_this_time"].shape[0]
        total_accepted_num = paddle.sum(share_inputs["accept_num"])
        real_seq_lens = paddle.where(
            share_inputs["seq_lens_encoder"][:real_bsz].squeeze(1) != 0,
            paddle.ones([1], dtype="int32"),
            share_inputs["seq_lens_this_time"].squeeze(1),
        )
        seq_start_idx = paddle.concat([paddle.zeros([1], dtype="int32"), paddle.cumsum(real_seq_lens, dtype="int32")])
        repeated_starts = paddle.repeat_interleave(seq_start_idx[:-1], share_inputs["accept_num"][:real_bsz])
        offsets = paddle.concat([paddle.arange(share_inputs["accept_num"][i].item()) for i in range(real_bsz)]).astype(
            "int32"
        )
        accepted_idx = repeated_starts + offsets

        accepted_logits = paddle.empty([total_accepted_num, logits.shape[1]], dtype=logits.dtype)
        for i in range(total_accepted_num):
            accepted_logits[i] = logits[accepted_idx[i]]

        batch_indices = paddle.arange(share_inputs["accept_num"].shape[0], dtype="int32")
        batch_id_per_token = paddle.repeat_interleave(batch_indices, share_inputs["accept_num"])
        for i in range(accepted_logits.shape[0]):
            if temperature[batch_id_per_token[i]] > 0 and temperature[batch_id_per_token[i]] != 1.0:
                accepted_logits[i] = accepted_logits[i].scale_(1 / temperature[batch_id_per_token[i]])

        entropy_tensor = get_entropy(accepted_logits)
        entropy = entropy_tensor.tolist()

        for i in range(real_bsz):
            for _ in range(share_inputs["accept_num"][i]):
                share_inputs["entropy_list"][i].append(entropy.pop(0))
            if (
                share_inputs["stop_flags"][i]
                and share_inputs["seq_lens_decoder"][i] != 0
                and len(share_inputs["entropy_list"][i]) != 0
            ):
                data_processor_logger.info(
                    f"req_id: {share_inputs['req_ids'][i]}, entropy: {sum(share_inputs['entropy_list'][i])/len(share_inputs['entropy_list'][i])}"
                )
                share_inputs["entropy_list"][i] = []
