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

import os
import random

import numpy as np
import paddle
from paddle.nn.quant import weight_quantize

from fastdeploy.model_executor.ops.gpu import (
    moe_expert_dispatch,
    moe_expert_ffn,
    moe_expert_ffn_wint2,
    moe_expert_reduce,
)


def print_tensor_info(t, name):
    if t is not None:
        print(f"-- [print_tensor_info] {name}: shape={t.shape}, dtype={t.dtype}, data_ptr={t.data_ptr():#x}")
    else:
        print(f"-- [print_tensor_info] {name}: tensor is {t}")


def load_all_tensors(tensor_names, dump_dir):
    tensor_dict = {}
    for name in tensor_names:
        key = name.replace(".pdparams", "").replace("_layer1", "")
        filepath = os.path.join(dump_dir, name)
        if os.path.exists(filepath):
            tensor_dict[key] = paddle.load(filepath)
            if isinstance(tensor_dict[key], paddle.Tensor):
                print_tensor_info(tensor_dict[key], name)
            else:
                print(f"-- {name}: {tensor_dict[key]}")
        else:
            tensor_dict[key] = None
            print(f"-- {name}: {filepath} does not exist.")
    return tensor_dict


def check_result(dtype, out_1, out_2, check_equal=False):
    def get_flattened_array(out):
        if isinstance(out, paddle.Tensor):
            if out.dtype == paddle.bfloat16:
                res = paddle.cast(out, dtype="float32").numpy()
            else:
                res = out.numpy()
        else:
            res = out
        return res.flatten()

    out_1_flatten = get_flattened_array(out_1)
    out_2_flatten = get_flattened_array(out_2)

    diff = np.abs(out_1_flatten - out_2_flatten)
    max_atol_idx = np.argmax(diff)
    print(
        f"-- max difference     : {np.max(diff)}, {out_1_flatten[max_atol_idx]} vs {out_2_flatten[max_atol_idx]}, idx={max_atol_idx}"
    )

    relative_error = np.abs(diff / (out_2_flatten + 1e-8))
    max_rtol_idx = np.nanargmax(relative_error)
    print(
        f"-- max relative error : {np.nanmax(relative_error)}, {out_1_flatten[max_rtol_idx]} vs {out_2_flatten[max_rtol_idx]}"
    )

    if check_equal:
        num_diffs = 0
        for i in range(out_1.size):
            if num_diffs >= 10:
                break

            if out_1_flatten[i] != out_2_flatten[i]:
                print(f"-- {i}: {out_1_flatten[i]} vs {out_2_flatten[i]}")
                num_diffs += 1
        np.testing.assert_array_equal(out_1, out_2)
    else:
        if dtype == "float32":
            if os.getenv("NVIDIA_TF32_OVERRIDE", "1") == "0":
                atol, rtol = 1e-5, 1e-5
            else:
                atol, rtol = 1e-3, 1e-3
        elif dtype == "float16":
            atol, rtol = 1e-3, 1e-3
        elif dtype == "bfloat16":
            atol, rtol = 1e-2, 1e-2

        np.testing.assert_allclose(
            out_1,
            out_2,
            atol=atol,
            rtol=rtol,
        )


def unzip_and_dequant_wint2(
    w, w_scale, w_code_scale, w_code_zp, w_super_scale=None, scale_compute_dtype=None, shuffled=False, group_size=64
):
    """
    w                 uint8             [num_experts, in_feature_size // pack_num, out_feature_size]
    w_scale                             [num_experts, in_feature_size // group_size, out_feature_size]
    w_code_scale      float32           [num_experts, out_feature_size]
    w_code_zp         float32           [num_experts, out_feature_size]
    w_super_scale     w_scale.dtype     [num_experts, out_feature_size]
    output:           w_scale.dtype     [num_experts, in_feature_size, out_feature_size]
    """

    def w_round(x):
        return paddle.floor(x + 0.5)

    # step0: w dtype: uint8, shape: [num_experts, in_feature_size // pack_num, out_feature_size]
    # where pack_num = 4
    pack_num = 4
    bzp = 32
    num_experts, pack_in_feature_size, out_feature_size = w.shape

    in_feature_size = pack_in_feature_size * pack_num
    # step1: w need to unzip to shape: [num_experts, in_feature_size, out_feature_size]
    # here we use broadcast operation to implcitly expand the last dimension
    w = w.transpose(perm=[0, 2, 1]).reshape([num_experts, out_feature_size, pack_in_feature_size, 1])

    # for support repeat_interleave, w cast to int32
    w = w.cast("int32")
    w = w.repeat_interleave(pack_num, axis=-1)
    w = w.reshape([num_experts, out_feature_size, in_feature_size])
    w = w.transpose(perm=[0, 2, 1])

    # step2: w need to first dequant
    # w_code_scale shape: [num_experts, out_feature_size]
    # w_code_zp shape: [num_experts, out_feature_size]
    w_code_scale = w_code_scale.reshape([num_experts, 1, out_feature_size])
    w_code_zp = w_code_zp.reshape([num_experts, 1, out_feature_size])

    w = w_round(w.cast("float32") * w_code_scale + w_code_zp).cast("int32")

    # step3: w need to shifted and mask the original weight to unzip
    bit_shift = paddle.to_tensor([9, 6, 3, 0], dtype="int32")
    in_feature_bit_shift = bit_shift[paddle.arange(in_feature_size) % pack_num]
    in_feature_bit_shift = in_feature_bit_shift.reshape([1, in_feature_size, 1])
    mask = paddle.to_tensor(0x3F, dtype="int32")

    if scale_compute_dtype is None:
        scale_compute_dtype = w_super_scale.dtype if w_super_scale is not None else w_scale.dtype

    group_num = in_feature_size // group_size

    # step4: w_scale need to shift and mask and dequant
    if w_scale.dtype == paddle.uint8:
        # w_scale shape: [num_experts, in_feature_size // group_size, out_feature_size]
        # w_scale packed shape: [num_experts, group_num // 2, out_feature_size]
        w_scale = w_scale.cast("int32")
        w_scale = w_scale.reshape([num_experts, group_num // 2, 1, out_feature_size])
        w_scale = w_scale.repeat_interleave(2, axis=2)
        w_scale = (w_scale >> paddle.to_tensor([4, 0], dtype="int32").reshape([1, 1, 2, 1])) & paddle.to_tensor(
            0xF, dtype="int32"
        )
        w_scale = w_scale.reshape([num_experts, group_num, out_feature_size]).cast(scale_compute_dtype)

    # step5: w need to shift and mask and second dequant
    w = ((w >> in_feature_bit_shift) & mask).cast(w_scale.dtype)

    if w_super_scale is not None:
        # w_super_scale shape: [num_experts, out_feature_size]
        w_super_scale = w_super_scale.reshape([num_experts, 1, out_feature_size])
        w_scale = w_scale * w_super_scale

    # w_scale reshape to [num_experts, in_feature_size, out_feature_size]
    w_scale = w_scale.reshape([num_experts, in_feature_size // group_size, 1, out_feature_size])
    w_scale = w_scale.repeat_interleave(group_size, axis=2).reshape([num_experts, in_feature_size, out_feature_size])

    w = (w - bzp).cast(w_scale.dtype) * w_scale

    if shuffled:
        w = w.reshape([num_experts, in_feature_size // 64, 4, 8, 2, out_feature_size])
        w = paddle.transpose(w, perm=[0, 1, 3, 2, 4, 5])
        w = w.reshape([num_experts, in_feature_size, out_feature_size])
    return w.cast(w_super_scale.dtype)


class MoEArguments:
    def __init__(
        self,
        quant_method,
        gate_weight,
        ffn1_weight,
        ffn2_weight,
        ffn1_weight_scale,
        ffn2_weight_scale,
        ffn1_local_scale=None,
        ffn1_code_scale=None,
        ffn1_code_zp=None,
        ffn2_local_scale=None,
        ffn2_code_scale=None,
        ffn2_code_zp=None,
        gate_correction_bias=None,
        topk=8,
    ):
        self.quant_method = quant_method

        self.gate_weight = gate_weight
        self.gate_correction_bias = gate_correction_bias
        self.topk = topk

        self.ffn1_weight = ffn1_weight
        self.ffn2_weight = ffn2_weight
        self.ffn1_weight_scale = ffn1_weight_scale
        self.ffn2_weight_scale = ffn2_weight_scale
        self.ffn1_local_scale = ffn1_local_scale
        self.ffn1_code_scale = ffn1_code_scale
        self.ffn1_code_zp = ffn1_code_zp
        self.ffn2_local_scale = ffn2_local_scale
        self.ffn2_code_scale = ffn2_code_scale
        self.ffn2_code_zp = ffn2_code_zp

        if quant_method == "none":
            self.dtype = ffn1_weight.dtype
        else:
            self.dtype = ffn1_weight_scale.dtype

        self.num_experts = ffn1_weight.shape[0]
        if ffn1_weight_scale is not None:
            self.intermediate_size = ffn1_weight_scale.shape[1] // 2
        else:
            self.intermediate_size = ffn1_weight.shape[2] // 2
        if ffn2_weight_scale is not None:
            self.hidden_size = ffn2_weight_scale.shape[1]
        else:
            self.hidden_size = ffn2_weight.shape[2]

    def convert_to_bf16(self, shuffled=False):
        if self.quant_method == "weight_only_int2":
            assert (
                self.dtype != self.ffn1_weight.dtype
            ), f"dtype:{self.dtype} vs weight_dtype: {self.ffn1_weights.dtype}"

            ffn1_weight = unzip_and_dequant_wint2(
                w=self.ffn1_weight,
                w_scale=self.ffn1_local_scale,
                w_code_scale=self.ffn1_code_scale,
                w_code_zp=self.ffn1_code_zp,
                w_super_scale=self.ffn1_weight_scale,
                shuffled=shuffled,
                group_size=64,
            )
            ffn2_weight = unzip_and_dequant_wint2(
                w=self.ffn2_weight,
                w_scale=self.ffn2_local_scale,
                w_code_scale=self.ffn2_code_scale,
                w_code_zp=self.ffn2_code_zp,
                w_super_scale=self.ffn2_weight_scale,
                shuffled=shuffled,
                group_size=64,
            )
            other = MoEArguments(
                quant_method="none",
                gate_weight=self.gate_weight,
                ffn1_weight=ffn1_weight,
                ffn2_weight=ffn2_weight,
                ffn1_weight_scale=None,
                ffn2_weight_scale=None,
                gate_correction_bias=self.gate_correction_bias,
                topk=self.topk,
            )
            return other
        else:
            assert False, "Not supported now!"

    def convert_to_wint4(self):
        assert self.quant_method == "none"
        assert self.dtype == self.ffn1_weight.dtype, f"dtype:{self.dtype} vs weight_dtype: {self.ffn1_weights.dtype}"

        def quantize_ffn_weight(ffn_weight):
            weight_list = []
            scale_list = []
            for i in range(ffn_weight.shape[0]):
                quant_weight, scale = weight_quantize(ffn_weight[i, :, :], algo="weight_only_int4", arch=80)
                weight_list.append(quant_weight)
                scale_list.append(scale)
            quanted_weight = paddle.stack(weight_list, axis=0)
            scale = paddle.stack(scale_list, axis=0)
            return quanted_weight, scale

        ffn1_weight, ffn1_weight_scale = quantize_ffn_weight(self.ffn1_weight)
        ffn2_weight, ffn2_weight_scale = quantize_ffn_weight(self.ffn2_weight)

        other = MoEArguments(
            quant_method="weight_only_int4",
            gate_weight=self.gate_weight,
            ffn1_weight=ffn1_weight,
            ffn2_weight=ffn2_weight,
            ffn1_weight_scale=ffn1_weight_scale,
            ffn2_weight_scale=ffn2_weight_scale,
            gate_correction_bias=self.gate_correction_bias,
            topk=self.topk,
        )
        return other

    def print(self):
        print("")
        print(f"-- [MoEArguments] dtype: {self.dtype}")
        print(f"-- [MoEArguments] num_experts: {self.num_experts}")
        print(f"-- [MoEArguments] intermediate_size: {self.intermediate_size}")
        print(f"-- [MoEArguments] hidden_size: {self.hidden_size}")
        print_tensor_info(self.gate_correction_bias, "gate_correction_bias")
        print_tensor_info(self.ffn1_weight, "ffn1_weight")
        print_tensor_info(self.ffn2_weight, "ffn2_weight")
        print_tensor_info(self.ffn1_weight_scale, "ffn1_weight_scale")
        print_tensor_info(self.ffn2_weight_scale, "ffn2_weight_scale")
        print_tensor_info(self.ffn1_local_scale, "ffn1_local_scale")
        print_tensor_info(self.ffn2_local_scale, "ffn2_local_scale")
        print_tensor_info(self.ffn1_code_scale, "ffn1_code_scale")
        print_tensor_info(self.ffn2_code_scale, "ffn2_code_scale")
        print_tensor_info(self.ffn1_code_zp, "ffn1_code_zp")
        print_tensor_info(self.ffn2_code_zp, "ffn2_code_zp")


def prepare_args_wint2(test_dir):
    tensor_names = [
        "x",
        "gate_weight",
        "topk_ids",
        "gate_correction_bias",
        "ffn1_weight",
        "ffn2_weight",
        "ffn1_super_scales",
        "ffn2_super_scales",
        "ffn1_weight_scale",
        "ffn1_code_scale",
        "ffn1_code_zp",
        "ffn2_weight_scale",
        "ffn2_code_scale",
        "ffn2_code_zp",
    ]
    tensor_dict = load_all_tensors(tensor_names, test_dir)
    topk = tensor_dict["topk_ids"].shape[1]

    moe_args = MoEArguments(
        quant_method="weight_only_int2",
        gate_weight=tensor_dict["gate_weight"],
        ffn1_weight=tensor_dict["ffn1_weight"],
        ffn2_weight=tensor_dict["ffn2_weight"],
        ffn1_weight_scale=tensor_dict["ffn1_super_scales"],
        ffn2_weight_scale=tensor_dict["ffn2_super_scales"],
        ffn1_local_scale=tensor_dict["ffn1_weight_scale"],
        ffn1_code_scale=tensor_dict["ffn1_code_scale"],
        ffn1_code_zp=tensor_dict["ffn1_code_zp"],
        ffn2_local_scale=tensor_dict["ffn2_weight_scale"],
        ffn2_code_scale=tensor_dict["ffn2_code_scale"],
        ffn2_code_zp=tensor_dict["ffn2_code_zp"],
        gate_correction_bias=tensor_dict["gate_correction_bias"],
        topk=topk,
    )
    return moe_args


def run_moe_decode_cutlass(moe_args, quant_method, hidden_states, scores):
    # print(f"-- [run_moe_decode_cutlass] {quant_method}")

    def rearrange_weights(w):
        # [num_experts, in_feature_size, out_feature_size]
        w_shape = w.shape
        # [num_experts, in_feature_size / 64, 64, out_feature_size / 8, 8]
        w = w.reshape([w_shape[0], w_shape[1] // 16, 16, w_shape[2] // 8, 8])
        # [num_experts, out_feature_size / 8, in_feature_size // 64, 8, 64]
        w = paddle.transpose(w, perm=[0, 3, 1, 4, 2])
        # w = w.reshape([w_shape[0], w_shape[2] // 8, w_shape[1] // 16, 128])
        w = w.reshape(w_shape)
        return w

    if quant_method == "weight_only_int2":
        ffn1_weight = rearrange_weights(moe_args.ffn1_weight)
        ffn2_weight = rearrange_weights(moe_args.ffn2_weight)

    cache = paddle.empty((int(512e6 // 4),), dtype="int32")

    warmup, repeat = 5, 100
    gpu_timecosts = []
    for i in range(warmup + repeat):
        start_event = paddle.device.Event(enable_timing=True)
        end_event = paddle.device.Event(enable_timing=True)
        cache.zero_()  # fast_flush
        start_event.record()
        (
            permute_input,
            token_nums_per_expert,
            permute_indices_per_token,
            topk_weights,
            topk_indices,
            expert_idx_per_token,
        ) = moe_expert_dispatch(
            input=hidden_states,
            gating_output=scores,
            gating_correction_bias=moe_args.gate_correction_bias,
            w4a8_in_scale=None,
            moe_topk=moe_args.topk,
            group_moe=False,
            topk_only_mode=moe_args.gate_correction_bias is None,
        )

        if quant_method == "weight_only_int2":
            ffn_out = moe_expert_ffn_wint2(
                permute_input,
                token_nums_per_expert,
                ffn1_weight,
                ffn2_weight,
                None,
                moe_args.ffn1_weight_scale,
                moe_args.ffn2_weight_scale,
                moe_args.ffn1_local_scale,
                moe_args.ffn1_code_scale,
                moe_args.ffn1_code_zp,
                moe_args.ffn2_local_scale,
                moe_args.ffn2_code_scale,
                moe_args.ffn2_code_zp,
                False,
            )
        else:
            ffn_out = moe_expert_ffn(
                permute_input,
                token_nums_per_expert,
                moe_args.ffn1_weight,
                moe_args.ffn2_weight,
                None,
                moe_args.ffn1_weight_scale,
                moe_args.ffn2_weight_scale,
                None,
                None,
                quant_method,
                False,
            )

        moe_out = moe_expert_reduce(
            ffn_out,
            topk_weights,
            permute_indices_per_token,
            topk_indices,
            None,
            norm_topk_prob=True,
            routed_scaling_factor=1.0,
        )
        end_event.record()
        gpu_timecosts.append(start_event.elapsed_time(end_event))
        cache += int(random.random() * 1000)  # change cache

    paddle.device.synchronize()
    del cache
    gpu_timecosts = gpu_timecosts[warmup:]
    return moe_out, np.quantile(gpu_timecosts, 0.5)


def test_main(test_dir):
    moe_args = prepare_args_wint2(test_dir)
    moe_args.print()

    quant_method = "weight_only_int2"
    check_acc = False

    moe_args_bf16 = moe_args.convert_to_bf16(shuffled=True)
    moe_args_wint4 = moe_args_bf16.convert_to_wint4()

    for num_tokens in [1, 2, 4, 16, 64, 128, 512, 1024]:
        hidden_states = paddle.randn([num_tokens, moe_args.hidden_size]).cast(moe_args.dtype)
        gate_out = paddle.matmul(hidden_states.cast("float32"), moe_args.gate_weight)
        scores = paddle.nn.functional.softmax(gate_out, axis=-1)

        timecost_wint2, timecost_bf16, timecost_wint4 = 0.0, 0.0, 0.0
        out_wint2, timecost_wint2 = run_moe_decode_cutlass(moe_args, quant_method, hidden_states, scores)

        out_bf16, timecost_bf16 = run_moe_decode_cutlass(moe_args_bf16, "none", hidden_states, scores)
        out_wint4, timecost_wint4 = run_moe_decode_cutlass(moe_args_wint4, "weight_only_int4", hidden_states, scores)

        print(
            f"[Time Cost] num_tokens: {num_tokens}, {quant_method}: {timecost_wint2:.5f} ms; bf16: {timecost_bf16:.5f} ms; wint4: {timecost_wint4:0.5f} ms"
        )

        if check_acc:
            check_result("bfloat16", out_wint2, out_bf16, check_equal=False)


if __name__ == "__main__":
    paddle.seed(1024)
    test_dir = os.path.dirname(os.path.abspath(__file__)) + "/ernie45t_tp1_wint2_params"
    test_main(test_dir)
