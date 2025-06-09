// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/extension.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

__global__ void System2GroupKernel(int* group_ids,
                                   int* group_lens,
                                   int* group_lens_without_encoder,
                                   int* dec_group_num,
                                   const int* seq_lens_this_time,
                                   const int* seq_lens_encoder,
                                   const int* system_ids,
                                   const int bsz,
                                   const int max_bsz) {
    const int ti = threadIdx.x;
    if (ti < bsz) {
        if (seq_lens_this_time[ti] <=
            0) {  // 终止位置不参与分组，encoder需要是一个特定的system
                  // id，在seqs2seqs里处理
            return;
        }
        int group_id = system_ids[ti];
        int group_len_now = atomicAdd(&group_lens[group_id], 1);
        if (seq_lens_encoder[ti] <= 0) {  // is decoder
            atomicAdd(dec_group_num, 1);
            atomicAdd(&group_lens_without_encoder[group_id], 1);
        }
        group_ids[group_id * max_bsz + group_len_now] = ti;
    }
}

std::vector<paddle::Tensor> System2Group(
    const paddle::Tensor& system_ids,
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& seq_lens_encoder) {
    auto cu_stream = seq_lens_this_time.stream();
    const int bsz = seq_lens_this_time.shape()[0];
    const int max_bsz = seq_lens_encoder.shape()[0];

    auto group_ids = paddle::full({bsz, max_bsz},
                                  -1,
                                  paddle::DataType::INT32,
                                  seq_lens_this_time.place());
    auto group_lens = paddle::full(
        {bsz, 1}, 0, paddle::DataType::INT32, seq_lens_this_time.place());
    auto group_lens_without_encoder = paddle::full(
        {bsz, 1}, 0, paddle::DataType::INT32, seq_lens_this_time.place());
    auto dec_group_num = paddle::full(
        {1}, 0, paddle::DataType::INT32, seq_lens_this_time.place());

    const int blockSize = (bsz + 32 - 1) / 32 * 32;
    System2GroupKernel<<<1, blockSize, 0, cu_stream>>>(
        group_ids.data<int>(),
        group_lens.data<int>(),
        group_lens_without_encoder.data<int>(),
        dec_group_num.data<int>(),
        seq_lens_this_time.data<int>(),
        seq_lens_encoder.data<int>(),
        system_ids.data<int>(),
        bsz,
        max_bsz);
    return {group_ids, group_lens, group_lens_without_encoder, dec_group_num};
}

std::vector<std::vector<int64_t>> System2GroupInferShape(
    const std::vector<int64_t>& system_ids_shape,
    const std::vector<int64_t>& seq_lens_this_time_shape,
    const std::vector<int64_t>& seq_lens_encoder_shape) {
    int64_t bsz = seq_lens_this_time_shape[0];
    int64_t max_bsz = seq_lens_encoder_shape[0];
    return {{bsz, max_bsz}, {bsz, 1}, {bsz, 1}, {1}};
}

std::vector<paddle::DataType> System2GroupInferDtype(
    const paddle::DataType& system_ids_dtype,
    const paddle::DataType& seq_lens_this_time_dtype,
    const paddle::DataType& seq_lens_encoder_dtype) {
    return {seq_lens_this_time_dtype,
            seq_lens_this_time_dtype,
            seq_lens_this_time_dtype,
            seq_lens_this_time_dtype};
}

PD_BUILD_STATIC_OP(system2group)
    .Inputs({"system_ids", "seq_lens_this_time", "seq_lens_encoder"})
    .Outputs({"group_ids",
              "group_lens",
              "group_lens_without_encoder",
              "dec_group_num"})
    .SetKernelFn(PD_KERNEL(System2Group))
    .SetInferShapeFn(PD_INFER_SHAPE(System2GroupInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(System2GroupInferDtype));
