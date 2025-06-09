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

__global__ void GetPaddingOffsetSystemKernel(int64_t* output_data,
                                             int* padding_offset,
                                             int* padding_offset_merged,
                                             int* cum_offsets_out,
                                             int* cum_offsets_out_merged,
                                             const int64_t* input_data,
                                             const int* seq_mapping,
                                             const int* cum_offsets,
                                             const int* cum_offsets_merged,
                                             const int* seq_lens,
                                             const int* seq_lens_merged,
                                             const int max_seq_len) {
    // get padding offset of each batch
    const int bi = blockIdx.x;
    const int real_bi = seq_mapping[bi];
    const int ti = threadIdx.x;
    const int cum_offset = bi == 0 ? 0 : cum_offsets[bi - 1];

    const int base_offset = bi * max_seq_len - cum_offset;
    const int src_base_offset = real_bi * max_seq_len;
    for (int i = ti; i < seq_lens[bi]; i += blockDim.x) {
        const int offset_now = base_offset + i;
        padding_offset[offset_now] = cum_offset;
        output_data[offset_now] = input_data[src_base_offset + i];
    }

    const int seq_len_merged = seq_lens_merged[bi];
    const int cum_offset_merged = bi == 0 ? 0 : cum_offsets_merged[bi - 1];
    const int merged_base_offset = bi * max_seq_len - cum_offset_merged;
    for (int i = ti; i < seq_len_merged; i += blockDim.x) {
        padding_offset_merged[merged_base_offset + i] = cum_offset_merged;
    }
    if (ti == 0) {
        cum_offsets_out[bi] = cum_offset;
        cum_offsets_out_merged[bi] = cum_offset_merged;
    }
}

std::vector<paddle::Tensor> GetPaddingOffsetSystem(
    const paddle::Tensor& input_ids,
    const paddle::Tensor& cum_offsets,
    const paddle::Tensor& cum_offsets_merged,
    const paddle::Tensor& token_num,
    const paddle::Tensor& seq_len,
    const paddle::Tensor& seq_len_merged,
    const paddle::Tensor& seq_mapping) {
    auto cu_stream = input_ids.stream();
    std::vector<int64_t> input_ids_shape = input_ids.shape();
    const int bsz = seq_len.shape()[0];
    const int seq_length = input_ids_shape[1];
    auto cum_offsets_out = cum_offsets.copy_to(cum_offsets.place(), false);
    auto cum_offsets_merged_out =
        cum_offsets_merged.copy_to(cum_offsets_merged.place(), false);
    auto cpu_token_num = token_num.copy_to(paddle::CPUPlace(), false);

    const int token_num_data = cpu_token_num.data<int64_t>()[0];
    auto x_remove_padding = paddle::full(
        {token_num_data}, 0, paddle::DataType::INT64, input_ids.place());
    auto padding_offset = paddle::full(
        {token_num_data}, 0, paddle::DataType::INT32, input_ids.place());
    auto padding_offset_merged = paddle::full(
        {token_num_data}, 0, paddle::DataType::INT32, input_ids.place());
    int blockSize = min((token_num_data + 32 - 1) / 32 * 32, 128);
    GetPaddingOffsetSystemKernel<<<bsz, 1024, 0, cu_stream>>>(
        x_remove_padding.data<int64_t>(),
        padding_offset.data<int>(),
        padding_offset_merged.data<int>(),
        cum_offsets_out.data<int>(),
        cum_offsets_merged_out.data<int>(),
        input_ids.data<int64_t>(),
        seq_mapping.data<int>(),
        cum_offsets.data<int>(),
        cum_offsets_merged.data<int>(),
        seq_len.data<int>(),
        seq_len_merged.data<int>(),
        seq_length);
    return {x_remove_padding,
            cum_offsets_out,
            cum_offsets_merged_out,
            padding_offset,
            padding_offset_merged};  // , enc_token_num, dec_token_num};
}

std::vector<std::vector<int64_t>> GetPaddingOffsetSystemInferShape(
    const std::vector<int64_t>& input_ids_shape,
    const std::vector<int64_t>& cum_offsets_shape,
    const std::vector<int64_t>& cum_offsets_merged_shape,
    const std::vector<int64_t>& token_num_shape,
    const std::vector<int64_t>& seq_len_shape,
    const std::vector<int64_t>& seq_len_merged_shape,
    const std::vector<int64_t>& seq_mapping_shape) {
    int64_t bsz = seq_len_shape[0];
    return {{-1}, {bsz}, {bsz}, {-1}, {-1}};
}

std::vector<paddle::DataType> GetPaddingOffsetSystemInferDtype(
    const paddle::DataType& input_ids_dtype,
    const paddle::DataType& cum_offsets_dtype,
    const paddle::DataType& cum_offsets_merged_dtype,
    const paddle::DataType& token_num_dtype,
    const paddle::DataType& seq_len_dtype,
    const paddle::DataType& seq_len_merged_dtype,
    const paddle::DataType& seq_mapping_dtype) {
    return {input_ids_dtype,
            seq_len_dtype,
            seq_len_dtype,
            seq_len_dtype,
            seq_len_dtype};
}

PD_BUILD_STATIC_OP(get_padding_offset_system)
    .Inputs({"input_ids",
             "cum_offsets",
             "cum_offsets_merged",
             "token_num",
             "seq_len",
             "seq_len_merged",
             "seq_mapping"})
    .Outputs({"x_remove_padding",
              "cum_offsets_out",
              "cum_offsets_merged_out",
              "padding_offset",
              "padding_offset_merged"})
    .SetKernelFn(PD_KERNEL(GetPaddingOffsetSystem))
    .SetInferShapeFn(PD_INFER_SHAPE(GetPaddingOffsetSystemInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(GetPaddingOffsetSystemInferDtype));
