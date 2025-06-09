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

__global__ void GetSeqMapping(int* merged_seq_lens_this_time,
                              int* merged_seq_lens_encoder,
                              int* merged_seq_lens_decoder,
                              int* new_system_lens,
                              int* new_seq_lens_this_time,
                              int* new_seq_lens_encoder,
                              int* new_seq_lens_decoder,
                              int* seq_mapping,
                              int* dec_group_num,
                              const int* seq_lens_this_time,
                              const int* seq_lens_encoder,
                              const int* seq_lens_decoder,
                              const int* group_ids,
                              const int* group_lens,
                              const int* cum_group_lens,
                              const int* system_lens,
                              const int max_bsz) {
    const int bi = blockIdx.x;
    const int ti = threadIdx.x;
    const int group_len = group_lens[bi];
    if (group_len <= 0) return;
    const int start_bi = bi == 0 ? 0 : cum_group_lens[bi - 1];
    if (ti == 0) {
        const int* group_ids_now = group_ids + bi * max_bsz;
        int seq_len_sum = 0;
        int seq_len_encoder_sum = 0;
        int dec_count = 0;
        int system_len = 0;
        for (int i = 0; i < group_len; i++) {
            const int group_id = group_ids_now[i];
            const int seq_len_this_time = seq_lens_this_time[group_id];
            const int seq_len_encoder = seq_lens_encoder[group_id];
            if (seq_len_encoder <= 0) {  // decoder
                seq_len_sum += seq_len_this_time;
                seq_len_encoder_sum += seq_len_encoder;
                new_seq_lens_this_time[start_bi + dec_count] =
                    seq_len_this_time;
                new_seq_lens_encoder[start_bi + dec_count] = seq_len_encoder;
                new_seq_lens_decoder[start_bi + dec_count] =
                    seq_lens_decoder[group_id];
                seq_mapping[start_bi + dec_count] = group_id;
                system_len = system_lens[group_id];
                new_system_lens[group_id] = system_len;
                dec_count++;
            } else {  // encoder
                int encoder_bid = atomicAdd(dec_group_num, 1);
                new_seq_lens_this_time[encoder_bid] = seq_len_this_time;
                new_seq_lens_encoder[encoder_bid] = seq_len_encoder;
                new_seq_lens_decoder[encoder_bid] = seq_lens_decoder[group_id];
                seq_mapping[encoder_bid] = group_id;
                new_system_lens[group_id] = 0;
            }
        }
        if (dec_count >= 1) {
            merged_seq_lens_this_time[bi] = seq_len_sum;
            merged_seq_lens_encoder[bi] = seq_len_encoder_sum;
            if (dec_count == 1) {
                merged_seq_lens_decoder[bi] = 0;
            } else {
                merged_seq_lens_decoder[bi] = system_len;
            }
        }
        if (dec_count <= 1) {
            for (int i = 0; i < group_len; i++) {
                const int group_id = group_ids_now[i];
                new_system_lens[group_id] = 0;
            }
        }
    }
}

std::vector<paddle::Tensor> Seqs2Seqs(const paddle::Tensor& seq_lens_this_time,
                                      const paddle::Tensor& seq_lens_encoder,
                                      const paddle::Tensor& seq_lens_decoder,
                                      const paddle::Tensor& group_ids,
                                      const paddle::Tensor& group_lens,
                                      const paddle::Tensor& dec_group_num,
                                      const paddle::Tensor& cum_group_lens,
                                      const paddle::Tensor& system_lens) {
    auto cu_stream = seq_lens_this_time.stream();
    const int bsz = seq_lens_this_time.shape()[0];
    const int max_bsz = seq_lens_encoder.shape()[0];

    auto merged_seq_lens_this_time = paddle::full(
        {bsz, 1}, 0, paddle::DataType::INT32, seq_lens_this_time.place());
    auto merged_seq_lens_encoder = paddle::full(
        {bsz, 1}, 0, paddle::DataType::INT32, seq_lens_this_time.place());
    auto merged_seq_lens_decoder = paddle::full(
        {bsz, 1}, 0, paddle::DataType::INT32, seq_lens_this_time.place());
    auto new_system_lens = paddle::full(
        {bsz, 1}, 0, paddle::DataType::INT32, seq_lens_this_time.place());
    auto new_seq_lens_this_time = paddle::full(
        {bsz, 1}, 0, paddle::DataType::INT32, seq_lens_this_time.place());
    auto new_seq_lens_encoder = paddle::full(
        {bsz, 1}, 0, paddle::DataType::INT32, seq_lens_this_time.place());
    auto new_seq_lens_decoder = paddle::full(
        {bsz, 1}, 0, paddle::DataType::INT32, seq_lens_this_time.place());
    auto seq_mapping = paddle::full(
        {bsz, 1}, -1, paddle::DataType::INT32, seq_lens_this_time.place());

    constexpr int BlockSize = 32;
    // const int blockSize = (max_bsz + 32 - 1) / 32 * 32;
    GetSeqMapping<<<bsz, BlockSize, 0, cu_stream>>>(
        merged_seq_lens_this_time.data<int>(),
        merged_seq_lens_encoder.data<int>(),
        merged_seq_lens_decoder.data<int>(),
        new_system_lens.data<int>(),
        new_seq_lens_this_time.data<int>(),
        new_seq_lens_encoder.data<int>(),
        new_seq_lens_decoder.data<int>(),
        seq_mapping.data<int>(),
        const_cast<int*>(dec_group_num.data<int>()),
        seq_lens_this_time.data<int>(),
        seq_lens_encoder.data<int>(),
        seq_lens_decoder.data<int>(),
        group_ids.data<int>(),
        group_lens.data<int>(),
        cum_group_lens.data<int>(),
        system_lens.data<int>(),
        max_bsz);
    return {merged_seq_lens_this_time,
            merged_seq_lens_encoder,
            merged_seq_lens_decoder,
            new_system_lens,
            new_seq_lens_this_time,
            new_seq_lens_encoder,
            new_seq_lens_decoder,
            seq_mapping};  // , enc_token_num, dec_token_num};
}

std::vector<std::vector<int64_t>> Seqs2SeqsInferShape(
    const std::vector<int64_t>& seq_lens_this_time_shape,
    const std::vector<int64_t>& seq_lens_encoder_shape,
    const std::vector<int64_t>& seq_lens_decoder_shape,
    const std::vector<int64_t>& group_ids_shape,
    const std::vector<int64_t>& group_lens_shape,
    const std::vector<int64_t>& dec_group_num_shape,
    const std::vector<int64_t>& cum_group_lens_shape,
    const std::vector<int64_t>& system_lens_shape) {
    int64_t bsz = seq_lens_this_time_shape[0];
    return {{bsz, 1},
            {bsz, 1},
            {bsz, 1},
            {bsz, 1},
            {bsz, 1},
            {bsz, 1},
            {bsz, 1},
            {bsz, 1}};
}

std::vector<paddle::DataType> Seqs2SeqsInferDtype(
    const paddle::DataType& seq_lens_this_time_dtype,
    const paddle::DataType& seq_lens_encoder_dtype,
    const paddle::DataType& seq_lens_decoder_dtype,
    const paddle::DataType& group_ids_dtype,
    const paddle::DataType& group_lens_dtype,
    const paddle::DataType& dec_group_num_dtype,
    const paddle::DataType& cum_group_lens_dtype,
    const paddle::DataType& system_lens_dtype) {
    return {seq_lens_this_time_dtype,
            seq_lens_this_time_dtype,
            seq_lens_this_time_dtype,
            seq_lens_this_time_dtype,
            seq_lens_this_time_dtype,
            seq_lens_this_time_dtype,
            seq_lens_this_time_dtype,
            seq_lens_this_time_dtype};
}

PD_BUILD_STATIC_OP(seqs2seqs)
    .Inputs({"seq_lens_this_time",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "group_ids",
             "group_lens",
             "dec_group_num",
             "cum_group_lens",
             "system_lens"})
    .Outputs({"merged_seq_lens_this_time",
              "merged_seq_lens_encoder",
              "merged_seq_lens_decoder",
              "new_system_lens",
              "new_seq_lens_this_time",
              "new_seq_lens_encoder",
              "new_seq_lens_decoder",
              "seq_mapping"})
    .SetKernelFn(PD_KERNEL(Seqs2Seqs))
    .SetInferShapeFn(PD_INFER_SHAPE(Seqs2SeqsInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(Seqs2SeqsInferDtype));
