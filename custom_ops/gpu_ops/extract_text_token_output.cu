// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "helper.h"

template <int THREADBLOCK_SIZE>
__global__ void extract_text_token_output_kernel(int *max_seq_len,
                                int *max_seq_len_index,
                                int *mm_token_num_len,
                                int *seq_lens_this_time,
                                int *cu_seqlens_q,
                                float *score_text,
                                float *output,
                                const int bsz,
                                const int hidden_size) {
    int bsz_index = threadIdx.x;
    int block_idx = blockIdx.x;
    if (bsz_index >= bsz) return;

    int max_seq_len_data = max_seq_len[0];
    int max_seq_len_index_data = max_seq_len_index[0];
    int mm_token_num_len_data = mm_token_num_len[0];
    int true_bsz = cu_seqlens_q[bsz_index + 1] - 1;
    if (bsz_index >= max_seq_len_index_data) {
        true_bsz = true_bsz - mm_token_num_len_data;
    }
    if (max_seq_len_data == mm_token_num_len_data && bsz_index == max_seq_len_index_data) {
        output[bsz_index * hidden_size + block_idx] = 0.0;
    } else {
        if (seq_lens_this_time[bsz_index] != 0) {
            output[bsz_index * hidden_size + block_idx] = score_text[true_bsz * hidden_size + block_idx];
        }
    }
    __syncthreads();
}

std::vector<paddle::Tensor> ExtractTextTokenOutput(
            const paddle::Tensor& max_seq_len,
            const paddle::Tensor& max_seq_len_index,
            const paddle::Tensor& mm_token_num_len,
            const paddle::Tensor& seq_lens_this_time,
            const paddle::Tensor& cu_seqlens_q,
            const paddle::Tensor& score_text) {

    const int bsz = seq_lens_this_time.shape()[0];
    const int hidden_size = score_text.shape()[1];
    paddle::Tensor output = paddle::full({bsz, hidden_size}, 1, paddle::DataType::FLOAT32, score_text.place());

    extract_text_token_output_kernel<1024><<<hidden_size, 1024, 0, score_text.stream()>>>(
      const_cast<int*>(max_seq_len.data<int>()),
      const_cast<int*>(max_seq_len_index.data<int>()),
      const_cast<int*>(mm_token_num_len.data<int>()),
      const_cast<int*>(seq_lens_this_time.data<int>()),
      const_cast<int*>(cu_seqlens_q.data<int>()),
      const_cast<float*>(score_text.data<float>()),
      output.data<float>(),
      bsz,
      hidden_size
    );
    return {output};
}

std::vector<std::vector<int64_t>> ExtractTextTokenOutputInferShape(const std::vector<int64_t>& max_seq_len_shape,
                                                             const std::vector<int64_t>& max_seq_len_index_shape,
                                                             const std::vector<int64_t>& mm_token_num_len_shape,
                                                             const std::vector<int64_t>& seq_lens_this_time_shape,
                                                             const std::vector<int64_t>& cu_seqlens_q_shape,
                                                             const std::vector<int64_t>& score_text_shape) {
    const int bsz = seq_lens_this_time_shape[0];
    const int hidden_size = score_text_shape[1];
    return {{bsz, hidden_size}};
}

std::vector<paddle::DataType> ExtractTextTokenOutputInferDtype(const paddle::DataType& max_seq_len_dtype,
                                                         const paddle::DataType& max_seq_len_index_dtype,
                                                         const paddle::DataType& mm_token_num_len_dtype,
                                                         const paddle::DataType& seq_lens_this_time_dtype,
                                                         const paddle::DataType& cu_seqlens_q_dtype,
                                                         const paddle::DataType& score_text_dtype) {
    return {score_text_dtype};
}

PD_BUILD_STATIC_OP(extract_text_token_output)
    .Inputs({"max_seq_len",
             "max_seq_len_index",
             "mm_token_num_len",
             "seq_lens_this_time",
             "cu_seqlens_q",
             "score_text"})
    .Outputs({"output"})
    .SetKernelFn(PD_KERNEL(ExtractTextTokenOutput))
    .SetInferShapeFn(PD_INFER_SHAPE(ExtractTextTokenOutputInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(ExtractTextTokenOutputInferDtype));
