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

#include "helper.h"
#include "paddle/extension.h"

template <typename T, int VecSize>
__global__ void HydraFetchHiddenStatesKernel(const T* output_hidden_states,
                                             const int* output_padding_offset,
                                             const int* accept_token_num,
                                             T* hidden_states,
                                             const int bsz,
                                             const int max_seq_len,
                                             const int hidden_size) {
    const int token_id = blockIdx.x;
    const int ori_token_id = token_id + output_padding_offset[token_id];
    const int bid = ori_token_id / max_seq_len;
    const int start_ori_token_id = bid * max_seq_len;
    const int local_token_id = ori_token_id - start_ori_token_id;

    if (local_token_id != accept_token_num[bid] - 1) return;

    using LoadT = AlignedVector<T, VecSize>;

    LoadT vec;

    for (int idx = threadIdx.x * VecSize; idx < hidden_size;
         idx += blockDim.x * VecSize) {
        Load(&output_hidden_states[token_id * hidden_size + idx], &vec);
        Store(vec, &hidden_states[bid * hidden_size + idx]);
    }
}

template <paddle::DataType D>
std::vector<paddle::Tensor> HydraFetchHiddenStatesImpl(
    const paddle::Tensor& output_hidden_states,
    const paddle::Tensor& output_padding_offset,
    const paddle::Tensor& accept_token_num,
    const int bsz,
    const int max_seq_length) {
    typedef PDTraits<D> traits_;
    typedef typename traits_::DataType DataType_;
    typedef typename traits_::data_t data_t;

    auto cu_stream = output_hidden_states.stream();

    auto output_token_num = output_hidden_states.shape()[0];
    auto hidden_size = output_hidden_states.shape()[1];

    auto hidden_states = paddle::full({bsz, hidden_size},
                                      0,
                                      output_hidden_states.dtype(),
                                      output_hidden_states.place());

    constexpr int VecSize = 16 / sizeof(data_t);

    HydraFetchHiddenStatesKernel<data_t, VecSize>
        <<<output_token_num, 256, 0, cu_stream>>>(
            output_hidden_states.data<data_t>(),
            output_padding_offset.data<int>(),
            accept_token_num.data<int>(),
            hidden_states.data<data_t>(),
            bsz,
            max_seq_length,
            hidden_size);
    return {hidden_states};  // , enc_token_num, dec_token_num};
}

std::vector<paddle::Tensor> HydraFetchHiddenStates(
    const paddle::Tensor& output_hidden_states,
    const paddle::Tensor& output_padding_offset,
    const paddle::Tensor& accept_token_num,
    const int bsz,
    const int max_seq_length) {
    switch (output_hidden_states.dtype()) {
        case paddle::DataType::BFLOAT16: {
            return HydraFetchHiddenStatesImpl<paddle::DataType::BFLOAT16>(
                output_hidden_states,
                output_padding_offset,
                accept_token_num,
                bsz,
                max_seq_length);
        }
        case paddle::DataType::FLOAT16: {
            return HydraFetchHiddenStatesImpl<paddle::DataType::FLOAT16>(
                output_hidden_states,
                output_padding_offset,
                accept_token_num,
                bsz,
                max_seq_length);
        }
        default: {
            PD_THROW(
                "NOT supported data type. "
                "Only float16, bfloat16 are supported. ");
            break;
        }
    }
}

std::vector<std::vector<int64_t>> HydraFetchHiddenStatesInferShape(
    const std::vector<int64_t>& output_hidden_states_shape,
    const std::vector<int64_t>& output_padding_offset_shape,
    const std::vector<int64_t>& accept_token_num_shape,
    const int bsz,
    const int max_seq_length) {
    return {{bsz, output_hidden_states_shape[1]}};
}

std::vector<paddle::DataType> HydraFetchHiddenStatesInferDtype(
    const paddle::DataType& output_hidden_states_dtype,
    const paddle::DataType& output_padding_offset_dtype,
    const paddle::DataType& accept_token_num_dtype) {
    return {output_hidden_states_dtype};
}

PD_BUILD_STATIC_OP(hydra_fetch_hidden_states)
    .Inputs({"output_hidden_states",
             "output_padding_offset",
             "accept_token_num"})
    .Outputs({"hidden_states"})
    .Attrs({"bsz: int", "max_seq_length,: int"})
    .SetKernelFn(PD_KERNEL(HydraFetchHiddenStates))
    .SetInferShapeFn(PD_INFER_SHAPE(HydraFetchHiddenStatesInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(HydraFetchHiddenStatesInferDtype));
