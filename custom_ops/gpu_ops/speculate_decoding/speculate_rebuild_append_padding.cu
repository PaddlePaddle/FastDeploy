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
#include "helper.h"

template <typename T, int VecSize>
__global__ void RebuildAppendPaddingKernel(
                               T *out,
                               const T *full_hidden_states,
                               const int *cum_offset,
                               const int *seq_len_encoder,
                               const int *seq_len_decoder,
                               const int *output_padding_offset,
                               const int seq_len,
                               const int dim_embed,
                               const size_t elem_nums) {
  using LoadT = AlignedVector<T, VecSize>;  
  LoadT src_vec;
  const int64_t global_idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int64_t i = global_idx * VecSize; i < elem_nums; i += gridDim.x * blockDim.x * VecSize) {
    const int out_token_id = i / dim_embed;
    const int ori_token_id = out_token_id + output_padding_offset[out_token_id];
    const int bi = ori_token_id / seq_len;
    int seq_id = 0;

    if (seq_len_decoder[bi] == 0 && seq_len_encoder[bi] == 0) continue;
    else if (seq_len_encoder[bi] != 0) {
      seq_id = seq_len_encoder[bi] - 1;
    }

    const int input_token_id = ori_token_id - cum_offset[bi] + seq_id;
    const int bias_idx = i % dim_embed;
    
    Load<T, VecSize>(&full_hidden_states[input_token_id * dim_embed + bias_idx], &src_vec);
    Store<T, VecSize>(src_vec, &out[i]);
  }
}

template <paddle::DataType D>
std::vector<paddle::Tensor> DispatchDtype(
                  const paddle::Tensor& full_hidden_states,
                  const paddle::Tensor& cum_offsets,
                  const paddle::Tensor& seq_len_encoder,
                  const paddle::Tensor& seq_len_decoder,
                  const paddle::Tensor& output_padding_offset,
                  const int max_seq_len) {
  // src: [token_num, dim_embed]
  // dst: [batch_size, 1, dim_embed]

  typedef PDTraits<D> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t;


  int dim_embed = full_hidden_states.shape()[1];
  int output_token_num = output_padding_offset.shape()[0];
  int elem_nums = output_token_num * dim_embed;
  constexpr int PackSize = VEC_16B / sizeof(DataType_);
  assert(elem_nums % PackSize == 0);

  auto out = paddle::full({output_token_num, dim_embed}, 0, full_hidden_states.dtype(), full_hidden_states.place());

  int pack_num = elem_nums / PackSize;
  const int threads_per_block = 128;
  int grid_size = 1;
  GetNumBlocks(pack_num, &grid_size);

  RebuildAppendPaddingKernel<DataType_, PackSize><<<grid_size, threads_per_block, 0, full_hidden_states.stream()>>>(
          reinterpret_cast<DataType_*>(out.data<data_t>()), 
          reinterpret_cast<const DataType_*>(full_hidden_states.data<data_t>()), 
          cum_offsets.data<int32_t>(), 
          seq_len_encoder.data<int32_t>(), 
          seq_len_decoder.data<int32_t>(), 
          output_padding_offset.data<int32_t>(), 
          max_seq_len, 
          dim_embed, 
          elem_nums);
  return {out};
}


std::vector<paddle::Tensor> RebuildAppendPadding(
                  const paddle::Tensor& full_hidden_states,
                  const paddle::Tensor& cum_offsets,
                  const paddle::Tensor& seq_len_encoder,
                  const paddle::Tensor& seq_len_decoder,
                  const paddle::Tensor& output_padding_offset,
                  const int max_seq_len) {

              
  switch (full_hidden_states.dtype()) {
    case paddle::DataType::BFLOAT16:
      return DispatchDtype<paddle::DataType::BFLOAT16>(
          full_hidden_states, cum_offsets, seq_len_encoder, seq_len_decoder, output_padding_offset, max_seq_len);
    case paddle::DataType::FLOAT16:
      return DispatchDtype<paddle::DataType::FLOAT16>(
          full_hidden_states, cum_offsets, seq_len_encoder, seq_len_decoder, output_padding_offset, max_seq_len);
    default:
      PD_THROW("Unsupported data type.");
  }

}


std::vector<std::vector<int64_t>> RebuildAppendPaddingInferShape(
                          const std::vector<int64_t>& full_hidden_states_shape,
                          const std::vector<int64_t>& cum_offsets_shape,
                          const std::vector<int64_t>& seq_len_encoder_shape,
                          const std::vector<int64_t>& seq_len_decoder_shape,
                          const std::vector<int64_t>& output_padding_offset_shape) {
  const int64_t output_token_num = output_padding_offset_shape[0];
  const int64_t dim_embed = full_hidden_states_shape[1];
  std::vector<int64_t> out_shape = {output_token_num, dim_embed};
  return {out_shape};
}

std::vector<paddle::DataType> RebuildAppendPaddingInferDtype(
                          const paddle::DataType& full_hidden_states_dtype,
                          const paddle::DataType& cum_offsets_dtype,
                          const paddle::DataType& seq_len_encoder_dtype,
                          const paddle::DataType& seq_len_decoder_dtype,
                          const paddle::DataType& output_padding_offset_dtype) {
  return {full_hidden_states_dtype};
}


PD_BUILD_STATIC_OP(speculate_rebuild_append_padding)
    .Inputs({"full_hidden_states", 
             "cum_offsets",
             "seq_len_encoder",
             "seq_len_decoder",
             "output_padding_offset"})
    .Attrs({"max_seq_len: int"})
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(RebuildAppendPadding))
    .SetInferShapeFn(PD_INFER_SHAPE(RebuildAppendPaddingInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(RebuildAppendPaddingInferDtype));