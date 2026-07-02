// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

template <typename T, int VecSize>
__global__ void MixedReorderHiddenStatesKernel(const T* input,
                                               T* output,
                                               const int* seq_lens_encoder,
                                               const int* seq_lens_decoder,
                                               const int* seq_lens_this_time,
                                               const int prefill_num_tokens,
                                               const int hidden_dim,
                                               const bool reverse) {
  using LoadT = AlignedVector<T, VecSize>;

  const int bid = blockIdx.x;
  const int seq_len = seq_lens_this_time[bid];
  if (seq_len <= 0) {
    return;
  }

  const bool is_prefill = seq_lens_encoder[bid] > 0;
  const bool is_decode = !is_prefill && seq_lens_decoder[bid] > 0;
  if (!is_prefill && !is_decode) {
    return;
  }

  int original_start = 0;
  int reordered_start = 0;
  int decode_rank = 0;
  for (int i = 0; i < bid; ++i) {
    original_start += seq_lens_this_time[i];
    if (seq_lens_encoder[i] > 0) {
      reordered_start += seq_lens_encoder[i];
    } else if (seq_lens_decoder[i] > 0) {
      ++decode_rank;
    }
  }

  int copy_seq_len = seq_len;
  if (is_decode) {
    reordered_start = prefill_num_tokens + decode_rank;
    // Decode mixed attention consumes one query token for each decode request.
    copy_seq_len = 1;
  }

  for (int idx = threadIdx.x * VecSize; idx < copy_seq_len * hidden_dim;
       idx += blockDim.x * VecSize) {
    const int token_offset = idx / hidden_dim;
    const int hidden_offset = idx % hidden_dim;
    const int original_offset =
        (original_start + token_offset) * hidden_dim + hidden_offset;
    const int reordered_offset =
        (reordered_start + token_offset) * hidden_dim + hidden_offset;

    LoadT src_vec;
    if (reverse) {
      Load<T, VecSize>(&input[reordered_offset], &src_vec);
      Store<T, VecSize>(src_vec, &output[original_offset]);
    } else {
      Load<T, VecSize>(&input[original_offset], &src_vec);
      Store<T, VecSize>(src_vec, &output[reordered_offset]);
    }
  }
}

template <paddle::DataType D, int VecSize>
void LaunchMixedReorderHiddenStates(const paddle::Tensor& hidden_states,
                                    const paddle::Tensor& seq_lens_encoder,
                                    const paddle::Tensor& seq_lens_decoder,
                                    const paddle::Tensor& seq_lens_this_time,
                                    const int prefill_num_tokens,
                                    const bool reverse,
                                    paddle::Tensor* out) {
  typedef PDTraits<D> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t;

  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(
          hidden_states.place()));
  auto stream = dev_ctx->stream();

  const auto hidden_shape = hidden_states.shape();
  const int hidden_dim = hidden_shape[1];
  const int max_num_seqs = seq_lens_this_time.shape()[0];
  const int block_size = 128;

  MixedReorderHiddenStatesKernel<DataType_, VecSize>
      <<<max_num_seqs, block_size, 0, stream>>>(
          reinterpret_cast<const DataType_*>(hidden_states.data<data_t>()),
          reinterpret_cast<DataType_*>(out->data<data_t>()),
          seq_lens_encoder.data<int>(),
          seq_lens_decoder.data<int>(),
          seq_lens_this_time.data<int>(),
          prefill_num_tokens,
          hidden_dim,
          reverse);
}

template <paddle::DataType D>
paddle::Tensor MixedReorderHiddenStatesImpl(
    const paddle::Tensor& hidden_states,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& seq_lens_this_time,
    const int prefill_num_tokens,
    const bool reverse) {
  typedef PDTraits<D> traits_;
  typedef typename traits_::DataType DataType_;

  const auto hidden_shape = hidden_states.shape();
  PADDLE_ENFORCE_EQ(hidden_shape.size(),
                    2,
                    common::errors::InvalidArgument(
                        "hidden_states must be a 2-D tensor, but got %d dims.",
                        hidden_shape.size()));

  auto out = GetEmptyTensor({hidden_shape[0], hidden_shape[1]},
                            hidden_states.dtype(),
                            hidden_states.place());

  constexpr int PackSize = VEC_16B / sizeof(DataType_);
  if (hidden_shape[1] % PackSize == 0) {
    LaunchMixedReorderHiddenStates<D, PackSize>(hidden_states,
                                                seq_lens_encoder,
                                                seq_lens_decoder,
                                                seq_lens_this_time,
                                                prefill_num_tokens,
                                                reverse,
                                                &out);
  } else {
    LaunchMixedReorderHiddenStates<D, 1>(hidden_states,
                                         seq_lens_encoder,
                                         seq_lens_decoder,
                                         seq_lens_this_time,
                                         prefill_num_tokens,
                                         reverse,
                                         &out);
  }

  return out;
}

paddle::Tensor MixedReorderHiddenStatesFunc(
    const paddle::Tensor& hidden_states,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& seq_lens_this_time,
    int prefill_num_tokens,
    bool reverse) {
  switch (hidden_states.type()) {
    case paddle::DataType::BFLOAT16: {
      return MixedReorderHiddenStatesImpl<paddle::DataType::BFLOAT16>(
          hidden_states,
          seq_lens_encoder,
          seq_lens_decoder,
          seq_lens_this_time,
          prefill_num_tokens,
          reverse);
    }
    case paddle::DataType::FLOAT16: {
      return MixedReorderHiddenStatesImpl<paddle::DataType::FLOAT16>(
          hidden_states,
          seq_lens_encoder,
          seq_lens_decoder,
          seq_lens_this_time,
          prefill_num_tokens,
          reverse);
    }
    case paddle::DataType::FLOAT32: {
      return MixedReorderHiddenStatesImpl<paddle::DataType::FLOAT32>(
          hidden_states,
          seq_lens_encoder,
          seq_lens_decoder,
          seq_lens_this_time,
          prefill_num_tokens,
          reverse);
    }
    default: {
      PD_THROW(
          "NOT supported data type. "
          "Only float16, bfloat16 and float32 are supported. ");
    }
  }
}

std::vector<paddle::Tensor> MixedReorderHiddenStates(
    const paddle::Tensor& hidden_states,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& seq_lens_this_time,
    int prefill_num_tokens,
    bool reverse) {
  return {MixedReorderHiddenStatesFunc(hidden_states,
                                       seq_lens_encoder,
                                       seq_lens_decoder,
                                       seq_lens_this_time,
                                       prefill_num_tokens,
                                       reverse)};
}

std::vector<std::vector<int64_t>> MixedReorderHiddenStatesInferShape(
    const std::vector<int64_t>& hidden_states_shape,
    const std::vector<int64_t>& seq_lens_encoder_shape,
    const std::vector<int64_t>& seq_lens_decoder_shape,
    const std::vector<int64_t>& seq_lens_this_time_shape) {
  return {hidden_states_shape};
}

std::vector<paddle::DataType> MixedReorderHiddenStatesInferDtype(
    const paddle::DataType& hidden_states_dtype,
    const paddle::DataType& seq_lens_encoder_dtype,
    const paddle::DataType& seq_lens_decoder_dtype,
    const paddle::DataType& seq_lens_this_time_dtype) {
  return {hidden_states_dtype};
}

PD_BUILD_STATIC_OP(mixed_reorder_hidden_states)
    .Inputs({"hidden_states",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "seq_lens_this_time"})
    .Outputs({"out"})
    .Attrs({"prefill_num_tokens:int", "reverse:bool"})
    .SetKernelFn(PD_KERNEL(MixedReorderHiddenStates))
    .SetInferShapeFn(PD_INFER_SHAPE(MixedReorderHiddenStatesInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MixedReorderHiddenStatesInferDtype));
