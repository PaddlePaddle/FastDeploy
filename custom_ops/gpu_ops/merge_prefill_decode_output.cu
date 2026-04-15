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

#include "paddle/extension.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

template <int warps, typename T>
__global__ void FillEncoderDecoderResKernel(T *encoder_res_data,
                                            T *decoder_res_data,
                                            const int *seq_lens_encoder,
                                            const int *seq_lens_decoder,
                                            const int *seq_lens_this_time,
                                            const int *cu_seq_q,
                                            const int head_num,
                                            const int head_dim) {
  const int bidb = blockIdx.x;
  const int bidh = blockIdx.y;
  const int bidt = blockIdx.z * warps;
  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int land_id = tid % 32;
  const int token_id = bidt + warp_id;

  const int seq_len_encoder = seq_lens_encoder[bidb];
  const int seq_len_decoder = seq_lens_decoder[bidb];
  const int seq_len_this_time = seq_lens_this_time[bidb];

  if (seq_len_encoder > 0 || seq_len_decoder == 0 ||
      token_id >= seq_len_this_time) {
    return;
  }

  const int base_idx =
      ((cu_seq_q[bidb] + token_id) * head_num + bidh) * head_dim;

  if (head_dim == 128) {
    const int load_idx = base_idx + land_id * 4;
    *reinterpret_cast<float2 *>(encoder_res_data + load_idx) =
        *reinterpret_cast<float2 *>(decoder_res_data + load_idx);
  } else if (head_dim == 192) {
    const int load_idx = base_idx + land_id * 4;
    *reinterpret_cast<float2 *>(encoder_res_data + load_idx) =
        *reinterpret_cast<float2 *>(decoder_res_data + load_idx);
    if (land_id < 16) {
      *reinterpret_cast<float2 *>(encoder_res_data + load_idx + 128) =
          *reinterpret_cast<float2 *>(decoder_res_data + load_idx + 128);
    }
  } else if (head_dim == 256) {
    // float4 = 单条LDG.128，性能最优
    const int load_idx = base_idx + land_id * 8;
    *reinterpret_cast<float4 *>(encoder_res_data + load_idx) =
        *reinterpret_cast<float4 *>(decoder_res_data + load_idx);
  }
}

#define LAUNCH_KERNEL(T, WARPS)                           \
  FillEncoderDecoderResKernel<WARPS>                      \
      <<<grid_dims, head_dim, 0, encoder_res.stream()>>>( \
          const_cast<T *>(encoder_res.data<T>()),         \
          const_cast<T *>(decoder_res.data<T>()),         \
          seq_lens_encoder.data<int>(),                   \
          seq_lens_decoder.data<int>(),                   \
          seq_lens_this_time.data<int>(),                 \
          cu_seq_q.data<int>(),                           \
          head_num,                                       \
          head_dim)

#define LAUNCH_KERNEL_BY_HEAD_DIM(T) \
  if (head_dim == 128)               \
    LAUNCH_KERNEL(T, 4);             \
  else if (head_dim == 192)          \
    LAUNCH_KERNEL(T, 6);             \
  else if (head_dim == 256)          \
  LAUNCH_KERNEL(T, 8)

void MergePrefillDecodeOutput(const paddle::Tensor &encoder_res,
                              const paddle::Tensor &decoder_res,
                              const paddle::Tensor &seq_lens_encoder,
                              const paddle::Tensor &seq_lens_decoder,
                              const paddle::Tensor &seq_lens_this_time,
                              const paddle::Tensor &cu_seq_q,
                              const int head_num,
                              const int head_dim,
                              const int max_token) {
  if (head_dim != 128 && head_dim != 192 && head_dim != 256) {
    PD_THROW("Only supported head_dim = 128, 192 or 256");
  }
  const int batch_size = seq_lens_encoder.shape()[0];
  const int warps = head_dim / 32;
  const int tokens_block = (max_token + warps - 1) / warps;
  dim3 grid_dims(batch_size, head_num, tokens_block);

  if (encoder_res.dtype() == paddle::DataType::FLOAT16) {
    using T = phi::dtype::float16;
    LAUNCH_KERNEL_BY_HEAD_DIM(T);
  } else if (encoder_res.dtype() == paddle::DataType::BFLOAT16) {
    using T = phi::dtype::bfloat16;
    LAUNCH_KERNEL_BY_HEAD_DIM(T);
  }
}

PD_BUILD_STATIC_OP(merge_prefill_decode_output)
    .Inputs({"encoder_res",
             "decoder_res",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "seq_lens_this_time",
             "cu_seq_q"})
    .Outputs({"res"})
    .Attrs({"head_num: int", "head_dim: int", "max_token: int"})
    .SetInplaceMap({{"encoder_res", "res"}})
    .SetKernelFn(PD_KERNEL(MergePrefillDecodeOutput));
