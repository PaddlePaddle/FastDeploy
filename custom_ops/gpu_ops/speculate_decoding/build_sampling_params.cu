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
#include "paddle/extension.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

constexpr int64_t MAX_INFER_SEED = 9223372036854775806;

__global__ void BuildSamplingParamsKernel(float *top_p_padding,
                                          int64_t *top_k_padding,
                                          int64_t *topp_seed,
                                          const float *top_p,
                                          const int64_t *top_k,
                                          int64_t *infer_seed,
                                          const int *cu_seqlens_q_output,
                                          const int64_t increment_value) {
  const int tid = threadIdx.x;
  const int bi = blockIdx.x;
  int cur_seq_len_q_output_start = cu_seqlens_q_output[bi];
  int cur_seq_len_q_output_end = cu_seqlens_q_output[bi + 1];
  const float bi_top_p = top_p[bi];
  const int64_t bi_top_k = top_k[bi];
  int64_t bi_infer_seed = (infer_seed[bi] + tid * 4) % MAX_INFER_SEED;

  for (int i = tid; i < cur_seq_len_q_output_end - cur_seq_len_q_output_start;
       i += blockDim.x) {
    int pad_idx = cur_seq_len_q_output_start + i;
    top_p_padding[pad_idx] = bi_top_p;
    top_k_padding[pad_idx] = bi_top_k;
    topp_seed[pad_idx] = bi_infer_seed;
    bi_infer_seed = (bi_infer_seed + blockDim.x * 4) % MAX_INFER_SEED;
  }

  if (tid == 0) {
    infer_seed[bi] = (infer_seed[bi] + increment_value) % MAX_INFER_SEED;
  }
}

std::vector<paddle::Tensor> BuildSamplingParams(
    const paddle::Tensor &top_p,
    const paddle::Tensor &top_k,
    paddle::Tensor &infer_seed,
    const paddle::Tensor &seq_lens_this_time,
    const paddle::Tensor &cu_seqlens_q_output,
    const int64_t token_num_output_cpu,
    const int64_t increment_value) {
  auto cu_stream = seq_lens_this_time.stream();
  int real_bsz = seq_lens_this_time.shape()[0];
  paddle::Tensor top_p_padding = paddle::empty({token_num_output_cpu, 1},
                                               paddle::DataType::FLOAT32,
                                               seq_lens_this_time.place());
  paddle::Tensor top_k_padding = paddle::empty({token_num_output_cpu, 1},
                                               paddle::DataType::INT64,
                                               seq_lens_this_time.place());
  paddle::Tensor topp_seed = paddle::empty({token_num_output_cpu, 1},
                                           paddle::DataType::INT64,
                                           seq_lens_this_time.place());

  BuildSamplingParamsKernel<<<real_bsz, 64, 0, cu_stream>>>(
      top_p_padding.data<float>(),
      top_k_padding.data<int64_t>(),
      topp_seed.data<int64_t>(),
      top_p.data<float>(),
      top_k.data<int64_t>(),
      infer_seed.data<int64_t>(),
      cu_seqlens_q_output.data<int>(),
      increment_value);

  return {top_p_padding, top_k_padding, topp_seed};
}

PD_BUILD_STATIC_OP(build_sampling_params)
    .Inputs({"top_p",
             "top_k",
             "infer_seed",
             "seq_lens_this_time",
             "cu_seqlens_q_output"})
    .Outputs({"top_p_padding", "top_k_padding", "topp_seed"})
    .Attrs({"token_num_output_cpu: int64_t", "increment_value: int64_t"})
    .SetKernelFn(PD_KERNEL(BuildSamplingParams));
