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

template <typename T>
__global__ void BuildSamplingParamLogProbKernel(
    T* output_params,
    const T* input_params,
    const int32_t* token_num_per_batch,
    const int64_t token_num_output_cpu) {
  const int bi = blockIdx.x;
  const int tid = threadIdx.x;

  // Compute start offset: sum of token_num_per_batch[0..bi-1]
  int start_offset = 0;
  for (int i = 0; i < bi; i++) {
    start_offset += token_num_per_batch[i];
  }

  int cur_token_num = token_num_per_batch[bi];

  if (cur_token_num <= 0) {
    return;
  }

  // Read per-batch param into register
  T val = input_params[bi];

  // Fill output_params with bounds check against total output size
  for (int i = tid; i < cur_token_num; i += blockDim.x) {
    int64_t idx = static_cast<int64_t>(start_offset) + i;
    if (idx < token_num_output_cpu) {
      output_params[idx] = val;
    }
  }
}

std::vector<paddle::Tensor> BuildSamplingParamLogProb(
    const paddle::Tensor& input_params,
    const paddle::Tensor& token_num_per_batch,
    const int64_t token_num_output_cpu) {
  auto cu_stream = input_params.stream();
  // Initialize output to safe defaults for use as divisors:
  // int32/float32 -> 1, bool -> false
  paddle::Tensor output_params;
  switch (input_params.dtype()) {
    case paddle::DataType::BOOL:
      output_params = paddle::full({token_num_output_cpu},
                                   false,
                                   input_params.dtype(),
                                   input_params.place());
      break;
    case paddle::DataType::INT32:
      output_params = paddle::full({token_num_output_cpu},
                                   1,
                                   input_params.dtype(),
                                   input_params.place());
      break;
    case paddle::DataType::FLOAT32:
      output_params = paddle::full({token_num_output_cpu},
                                   1.0f,
                                   input_params.dtype(),
                                   input_params.place());
      break;
    default:
      PD_THROW(
          "Unsupported data type for BuildSamplingParamLogProb. "
          "Only bool, int32, float32 are supported.");
  }

  int32_t num_blocks = token_num_per_batch.shape()[0];
  switch (input_params.dtype()) {
    case paddle::DataType::BOOL: {
      BuildSamplingParamLogProbKernel<bool><<<num_blocks, 256, 0, cu_stream>>>(
          output_params.data<bool>(),
          input_params.data<bool>(),
          token_num_per_batch.data<int32_t>(),
          token_num_output_cpu);
      break;
    }
    case paddle::DataType::INT32: {
      BuildSamplingParamLogProbKernel<int32_t>
          <<<num_blocks, 256, 0, cu_stream>>>(
              output_params.data<int32_t>(),
              input_params.data<int32_t>(),
              token_num_per_batch.data<int32_t>(),
              token_num_output_cpu);
      break;
    }
    case paddle::DataType::FLOAT32: {
      BuildSamplingParamLogProbKernel<float><<<num_blocks, 256, 0, cu_stream>>>(
          output_params.data<float>(),
          input_params.data<float>(),
          token_num_per_batch.data<int32_t>(),
          token_num_output_cpu);
      break;
    }
    default: {
      PD_THROW(
          "Unsupported data type for BuildSamplingParamLogProb. "
          "Only bool, int32, float32 are supported.");
    }
  }

  return {output_params};
}

PD_BUILD_STATIC_OP(build_sampling_params_logprob)
    .Inputs({"input_params", "token_num_per_batch"})
    .Outputs({"output_params"})
    .Attrs({"token_num_output_cpu: int64_t"})
    .SetKernelFn(PD_KERNEL(BuildSamplingParamLogProb));
