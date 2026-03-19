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

// Fused kernel: cast(input, fp32) -> sigmoid -> scores, scores + bias ->
// scores_with_bias
//
// For each element (token i, expert j):
//   scores[i][j] = sigmoid(float(input[i][j]))
//   scores_with_bias[i][j] = scores[i][j] + bias[j]
//
// Input:  input [num_tokens, num_experts] bf16/fp16/fp32
//         bias  [num_experts] or [1, num_experts] fp32
// Output: scores [num_tokens, num_experts] fp32
//         scores_with_bias [num_tokens, num_experts] fp32

template <typename InT>
__global__ void fused_cast_sigmoid_bias_kernel(
    const InT* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ scores,
    float* __restrict__ scores_with_bias,
    const int num_experts) {
  const int64_t token_idx = blockIdx.x;
  const int64_t offset = token_idx * num_experts;

  for (int j = threadIdx.x; j < num_experts; j += blockDim.x) {
    float val = static_cast<float>(input[offset + j]);
    // sigmoid: 1 / (1 + exp(-x))
    float s = 1.0f / (1.0f + expf(-val));
    scores[offset + j] = s;
    scores_with_bias[offset + j] = s + bias[j];
  }
}

// Vectorized version for better memory throughput
template <typename InT, int kVecSize>
__global__ void fused_cast_sigmoid_bias_vec_kernel(
    const InT* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ scores,
    float* __restrict__ scores_with_bias,
    const int num_experts) {
  const int64_t token_idx = blockIdx.x;
  const int64_t offset = token_idx * num_experts;

  using in_vec_t = AlignedVector<InT, kVecSize>;
  using out_vec_t = AlignedVector<float, kVecSize>;

  const int vec_count = num_experts / kVecSize;
  for (int idx = threadIdx.x; idx < vec_count; idx += blockDim.x) {
    const int base = idx * kVecSize;
    in_vec_t in_vec;
    out_vec_t bias_vec;
    Load(input + offset + base, &in_vec);
    Load(bias + base, &bias_vec);

    out_vec_t s_vec, sb_vec;
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      float val = static_cast<float>(in_vec[i]);
      float s = 1.0f / (1.0f + expf(-val));
      s_vec[i] = s;
      sb_vec[i] = s + bias_vec[i];
    }

    Store(s_vec, scores + offset + base);
    Store(sb_vec, scores_with_bias + offset + base);
  }

  // Handle remaining elements
  const int remaining_start = vec_count * kVecSize;
  for (int j = remaining_start + threadIdx.x; j < num_experts;
       j += blockDim.x) {
    float val = static_cast<float>(input[offset + j]);
    float s = 1.0f / (1.0f + expf(-val));
    scores[offset + j] = s;
    scores_with_bias[offset + j] = s + bias[j];
  }
}

std::vector<paddle::Tensor> FusedCastSigmoidBias(const paddle::Tensor& input,
                                                 const paddle::Tensor& bias) {
  auto input_shape = input.shape();
  PD_CHECK(input_shape.size() == 2,
           "input must be 2D [num_tokens, num_experts]");
  auto bias_shape = bias.shape();
  // Support both [num_experts] and [1, num_experts] bias shapes
  PD_CHECK(
      bias_shape.size() == 1 || (bias_shape.size() == 2 && bias_shape[0] == 1),
      "bias must be 1D [num_experts] or 2D [1, num_experts]");

  int64_t num_tokens = input_shape[0];
  int64_t num_experts = input_shape[1];
  int64_t bias_numel = (bias_shape.size() == 1) ? bias_shape[0] : bias_shape[1];
  PD_CHECK(bias_numel == num_experts, "bias size must match num_experts");

  auto place = input.place();
  auto stream = input.stream();

  auto scores = paddle::empty(
      {num_tokens, num_experts}, paddle::DataType::FLOAT32, place);
  auto scores_with_bias = paddle::empty(
      {num_tokens, num_experts}, paddle::DataType::FLOAT32, place);

  if (num_tokens == 0) {
    return {scores, scores_with_bias};
  }

  dim3 grid(num_tokens);
  int block_size = std::min(static_cast<int64_t>(1024), num_experts);
  // Round up to warp size
  block_size = ((block_size + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
  dim3 block(block_size);

  DISPATCH_FLOAT_FP6_DTYPE(input.dtype(), scalar_t, {
    constexpr int kVecSize = 16 / sizeof(scalar_t);
    if (num_experts % kVecSize == 0 && num_experts >= kVecSize) {
      fused_cast_sigmoid_bias_vec_kernel<scalar_t, kVecSize>
          <<<grid, block, 0, stream>>>(input.data<scalar_t>(),
                                       bias.data<float>(),
                                       scores.data<float>(),
                                       scores_with_bias.data<float>(),
                                       num_experts);
    } else {
      fused_cast_sigmoid_bias_kernel<scalar_t>
          <<<grid, block, 0, stream>>>(input.data<scalar_t>(),
                                       bias.data<float>(),
                                       scores.data<float>(),
                                       scores_with_bias.data<float>(),
                                       num_experts);
    }
  });

  return {scores, scores_with_bias};
}

std::vector<paddle::DataType> FusedCastSigmoidBiasInferDtype(
    const paddle::DataType& input_dtype, const paddle::DataType& bias_dtype) {
  return {paddle::DataType::FLOAT32, paddle::DataType::FLOAT32};
}

std::vector<std::vector<int64_t>> FusedCastSigmoidBiasInferShape(
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& bias_shape) {
  return {input_shape, input_shape};
}

PD_BUILD_STATIC_OP(fused_cast_sigmoid_bias)
    .Inputs({"input", "bias"})
    .Outputs({"scores", "scores_with_bias"})
    .SetKernelFn(PD_KERNEL(FusedCastSigmoidBias))
    .SetInferShapeFn(PD_INFER_SHAPE(FusedCastSigmoidBiasInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FusedCastSigmoidBiasInferDtype));
