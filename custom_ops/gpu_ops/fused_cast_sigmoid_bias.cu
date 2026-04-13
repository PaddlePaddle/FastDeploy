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

// Fused kernel: cast(input, cast_type) -> sigmoid -> scores, scores + bias ->
// scores_with_bias
//
// For each element (token i, expert j):
//   scores[i][j] = OutT(sigmoid(float(input[i][j])))
//   scores_with_bias[i][j] = OutT(sigmoid(float(input[i][j])) + bias[j])
//
// Input:  input [num_tokens, num_experts] bf16/fp16/fp32
//         bias  [num_experts] or [1, num_experts] fp32
// Output: scores [num_tokens, num_experts] cast_type (fp32/fp16/bf16)
//         scores_with_bias [num_tokens, num_experts] cast_type (fp32/fp16/bf16)
//
// Precision guarantee:
//   All intermediate computations (cast, sigmoid, bias addition) are performed
//   in float32, regardless of input/output types. The cast to OutT only happens
//   at the final store. This matches the reference implementation:
//     gate_fp32 = gate_out.cast("float32")
//     scores_fp32 = sigmoid(gate_fp32)
//     scores_with_bias_fp32 = scores_fp32 + bias  // bias is always float32
//     scores = scores_fp32.cast(cast_type)
//     scores_with_bias = scores_with_bias_fp32.cast(cast_type)
//
//   When cast_type is "float32", the fused kernel is numerically identical to
//   the reference. For fp16/bf16 output, the only precision loss comes from
//   the final static_cast<OutT>, equivalent to .cast() in the reference path.
//
//   Note: bias is intentionally kept as float32 (not converted to OutT) to
//   ensure the addition s + bias[j] is always computed in full float32
//   precision before the final downcast.

template <typename InT, typename OutT>
__global__ void fused_cast_sigmoid_bias_kernel(
    const InT* __restrict__ input,
    const float* __restrict__ bias,
    OutT* __restrict__ scores,
    OutT* __restrict__ scores_with_bias,
    const int num_experts) {
  const int64_t token_idx = blockIdx.x;
  const int64_t offset = token_idx * num_experts;

  for (int j = threadIdx.x; j < num_experts; j += blockDim.x) {
    // All intermediate computation in float32 for precision
    float val = static_cast<float>(input[offset + j]);
    float s = 1.0f / (1.0f + expf(-val));
    // s (float32) + bias[j] (float32) -> float32 addition, then downcast
    scores[offset + j] = static_cast<OutT>(s);
    scores_with_bias[offset + j] = static_cast<OutT>(s + bias[j]);
  }
}

// Vectorized version for better memory throughput
template <typename InT, typename OutT, int kVecSize>
__global__ void fused_cast_sigmoid_bias_vec_kernel(
    const InT* __restrict__ input,
    const float* __restrict__ bias,  // kept as float32 for full-precision add
    OutT* __restrict__ scores,
    OutT* __restrict__ scores_with_bias,
    const int num_experts) {
  const int64_t token_idx = blockIdx.x;
  const int64_t offset = token_idx * num_experts;

  using in_vec_t = AlignedVector<InT, kVecSize>;
  using out_vec_t = AlignedVector<OutT, kVecSize>;
  using bias_vec_t = AlignedVector<float, kVecSize>;  // float32 bias vectors

  const int vec_count = num_experts / kVecSize;
  for (int idx = threadIdx.x; idx < vec_count; idx += blockDim.x) {
    const int base = idx * kVecSize;
    in_vec_t in_vec;
    bias_vec_t bias_vec;
    Load(input + offset + base, &in_vec);
    Load(bias + base, &bias_vec);

    out_vec_t s_vec, sb_vec;
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      // All intermediate computation in float32 for precision
      float val = static_cast<float>(in_vec[i]);
      float s = 1.0f / (1.0f + expf(-val));
      // s (float32) + bias_vec[i] (float32) -> float32 addition, then downcast
      s_vec[i] = static_cast<OutT>(s);
      sb_vec[i] = static_cast<OutT>(s + bias_vec[i]);
    }

    Store(s_vec, scores + offset + base);
    Store(sb_vec, scores_with_bias + offset + base);
  }

  // Handle remaining elements (same float32 precision guarantee)
  const int remaining_start = vec_count * kVecSize;
  for (int j = remaining_start + threadIdx.x; j < num_experts;
       j += blockDim.x) {
    float val = static_cast<float>(input[offset + j]);
    float s = 1.0f / (1.0f + expf(-val));
    scores[offset + j] = static_cast<OutT>(s);
    scores_with_bias[offset + j] = static_cast<OutT>(s + bias[j]);
  }
}

static paddle::DataType ParseCastType(const std::string& cast_type) {
  if (cast_type == "float32") return paddle::DataType::FLOAT32;
  if (cast_type == "float16") return paddle::DataType::FLOAT16;
  if (cast_type == "bfloat16") return paddle::DataType::BFLOAT16;
  PD_THROW("Unsupported cast_type: " + cast_type +
           ". Only float32, float16, bfloat16 are supported.");
}

std::vector<paddle::Tensor> FusedCastSigmoidBias(const paddle::Tensor& input,
                                                 const paddle::Tensor& bias,
                                                 std::string cast_type) {
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
  PD_CHECK(bias.dtype() == paddle::DataType::FLOAT32,
           "bias must be float32, got ",
           bias.dtype());

  auto place = input.place();
  auto stream = input.stream();
  auto out_dtype = ParseCastType(cast_type);

  auto scores = paddle::empty({num_tokens, num_experts}, out_dtype, place);
  auto scores_with_bias =
      paddle::empty({num_tokens, num_experts}, out_dtype, place);

  if (num_tokens == 0) {
    return {scores, scores_with_bias};
  }

  dim3 grid(num_tokens);
  int block_size = std::min(static_cast<int64_t>(1024), num_experts);
  // Round up to warp size
  block_size = ((block_size + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
  dim3 block(block_size);

  DISPATCH_FLOAT_FP6_DTYPE(input.dtype(), in_scalar_t, {
    DISPATCH_FLOAT_FP6_DTYPE(out_dtype, out_scalar_t, {
      constexpr int kVecSize = 16 / sizeof(in_scalar_t);
      if (num_experts % kVecSize == 0 && num_experts >= kVecSize) {
        fused_cast_sigmoid_bias_vec_kernel<in_scalar_t, out_scalar_t, kVecSize>
            <<<grid, block, 0, stream>>>(input.data<in_scalar_t>(),
                                         bias.data<float>(),
                                         scores.data<out_scalar_t>(),
                                         scores_with_bias.data<out_scalar_t>(),
                                         num_experts);
      } else {
        fused_cast_sigmoid_bias_kernel<in_scalar_t, out_scalar_t>
            <<<grid, block, 0, stream>>>(input.data<in_scalar_t>(),
                                         bias.data<float>(),
                                         scores.data<out_scalar_t>(),
                                         scores_with_bias.data<out_scalar_t>(),
                                         num_experts);
      }
    });
  });

  return {scores, scores_with_bias};
}

std::vector<paddle::DataType> FusedCastSigmoidBiasInferDtype(
    const paddle::DataType& input_dtype,
    const paddle::DataType& bias_dtype,
    std::string cast_type) {
  auto out_dtype = ParseCastType(cast_type);
  return {out_dtype, out_dtype};
}

std::vector<std::vector<int64_t>> FusedCastSigmoidBiasInferShape(
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& bias_shape) {
  return {input_shape, input_shape};
}

PD_BUILD_STATIC_OP(fused_cast_sigmoid_bias)
    .Inputs({"input", "bias"})
    .Outputs({"scores", "scores_with_bias"})
    .Attrs({"cast_type: std::string"})
    .SetKernelFn(PD_KERNEL(FusedCastSigmoidBias))
    .SetInferShapeFn(PD_INFER_SHAPE(FusedCastSigmoidBiasInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FusedCastSigmoidBiasInferDtype));
