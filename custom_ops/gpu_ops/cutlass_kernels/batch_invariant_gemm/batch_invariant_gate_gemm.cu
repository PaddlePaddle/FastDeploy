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

#include "batch_invariant_gate_gemm.cuh"

namespace fastdeploy {

// Tile configs for Gate GEMM: N=256 fits in one N-tile
// bf16/fp16: 128x256x64 — one N-tile covers full N dimension
using TileShape_bf16 = cute::Shape<cute::_128, cute::_256, cute::_64>;
// fp32: 128x128x64 — smaller N-tile due to larger element size
using TileShape_fp32 = cute::Shape<cute::_128, cute::_128, cute::_64>;
using ClusterShape_1x1x1 = cute::Shape<cute::_1, cute::_1, cute::_1>;

// --- Explicit template instantiation types ---

// bf16, no bias
using GateGemm_bf16_nobias = GateGemmSm90<cutlass::bfloat16_t,
                                          cutlass::bfloat16_t,
                                          false,
                                          TileShape_bf16,
                                          ClusterShape_1x1x1>;

// bf16, with bias
using GateGemm_bf16_bias = GateGemmSm90<cutlass::bfloat16_t,
                                        cutlass::bfloat16_t,
                                        true,
                                        TileShape_bf16,
                                        ClusterShape_1x1x1>;

// fp16, no bias
using GateGemm_fp16_nobias = GateGemmSm90<cutlass::half_t,
                                          cutlass::half_t,
                                          false,
                                          TileShape_bf16,
                                          ClusterShape_1x1x1>;

// fp16, with bias
using GateGemm_fp16_bias = GateGemmSm90<cutlass::half_t,
                                        cutlass::half_t,
                                        true,
                                        TileShape_bf16,
                                        ClusterShape_1x1x1>;

// fp32, no bias
using GateGemm_fp32_nobias =
    GateGemmSm90<float, float, false, TileShape_fp32, ClusterShape_1x1x1>;

// fp32, with bias
using GateGemm_fp32_bias =
    GateGemmSm90<float, float, true, TileShape_fp32, ClusterShape_1x1x1>;

template <typename GemmNoBias, typename GemmBias>
void dispatch_gate_gemm(paddle::Tensor &c,
                        paddle::Tensor const &a,
                        paddle::Tensor const &b,
                        paddle::optional<paddle::Tensor> const &bias) {
  int M = a.dims()[0];
  int N = b.dims()[0];
  int K = a.dims()[1];

  if (bias) {
    launch_gate_gemm<GemmBias>(c, a, b, bias->data(), M, N, K);
  } else {
    launch_gate_gemm<GemmNoBias>(c, a, b, nullptr, M, N, K);
  }
}

}  // namespace fastdeploy

void BatchInvariantGateGemm(paddle::Tensor &c,
                            paddle::Tensor const &a,
                            paddle::Tensor const &b,
                            paddle::optional<paddle::Tensor> const &bias) {
  // a: [M, K] row-major, b: [N, K] (pre-transposed weight), c: [M, N] output
  PD_CHECK(a.dims().size() == 2 && b.dims().size() == 2 && c.dims().size() == 2,
           "All inputs must be 2D tensors");
  PD_CHECK(a.dims()[1] == b.dims()[1],
           "K dimension mismatch: a.shape[1]=",
           a.dims()[1],
           " vs b.shape[1]=",
           b.dims()[1]);
  PD_CHECK(c.dims()[0] == a.dims()[0] && c.dims()[1] == b.dims()[0],
           "Output shape mismatch");
  PD_CHECK(a.dtype() == b.dtype(), "Input dtypes must match");

  // Row-major check
  PD_CHECK(a.strides()[1] == 1, "A must be row-major (contiguous)");
  PD_CHECK(c.strides()[1] == 1, "C must be row-major (contiguous)");

  if (bias) {
    PD_CHECK(
        bias->numel() == b.dims()[0], "Bias size must equal N=", b.dims()[0]);
    PD_CHECK(bias->is_contiguous() && bias->dims().size() == 1);
  }

  auto dtype = a.dtype();
  if (dtype == paddle::DataType::BFLOAT16) {
    fastdeploy::dispatch_gate_gemm<fastdeploy::GateGemm_bf16_nobias,
                                   fastdeploy::GateGemm_bf16_bias>(
        c, a, b, bias);
  } else if (dtype == paddle::DataType::FLOAT16) {
    fastdeploy::dispatch_gate_gemm<fastdeploy::GateGemm_fp16_nobias,
                                   fastdeploy::GateGemm_fp16_bias>(
        c, a, b, bias);
  } else if (dtype == paddle::DataType::FLOAT32) {
    fastdeploy::dispatch_gate_gemm<fastdeploy::GateGemm_fp32_nobias,
                                   fastdeploy::GateGemm_fp32_bias>(
        c, a, b, bias);
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "batch_invariant_gate_gemm: unsupported dtype"));
  }
}

PD_BUILD_STATIC_OP(batch_invariant_gate_gemm)
    .Inputs({"c", "a", "b", paddle::Optional("bias")})
    .Outputs({"c_out"})
    .SetInplaceMap({{"c", "c_out"}})
    .SetKernelFn(PD_KERNEL(BatchInvariantGateGemm));
