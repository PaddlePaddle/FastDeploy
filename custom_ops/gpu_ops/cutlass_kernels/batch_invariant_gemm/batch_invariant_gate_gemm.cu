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

// Tile configs for Gate GEMM (N=256, K=7168)
// bf16/fp16: 128x32x64 — Cooperative, 8 N-tiles for N=256
using TileShape_bf16 = cute::Shape<cute::_128, cute::_32, cute::_64>;
// bf16/fp16: 64x32x64 — Non-Cooperative StreamK, 2 M-tiles for M=128
using TileShape_small = cute::Shape<cute::_64, cute::_32, cute::_64>;
// P1: K_tile=128 — halves K iterations (112→56), better TMA pipeline efficiency
using TileShape_small_k128 = cute::Shape<cute::_64, cute::_32, cute::_128>;
// fp32: 128x128x64
using TileShape_fp32 = cute::Shape<cute::_128, cute::_128, cute::_64>;
using ClusterShape_1x1x1 = cute::Shape<cute::_1, cute::_1, cute::_1>;
// P2: Cluster<1,2,1> — 2 N-adjacent CTAs share A via TMA multicast
using ClusterShape_1x2x1 = cute::Shape<cute::_1, cute::_2, cute::_1>;

// --- Cooperative (M_tile=128) instantiations ---

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

// --- Non-Cooperative StreamK (M_tile=64) instantiations ---

using GateGemm_bf16_nobias_small = GateGemmSm90StreamK<cutlass::bfloat16_t,
                                                       cutlass::bfloat16_t,
                                                       false,
                                                       TileShape_small,
                                                       ClusterShape_1x1x1>;

using GateGemm_bf16_bias_small = GateGemmSm90StreamK<cutlass::bfloat16_t,
                                                     cutlass::bfloat16_t,
                                                     true,
                                                     TileShape_small,
                                                     ClusterShape_1x1x1>;

using GateGemm_fp16_nobias_small = GateGemmSm90StreamK<cutlass::half_t,
                                                       cutlass::half_t,
                                                       false,
                                                       TileShape_small,
                                                       ClusterShape_1x1x1>;

using GateGemm_fp16_bias_small = GateGemmSm90StreamK<cutlass::half_t,
                                                     cutlass::half_t,
                                                     true,
                                                     TileShape_small,
                                                     ClusterShape_1x1x1>;

// --- P1: K_tile=128 instantiations (bf16 only for benchmarking) ---

using GateGemm_bf16_nobias_small_k128 =
    GateGemmSm90StreamK<cutlass::bfloat16_t,
                        cutlass::bfloat16_t,
                        false,
                        TileShape_small_k128,
                        ClusterShape_1x1x1>;

using GateGemm_bf16_bias_small_k128 = GateGemmSm90StreamK<cutlass::bfloat16_t,
                                                          cutlass::bfloat16_t,
                                                          true,
                                                          TileShape_small_k128,
                                                          ClusterShape_1x1x1>;

// --- P2: ClusterShape<1,2,1> instantiations (bf16 only) ---

using GateGemm_bf16_nobias_small_cluster =
    GateGemmSm90StreamK<cutlass::bfloat16_t,
                        cutlass::bfloat16_t,
                        false,
                        TileShape_small,
                        ClusterShape_1x2x1>;

using GateGemm_bf16_bias_small_cluster =
    GateGemmSm90StreamK<cutlass::bfloat16_t,
                        cutlass::bfloat16_t,
                        true,
                        TileShape_small,
                        ClusterShape_1x2x1>;

// --- P1+P2: K_tile=128 + ClusterShape<1,2,1> (bf16 only) ---

using GateGemm_bf16_nobias_small_k128_cluster =
    GateGemmSm90StreamK<cutlass::bfloat16_t,
                        cutlass::bfloat16_t,
                        false,
                        TileShape_small_k128,
                        ClusterShape_1x2x1>;

using GateGemm_bf16_bias_small_k128_cluster =
    GateGemmSm90StreamK<cutlass::bfloat16_t,
                        cutlass::bfloat16_t,
                        true,
                        TileShape_small_k128,
                        ClusterShape_1x2x1>;

// Runtime M-based dispatch:
// M >= 128: use TileShape<64,32,64> (2 M-tiles → 64 CTAs, 48% SM)
// M < 128:  use TileShape<128,32,64> (1 M-tile, same CTA count as small tile)
// Override with CUTLASS_GATE_GEMM_TILE=large|small for debugging.
template <typename GemmNoBias,
          typename GemmBias,
          typename GemmNoBiasSmall,
          typename GemmBiasSmall>
void dispatch_gate_gemm(paddle::Tensor &c,
                        paddle::Tensor const &a,
                        paddle::Tensor const &b,
                        paddle::optional<paddle::Tensor> const &bias) {
  int M = a.dims()[0];
  int N = b.dims()[0];
  int K = a.dims()[1];

  bool use_small = (M >= 128);
  const char *env_tile = std::getenv("CUTLASS_GATE_GEMM_TILE");
  if (env_tile) {
    if (std::string(env_tile) == "large")
      use_small = false;
    else if (std::string(env_tile) == "small")
      use_small = true;
  }

  if (use_small) {
    if (bias) {
      launch_gate_gemm<GemmBiasSmall>(c, a, b, bias->data(), M, N, K);
    } else {
      launch_gate_gemm<GemmNoBiasSmall>(c, a, b, nullptr, M, N, K);
    }
  } else {
    if (bias) {
      launch_gate_gemm<GemmBias>(c, a, b, bias->data(), M, N, K);
    } else {
      launch_gate_gemm<GemmNoBias>(c, a, b, nullptr, M, N, K);
    }
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
    // P1/P2 optimization variants (bf16 only).
    // CUTLASS_GATE_GEMM_OPT: k128 | cluster | k128_cluster | (empty=baseline)
    const char *env_opt = std::getenv("CUTLASS_GATE_GEMM_OPT");
    std::string opt = env_opt ? env_opt : "";

    if (opt == "k128_cluster") {
      fastdeploy::dispatch_gate_gemm<
          fastdeploy::GateGemm_bf16_nobias,
          fastdeploy::GateGemm_bf16_bias,
          fastdeploy::GateGemm_bf16_nobias_small_k128_cluster,
          fastdeploy::GateGemm_bf16_bias_small_k128_cluster>(c, a, b, bias);
    } else if (opt == "k128") {
      fastdeploy::dispatch_gate_gemm<
          fastdeploy::GateGemm_bf16_nobias,
          fastdeploy::GateGemm_bf16_bias,
          fastdeploy::GateGemm_bf16_nobias_small_k128,
          fastdeploy::GateGemm_bf16_bias_small_k128>(c, a, b, bias);
    } else if (opt == "cluster") {
      fastdeploy::dispatch_gate_gemm<
          fastdeploy::GateGemm_bf16_nobias,
          fastdeploy::GateGemm_bf16_bias,
          fastdeploy::GateGemm_bf16_nobias_small_cluster,
          fastdeploy::GateGemm_bf16_bias_small_cluster>(c, a, b, bias);
    } else {
      fastdeploy::dispatch_gate_gemm<fastdeploy::GateGemm_bf16_nobias,
                                     fastdeploy::GateGemm_bf16_bias,
                                     fastdeploy::GateGemm_bf16_nobias_small,
                                     fastdeploy::GateGemm_bf16_bias_small>(
          c, a, b, bias);
    }
  } else if (dtype == paddle::DataType::FLOAT16) {
    fastdeploy::dispatch_gate_gemm<fastdeploy::GateGemm_fp16_nobias,
                                   fastdeploy::GateGemm_fp16_bias,
                                   fastdeploy::GateGemm_fp16_nobias_small,
                                   fastdeploy::GateGemm_fp16_bias_small>(
        c, a, b, bias);
  } else if (dtype == paddle::DataType::FLOAT32) {
    fastdeploy::dispatch_gate_gemm<fastdeploy::GateGemm_fp32_nobias,
                                   fastdeploy::GateGemm_fp32_bias,
                                   fastdeploy::GateGemm_fp32_nobias,
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
