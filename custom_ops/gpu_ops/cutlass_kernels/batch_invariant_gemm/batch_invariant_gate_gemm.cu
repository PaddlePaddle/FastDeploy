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

// --- Non-Cooperative tile configs (M_tile=64, no M constraint) ---
using TileShape_k64 = cute::Shape<cute::_64, cute::_32, cute::_64>;
using TileShape_k128 = cute::Shape<cute::_64, cute::_32, cute::_128>;
using TileShape_n16 = cute::Shape<cute::_64, cute::_16, cute::_128>;
using TileShape_n128 = cute::Shape<cute::_64, cute::_128, cute::_64>;
using TileShape_n64 = cute::Shape<cute::_64, cute::_64, cute::_64>;
using TileShape_n64_k128 = cute::Shape<cute::_64, cute::_64, cute::_128>;
using TileShape_n256 = cute::Shape<cute::_64, cute::_256, cute::_64>;

// --- Cooperative tile configs (M_tile >= 128 required) ---
using CoopTileShape_k64 = cute::Shape<cute::_128, cute::_32, cute::_64>;
using CoopTileShape_k128 = cute::Shape<cute::_128, cute::_32, cute::_128>;
using CoopTileShape_n128 = cute::Shape<cute::_128, cute::_128, cute::_64>;
using CoopTileShape_n256 = cute::Shape<cute::_128, cute::_256, cute::_64>;

using ClusterShape_1x1x1 = cute::Shape<cute::_1, cute::_1, cute::_1>;
using ClusterShape_1x2x1 = cute::Shape<cute::_1, cute::_2, cute::_1>;

// --- Default (K128) instantiations ---

// bf16
using GateGemm_bf16_nobias = GateGemmSm90StreamK<cutlass::bfloat16_t,
                                                 cutlass::bfloat16_t,
                                                 false,
                                                 TileShape_k128,
                                                 ClusterShape_1x1x1>;

using GateGemm_bf16_bias = GateGemmSm90StreamK<cutlass::bfloat16_t,
                                               cutlass::bfloat16_t,
                                               true,
                                               TileShape_k128,
                                               ClusterShape_1x1x1>;

// fp16
using GateGemm_fp16_nobias = GateGemmSm90StreamK<cutlass::half_t,
                                                 cutlass::half_t,
                                                 false,
                                                 TileShape_k128,
                                                 ClusterShape_1x1x1>;

using GateGemm_fp16_bias = GateGemmSm90StreamK<cutlass::half_t,
                                               cutlass::half_t,
                                               true,
                                               TileShape_k128,
                                               ClusterShape_1x1x1>;

// fp32 (K64 only — fp32 elements are 4B, K128 tile would exceed shared memory)
using GateGemm_fp32_nobias =
    GateGemmSm90StreamK<float, float, false, TileShape_k64, ClusterShape_1x1x1>;

using GateGemm_fp32_bias =
    GateGemmSm90StreamK<float, float, true, TileShape_k64, ClusterShape_1x1x1>;

// --- N16 variant (opt="n16", bf16 only for benchmarking) ---

using GateGemm_bf16_nobias_n16 = GateGemmSm90StreamK<cutlass::bfloat16_t,
                                                     cutlass::bfloat16_t,
                                                     false,
                                                     TileShape_n16,
                                                     ClusterShape_1x1x1>;

using GateGemm_bf16_bias_n16 = GateGemmSm90StreamK<cutlass::bfloat16_t,
                                                   cutlass::bfloat16_t,
                                                   true,
                                                   TileShape_n16,
                                                   ClusterShape_1x1x1>;

// --- K64 fallback (opt="k64") ---

using GateGemm_bf16_nobias_k64 = GateGemmSm90StreamK<cutlass::bfloat16_t,
                                                     cutlass::bfloat16_t,
                                                     false,
                                                     TileShape_k64,
                                                     ClusterShape_1x1x1>;

using GateGemm_bf16_bias_k64 = GateGemmSm90StreamK<cutlass::bfloat16_t,
                                                   cutlass::bfloat16_t,
                                                   true,
                                                   TileShape_k64,
                                                   ClusterShape_1x1x1>;

// --- N128 variant (opt="n128", large-N shapes like qkv N=4608) ---

using GateGemm_bf16_nobias_n128 = GateGemmSm90StreamK<cutlass::bfloat16_t,
                                                      cutlass::bfloat16_t,
                                                      false,
                                                      TileShape_n128,
                                                      ClusterShape_1x1x1>;

using GateGemm_bf16_bias_n128 = GateGemmSm90StreamK<cutlass::bfloat16_t,
                                                    cutlass::bfloat16_t,
                                                    true,
                                                    TileShape_n128,
                                                    ClusterShape_1x1x1>;

using GateGemm_fp16_nobias_n128 = GateGemmSm90StreamK<cutlass::half_t,
                                                      cutlass::half_t,
                                                      false,
                                                      TileShape_n128,
                                                      ClusterShape_1x1x1>;

using GateGemm_fp16_bias_n128 = GateGemmSm90StreamK<cutlass::half_t,
                                                    cutlass::half_t,
                                                    true,
                                                    TileShape_n128,
                                                    ClusterShape_1x1x1>;

// --- N256 variant (opt="n256", very large-N shapes) ---

using GateGemm_bf16_nobias_n256 = GateGemmSm90StreamK<cutlass::bfloat16_t,
                                                      cutlass::bfloat16_t,
                                                      false,
                                                      TileShape_n256,
                                                      ClusterShape_1x1x1>;

using GateGemm_bf16_bias_n256 = GateGemmSm90StreamK<cutlass::bfloat16_t,
                                                    cutlass::bfloat16_t,
                                                    true,
                                                    TileShape_n256,
                                                    ClusterShape_1x1x1>;

using GateGemm_fp16_nobias_n256 = GateGemmSm90StreamK<cutlass::half_t,
                                                      cutlass::half_t,
                                                      false,
                                                      TileShape_n256,
                                                      ClusterShape_1x1x1>;

using GateGemm_fp16_bias_n256 = GateGemmSm90StreamK<cutlass::half_t,
                                                    cutlass::half_t,
                                                    true,
                                                    TileShape_n256,
                                                    ClusterShape_1x1x1>;

using GateGemm_fp16_nobias_k64 = GateGemmSm90StreamK<cutlass::half_t,
                                                     cutlass::half_t,
                                                     false,
                                                     TileShape_k64,
                                                     ClusterShape_1x1x1>;

using GateGemm_fp16_bias_k64 = GateGemmSm90StreamK<cutlass::half_t,
                                                   cutlass::half_t,
                                                   true,
                                                   TileShape_k64,
                                                   ClusterShape_1x1x1>;

// --- N64 variant (opt="n64", bf16 only — sweetspot between n32 and n128) ---

using GateGemm_bf16_nobias_n64 = GateGemmSm90StreamK<cutlass::bfloat16_t,
                                                     cutlass::bfloat16_t,
                                                     false,
                                                     TileShape_n64,
                                                     ClusterShape_1x1x1>;

using GateGemm_bf16_bias_n64 = GateGemmSm90StreamK<cutlass::bfloat16_t,
                                                   cutlass::bfloat16_t,
                                                   true,
                                                   TileShape_n64,
                                                   ClusterShape_1x1x1>;

// --- N64_K128 variant (opt="n64_k128", bf16 only) ---

using GateGemm_bf16_nobias_n64_k128 = GateGemmSm90StreamK<cutlass::bfloat16_t,
                                                          cutlass::bfloat16_t,
                                                          false,
                                                          TileShape_n64_k128,
                                                          ClusterShape_1x1x1>;

using GateGemm_bf16_bias_n64_k128 = GateGemmSm90StreamK<cutlass::bfloat16_t,
                                                        cutlass::bfloat16_t,
                                                        true,
                                                        TileShape_n64_k128,
                                                        ClusterShape_1x1x1>;

// --- N128 + ClusterShape 1x2x1 (opt="n128_c2", bf16 only — TMA multicast) ---

using GateGemm_bf16_nobias_n128_c2 = GateGemmSm90StreamK<cutlass::bfloat16_t,
                                                         cutlass::bfloat16_t,
                                                         false,
                                                         TileShape_n128,
                                                         ClusterShape_1x2x1>;

using GateGemm_bf16_bias_n128_c2 = GateGemmSm90StreamK<cutlass::bfloat16_t,
                                                       cutlass::bfloat16_t,
                                                       true,
                                                       TileShape_n128,
                                                       ClusterShape_1x2x1>;

// =====================================================================
// Cooperative instantiations (GateGemmSm90, M_tile=128)
// Higher throughput per CTA with 2 consumer WGs, but M_tile >= 128.
// Select via CUTLASS_GATE_GEMM_OPT=coop (default k64) or coop_k128.
// =====================================================================

// --- Coop default (K64) ---
using CoopGemm_bf16_nobias = GateGemmSm90<cutlass::bfloat16_t,
                                          cutlass::bfloat16_t,
                                          false,
                                          CoopTileShape_k64,
                                          ClusterShape_1x1x1>;

using CoopGemm_bf16_bias = GateGemmSm90<cutlass::bfloat16_t,
                                        cutlass::bfloat16_t,
                                        true,
                                        CoopTileShape_k64,
                                        ClusterShape_1x1x1>;

using CoopGemm_fp16_nobias = GateGemmSm90<cutlass::half_t,
                                          cutlass::half_t,
                                          false,
                                          CoopTileShape_k64,
                                          ClusterShape_1x1x1>;

using CoopGemm_fp16_bias = GateGemmSm90<cutlass::half_t,
                                        cutlass::half_t,
                                        true,
                                        CoopTileShape_k64,
                                        ClusterShape_1x1x1>;

// --- Coop K128 variant ---
using CoopGemm_bf16_nobias_k128 = GateGemmSm90<cutlass::bfloat16_t,
                                               cutlass::bfloat16_t,
                                               false,
                                               CoopTileShape_k128,
                                               ClusterShape_1x1x1>;

using CoopGemm_bf16_bias_k128 = GateGemmSm90<cutlass::bfloat16_t,
                                             cutlass::bfloat16_t,
                                             true,
                                             CoopTileShape_k128,
                                             ClusterShape_1x1x1>;

using CoopGemm_fp16_nobias_k128 = GateGemmSm90<cutlass::half_t,
                                               cutlass::half_t,
                                               false,
                                               CoopTileShape_k128,
                                               ClusterShape_1x1x1>;

using CoopGemm_fp16_bias_k128 = GateGemmSm90<cutlass::half_t,
                                             cutlass::half_t,
                                             true,
                                             CoopTileShape_k128,
                                             ClusterShape_1x1x1>;

// --- Coop N128 variant ---
using CoopGemm_bf16_nobias_n128 = GateGemmSm90<cutlass::bfloat16_t,
                                               cutlass::bfloat16_t,
                                               false,
                                               CoopTileShape_n128,
                                               ClusterShape_1x1x1>;

using CoopGemm_bf16_bias_n128 = GateGemmSm90<cutlass::bfloat16_t,
                                             cutlass::bfloat16_t,
                                             true,
                                             CoopTileShape_n128,
                                             ClusterShape_1x1x1>;

using CoopGemm_fp16_nobias_n128 = GateGemmSm90<cutlass::half_t,
                                               cutlass::half_t,
                                               false,
                                               CoopTileShape_n128,
                                               ClusterShape_1x1x1>;

using CoopGemm_fp16_bias_n128 = GateGemmSm90<cutlass::half_t,
                                             cutlass::half_t,
                                             true,
                                             CoopTileShape_n128,
                                             ClusterShape_1x1x1>;

// --- Coop N256 variant ---
using CoopGemm_bf16_nobias_n256 = GateGemmSm90<cutlass::bfloat16_t,
                                               cutlass::bfloat16_t,
                                               false,
                                               CoopTileShape_n256,
                                               ClusterShape_1x1x1>;

using CoopGemm_bf16_bias_n256 = GateGemmSm90<cutlass::bfloat16_t,
                                             cutlass::bfloat16_t,
                                             true,
                                             CoopTileShape_n256,
                                             ClusterShape_1x1x1>;

using CoopGemm_fp16_nobias_n256 = GateGemmSm90<cutlass::half_t,
                                               cutlass::half_t,
                                               false,
                                               CoopTileShape_n256,
                                               ClusterShape_1x1x1>;

using CoopGemm_fp16_bias_n256 = GateGemmSm90<cutlass::half_t,
                                             cutlass::half_t,
                                             true,
                                             CoopTileShape_n256,
                                             ClusterShape_1x1x1>;

// Dispatch: kernel type is selected externally via CUTLASS_GATE_GEMM_OPT,
// NOT based on M. This preserves batch invariance (same kernel for all M).
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

  // CUTLASS_GATE_GEMM_OPT: k64 | n16 | n64 | n64_k128 | n128 | n128_c2 |
  //   n256 | coop | coop_k128 | coop_n128 | coop_n256 | (empty=k128 default)
  // NOTE: not using cached_gate_gemm_opt() here — env var must be switchable
  // at runtime for benchmark sweeps. Cache is safe only in production where
  // the value is set once before first kernel call.
  const char *env_opt = std::getenv("CUTLASS_GATE_GEMM_OPT");
  std::string opt = env_opt ? env_opt : "";

  auto dtype = a.dtype();
  if (dtype == paddle::DataType::BFLOAT16) {
    if (opt == "coop_n256") {
      fastdeploy::dispatch_gate_gemm<fastdeploy::CoopGemm_bf16_nobias_n256,
                                     fastdeploy::CoopGemm_bf16_bias_n256>(
          c, a, b, bias);
    } else if (opt == "coop_n128") {
      fastdeploy::dispatch_gate_gemm<fastdeploy::CoopGemm_bf16_nobias_n128,
                                     fastdeploy::CoopGemm_bf16_bias_n128>(
          c, a, b, bias);
    } else if (opt == "coop_k128") {
      fastdeploy::dispatch_gate_gemm<fastdeploy::CoopGemm_bf16_nobias_k128,
                                     fastdeploy::CoopGemm_bf16_bias_k128>(
          c, a, b, bias);
    } else if (opt == "coop") {
      fastdeploy::dispatch_gate_gemm<fastdeploy::CoopGemm_bf16_nobias,
                                     fastdeploy::CoopGemm_bf16_bias>(
          c, a, b, bias);
    } else if (opt == "n256") {
      fastdeploy::dispatch_gate_gemm<fastdeploy::GateGemm_bf16_nobias_n256,
                                     fastdeploy::GateGemm_bf16_bias_n256>(
          c, a, b, bias);
    } else if (opt == "n128") {
      fastdeploy::dispatch_gate_gemm<fastdeploy::GateGemm_bf16_nobias_n128,
                                     fastdeploy::GateGemm_bf16_bias_n128>(
          c, a, b, bias);
    } else if (opt == "n128_c2") {
      fastdeploy::dispatch_gate_gemm<fastdeploy::GateGemm_bf16_nobias_n128_c2,
                                     fastdeploy::GateGemm_bf16_bias_n128_c2>(
          c, a, b, bias);
    } else if (opt == "n64_k128") {
      fastdeploy::dispatch_gate_gemm<fastdeploy::GateGemm_bf16_nobias_n64_k128,
                                     fastdeploy::GateGemm_bf16_bias_n64_k128>(
          c, a, b, bias);
    } else if (opt == "n64") {
      fastdeploy::dispatch_gate_gemm<fastdeploy::GateGemm_bf16_nobias_n64,
                                     fastdeploy::GateGemm_bf16_bias_n64>(
          c, a, b, bias);
    } else if (opt == "n16") {
      fastdeploy::dispatch_gate_gemm<fastdeploy::GateGemm_bf16_nobias_n16,
                                     fastdeploy::GateGemm_bf16_bias_n16>(
          c, a, b, bias);
    } else if (opt == "k64") {
      fastdeploy::dispatch_gate_gemm<fastdeploy::GateGemm_bf16_nobias_k64,
                                     fastdeploy::GateGemm_bf16_bias_k64>(
          c, a, b, bias);
    } else {
      fastdeploy::dispatch_gate_gemm<fastdeploy::GateGemm_bf16_nobias,
                                     fastdeploy::GateGemm_bf16_bias>(
          c, a, b, bias);
    }
  } else if (dtype == paddle::DataType::FLOAT16) {
    if (opt == "coop_n256") {
      fastdeploy::dispatch_gate_gemm<fastdeploy::CoopGemm_fp16_nobias_n256,
                                     fastdeploy::CoopGemm_fp16_bias_n256>(
          c, a, b, bias);
    } else if (opt == "coop_n128") {
      fastdeploy::dispatch_gate_gemm<fastdeploy::CoopGemm_fp16_nobias_n128,
                                     fastdeploy::CoopGemm_fp16_bias_n128>(
          c, a, b, bias);
    } else if (opt == "coop_k128") {
      fastdeploy::dispatch_gate_gemm<fastdeploy::CoopGemm_fp16_nobias_k128,
                                     fastdeploy::CoopGemm_fp16_bias_k128>(
          c, a, b, bias);
    } else if (opt == "coop") {
      fastdeploy::dispatch_gate_gemm<fastdeploy::CoopGemm_fp16_nobias,
                                     fastdeploy::CoopGemm_fp16_bias>(
          c, a, b, bias);
    } else if (opt == "n256") {
      fastdeploy::dispatch_gate_gemm<fastdeploy::GateGemm_fp16_nobias_n256,
                                     fastdeploy::GateGemm_fp16_bias_n256>(
          c, a, b, bias);
    } else if (opt == "n128") {
      fastdeploy::dispatch_gate_gemm<fastdeploy::GateGemm_fp16_nobias_n128,
                                     fastdeploy::GateGemm_fp16_bias_n128>(
          c, a, b, bias);
    } else if (opt == "k64") {
      fastdeploy::dispatch_gate_gemm<fastdeploy::GateGemm_fp16_nobias_k64,
                                     fastdeploy::GateGemm_fp16_bias_k64>(
          c, a, b, bias);
    } else {
      fastdeploy::dispatch_gate_gemm<fastdeploy::GateGemm_fp16_nobias,
                                     fastdeploy::GateGemm_fp16_bias>(
          c, a, b, bias);
    }
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
