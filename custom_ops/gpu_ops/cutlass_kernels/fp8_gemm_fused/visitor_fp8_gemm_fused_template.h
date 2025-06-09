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
#pragma once

#include "per_channel_fp8_fp8_half_gemm.h"  // NOLINT

template <typename Gemm>
typename Gemm::Arguments prepar_gemm_args_sm89(void* D, void const* A, void const* B, void const* C_bias,
    int m, int n, int k, float const* scale_d0, float const* scale_d1)
{
    using ElementT = typename Gemm::ElementA;
    using ElementOutput = typename Gemm::ElementD;
    using ElementComputeEpilogue = float;

    int const lda = k;
    int const ldb = k;
    int const ldc = n;

    typename Gemm::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, // Mode
        {m, n, k},                                                         // Problem size
        1,                                                                 // Split-k factor
        {},                                                                // Epilogue args
        reinterpret_cast<ElementT const*>(A),                              // a pointer
        reinterpret_cast<ElementT const*>(B),                              // b pointer
        nullptr,                                                           // c pointer (unused)
        nullptr,                                                           // d pointer (unused)
        m * k,                                                             // batch stride a (unused)
        n * k,                                                             // batch stride b (unused)
        m * n,                                                             // batch stride c (unused)
        m * n,                                                             // batch stride d (unused)
        lda,                                                               // stride a
        ldb,                                                               // stride b
        ldc,                                                               // stride c (unused)
        ldc);                                                              // stride d (unused)

    args.epilogue = {
        {
            {
                {
                    {},                                      // Accumulator
                    {reinterpret_cast<ElementComputeEpilogue const*>(scale_d1), ElementComputeEpilogue(0),
                        {cute::_0{}, cute::_1{}, cute::_0{}}},
                    {} // Multiplies
                },
                {reinterpret_cast<ElementComputeEpilogue const*>(scale_d0), ElementComputeEpilogue(0), {cute::_0{}, cute::_0{}, cute::_0{}}},
                {} // Multiplies
            },                                                                                                          // Accum
            {reinterpret_cast<ElementOutput const*>(C_bias), ElementOutput(0), {cute::_0{}, cute::_1{}, cute::_0{}}},                 // Bias
            {}                                                                                             // Compute0
        },
        {reinterpret_cast<ElementOutput*>(D), {n, cute::_1{}, cute::_0{}}}
    };
    return args;
}

template <typename Gemm>
bool per_channel_fp8_fp8_gemm_scale_bias(GemmEpilogueAllParams params, typename Gemm::Arguments args) {
  Gemm per_channel_fp8_gemm;

  cutlass::Status status = per_channel_fp8_gemm.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "per_channel_fp8_gemm::can_implement() failed" << std::endl;
    return false;
  }

  size_t workspace_size = Gemm::get_workspace_size(args);
  phi::Allocator* allocator = paddle::GetAllocator(params.place);
  auto workspace = allocator->Allocate(workspace_size);

  //
  // Run the GEMM
  //
  status = per_channel_fp8_gemm(args, workspace->ptr(), params.stream);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "per_channel_fp8_gemm::run() failed" << std::endl;
    return false;
  }
  return true;
}


template <typename InputType, typename OutType,
            typename ThreadBlockShape, typename WarpShape,
            typename MMAShape, int Stages, bool hasbias, typename SM>
bool dispatch_visitor_fuse_gemm(GemmEpilogueAllParams params) {
    using AccumElementType = float;
    using Gemm = typename DeviceGemmFp8RowwiseSm89<InputType, OutType, AccumElementType, ThreadBlockShape, WarpShape, MMAShape,
        Stages>::Gemm;
    auto args = prepar_gemm_args_sm89<Gemm>(params.D, params.A, params.B, params.bias, params.M, params.N, params.K, params.scalar_scale, params.channel_scale);
    per_channel_fp8_fp8_gemm_scale_bias<Gemm>(params, args);
}
