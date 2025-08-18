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

#include "helper.h"
#include "paddle/phi/backends/context_pool.h"
#include "sample_kernels/sampling.cuh"

std::vector<paddle::Tensor> TopPSamplingReject(const paddle::Tensor &probs,
                                               const paddle::Tensor &top_p,
                                               const paddle::optional<paddle::Tensor> &top_k,
                                               int64_t seed) {
  std::vector<int64_t> probs_shape = probs.shape();
  unsigned int batch_size = probs_shape[0];
  unsigned int vocab_size = probs_shape[1];
  uint64_t philox_seed = seed;
  uint64_t philox_offset = 0;
  auto cu_stream = probs.stream();

  // need_batch_random
  if (seed == -1) {
#ifdef PADDLE_WITH_COREX
    auto dev_ctx = static_cast<const phi::CustomContext*>(paddle::experimental::DeviceContextPool::Instance().Get(probs.place()));
#else
    phi::GPUContext* dev_ctx = static_cast<phi::GPUContext*>(phi::DeviceContextPool::Instance().Get(probs.place()));
#endif
    auto gen_cuda = dev_ctx->GetGenerator();
    auto seed_offset = gen_cuda->IncrementOffset(32 * batch_size);
    philox_seed = seed_offset.first;
    philox_offset = seed_offset.second;
  }

  auto samples =
      paddle::empty({batch_size, 1}, paddle::DataType::INT64, probs.place());

  cudaError_t status;

  if (top_k) {
    status = sampling::TopKTopPSamplingFromProb<float, int64_t>(
        const_cast<float *>(probs.data<float>()), samples.data<int64_t>(),
        batch_size, top_p.data<float>(), top_k.get().data<int64_t>(), vocab_size,
        true, philox_seed, philox_offset, cu_stream);
  }
  else {
    status = sampling::TopPSamplingFromProb<float, int64_t>(
        const_cast<float *>(probs.data<float>()), samples.data<int64_t>(),
        batch_size, top_p.data<float>(), vocab_size,
        true, philox_seed, philox_offset, cu_stream);
  }

  PD_CHECK(status == cudaSuccess, "SamplingFromProbs failed with error code " +
                                      std::string(cudaGetErrorString(status)));

  return {samples};
}

std::vector<std::vector<int64_t>>
TopPSamplingRejectInferShape(const std::vector<int64_t> &probs_shape,
                             const std::vector<int64_t> &top_p_shape,
                             const paddle::optional<std::vector<int64_t>> &top_k_shape) {
  int64_t bs = probs_shape[0];
  return {{bs, 1}};
}

std::vector<paddle::DataType>
TopPSamplingRejectInferDtype(const paddle::DataType &probs_dtype,
                             const paddle::DataType &top_p_dtype,
                             const paddle::optional<paddle::DataType> &top_k_dtype) {
  return {paddle::DataType::INT64};
}

PD_BUILD_STATIC_OP(rejection_top_p_sampling)
    .Inputs({"probs", "top_p", paddle::Optional("top_k")})
    .Outputs({"samples"})
    .Attrs({"seed: int64_t"})
    .SetKernelFn(PD_KERNEL(TopPSamplingReject))
    .SetInferShapeFn(PD_INFER_SHAPE(TopPSamplingRejectInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(TopPSamplingRejectInferDtype));
