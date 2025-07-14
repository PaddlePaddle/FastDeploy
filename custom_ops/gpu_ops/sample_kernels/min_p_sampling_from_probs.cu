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
#include "paddle/phi/backends/context_pool.h"
#include "sample_kernels/sampling.cuh"

std::vector<paddle::Tensor> MinPSamplingFromProbs(const paddle::Tensor &probs,
                                               const paddle::Tensor &min_p,
                                               int seed) {
    std::vector<int64_t> probs_shape = probs.shape();
    unsigned int batch_size = probs_shape[0];
    unsigned int vocab_size = probs_shape[1];
    uint64_t philox_seed = seed;
    uint64_t philox_offset = 0;
    auto cu_stream = probs.stream();

    auto samples =
        paddle::empty({batch_size, 1}, paddle::DataType::INT64, probs.place());

    cudaError_t status;

    status = sampling::MinPSamplingFromProb<float, int64_t>(
    const_cast<float *>(probs.data<float>()),samples.data<int64_t>(),
    batch_size,min_p.data<float>(),vocab_size,true,philox_seed,philox_offset,cu_stream);

  PD_CHECK(status == cudaSuccess, "SamplingFromProbs failed with error code " +
                                      std::string(cudaGetErrorString(status)));

  return {samples};
}

std::vector<std::vector<int64_t>>
MinPSamplingFromProbsInferShape(const std::vector<int64_t> &probs_shape,
                             const paddle::optional<std::vector<int64_t>> &min_p_shape) {
  int64_t bs = probs_shape[0];
  return {{bs, 1}};
}

std::vector<paddle::DataType>
MinPSamplingFromProbsInferDtype(const paddle::DataType &probs_dtype,
                             const paddle::optional<paddle::DataType> &min_p_dtype) {
  return {paddle::DataType::INT64};
}


PD_BUILD_STATIC_OP(min_p_sampling)
    .Inputs({"probs", "min_p"})
    .Outputs({"samples"})
    .Attrs({"seed: int"})
    .SetKernelFn(PD_KERNEL(MinPSamplingFromProbs))
    .SetInferShapeFn(PD_INFER_SHAPE(MinPSamplingFromProbsInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MinPSamplingFromProbsInferDtype));
