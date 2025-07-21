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
                                               const paddle::Tensor &min_p) {
    std::vector<int64_t> probs_shape = probs.shape();
    unsigned int batch_size = probs_shape[0];
    unsigned int vocab_size = probs_shape[1];
    auto cu_stream = probs.stream();

    auto renorm_probs =
      GetEmptyTensor(probs.dims(), paddle::DataType::FLOAT32, probs.place());

    cudaError_t status;

    status = sampling::MinPSamplingFromProb<float, int>(
        const_cast<float *>(probs.data<float>()),
        const_cast<float *>(min_p.data<float>()),
        renorm_probs.data<float>(),
        batch_size,
        vocab_size,
        true,  // deterministic
        cu_stream);


  PD_CHECK(status == cudaSuccess, "SamplingFromProbs failed with error code " +
                                      std::string(cudaGetErrorString(status)));

  return {renorm_probs};
}

std::vector<std::vector<int64_t>>
MinPSamplingFromProbsInferShape(const std::vector<int64_t> &probs_shape,
                             const paddle::optional<std::vector<int64_t>> &min_p_shape) {
  return {probs_shape};
}

std::vector<paddle::DataType>
MinPSamplingFromProbsInferDtype(const paddle::DataType &probs_dtype,
                             const paddle::optional<paddle::DataType> &min_p_dtype) {
  return {probs_dtype};
}


PD_BUILD_STATIC_OP(min_p_sampling)
    .Inputs({"probs", "min_p"})
    .Outputs({"renorm_probs"})
    .SetKernelFn(PD_KERNEL(MinPSamplingFromProbs))
    .SetInferShapeFn(PD_INFER_SHAPE(MinPSamplingFromProbsInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MinPSamplingFromProbsInferDtype));
