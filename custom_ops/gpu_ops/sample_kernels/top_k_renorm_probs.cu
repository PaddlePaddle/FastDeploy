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

std::vector<paddle::Tensor> TopKRenorm(const paddle::Tensor &probs,
                                       const paddle::Tensor &top_k) {
  std::vector<int64_t> probs_shape = probs.shape();
  uint32_t batch_size = probs_shape[0];
  uint32_t vocab_size = probs_shape[1];
  auto cu_stream = probs.stream();

  auto renorm_probs =
      GetEmptyTensor(probs.dims(), paddle::DataType::FLOAT32, probs.place());

  cudaError_t status;


  status = sampling::TopKRenormProb<float>(
    const_cast<float *>(probs.data<float>()),
    renorm_probs.data<float>(),
    const_cast<int64_t *>(top_k.data<int64_t>()),
    batch_size, vocab_size, cu_stream);

  PD_CHECK(status == cudaSuccess, "TopKRenormProb failed with error code " +
                                      std::string(cudaGetErrorString(status)));

  return {renorm_probs};
}

std::vector<std::vector<int64_t>>
TopKRenormInferShape(const std::vector<int64_t> &probs_shape,
                    const std::vector<int64_t> &top_k_shape) {
  return {probs_shape};
}

std::vector<paddle::DataType>
TopKRenormInferDtype(const paddle::DataType &probs_dtype,
                    const paddle::DataType &top_k_shape) {
  return {probs_dtype};
}

PD_BUILD_STATIC_OP(top_k_renorm_probs)
    .Inputs({"probs", "top_k"})
    .Outputs({"renorm_probs"})
    .SetKernelFn(PD_KERNEL(TopKRenorm))
    .SetInferShapeFn(PD_INFER_SHAPE(TopKRenormInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(TopKRenormInferDtype));
