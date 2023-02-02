// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include "fastdeploy/vision/common/processors/mat.h"

#ifdef WITH_GPU
#include <cuda_runtime_api.h>
#endif

namespace fastdeploy {
namespace vision {

enum MatBatchLayout { NHWC, NCHW };

struct FASTDEPLOY_DECL MatBatch {
  MatBatch() = default;

  // MatBatch is intialized with a list of mats,
  // the data is stored in the mats separately.
  // Call Tensor() function to get a batched 4-dimension tensor.
  explicit MatBatch(std::vector<Mat>* _mats) {
    mats = _mats;
    layout = MatBatchLayout::NHWC;
    mat_type = ProcLib::OPENCV;
  }

  // Get the batched 4-dimension tensor.
  FDTensor* Tensor();

  void SetTensor(FDTensor* tensor);

 private:
#ifdef WITH_GPU
  cudaStream_t stream = nullptr;
#endif
  FDTensor fd_tensor;

 public:
  FDTensor* input_cache;
  FDTensor* output_cache;
#ifdef WITH_GPU
  cudaStream_t Stream() const { return stream; }
  void SetStream(cudaStream_t s);
#endif

  std::vector<Mat>* mats;
  ProcLib mat_type = ProcLib::OPENCV;
  MatBatchLayout layout = MatBatchLayout::NHWC;
  Device device = Device::CPU;

  // False: the data is stored in the mats separately
  // True: the data is stored in the fd_tensor continuously in 4 dimensions
  bool has_batched_tensor = false;
};

typedef MatBatch FDMatBatch;

// Create a batched input tensor on GPU and save into input_cache.
// If the MatBatch is on GPU, return the Tensor() directly.
// If the MatBatch is on CPU, then copy the CPU tensors to GPU and get a GPU
// batched input tensor.
FDTensor* CreateCachedGpuInputTensor(MatBatch* mat_batch);

}  // namespace vision
}  // namespace fastdeploy