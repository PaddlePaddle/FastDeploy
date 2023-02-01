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
#include "fastdeploy/vision/common/processors/mat_batch.h"

namespace fastdeploy {
namespace vision {

#ifdef WITH_GPU
void MatBatch::SetStream(cudaStream_t s) {
  stream = s;
  for (size_t i = 0; i < mats->size(); ++i) {
    (*mats)[i].SetStream(s);
  }
}
#endif

FDTensor* MatBatch::Tensor() { return &fd_tensor; }

void MatBatch::SetTensor(FDTensor* tensor) {
  fd_tensor.SetExternalData(tensor->Shape(), tensor->Dtype(), tensor->Data(),
                            tensor->device, tensor->device_id);
}

FDTensor* CreateCachedGpuInputTensor(MatBatch* mat_batch) {
#ifdef WITH_GPU
  auto mats = mat_batch->mats;
  FDASSERT(CheckShapeConsistency(mats), "Mats shapes are not consistent.")
  FDTensor* src = (*mats)[0].Tensor();
  if (mat_batch->device == Device::GPU) {
    if (mat_batch->has_batched_tensor) {
      return mat_batch->Tensor();
    }
    // Mats on GPU, but each mat has its own tensor,
    // to get a batch tensor, we need to copy these tensors to a batch tensor
    auto new_shape = src->Shape();
    new_shape.insert(new_shape.begin(), mat_batch->mats->size());

    mat_batch->input_cache->Resize(new_shape, src->Dtype(), "batch_input_cache",
                                   Device::GPU);
    mat_batch->input_cache->PrintInfo();
    for (size_t i = 0; i < mats->size(); ++i) {
      uint8_t* p = reinterpret_cast<uint8_t*>(mat_batch->input_cache->Data());
      int num_bytes = (*mats)[i].Tensor()->Nbytes();
      FDASSERT(cudaMemcpyAsync(p + i * num_bytes, (*mats)[i].Tensor()->Data(),
                               num_bytes, cudaMemcpyDeviceToDevice,
                               (*mats)[i].Stream()) == 0,
               "[ERROR] Error occurs while copy memory from GPU to GPU.");
    }
    return mat_batch->input_cache;
  } else if (mat_batch->device == Device::CPU) {
    // Mats on CPU, we need copy these tensors from CPU to GPU
    FDASSERT(src->Shape().size() == 3, "The CPU tensor must has 3 dims.")
    auto new_shape = src->Shape();
    new_shape.insert(new_shape.begin(), mats->size());
    mat_batch->input_cache->Resize(new_shape, src->Dtype(), "batch_input_cache",
                                   Device::GPU);
    for (size_t i = 0; i < mats->size(); ++i) {
      uint8_t* p = reinterpret_cast<uint8_t*>(mat_batch->input_cache->Data());
      int num_bytes = (*mats)[i].Tensor()->Nbytes();
      FDASSERT(cudaMemcpyAsync(p + i * num_bytes, (*mats)[i].Tensor()->Data(),
                               num_bytes, cudaMemcpyHostToDevice,
                               (*mats)[i].Stream()) == 0,
               "[ERROR] Error occurs while copy memory from CPU to GPU.");
    }
    return mat_batch->input_cache;
  } else {
    FDASSERT(false, "FDMat is on unsupported device: %d", src->device);
  }
#else
  FDASSERT(false, "FastDeploy didn't compile with WITH_GPU.");
#endif
  return nullptr;
}

}  // namespace vision
}  // namespace fastdeploy
