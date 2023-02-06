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
void FDMatBatch::SetStream(cudaStream_t s) {
  stream = s;
  for (size_t i = 0; i < mats->size(); ++i) {
    (*mats)[i].SetStream(s);
  }
}
#endif

FDTensor* FDMatBatch::Tensor() {
  if (has_batched_tensor) {
    return &fd_tensor;
  }
  FDASSERT(CheckShapeConsistency(mats), "Mats shapes are not consistent.")
  // Each mat has its own tensor,
  // to get a batched tensor, we need copy these tensors to a batched tensor
  FDTensor* src = (*mats)[0].Tensor();
  auto new_shape = src->Shape();
  new_shape.insert(new_shape.begin(), mats->size());
  input_cache->Resize(new_shape, src->Dtype(), "batch_input_cache", device);
  for (size_t i = 0; i < mats->size(); ++i) {
    FDASSERT(device == (*mats)[i].Tensor()->device,
             "Mats and MatBatch are not on the same device");
    uint8_t* p = reinterpret_cast<uint8_t*>(input_cache->Data());
    int num_bytes = (*mats)[i].Tensor()->Nbytes();
    FDTensor::CopyBuffer(p + i * num_bytes, (*mats)[i].Tensor()->Data(),
                         num_bytes, device, false);
  }
  SetTensor(input_cache);
  return &fd_tensor;
}

void FDMatBatch::SetTensor(FDTensor* tensor) {
  fd_tensor.SetExternalData(tensor->Shape(), tensor->Dtype(), tensor->Data(),
                            tensor->device, tensor->device_id);
  has_batched_tensor = true;
}

FDTensor* CreateCachedGpuInputTensor(FDMatBatch* mat_batch) {
#ifdef WITH_GPU
  auto mats = mat_batch->mats;
  FDASSERT(CheckShapeConsistency(mats), "Mats shapes are not consistent.")
  FDTensor* src = (*mats)[0].Tensor();
  if (mat_batch->device == Device::GPU) {
    return mat_batch->Tensor();
  } else if (mat_batch->device == Device::CPU) {
    // Mats on CPU, we need copy them to GPU and then get a batched GPU tensor
    for (size_t i = 0; i < mats->size(); ++i) {
      FDTensor* tensor = CreateCachedGpuInputTensor(&(*mats)[i]);
      (*mats)[i].SetTensor(tensor);
    }
    return mat_batch->Tensor();
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
