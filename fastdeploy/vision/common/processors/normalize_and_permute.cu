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

#ifdef WITH_GPU
#include "fastdeploy/vision/common/processors/normalize_and_permute.h"

namespace fastdeploy {
namespace vision {

__global__ void NormalizeAndPermuteKernel(uint8_t* src, float* dst,
                                          const float* alpha, const float* beta,
                                          int num_channel, bool swap_rb,
                                          int edge) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= edge) return;

  if (swap_rb) {
    uint8_t tmp = src[num_channel * idx];
    src[num_channel * idx] = src[num_channel * idx + 2];
    src[num_channel * idx + 2] = tmp;
  }

  for (int i = 0; i < num_channel; ++i) {
    dst[idx + edge * i] = src[num_channel * idx + i] * alpha[i] + beta[i];
  }
}

bool NormalizeAndPermute::ImplByCuda(Mat* mat) {
  // Prepare input tensor
  std::string tensor_name = Name() + "_cvcuda_src";
  FDTensor* src = CreateCachedGpuInputTensor(mat, tensor_name);

  // Prepare output tensor
  tensor_name = Name() + "_dst";
  FDTensor* dst = UpdateAndGetReusedTensor(src->Shape(), FDDataType::FP32,
                                           tensor_name, Device::GPU);

  // Copy alpha and beta to GPU
  tensor_name = Name() + "_alpha";
  FDMat alpha_mat =
      FDMat::Create(1, 1, alpha_.size(), FDDataType::FP32, alpha_.data());
  FDTensor* alpha = CreateCachedGpuInputTensor(&alpha_mat, tensor_name);

  tensor_name = Name() + "_beta";
  FDMat beta_mat =
      FDMat::Create(1, 1, beta_.size(), FDDataType::FP32, beta_.data());
  FDTensor* beta = CreateCachedGpuInputTensor(&beta_mat, tensor_name);

  int jobs = mat->Width() * mat->Height();
  int threads = 256;
  int blocks = ceil(jobs / (float)threads);
  NormalizeAndPermuteKernel<<<blocks, threads, 0, mat->Stream()>>>(
      reinterpret_cast<uint8_t*>(src->Data()),
      reinterpret_cast<float*>(dst->Data()),
      reinterpret_cast<float*>(alpha->Data()),
      reinterpret_cast<float*>(beta->Data()), mat->Channels(), swap_rb_, jobs);

  mat->SetTensor(dst);
  mat->device = Device::GPU;
  mat->layout = Layout::CHW;
  return true;
}

#ifdef ENABLE_CVCUDA
bool NormalizeAndPermute::ImplByCvCuda(Mat* mat) {
  std::cout << Name() << " cvcuda" << std::endl;
  return ImplByCuda(mat);
}
#endif

}  // namespace vision
}  // namespace fastdeploy
#endif
