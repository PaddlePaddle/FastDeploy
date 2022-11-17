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

#include "fastdeploy/vision/common/processors/normalize_and_permute.h"

namespace fastdeploy {
namespace vision {

__global__ void NormalizeAndPermuteKernel(
    uint8_t* src, float* dst, const float* alpha, const float* beta,
    int num_channel, bool swap_rb, int edge) {
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
  cv::Mat* im = mat->GetOpenCVMat();
  std::string buf_name = Name() + "_src";
  std::vector<int64_t> shape = {im->rows, im->cols, im->channels()};
  FDTensor* src = UpdateAndGetReusedBuffer(shape, im->type(), buf_name,
                                           Device::GPU);
  FDASSERT(cudaMemcpy(src->Data(), im->ptr(), src->Nbytes(),
                      cudaMemcpyHostToDevice) == 0,
           "Error occurs while copy memory from CPU to GPU.");

  buf_name = Name() + "_dst";
  FDTensor* dst = UpdateAndGetReusedBuffer(shape, CV_32FC(im->channels()),
                                           buf_name, Device::GPU);
  cv::Mat res(im->rows, im->cols, CV_32FC(im->channels()), dst->Data());

  buf_name = Name() + "_alpha";
  FDTensor* alpha = UpdateAndGetReusedBuffer({(int64_t)alpha_.size()}, CV_32FC1,
                                             buf_name, Device::GPU);
  FDASSERT(cudaMemcpy(alpha->Data(), alpha_.data(), alpha->Nbytes(),
                      cudaMemcpyHostToDevice) == 0,
           "Error occurs while copy memory from CPU to GPU.");

  buf_name = Name() + "_beta";
  FDTensor* beta = UpdateAndGetReusedBuffer({(int64_t)beta_.size()}, CV_32FC1,
                                             buf_name, Device::GPU);
  FDASSERT(cudaMemcpy(beta->Data(), beta_.data(), beta->Nbytes(),
                      cudaMemcpyHostToDevice) == 0,
           "Error occurs while copy memory from CPU to GPU.");

  int jobs = im->cols * im->rows;
  int threads = 256;
  int blocks = ceil(jobs / (float)threads);
  NormalizeAndPermuteKernel<<<blocks, threads, 0, NULL>>>(
      reinterpret_cast<uint8_t*>(src->Data()),
      reinterpret_cast<float*>(dst->Data()),
      reinterpret_cast<float*>(alpha->Data()),
      reinterpret_cast<float*>(beta->Data()), im->channels(), swap_rb_, jobs);

  mat->SetMat(res);
  mat->device = Device::GPU;
  mat->layout = Layout::CHW;
  return true;
}

}  // namespace vision
}  // namespace fastdeploy
