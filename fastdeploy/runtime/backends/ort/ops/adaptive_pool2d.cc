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

#ifndef NON_64_PLATFORM

#include "adaptive_pool2d.h"

namespace fastdeploy {
struct OrtTensorDimensions : std::vector<int64_t> {
  OrtTensorDimensions(Ort::CustomOpApi ort, const OrtValue* value) {
    OrtTensorTypeAndShapeInfo* info = ort.GetTensorTypeAndShape(value);
    std::vector<int64_t>::operator=(ort.GetTensorShape(info));
    ort.ReleaseTensorTypeAndShapeInfo(info);
  }
};

void AdaptivePool2dKernel::CpuAdaptivePool(
    const std::vector<int64_t>& input_size,
    const std::vector<int64_t>& output_size, const float* input_data,
    float* output_data) {
  int64_t in_bc_offset = input_size[2] * input_size[3];
  int64_t out_bc_offset = output_size[2] * output_size[3];
  for (int64_t b = 0; b < output_size[0]; b++) {
    for (int64_t c = 0; c < output_size[1]; c++) {
      for (int64_t h = 0; h < output_size[2]; h++) {
        int64_t hstart =
            std::floor(static_cast<float>(h * input_size[2]) / output_size[2]);
        int64_t hend = std::ceil(static_cast<float>((h + 1) * input_size[2]) /
                                 output_size[2]);
        for (int64_t w = 0; w < output_size[3]; w++) {
          int64_t wstart = std::floor(static_cast<float>(w * input_size[3]) /
                                      output_size[3]);
          int64_t wend = std::ceil(static_cast<float>((w + 1) * input_size[3]) /
                                   output_size[3]);
          int64_t out_offset = h * output_size[3] + w;
          output_data[out_offset] = 0;
          for (auto i = hstart; i < hend; i++) {
            for (auto j = wstart; j < wend; j++) {
              if (pooling_type_ == "avg") {
                output_data[out_offset] += input_data[i * input_size[3] + j];
              }
              if (pooling_type_ == "max") {
                output_data[out_offset] = std::max(
                    output_data[out_offset], input_data[i * input_size[3] + j]);
              }
            }
          }
          if (pooling_type_ == "avg") {
            output_data[out_offset] /= ((hend - hstart) * (wend - wstart));
          }
        }
      }
      output_data += out_bc_offset;
      input_data += in_bc_offset;
    }
  }
}

void AdaptivePool2dKernel::Compute(OrtKernelContext* context) {
  const OrtValue* input = ort_.KernelContext_GetInput(context, 0);

  const float* input_data =
      reinterpret_cast<const float*>(ort_.GetTensorData<float>(input));

  OrtTensorDimensions input_dim(ort_, input);
  output_size_[0] = input_dim[0];
  std::vector<int64_t> input_size;
  for (auto i : input_dim) {
    input_size.push_back(i);
  }

  OrtValue* output = ort_.KernelContext_GetOutput(
      context, 0, output_size_.data(), output_size_.size());

  float* output_data = ort_.GetTensorMutableData<float>(output);
  if (!strcmp(this->provider_, "CUDAExecutionProvider")) {
#ifdef WITH_GPU
    auto compute_stream = ort_.KernelContext_GetGPUComputeStream(context);
    CudaAdaptivePool(input_size, output_size_, output_data, input_data,
                     compute_stream, pooling_type_);
#else
    FDWARNING << "FastDeploy didn't compile with WITH_GPU. "
              << "Will force to use CPU to run." << std::endl;
    CpuAdaptivePool(input_size, output_size_, input_data, output_data);
#endif
  } else {
    CpuAdaptivePool(input_size, output_size_, input_data, output_data);
  }
}

void AdaptivePool2dKernel::GetAttribute(const OrtKernelInfo* info) {
  pooling_type_ =
      ort_.KernelInfoGetAttribute<std::string>(info, "pooling_type");
  output_size_ =
      ort_.KernelInfoGetAttribute<std::vector<int64_t>>(info, "output_size");
  FDASSERT(
      output_size_.size() == 4 && output_size_[2] > 0 && output_size_[3] > 0,
      "The output size of adaptive pool must be positive.");
}
}  // namespace fastdeploy

#endif