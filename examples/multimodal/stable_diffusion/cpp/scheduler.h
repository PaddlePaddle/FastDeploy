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

#include "fastdeploy/core/fd_tensor.h"

namespace fastdeploy {

class Scheduler {
 public:
  virtual void SetTimesteps(int num_inference_steps) = 0;
  virtual FDTensor GetTimesteps() = 0;
  virtual void Step(const FDTensor& model_output, int timestep,
                    const FDTensor& sample, FDTensor* prev_sample) = 0;
  virtual void ScaleModelInput(const FDTensor& sample, FDTensor* out,
                               const std::vector<FDTensor>& timesteps = {}) = 0;
  virtual void AddNoise(const FDTensor& original_samples, const FDTensor& noise,
                        const FDTensor& timesteps, FDTensor* out) = 0;
  virtual float InitNoiseSigma() = 0;
};

}  // namespace fastdeploy
