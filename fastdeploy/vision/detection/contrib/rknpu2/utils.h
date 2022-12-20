// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.  //NOLINT
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
#include <stdint.h>
#include <cmath>
#include <vector>

namespace fastdeploy {
namespace vision {
namespace detection {
float Clamp(float val, int min, int max);
float Sigmoid(float x);
float UnSigmoid(float y);
inline static int32_t __clip(float val, float min, float max);
int8_t QntF32ToAffine(float f32, int32_t zp, float scale);
float DeqntAffineToF32(int8_t qnt, int32_t zp, float scale);
int NMS(int valid_count, std::vector<float>& output_locations,
        std::vector<int>& class_id, std::vector<int>& order, float threshold,
        bool class_agnostic);

}  // namespace detection
}  // namespace vision
}  // namespace fastdeploy
