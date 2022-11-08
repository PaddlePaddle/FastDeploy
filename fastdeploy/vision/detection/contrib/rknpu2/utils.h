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
#include <cmath>
#include <cstdint>
#include <vector>
float sigmoid(float x);
float unsigmoid(float y);
inline static int32_t __clip(float val, float min, float max);
int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale);
float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale);
