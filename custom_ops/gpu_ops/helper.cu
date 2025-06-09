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

float bfloat16_to_float(__nv_bfloat16 x) {
    uint32_t tmp_x = *(reinterpret_cast<uint16_t*>(&x));
    tmp_x = tmp_x << 16;
    float float_x = *(reinterpret_cast<float*>(&tmp_x));
    return float_x;
}

template <typename T>
static void PrintMatrix(const T* mat_d,
                        int num,
                        std::string name,
                        int numOfCols) {
    std::vector<T> tmp(num);
    cudaMemcpy(tmp.data(), mat_d, sizeof(T) * num, cudaMemcpyDeviceToHost);

    std::ofstream outfile;
    outfile.open(name + ".dtxt", std::ios::out | std::ios::app);
    std::stringstream ss;

    for (int i = 0; i < num; ++i) {
        if (std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value ||
            std::is_same<T, int32_t>::value) {
            ss << static_cast<int>(tmp[i]) << " ";
        } else {
            ss << std::setprecision(8) << static_cast<float>(tmp[i]) << " ";
        }
        if (i % numOfCols == numOfCols - 1) {
            ss << std::endl;
        }
    }
    outfile << ss.str();
    outfile.close();
}
