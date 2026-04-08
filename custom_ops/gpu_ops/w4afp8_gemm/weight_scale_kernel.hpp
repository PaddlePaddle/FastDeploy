// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/extension.h"

template <typename T, int kPackSize>
__global__ void permute_scale_kernel(T* input_data, const int numel) {
  using LoadT = AlignedVector<T, kPackSize>;
  LoadT input_vec;
  LoadT dst_vec;
  const int load_idx = (blockIdx.x * blockDim.x + threadIdx.x) * kPackSize;
  if (load_idx >= numel) {
    return;
  }
  Load<T, kPackSize>(&input_data[load_idx], &input_vec);

  for (int i = 0; i < kPackSize; i += 2) {
    dst_vec[i] = input_vec[i / 2];
    dst_vec[i + 1] = input_vec[i / 2 + 8];
  }

  Store<T, kPackSize>(dst_vec, &input_data[load_idx]);
}

void W4AFp8GemmScalePermute(const paddle::Tensor& scale) {
  const int row = scale.dims().size() == 2 ? scale.dims()[0] : 1;
  const int col = scale.dims().size() == 2 ? scale.dims()[1] : scale.dims()[0];
  if (col % 16 != 0) {
    PD_THROW("Only supported when col is divisible by 16.");
  }
  const int numel = row * col;
  const int threads = 128;
  const int kPackSize = 16;
  const int grid_size = (numel / kPackSize + threads - 1) / threads;

  if (scale.dtype() == paddle::DataType::BFLOAT16) {
    permute_scale_kernel<phi::dtype::bfloat16, kPackSize>
        <<<grid_size, threads, 0, scale.stream()>>>(
            const_cast<phi::dtype::bfloat16*>(
                scale.data<phi::dtype::bfloat16>()),
            numel);
  } else if (scale.dtype() == paddle::DataType::FLOAT16) {
    permute_scale_kernel<phi::dtype::float16, kPackSize>
        <<<grid_size, threads, 0, scale.stream()>>>(
            const_cast<phi::dtype::float16*>(scale.data<phi::dtype::float16>()),
            numel);
  } else if (scale.dtype() == paddle::DataType::FLOAT32) {
    permute_scale_kernel<float, kPackSize>
        <<<grid_size, threads, 0, scale.stream()>>>(
            const_cast<float*>(scale.data<float>()), numel);
  }
}
