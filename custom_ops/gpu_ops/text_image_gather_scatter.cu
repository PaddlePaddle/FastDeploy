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

inline cudaError_t GetGridSize(int64_t n,
                               int block_size,
                               int num_waves,
                               int* num_blocks) {
  int dev;
  {
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) {
      return err;
    }
  }
  int sm_count;
  {
    cudaError_t err =
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess) {
      return err;
    }
  }
  int tpm;
  {
    cudaError_t err = cudaDeviceGetAttribute(
        &tpm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
    if (err != cudaSuccess) {
      return err;
    }
  }
  *num_blocks =
      std::max<int>(1,
                    std::min<int64_t>((n + block_size - 1) / block_size,
                                      sm_count * tpm / block_size * num_waves));
  return cudaSuccess;
}

template <typename T, int VecSize>
__global__ void text_image_scatter_kernel(T* input_ptr,
                                          T* text_gather_ptr,
                                          T* image_gather_ptr,
                                          int32_t* token_type_ids,
                                          int32_t* text_index,
                                          int32_t* image_index,
                                          const int64_t hidden_size,
                                          const int64_t total_element_num) {
  constexpr int HalfVecSize = VecSize / 2;
  using T_Vec = AlignedVector<T, VecSize>;
  T_Vec input_ptr_vec;
  T_Vec text_images_vec;

  int64_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t step = blockDim.x * gridDim.x * VecSize;

  for (int64_t element_idx = global_thread_id * VecSize;
       element_idx < total_element_num;
       element_idx += step) {
    int64_t token_idx = element_idx / hidden_size;
    int64_t hidden_offset = element_idx % hidden_size;
    int32_t token_type_ids_num = token_type_ids[token_idx];

    int64_t input_load_offset = token_idx * hidden_size + hidden_offset;

    Load<T, VecSize>(input_ptr + input_load_offset, &input_ptr_vec);
#pragma unroll
    for (int vi = 0; vi < VecSize; ++vi) {
      text_images_vec[vi] = input_ptr_vec[vi];
    }

    if (token_type_ids_num == 0) {
      int64_t text_load_offset =
          text_index[token_idx] * hidden_size + hidden_offset;
      Store<T, VecSize>(text_images_vec, text_gather_ptr + text_load_offset);

    } else if (token_type_ids_num == 1) {
      int64_t image_load_offset =
          image_index[token_idx] * hidden_size + hidden_offset;
      Store<T, VecSize>(text_images_vec, image_gather_ptr + image_load_offset);

    } else {
      // skip cuda graph padding value
      continue;
    }
  }
}

template <typename T, int VecSize>
__global__ void text_image_gather_kernel(T* output_ptr,
                                         T* text_gather_ptr,
                                         T* image_gather_ptr,
                                         int32_t* token_type_ids,
                                         int32_t* text_index,
                                         int32_t* image_index,
                                         const int64_t hidden_size,
                                         const int64_t total_element_num) {
  constexpr int HalfVecSize = VecSize / 2;
  using T_Vec = AlignedVector<T, VecSize>;
  T_Vec output_ptr_vec;
  T_Vec text_imgaes_vec;

  int64_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t step = blockDim.x * gridDim.x * VecSize;

  for (int64_t element_idx = global_thread_id * VecSize;
       element_idx < total_element_num;
       element_idx += step) {
    int64_t token_idx = element_idx / hidden_size;
    int64_t hidden_offset = element_idx % hidden_size;
    int32_t token_type_ids_num = token_type_ids[token_idx];

    if (token_type_ids_num == 0) {
      int64_t text_load_offset =
          text_index[token_idx] * hidden_size + hidden_offset;
      Load<T, VecSize>(text_gather_ptr + text_load_offset, &text_imgaes_vec);

    } else if (token_type_ids_num == 1) {
      int64_t image_load_offset =
          image_index[token_idx] * hidden_size + hidden_offset;
      Load<T, VecSize>(image_gather_ptr + image_load_offset, &text_imgaes_vec);
    } else {
      // skip cuda graph padding value
      continue;
    }

#pragma unroll
    for (int vi = 0; vi < VecSize; ++vi) {
      output_ptr_vec[vi] = text_imgaes_vec[vi];
    }

    int64_t input_load_offset = token_idx * hidden_size + hidden_offset;

    Store<T, VecSize>(output_ptr_vec, output_ptr + input_load_offset);
  }
}

template <paddle::DataType D>
void LaunchTextImageGatherScatter(paddle::Tensor& input,
                                  paddle::Tensor& text_input,
                                  paddle::Tensor& image_input,
                                  paddle::Tensor& token_type_ids,
                                  paddle::Tensor& text_index,
                                  paddle::Tensor& image_index,
                                  const bool is_scatter) {
  typedef PDTraits<D> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t;
  auto stream = input.stream();
  const auto& in_dims = input.dims();
  const int64_t token_num = in_dims[0];
  const int64_t hidden_size = in_dims[1];

  const int VecSize = 16 / sizeof(data_t);
  const int64_t tot_element_num = token_num * hidden_size;

  int64_t tot_pack_num = (tot_element_num + VecSize - 1) / VecSize;

  const int block_size = 128;
  int grid_index = (token_num + block_size - 1) / block_size;
  constexpr int32_t kNumWaves = 16;
  int grid_size_x = -1;

  PADDLE_ENFORCE_GPU_SUCCESS(
      GetGridSize(tot_pack_num, block_size, kNumWaves, &grid_size_x));
  dim3 grid_dim = dim3(grid_size_x, 1, 1);
  if (is_scatter) {
    text_image_scatter_kernel<DataType_, VecSize>
        <<<grid_dim, block_size, 0, stream>>>(
            reinterpret_cast<DataType_*>(input.data<data_t>()),
            reinterpret_cast<DataType_*>(text_input.data<data_t>()),
            reinterpret_cast<DataType_*>(image_input.data<data_t>()),
            reinterpret_cast<int32_t*>(token_type_ids.data<int32_t>()),
            reinterpret_cast<int32_t*>(text_index.data<int32_t>()),
            reinterpret_cast<int32_t*>(image_index.data<int32_t>()),
            hidden_size,
            tot_element_num);
  } else {
    text_image_gather_kernel<DataType_, VecSize>
        <<<grid_dim, block_size, 0, stream>>>(
            reinterpret_cast<DataType_*>(input.data<data_t>()),
            reinterpret_cast<DataType_*>(text_input.data<data_t>()),
            reinterpret_cast<DataType_*>(image_input.data<data_t>()),
            reinterpret_cast<int32_t*>(token_type_ids.data<int32_t>()),
            reinterpret_cast<int32_t*>(text_index.data<int32_t>()),
            reinterpret_cast<int32_t*>(image_index.data<int32_t>()),
            hidden_size,
            tot_element_num);
  }
}

std::vector<paddle::Tensor> TextImageGatherScatter(
    paddle::Tensor& input,
    paddle::Tensor& text_input,
    paddle::Tensor& image_input,
    paddle::Tensor& token_type_ids,
    paddle::Tensor& text_index,
    paddle::Tensor& image_index,
    const bool is_scatter) {
  switch (input.dtype()) {
    case paddle::DataType::BFLOAT16: {
      LaunchTextImageGatherScatter<paddle::DataType::BFLOAT16>(input,
                                                               text_input,
                                                               image_input,
                                                               token_type_ids,
                                                               text_index,
                                                               image_index,
                                                               is_scatter);
      break;
    }
    default: {
      PD_THROW("NOT supported data type. Only support BFLOAT16, but got",
               input.dtype());
    }
  }
  return {input, text_input, image_input};
}

PD_BUILD_STATIC_OP(text_image_gather_scatter)
    .Inputs({"input",
             "text_input",
             "image_input",
             "token_type_ids",
             "text_index",
             "image_index"})
    .Outputs({"output", "text_input_out", "image_input_out"})
    .Attrs({"is_scatter:bool"})
    .SetInplaceMap({{"input", "output"},
                    {"text_input", "text_input_out"},
                    {"image_input", "image_input_out"}})
    .SetKernelFn(PD_KERNEL(TextImageGatherScatter));
