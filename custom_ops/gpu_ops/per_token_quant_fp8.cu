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

constexpr float epsilon = 1e-10;

__host__ __device__ __forceinline__ int ceil_div(int x, int y) {
  return (x + y - 1) / y;
}

__host__ __device__ __forceinline__ int align(int x, int y) {
  return ceil_div(x, y) * y;
}

template <typename T, typename ScaleT, bool UseUE8M0>
__global__ void quant_per_token_per_block(
    const T *input,
    phi::dtype::float8_e4m3fn *quanted_res,
    ScaleT *quanted_scale,
    const int token_num,
    const int hidden_size,
    const int hidden_size_scale,
    const bool use_finegrained_range) {
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;
  const int num_warp = blockDim.x / 32;
  static constexpr int NUM_PER_THREADS = 128 / 32;  // 4
  static constexpr float MAX_VALUE = 448.f;
  // Note(ZKK) use ceil_div!!
  const int end_iter = (hidden_size + 127) / 128;  // warp_iter_num
  AlignedVector<T, NUM_PER_THREADS> load_vec;
  AlignedVector<float, NUM_PER_THREADS> load_vec_float;
  AlignedVector<phi::dtype::float8_e4m3fn, NUM_PER_THREADS> res_vec;
  for (int token_idx = bid; token_idx < token_num; token_idx += gridDim.x) {
    const T *input_now = input + static_cast<int64_t>(token_idx) * hidden_size;
    phi::dtype::float8_e4m3fn *quanted_res_now =
        quanted_res + static_cast<int64_t>(token_idx) * hidden_size;
    float *quanted_scale_now = reinterpret_cast<float *>(quanted_scale) +
                               token_idx * hidden_size_scale;
    // deal a block per warp
    for (int iter = warp_id; iter < end_iter; iter += num_warp) {
      const int start_offset = iter * 128;

      const bool is_valid_data =
          start_offset + lane_id * NUM_PER_THREADS < hidden_size;

      if (is_valid_data) {
        Load<T, NUM_PER_THREADS>(
            input_now + start_offset + lane_id * NUM_PER_THREADS, &load_vec);
      } else {
#pragma unroll
        for (int vid = 0; vid < NUM_PER_THREADS; vid++) load_vec[vid] = T(0.f);
      }
      // get max value per thread
      float max_value_thread = -5e4;
#pragma unroll
      for (int vid = 0; vid < NUM_PER_THREADS; vid++) {
        load_vec_float[vid] = static_cast<float>(load_vec[vid]);
        max_value_thread = max(abs(load_vec_float[vid]), max_value_thread);
      }
      // get max value per warp
      max_value_thread = max(__shfl_down_sync(0xffffffff, max_value_thread, 16),
                             max_value_thread);
      max_value_thread = max(__shfl_down_sync(0xffffffff, max_value_thread, 8),
                             max_value_thread);
      max_value_thread = max(__shfl_down_sync(0xffffffff, max_value_thread, 4),
                             max_value_thread);
      max_value_thread = max(__shfl_down_sync(0xffffffff, max_value_thread, 2),
                             max_value_thread);
      max_value_thread = max(__shfl_down_sync(0xffffffff, max_value_thread, 1),
                             max_value_thread);
      // broadcast max_value
      max_value_thread = __shfl_sync(0xFFFFFFFF, max_value_thread, 0);
      max_value_thread = max(max_value_thread, epsilon);

      if (use_finegrained_range) {
        max_value_thread *= 7.0f;
      }

      float scale_to_store = max_value_thread / MAX_VALUE;

      // quant
      if constexpr (UseUE8M0) {
        scale_to_store =
            exp2f(ceilf(log2f(fmaxf(scale_to_store, epsilon) + 5e-7f)));
#pragma unroll
        for (int vid = 0; vid < NUM_PER_THREADS; vid++) {
          res_vec[vid] = static_cast<phi::dtype::float8_e4m3fn>(
              load_vec_float[vid] / scale_to_store);
        }
      } else {
#pragma unroll
        for (int vid = 0; vid < NUM_PER_THREADS; vid++) {
          res_vec[vid] = static_cast<phi::dtype::float8_e4m3fn>(
              load_vec_float[vid] * MAX_VALUE / max_value_thread);
        }
      }
      // store
      if (is_valid_data)
        Store<phi::dtype::float8_e4m3fn, NUM_PER_THREADS>(
            res_vec,
            quanted_res_now + start_offset + lane_id * NUM_PER_THREADS);
      if (lane_id == 0) {
        if constexpr (UseUE8M0) {
          int exp = (reinterpret_cast<int &>(scale_to_store) >> 23) & 0xFF;
          const int pack_idx = iter >> 2;
          const int byte_idx = iter & 3;
          const int pack_num = ceil_div(hidden_size_scale, 4);
          int32_t *scale_now = quanted_scale;
          const int base_idx = token_idx * pack_num + pack_idx;
          reinterpret_cast<uint8_t *>(&scale_now[base_idx])[byte_idx] =
              static_cast<uint8_t>(exp);
        } else {
          quanted_scale_now[iter] = scale_to_store;
        }
      }
    }
  }
}

std::vector<paddle::Tensor> PerTokenQuant(paddle::Tensor &input,
                                          const int block_size,
                                          const bool use_ue8m0) {
  auto input_dim = input.dims();
  const int token_num = input_dim[0];
  const int hidden_size = input_dim[1];
  // Note(ZKK) here we use ceil_dive to support 4.5T runing on 8 GPUS
  // where moe_intermediate_size is 448, can not be divided by 128.
  const int hidden_size_scale = (hidden_size + block_size - 1) / block_size;

  auto quanted_x = GetEmptyTensor(
      {token_num, hidden_size}, paddle::DataType::FLOAT8_E4M3FN, input.place());

  const int gridx = min(132 * 8, token_num);
  const int blockx = min(1024, hidden_size / 128 * 32);

  bool use_finegrained_range = false;
  char *env_var = getenv("PER_TOKEN_QUANT_FP8_USE_FINEGRAINED_RANGE");
  if (env_var) {
    use_finegrained_range = static_cast<bool>(std::stoi(env_var));
  }

  if (use_ue8m0) {
    auto quanted_scale =
        GetEmptyTensor({token_num, ceil_div(hidden_size_scale, 4)},
                       paddle::DataType::INT32,
                       input.place());
    switch (input.dtype()) {
      case paddle::DataType::BFLOAT16:
        quant_per_token_per_block<paddle::bfloat16, int32_t, true>
            <<<gridx, blockx, 0, input.stream()>>>(
                input.data<paddle::bfloat16>(),
                quanted_x.data<phi::dtype::float8_e4m3fn>(),
                quanted_scale.data<int32_t>(),
                token_num,
                hidden_size,
                hidden_size_scale,
                use_finegrained_range);
        break;
      case paddle::DataType::FLOAT16:
        quant_per_token_per_block<paddle::float16, int32_t, true>
            <<<gridx, blockx, 0, input.stream()>>>(
                input.data<paddle::float16>(),
                quanted_x.data<phi::dtype::float8_e4m3fn>(),
                quanted_scale.data<int32_t>(),
                token_num,
                hidden_size,
                hidden_size_scale,
                use_finegrained_range);
        break;
      default:
        PD_THROW("Unsupported data type for PerTokenQuant");
    }
    return {quanted_x, quanted_scale};
  } else {
    auto quanted_scale = GetEmptyTensor({token_num, hidden_size_scale},
                                        paddle::DataType::FLOAT32,
                                        input.place());
    switch (input.dtype()) {
      case paddle::DataType::BFLOAT16:
        quant_per_token_per_block<paddle::bfloat16, float, false>
            <<<gridx, blockx, 0, input.stream()>>>(
                input.data<paddle::bfloat16>(),
                quanted_x.data<phi::dtype::float8_e4m3fn>(),
                quanted_scale.data<float>(),
                token_num,
                hidden_size,
                hidden_size_scale,
                use_finegrained_range);
        break;
      case paddle::DataType::FLOAT16:
        quant_per_token_per_block<paddle::float16, float, false>
            <<<gridx, blockx, 0, input.stream()>>>(
                input.data<paddle::float16>(),
                quanted_x.data<phi::dtype::float8_e4m3fn>(),
                quanted_scale.data<float>(),
                token_num,
                hidden_size,
                hidden_size_scale,
                use_finegrained_range);
        break;
      default:
        PD_THROW("Unsupported data type for PerTokenQuant");
    }
    return {quanted_x, quanted_scale};
  }
}

std::vector<std::vector<int64_t>> PerTokenQuantInferShape(
    std::vector<int64_t> input_shape, const int block_size) {
  const int token_num = input_shape[0];
  const int hidden_size = input_shape[1];
  const int hidden_size_scale = (hidden_size + block_size - 1) / block_size;
  if (GetSMVersion() >= 100) {
    return {{token_num, hidden_size},
            {token_num, ceil_div(hidden_size_scale, 4)}};
  }
  return {{token_num, hidden_size}, {token_num, hidden_size_scale}};
}

std::vector<paddle::DataType> PerTokenQuantInferDtype(
    paddle::DataType input_dtype, const int block_size) {
  if (GetSMVersion() >= 100) {
    return {paddle::DataType::FLOAT8_E4M3FN, paddle::DataType::INT32};
  }
  return {paddle::DataType::FLOAT8_E4M3FN, paddle::DataType::FLOAT32};
}

template <typename T, typename ScaleT, bool UseUE8M0>
__global__ void quant_per_token_per_block_padding(
    const T *input,
    phi::dtype::float8_e4m3fn *quanted_res,
    ScaleT *quanted_scale,
    const int token_num,
    const int padded_token_num,
    const int hidden_size,
    const int hidden_size_scale,
    const bool use_finegrained_range) {
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;
  const int num_warp = blockDim.x / 32;
  static constexpr int NUM_PER_THREADS = 128 / 32;  // 4
  static constexpr float MAX_VALUE = 448.f;
  const int end_iter = hidden_size / 128;  // warp_iter_num
  AlignedVector<T, NUM_PER_THREADS> load_vec;
  AlignedVector<float, NUM_PER_THREADS> load_vec_float;
  AlignedVector<phi::dtype::float8_e4m3fn, NUM_PER_THREADS> res_vec;
  for (int token_idx = bid; token_idx < token_num; token_idx += gridDim.x) {
    const T *input_now = input + static_cast<int64_t>(token_idx) * hidden_size;
    phi::dtype::float8_e4m3fn *quanted_res_now =
        quanted_res + static_cast<int64_t>(token_idx) * hidden_size;
    // deal a block per warp
    for (int iter = warp_id; iter < end_iter; iter += num_warp) {
      const int start_offset = iter * 128;
      Load<T, NUM_PER_THREADS>(
          input_now + start_offset + lane_id * NUM_PER_THREADS, &load_vec);
      // get max value per thread
      float max_value_thread = -5e4;
#pragma unroll
      for (int vid = 0; vid < NUM_PER_THREADS; vid++) {
        load_vec_float[vid] = static_cast<float>(load_vec[vid]);
        max_value_thread = max(abs(load_vec_float[vid]), max_value_thread);
      }
      // get max value per warp
      max_value_thread = max(__shfl_down_sync(0xffffffff, max_value_thread, 16),
                             max_value_thread);
      max_value_thread = max(__shfl_down_sync(0xffffffff, max_value_thread, 8),
                             max_value_thread);
      max_value_thread = max(__shfl_down_sync(0xffffffff, max_value_thread, 4),
                             max_value_thread);
      max_value_thread = max(__shfl_down_sync(0xffffffff, max_value_thread, 2),
                             max_value_thread);
      max_value_thread = max(__shfl_down_sync(0xffffffff, max_value_thread, 1),
                             max_value_thread);
      // broadcast max_value
      max_value_thread = __shfl_sync(0xFFFFFFFF, max_value_thread, 0);
      max_value_thread = max(max_value_thread, epsilon);

      if (use_finegrained_range) {
        max_value_thread *= 7.0f;
      }

      float scale_to_store = max_value_thread / MAX_VALUE;

      // quant
      if constexpr (UseUE8M0) {
        scale_to_store =
            exp2f(ceilf(log2f(fmaxf(scale_to_store, epsilon) + 5e-7f)));
#pragma unroll
        for (int vid = 0; vid < NUM_PER_THREADS; vid++) {
          res_vec[vid] = static_cast<phi::dtype::float8_e4m3fn>(
              load_vec_float[vid] / scale_to_store);
        }
      } else {
#pragma unroll
        for (int vid = 0; vid < NUM_PER_THREADS; vid++) {
          res_vec[vid] = static_cast<phi::dtype::float8_e4m3fn>(
              load_vec_float[vid] * MAX_VALUE / max_value_thread);
        }
      }
      // store
      Store<phi::dtype::float8_e4m3fn, NUM_PER_THREADS>(
          res_vec, quanted_res_now + start_offset + lane_id * NUM_PER_THREADS);
      if (lane_id == 0) {
        if constexpr (UseUE8M0) {
          // exp
          int exp = (reinterpret_cast<int &>(scale_to_store) >> 23) & 0xFF;

          const int pack_idx = iter >> 2;
          const int byte_idx = iter & 3;

          // pack
          const int pack_num = align(hidden_size_scale, 4) >> 2;

          // column-major base index
          int32_t *scale_now = quanted_scale;
          const int base_idx = token_idx + pack_idx * padded_token_num;

          // ---------------- store exp ----------------
          reinterpret_cast<uint8_t *>(&scale_now[base_idx])[byte_idx] =
              static_cast<uint8_t>(exp);
        } else {
          float *scale_now =
              quanted_scale + iter * padded_token_num + token_idx;
          *scale_now = scale_to_store;
        }
      }
    }
  }
}

std::vector<paddle::Tensor> PerTokenQuantPadding(paddle::Tensor &input,
                                                 const int block_size,
                                                 const bool use_ue8m0) {
  using ScaleDtype = float;
  auto input_dim = input.dims();
  const int token_num = input_dim[0];
  const int hidden_size = input_dim[1];

  PADDLE_ENFORCE(block_size == 128, "now only support block_size = 128");
  PADDLE_ENFORCE(hidden_size % 128 == 0,
                 "hidden_size must be divisible by 128");

  const int hidden_size_scale = hidden_size / block_size;
  auto quanted_x = GetEmptyTensor(
      {token_num, hidden_size}, paddle::DataType::FLOAT8_E4M3FN, input.place());

  const int tma_alignment_bytes = 16;
  const int tma_alignment_elements = tma_alignment_bytes / sizeof(ScaleDtype);

  const int padded_token_num =
      ((token_num + tma_alignment_elements - 1) / tma_alignment_elements) *
      tma_alignment_elements;

  const int gridx = min(132 * 8, token_num);
  const int blockx = min(1024, hidden_size / 128 * 32);

  bool use_finegrained_range = false;
  char *env_var = getenv("PER_TOKEN_QUANT_FP8_USE_FINEGRAINED_RANGE");
  if (env_var) {
    use_finegrained_range = static_cast<bool>(std::stoi(env_var));
  }
  if (use_ue8m0) {
    auto quanted_scale =
        GetEmptyTensor({padded_token_num, ceil_div(hidden_size_scale, 4)},
                       {1, padded_token_num},
                       paddle::DataType::INT32,
                       input.place());
    switch (input.dtype()) {
      case paddle::DataType::BFLOAT16:
        quant_per_token_per_block_padding<paddle::bfloat16, int32_t, true>
            <<<gridx, blockx, 0, input.stream()>>>(
                input.data<paddle::bfloat16>(),
                quanted_x.data<phi::dtype::float8_e4m3fn>(),
                quanted_scale.data<int32_t>(),
                token_num,
                padded_token_num,
                hidden_size,
                hidden_size_scale,
                use_finegrained_range);
        break;
      case paddle::DataType::FLOAT16:
        quant_per_token_per_block_padding<paddle::float16, int32_t, true>
            <<<gridx, blockx, 0, input.stream()>>>(
                input.data<paddle::float16>(),
                quanted_x.data<phi::dtype::float8_e4m3fn>(),
                quanted_scale.data<int32_t>(),
                token_num,
                padded_token_num,
                hidden_size,
                hidden_size_scale,
                use_finegrained_range);
        break;
      default:
        PD_THROW("Unsupported data type for PerTokenQuant");
    }
    return {quanted_x, quanted_scale};
  } else {
    auto quanted_scale = GetEmptyTensor({padded_token_num, hidden_size_scale},
                                        {1, padded_token_num},
                                        paddle::DataType::FLOAT32,
                                        input.place());
    switch (input.dtype()) {
      case paddle::DataType::BFLOAT16:
        quant_per_token_per_block_padding<paddle::bfloat16, float, false>
            <<<gridx, blockx, 0, input.stream()>>>(
                input.data<paddle::bfloat16>(),
                quanted_x.data<phi::dtype::float8_e4m3fn>(),
                quanted_scale.data<float>(),
                token_num,
                padded_token_num,
                hidden_size,
                hidden_size_scale,
                use_finegrained_range);
        break;
      case paddle::DataType::FLOAT16:
        quant_per_token_per_block_padding<paddle::float16, float, false>
            <<<gridx, blockx, 0, input.stream()>>>(
                input.data<paddle::float16>(),
                quanted_x.data<phi::dtype::float8_e4m3fn>(),
                quanted_scale.data<float>(),
                token_num,
                padded_token_num,
                hidden_size,
                hidden_size_scale,
                use_finegrained_range);
        break;
      default:
        PD_THROW("Unsupported data type for PerTokenQuant");
    }
    return {quanted_x, quanted_scale};
  }
}

std::vector<std::vector<int64_t>> PerTokenQuantPaddingInferShape(
    std::vector<int64_t> input_shape, const int block_size) {
  using ScaleDtype = float;

  const int token_num = input_shape[0];
  const int hidden_size = input_shape[1];
  const int hidden_size_scale = hidden_size / block_size;

  const int tma_alignment_bytes = 16;
  const int tma_alignment_elements = tma_alignment_bytes / sizeof(ScaleDtype);
  const int padded_token_num =
      ((token_num + tma_alignment_elements - 1) / tma_alignment_elements) *
      tma_alignment_elements;
  if (GetSMVersion() >= 100) {
    return {{token_num, hidden_size},
            {padded_token_num, ceil_div(hidden_size_scale, 4)}};
  }
  return {{token_num, hidden_size}, {padded_token_num, hidden_size_scale}};
}

std::vector<paddle::DataType> PerTokenQuantPaddingInferDtype(
    paddle::DataType input_dtype) {
  if (GetSMVersion() >= 100) {
    return {paddle::DataType::FLOAT8_E4M3FN, paddle::DataType::INT32};
  }
  return {paddle::DataType::FLOAT8_E4M3FN, paddle::DataType::FLOAT32};
}

PD_BUILD_STATIC_OP(per_token_quant)
    .Inputs({"input"})
    .Outputs({"output", "output_scale"})
    .Attrs({"block_size: int", "use_ue8m0: bool"})
    .SetKernelFn(PD_KERNEL(PerTokenQuant))
    .SetInferShapeFn(PD_INFER_SHAPE(PerTokenQuantInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(PerTokenQuantInferDtype));

PD_BUILD_STATIC_OP(per_token_quant_padding)
    .Inputs({"input"})
    .Outputs({"output", "output_scale"})
    .Attrs({"block_size: int", "use_ue8m0: bool"})
    .SetKernelFn(PD_KERNEL(PerTokenQuantPadding))
    .SetInferShapeFn(PD_INFER_SHAPE(PerTokenQuantPaddingInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(PerTokenQuantPaddingInferDtype));
