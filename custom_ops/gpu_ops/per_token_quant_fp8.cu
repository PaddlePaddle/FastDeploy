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
__global__ void masked_quant_per_token_per_block(
    const T *input,
    const int *recv_expert_count,
    phi::dtype::float8_e4m3fn *quanted_res,
    ScaleT *quanted_scale,
    const int token_num,
    const int hidden_size,
    const int hidden_size_scale,
    const int num_max_tokens_per_expert,
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
    const auto token_idx_in_expert = token_idx % num_max_tokens_per_expert;
    const auto expert_id = token_idx / num_max_tokens_per_expert;
    if (token_idx_in_expert >= recv_expert_count[expert_id]) {
      auto next_expert_start_idx = (expert_id + 1) * num_max_tokens_per_expert;
      auto num_iters_to_next_expert =
          (next_expert_start_idx - token_idx - 1) / gridDim.x;
      token_idx += num_iters_to_next_expert * gridDim.x;
      continue;
    }

    const T *input_now = input + token_idx * hidden_size;
    phi::dtype::float8_e4m3fn *quanted_res_now =
        quanted_res + token_idx * hidden_size;
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
        scale_to_store = exp2f(ceilf(log2f(fmaxf(scale_to_store, epsilon))));
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
          // 1. extract exponent
          const int exp = (__float_as_int(scale_to_store) >> 23) & 0xFF;

          // 2. pack information
          const int pack_idx = iter >> 2;  // iter / 4
          const int byte_idx = iter & 3;   // iter % 4

          // 3. layout parameters
          const int pack_num = ceil_div(hidden_size_scale, 4);
          const int token_stride = align(num_max_tokens_per_expert, 4);

          // 4. base pointer (int32 pack)
          auto *scale_pack = reinterpret_cast<int32_t *>(quanted_scale);

          // 5. column-major offset:
          //    [expert][pack][token]
          const int base_idx = expert_id * pack_num * token_stride +
                               pack_idx * token_stride + token_idx_in_expert;

          // 6. write one byte into pack
          reinterpret_cast<uint8_t *>(&scale_pack[base_idx])[byte_idx] =
              static_cast<uint8_t>(exp);

        } else {
          // float scale path (no packing)
          float *scale_ptr =
              quanted_scale +
              expert_id * hidden_size_scale * num_max_tokens_per_expert +
              iter * num_max_tokens_per_expert + token_idx_in_expert;

          *scale_ptr = scale_to_store;
        }
      }
    }
  }
}

std::vector<paddle::Tensor> MaskedPerTokenQuant(
    paddle::Tensor &input,
    paddle::Tensor &recv_expert_count,
    const int block_size,
    const bool use_ue8m0) {
  auto input_dim = input.dims();
  const int num_local_expert = input_dim[0];
  const int num_max_tokens_per_expert = input_dim[1];
  const int hidden_size = input_dim[2];
  const int hidden_size_scale = hidden_size / block_size;
  const int token_num = num_local_expert * num_max_tokens_per_expert;
  auto quanted_x =
      GetEmptyTensor({num_local_expert, num_max_tokens_per_expert, hidden_size},
                     paddle::DataType::FLOAT8_E4M3FN,
                     input.place());

  const int gridx = min(132 * 2, token_num);
  const int blockx = min(1024, hidden_size / 128 * 32);

  bool use_finegrained_range = false;
  char *env_var = getenv("PER_TOKEN_QUANT_FP8_USE_FINEGRAINED_RANGE");
  if (env_var) {
    use_finegrained_range = static_cast<bool>(std::stoi(env_var));
  }
  if (use_ue8m0) {
    auto quanted_scale = GetEmptyTensor(
        {num_local_expert,
         num_max_tokens_per_expert,
         ceil_div(hidden_size_scale, 4)},
        {ceil_div(hidden_size_scale, 4) * align(num_max_tokens_per_expert, 4),
         1,
         align(num_max_tokens_per_expert, 4)},
        paddle::DataType::INT32,
        input.place());
    switch (input.dtype()) {
      case paddle::DataType::BFLOAT16:
        masked_quant_per_token_per_block<paddle::bfloat16, int32_t, true>
            <<<gridx, blockx, 0, input.stream()>>>(
                input.data<paddle::bfloat16>(),
                recv_expert_count.data<int>(),
                quanted_x.data<phi::dtype::float8_e4m3fn>(),
                quanted_scale.data<int32_t>(),
                token_num,
                hidden_size,
                hidden_size_scale,
                num_max_tokens_per_expert,
                use_finegrained_range);
        break;
      case paddle::DataType::FLOAT16:
        masked_quant_per_token_per_block<paddle::float16, int32_t, true>
            <<<gridx, blockx, 0, input.stream()>>>(
                input.data<paddle::float16>(),
                recv_expert_count.data<int>(),
                quanted_x.data<phi::dtype::float8_e4m3fn>(),
                quanted_scale.data<int32_t>(),
                token_num,
                hidden_size,
                hidden_size_scale,
                num_max_tokens_per_expert,
                use_finegrained_range);
        break;
      default:
        PD_THROW("Unsupported data type for PerTokenQuant");
    }
    return {quanted_x, quanted_scale};
  } else {
    auto quanted_scale = GetEmptyTensor(
        {num_local_expert, num_max_tokens_per_expert, hidden_size_scale},
        {hidden_size_scale * num_max_tokens_per_expert,
         1,
         num_max_tokens_per_expert},
        paddle::DataType::FLOAT32,
        input.place());
    switch (input.dtype()) {
      case paddle::DataType::BFLOAT16:
        masked_quant_per_token_per_block<paddle::bfloat16, float, false>
            <<<gridx, blockx, 0, input.stream()>>>(
                input.data<paddle::bfloat16>(),
                recv_expert_count.data<int>(),
                quanted_x.data<phi::dtype::float8_e4m3fn>(),
                quanted_scale.data<float>(),
                token_num,
                hidden_size,
                hidden_size_scale,
                num_max_tokens_per_expert,
                use_finegrained_range);
        break;
      case paddle::DataType::FLOAT16:
        masked_quant_per_token_per_block<paddle::float16, float, false>
            <<<gridx, blockx, 0, input.stream()>>>(
                input.data<paddle::float16>(),
                recv_expert_count.data<int>(),
                quanted_x.data<phi::dtype::float8_e4m3fn>(),
                quanted_scale.data<float>(),
                token_num,
                hidden_size,
                hidden_size_scale,
                num_max_tokens_per_expert,
                use_finegrained_range);
        break;
      default:
        PD_THROW("Unsupported data type for PerTokenQuant");
    }
    return {quanted_x, quanted_scale};
  }
}

PD_BUILD_STATIC_OP(masked_per_token_quant)
    .Inputs({"input", "recv_expert_count"})
    .Outputs({"output", "output_scale"})
    .Attrs({"block_size: int", "use_ue8m0: bool"})
    .SetKernelFn(PD_KERNEL(MaskedPerTokenQuant));
