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

// Ignore CUTLASS warnings about type punning
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma once

#include "moe/fused_moe_helper.h"
#include "moe/fused_moe_op.h"
#pragma GCC diagnostic pop

#include <cooperative_groups.h>
#include "helper.h"

#define DISPATCH_NUM_EXPERTS_PER_RANK(                                         \
    num_experts_per_rank, NUM_EXPERTS_PER_RANK, ...)                           \
  switch (num_experts_per_rank) {                                              \
    case 2: {                                                                  \
      constexpr size_t NUM_EXPERTS_PER_RANK = 2;                               \
      __VA_ARGS__                                                              \
      break;                                                                   \
    }                                                                          \
    case 3: {                                                                  \
      constexpr size_t NUM_EXPERTS_PER_RANK = 3;                               \
      __VA_ARGS__                                                              \
      break;                                                                   \
    }                                                                          \
    case 6: {                                                                  \
      constexpr size_t NUM_EXPERTS_PER_RANK = 6;                               \
      __VA_ARGS__                                                              \
      break;                                                                   \
    }                                                                          \
    case 8: {                                                                  \
      constexpr size_t NUM_EXPERTS_PER_RANK = 8;                               \
      __VA_ARGS__                                                              \
      break;                                                                   \
    }                                                                          \
    case 9: {                                                                  \
      constexpr size_t NUM_EXPERTS_PER_RANK = 9;                               \
      __VA_ARGS__                                                              \
      break;                                                                   \
    }                                                                          \
    case 16: {                                                                 \
      constexpr size_t NUM_EXPERTS_PER_RANK = 16;                              \
      __VA_ARGS__                                                              \
      break;                                                                   \
    }                                                                          \
    case 32: {                                                                 \
      constexpr size_t NUM_EXPERTS_PER_RANK = 32;                              \
      __VA_ARGS__                                                              \
      break;                                                                   \
    }                                                                          \
    case 48: {                                                                 \
      constexpr size_t NUM_EXPERTS_PER_RANK = 48;                              \
      __VA_ARGS__                                                              \
      break;                                                                   \
    }                                                                          \
    case 64: {                                                                 \
      constexpr size_t NUM_EXPERTS_PER_RANK = 64;                              \
      __VA_ARGS__                                                              \
      break;                                                                   \
    }                                                                          \
    case 128: {                                                                \
      constexpr size_t NUM_EXPERTS_PER_RANK = 128;                             \
      __VA_ARGS__                                                              \
      break;                                                                   \
    }                                                                          \
    case 160: {                                                                \
      constexpr size_t NUM_EXPERTS_PER_RANK = 160;                             \
      __VA_ARGS__                                                              \
      break;                                                                   \
    }                                                                          \
    default: {                                                                 \
      std::ostringstream err_msg;                                              \
      err_msg << "Unsupported num_experts_per_rank: " << num_experts_per_rank; \
      throw std::invalid_argument(err_msg.str());                              \
    }                                                                          \
  }

namespace cg = cooperative_groups;

template <typename T>
__device__ T warpReduceSum(T val) {
  for (int lane_mask = 16; lane_mask > 0; lane_mask /= 2) {
    val += __shfl_down_sync(0xffffffff, val, lane_mask);
  }
  return val;
}

__global__ void get_expert_token_num(int64_t* topk_ids,
                                     int* out_workspace,  // num_experts * 2 + 2
                                     const int token_num,
                                     const int moe_topk,
                                     const int num_experts) {
  cg::grid_group grid = cg::this_grid();
  constexpr int KNWARPS = 512 / 32;
  __shared__ int warp_sum[KNWARPS * 2];
  int* expert_token_num = out_workspace;
  int* expert_token_num_padded = out_workspace + num_experts;
  int* token_num_all = out_workspace + num_experts * 2;
  int* token_num_all_padded = out_workspace + num_experts * 2 + 1;
  const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = global_idx; i < num_experts; i += blockDim.x * gridDim.x) {
    expert_token_num[i] = 0;
    expert_token_num_padded[i] = 0;
  }
  grid.sync();
  for (int i = global_idx; i < token_num * moe_topk;
       i += blockDim.x * gridDim.x) {
    const int topk_idx = topk_ids[i];
    atomicAdd(&expert_token_num[topk_idx], 1);
  }
  grid.sync();
  for (int i = global_idx; i < num_experts; i += blockDim.x * gridDim.x) {
    const int token_num_per_expert = expert_token_num[i];
    if (token_num_per_expert > 0) {
      expert_token_num_padded[i] =
          128 - token_num_per_expert % 128 + token_num_per_expert;
    }
  }
  grid.sync();
  if (blockIdx.x == 0) {
    int token_num_now = 0;
    int token_num_padded = 0;
    if (threadIdx.x < num_experts) {
      token_num_now = expert_token_num[threadIdx.x];
      token_num_padded = expert_token_num_padded[threadIdx.x];
    }
    const int laneId = threadIdx.x % 32;
    const int warpId = threadIdx.x / 32;

    int sum = warpReduceSum<int>(token_num_now);
    int sum_padded = warpReduceSum<int>(token_num_padded);
    __syncthreads();
    if (laneId == 0) {
      warp_sum[warpId] = sum;
      warp_sum[warpId + KNWARPS] = sum_padded;
    }
    __syncthreads();
    sum = (threadIdx.x < KNWARPS) ? warp_sum[laneId] : 0;
    sum_padded = (threadIdx.x < KNWARPS) ? warp_sum[laneId + KNWARPS] : 0;
    if (warpId == 0) {
      sum = warpReduceSum<int>(sum);
      sum_padded = warpReduceSum<int>(sum_padded);
    }
    if (threadIdx.x == 0) {
      *token_num_all = sum;
      *token_num_all_padded = sum_padded;
    }
  }
}

std::vector<std::vector<int>> GetExpertTokenNum(const paddle::Tensor& topk_ids,
                                                const int num_experts) {
  const int token_num = topk_ids.dims()[0];
  const int moe_topk = topk_ids.dims()[1];
  auto out_workspace = GetEmptyTensor(
      {num_experts * 2 + 2}, paddle::DataType::INT32, topk_ids.place());
  const int block_size = 512;
  const int grid_size = min(132 * 4, div_up(token_num * moe_topk, block_size));
  int64_t* topk_ids_ptr = const_cast<int64_t*>(topk_ids.data<int64_t>());
  int* out_workspace_ptr = out_workspace.data<int>();
  void* kernel_args[] = {(void*)(&topk_ids_ptr),
                         (void*)(&out_workspace_ptr),
                         (void*)&token_num,
                         (void*)&moe_topk,
                         (void*)&num_experts};
  cudaLaunchCooperativeKernel((void*)get_expert_token_num,
                              dim3(grid_size),
                              dim3(block_size),
                              kernel_args,
                              0,
                              topk_ids.stream());
  auto out_workspace_host = out_workspace.copy_to(paddle::CPUPlace(), true);
  int* out_workspace_host_ptr = out_workspace_host.data<int>();
  std::vector<int> expert_token_num(out_workspace_host_ptr,
                                    out_workspace_host_ptr + num_experts);
  std::vector<int> expert_token_num_padded(
      out_workspace_host_ptr + num_experts,
      out_workspace_host_ptr + num_experts * 2);
  std::vector<int> token_num_all(out_workspace_host_ptr + num_experts * 2,
                                 out_workspace_host_ptr + num_experts * 2 + 2);
  return {expert_token_num, expert_token_num_padded, token_num_all};
}

template <typename T>
__global__ void combine_prmt_back_kernel(
    const T* expanded_permuted_rows,
    T* reduced_unpermuted_output,
    const T* bias,
    const float* dst_weights,
    const int*
        expanded_source_row_to_expanded_dest_row,  // permute_indices_per_token
    const int* expert_for_source_row,              // dst_idx
    const int64_t cols,
    const int64_t k,
    const int64_t compute_bias,
    const bool norm_topk_prob,
    const float routed_scaling_factor,
    const int64_t num_rows) {
  static constexpr int VEC_SIZE = sizeof(int4) / sizeof(T);
  AlignedVector<T, VEC_SIZE> load_vec;
  AlignedVector<T, VEC_SIZE> bias_vec;
  AlignedVector<T, VEC_SIZE> res_vec;
  const int cols_int4 = cols / VEC_SIZE;
  for (int original_row = blockIdx.x; original_row < num_rows;
       original_row += gridDim.x) {
    T* reduced_row_ptr = reduced_unpermuted_output + original_row * cols;
    for (int tid = threadIdx.x; tid < cols_int4; tid += blockDim.x) {
#pragma unroll
      for (int vid = 0; vid < VEC_SIZE; vid++) {
        res_vec[vid] = 0;
      }
      for (int k_idx = 0; k_idx < k; ++k_idx) {  // k is num_experts_per_rank
        const int expanded_original_row = original_row + k_idx * num_rows;
        const int expanded_permuted_row =
            expanded_source_row_to_expanded_dest_row[expanded_original_row];
        if (expanded_permuted_row < 0) continue;
        const int64_t k_offset = original_row * k + k_idx;
        const float row_scale = dst_weights[expanded_permuted_row];
        const T* expanded_permuted_rows_row_ptr =
            expanded_permuted_rows +
            expanded_permuted_row * cols;  // prmt后的位置对应的值
        Load<T, VEC_SIZE>(expanded_permuted_rows_row_ptr + tid * VEC_SIZE,
                          &load_vec);
        const int expert_idx =
            expert_for_source_row[k_offset];  // 当前位置对应的专家
        const T* bias_ptr = bias ? bias + expert_idx * cols
                                 : nullptr;  // 当前专家对应的down_proj的bias
        if (bias_ptr) {
          Load<T, VEC_SIZE>(bias_ptr + tid * VEC_SIZE, &bias_vec);
#pragma unroll
          for (int vid = 0; vid < VEC_SIZE; vid++) {
            res_vec[vid] +=
                static_cast<T>(row_scale * static_cast<float>(load_vec[vid]) +
                               static_cast<float>(bias_vec[vid]));
          }
        } else {
#pragma unroll
          for (int vid = 0; vid < VEC_SIZE; vid++) {
            res_vec[vid] +=
                static_cast<T>(row_scale * static_cast<float>(load_vec[vid]));
          }
        }
      }
      Store<T, VEC_SIZE>(res_vec, reduced_row_ptr + tid * VEC_SIZE);
    }
  }
}

template <paddle::DataType T>
void MoeCombineKernel(const paddle::Tensor& ffn_out,
                      const paddle::Tensor& expert_scales_float,
                      const paddle::Tensor& permute_indices_per_token,
                      const paddle::Tensor& top_k_indices,
                      const paddle::optional<paddle::Tensor>& down_proj_bias,
                      const bool norm_topk_prob,
                      const float routed_scaling_factor,
                      const int num_rows,
                      const int hidden_size,
                      paddle::Tensor* output) {
  using namespace phi;
  typedef PDTraits<T> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t;
  auto stream = ffn_out.stream();
  const int threads = 1024;
  const int gridx = min(132 * 8, num_rows);
  const int num_experts_per_rank = top_k_indices.dims()[1];

  combine_prmt_back_kernel<<<gridx, threads, 0, stream>>>(
      ffn_out.data<data_t>(),
      output->data<data_t>(),
      down_proj_bias ? down_proj_bias->data<data_t>() : nullptr,
      expert_scales_float.data<float>(),
      permute_indices_per_token.data<int32_t>(),
      top_k_indices.data<int>(),
      hidden_size,
      num_experts_per_rank,
      static_cast<int>(1),  // compute bias
      norm_topk_prob,
      routed_scaling_factor,
      num_rows);
}

std::vector<paddle::Tensor> EPMoeExpertCombine(
    const paddle::Tensor& ffn_out,
    const paddle::Tensor& expert_scales_float,  // dst_weights
    const paddle::Tensor&
        permute_indices_per_token,        // permute_indices_per_token
    const paddle::Tensor& top_k_indices,  // dst_indices
    const paddle::optional<paddle::Tensor>& down_proj_bias,
    const bool norm_topk_prob,
    const float routed_scaling_factor) {
  const auto input_type = ffn_out.dtype();
  auto place = ffn_out.place();

  const int num_rows = top_k_indices.dims()[0];
  const int hidden_size = ffn_out.dims()[1];

  auto output = GetEmptyTensor({num_rows, hidden_size}, input_type, place);

  switch (input_type) {
    case paddle::DataType::BFLOAT16:
      MoeCombineKernel<paddle::DataType::BFLOAT16>(ffn_out,
                                                   expert_scales_float,
                                                   permute_indices_per_token,
                                                   top_k_indices,
                                                   down_proj_bias,
                                                   norm_topk_prob,
                                                   routed_scaling_factor,
                                                   num_rows,
                                                   hidden_size,
                                                   &output);
      break;
    case paddle::DataType::FLOAT16:
      MoeCombineKernel<paddle::DataType::BFLOAT16>(ffn_out,
                                                   expert_scales_float,
                                                   permute_indices_per_token,
                                                   top_k_indices,
                                                   down_proj_bias,
                                                   norm_topk_prob,
                                                   routed_scaling_factor,
                                                   num_rows,
                                                   hidden_size,
                                                   &output);
      break;
    default:
      PD_THROW("Unsupported data type for MoeDispatchKernel");
  }
  return {output};
}

template <typename T,
          typename OutT,
          int NUM_EXPERTS_PER_RANK = 8,
          int Kthread = 512,
          int RoundType = 1>
__global__ void permute_x_kernel(
    const T* src_x,
    const int64_t* topk_idx,
    const float* topk_weights,
    const int* token_nums_per_expert,
    const float* up_gate_proj_in_scale,
    const int moe_topk,
    const int num_rows,
    const int token_nums_this_rank,
    const int hidden_size,
    OutT* permute_x,                 // [token_nums_this_rank, hidden_size]
    int* permute_indices_per_token,  // [moe_topk, num_rows]
    float* dst_weights,              // [token_nums_this_rank]
    int* dst_indices,
    int* cumsum_idx_gpu,
    int64_t* token_nums_per_expert_cumsum,
    int64_t* expert_idx_per_token,  // [num_rows, moe_topk]
    float* dequant_scale,
    float max_bound = 127.0,
    float min_bound = -127.0) {
  const int src_token_idx = blockIdx.x;
  const int tid = threadIdx.x;
  constexpr int vec_size = sizeof(int4) / sizeof(T);
  __shared__ int write_idx;  // cumsum start idx
  __shared__ int token_nums_per_expert_cum[NUM_EXPERTS_PER_RANK];
  extern __shared__ char smem_[];
  T* data_smem = reinterpret_cast<T*>(smem_);
  AlignedVector<T, vec_size> src_vec;
  AlignedVector<OutT, vec_size> res_vec;
  if (tid == 0) {
    int sum_now = 0;
    for (int i = 0; i < NUM_EXPERTS_PER_RANK; i++) {
      sum_now += token_nums_per_expert[i];
      token_nums_per_expert_cum[i] = sum_now;
      if (blockIdx.x == 0) {
        token_nums_per_expert_cumsum[i] = sum_now;
      }
    }
  }
  __syncthreads();
  const int hidden_size_int4 = hidden_size / vec_size;
  for (int s_token_idx = src_token_idx; s_token_idx < num_rows;
       s_token_idx += gridDim.x) {
    const int64_t* topk_idx_now = topk_idx + s_token_idx * moe_topk;
#pragma unroll
    for (int expert_idx = 0; expert_idx < moe_topk; expert_idx++) {
      int expert_now = static_cast<int>(topk_idx_now[expert_idx]);
      if (expert_now == -1) continue;
      const int dst_chunk_start_idx =
          expert_now == 0 ? 0 : token_nums_per_expert_cum[expert_now - 1];
      if (tid == 0) {
        const int offset_now = atomicAdd(cumsum_idx_gpu + expert_now, 1);
        write_idx = offset_now;
      }
      __syncthreads();
      const int token_offset_now = write_idx;
      const int dst_token_idx = dst_chunk_start_idx + token_offset_now;
      permute_indices_per_token[expert_now * num_rows + s_token_idx] =
          dst_token_idx;
      dst_weights[dst_token_idx] =
          topk_weights[s_token_idx * moe_topk + expert_idx];
      dst_indices[s_token_idx * NUM_EXPERTS_PER_RANK + expert_now] = expert_now;
      // cp x
      if (dequant_scale) {  // dynamic quant
        float abs_max = 0.0f;
        for (int v_id = tid; v_id < hidden_size_int4; v_id += blockDim.x) {
          Load<T, vec_size>(&src_x[s_token_idx * hidden_size + v_id * vec_size],
                            &src_vec);
          Store<T, vec_size>(src_vec, &data_smem[v_id * vec_size]);
#pragma unroll
          for (int i = 0; i < vec_size; i++) {
            abs_max = fmaxf(abs_max, fabsf(static_cast<float>(src_vec[i])));
          }
        }
        abs_max = phi::BlockAllReduce<MaxOp, float, Kthread>(abs_max);
        float scale = 440.f / abs_max;  // use 440 so we do not have to clip
        dequant_scale[dst_token_idx] = abs_max;
        for (int v_id = tid; v_id < hidden_size_int4; v_id += blockDim.x) {
          Load<T, vec_size>(&data_smem[v_id * vec_size], &src_vec);
#pragma unroll
          for (int i = 0; i < vec_size; i++) {
            float quant_value = scale * static_cast<float>(src_vec[i]);
            // dynamic quant only supporet for w4afp8
            res_vec[i] = static_cast<OutT>(quant_value);
          }
          Store<OutT, vec_size>(
              res_vec,
              &permute_x[dst_token_idx * hidden_size + v_id * vec_size]);
        }
      } else {  // static quant or not quant
        for (int v_id = tid; v_id < hidden_size_int4; v_id += blockDim.x) {
          Load<T, vec_size>(&src_x[s_token_idx * hidden_size + v_id * vec_size],
                            &src_vec);
          if (up_gate_proj_in_scale) {
#pragma unroll
            for (int i = 0; i < vec_size; i++) {
              float quant_value = max_bound *
                                  up_gate_proj_in_scale[expert_now] *
                                  static_cast<float>(src_vec[i]);
              if constexpr (std::is_same<OutT, int8_t>::value) {
                // w4aint8
                if (RoundType == 0) {
                  res_vec[i] = static_cast<OutT>(
                      ClipFunc<float>(rint(quant_value), min_bound, max_bound));
                } else {
                  res_vec[i] = static_cast<OutT>(ClipFunc<float>(
                      round(quant_value), min_bound, max_bound));
                }
              } else {
                // w4afp8
                float value =
                    ClipFunc<float>(quant_value, min_bound, max_bound);
                res_vec[i] = static_cast<OutT>(value);
              }
            }
          } else {
#pragma unroll
            for (int i = 0; i < vec_size; i++) {
              res_vec[i] = static_cast<OutT>(src_vec[i]);
            }
          }
          Store<OutT, vec_size>(
              res_vec,
              &permute_x[dst_token_idx * hidden_size + v_id * vec_size]);
        }
      }
      expert_idx_per_token[dst_token_idx] = expert_now;
    }
  }
}

template <paddle::DataType T>
void EPMoeDispatchKernel(
    const paddle::Tensor& input,
    const paddle::Tensor& topk_ids,
    const paddle::Tensor& topk_weights,
    const paddle::Tensor& token_nums_per_expert,
    const paddle::optional<paddle::Tensor>& up_gate_proj_in_scale,
    const std::string& moe_quant_type,
    const int moe_topk,
    const int num_rows,
    const int token_nums_this_rank,
    const int hidden_size,
    const int num_experts_per_rank,
    paddle::Tensor* permute_input,
    paddle::Tensor* permute_indices_per_token,
    paddle::Tensor* dst_weights,
    paddle::Tensor* dst_indices,
    paddle::Tensor* cumsum_idx_gpu,
    paddle::Tensor* token_nums_per_expert_cumsum,
    paddle::Tensor* expert_idx_per_token,
    paddle::Tensor* dequant_scale) {
  using namespace phi;

  typedef PDTraits<T> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t;

  typedef PDTraits<paddle::DataType::FLOAT8_E4M3FN> traits_fp8;
  typedef typename traits_fp8::DataType DataType_fp8;
  typedef typename traits_fp8::data_t data_t_fp8;

  auto stream = input.stream();
  auto place = input.place();
  const int gridx = min(132 * 8, num_rows);
  if (moe_quant_type == "w4a8") {
    DISPATCH_NUM_EXPERTS_PER_RANK(
        num_experts_per_rank,
        NUM_EXPERTS_PER_RANK,
        permute_x_kernel<data_t, int8_t, NUM_EXPERTS_PER_RANK>
        <<<gridx, 512, 0, stream>>>(
            input.data<data_t>(),
            topk_ids.data<int64_t>(),
            topk_weights.data<float>(),
            token_nums_per_expert.data<int>(),
            up_gate_proj_in_scale ? up_gate_proj_in_scale.get().data<float>()
                                  : nullptr,
            moe_topk,
            num_rows,
            token_nums_this_rank,
            hidden_size,
            permute_input->data<int8_t>(),
            permute_indices_per_token->data<int>(),
            dst_weights->data<float>(),
            dst_indices->data<int>(),
            cumsum_idx_gpu->data<int>(),
            token_nums_per_expert_cumsum->data<int64_t>(),
            expert_idx_per_token->data<int64_t>(),
            nullptr,  // dequant_scale
            127.0,
            -127.0);)
  } else if (moe_quant_type == "w4afp8") {
    const int smem_size =
        up_gate_proj_in_scale ? 0 : hidden_size * sizeof(data_t);
    DISPATCH_NUM_EXPERTS_PER_RANK(
        num_experts_per_rank,
        NUM_EXPERTS_PER_RANK,
        permute_x_kernel<data_t, data_t_fp8, NUM_EXPERTS_PER_RANK, 512>
        <<<gridx, 512, smem_size, stream>>>(
            input.data<data_t>(),
            topk_ids.data<int64_t>(),
            topk_weights.data<float>(),
            token_nums_per_expert.data<int>(),
            up_gate_proj_in_scale ? up_gate_proj_in_scale.get().data<float>()
                                  : nullptr,
            moe_topk,
            num_rows,
            token_nums_this_rank,
            hidden_size,
            permute_input->data<data_t_fp8>(),
            permute_indices_per_token->data<int>(),
            dst_weights->data<float>(),
            dst_indices->data<int>(),
            cumsum_idx_gpu->data<int>(),
            token_nums_per_expert_cumsum->data<int64_t>(),
            expert_idx_per_token->data<int64_t>(),
            up_gate_proj_in_scale
                ? nullptr
                : dequant_scale
                      ->data<float>(),  // up_gate_proj_in_scale is used for
                                        // static quant, while dequant_scale is
                                        // used for dynamic quant
            448.0f,
            -448.0f);)
  } else {
    DISPATCH_NUM_EXPERTS_PER_RANK(
        num_experts_per_rank,
        NUM_EXPERTS_PER_RANK,
        permute_x_kernel<data_t, data_t, NUM_EXPERTS_PER_RANK>
        <<<gridx, 512, 0, stream>>>(
            input.data<data_t>(),
            topk_ids.data<int64_t>(),
            topk_weights.data<float>(),
            token_nums_per_expert.data<int>(),
            up_gate_proj_in_scale ? up_gate_proj_in_scale.get().data<float>()
                                  : nullptr,
            moe_topk,
            num_rows,
            token_nums_this_rank,
            hidden_size,
            permute_input->data<data_t>(),
            permute_indices_per_token->data<int>(),
            dst_weights->data<float>(),
            dst_indices->data<int>(),
            cumsum_idx_gpu->data<int>(),
            token_nums_per_expert_cumsum->data<int64_t>(),
            expert_idx_per_token->data<int64_t>(),
            nullptr,  // dequant scale
            127.0,
            -127.0);)
  }
}

std::vector<paddle::Tensor> EPMoeExpertDispatch(
    const paddle::Tensor& input,
    const paddle::Tensor& topk_ids,
    const paddle::Tensor& topk_weights,
    const paddle::optional<paddle::Tensor>& up_gate_proj_in_scale,
    const std::vector<int>& token_nums_per_expert,
    const int token_nums_this_rank,
    const std::string& moe_quant_type) {
  const auto input_type = input.dtype();
  const int moe_topk = topk_ids.dims()[1];
  auto place = input.place();
  int token_rows = 0;
  auto input_dims = input.dims();

  if (input_dims.size() == 3) {
    token_rows = input_dims[0] * input_dims[1];
  } else {
    token_rows = input_dims[0];
  }
  const int num_rows = token_rows;
  const int hidden_size = input.dims()[input_dims.size() - 1];
  const int num_experts_per_rank = token_nums_per_expert.size();

  auto permute_input = GetEmptyTensor(
      {token_nums_this_rank, hidden_size},
      moe_quant_type == "w4a8"     ? paddle::DataType::INT8
      : moe_quant_type == "w4afp8" ? paddle::DataType::FLOAT8_E4M3FN
                                   : input_type,
      place);
  auto num_experts_per_rank_tensor =
      GetEmptyTensor({num_experts_per_rank}, paddle::DataType::INT32, place);
  auto expert_idx_per_token =
      GetEmptyTensor({token_nums_this_rank}, paddle::DataType::INT64, place);
  cudaMemcpyAsync(num_experts_per_rank_tensor.data<int>(),
                  token_nums_per_expert.data(),
                  num_experts_per_rank * sizeof(int),
                  cudaMemcpyHostToDevice,
                  input.stream());
  // cudaMemcpy(num_experts_per_rank_tensor.data<int>(),
  // token_nums_per_expert.data(), num_experts_per_rank * sizeof(int),
  // cudaMemcpyHostToDevice);
  auto token_nums_per_expert_cumsum =
      GetEmptyTensor({num_experts_per_rank}, paddle::DataType::INT64, place);
  auto dst_weights =
      GetEmptyTensor({token_nums_this_rank}, paddle::DataType::FLOAT32, place);
  auto dst_indices = GetEmptyTensor(
      {num_rows, num_experts_per_rank}, paddle::DataType::INT32, place);
  auto permute_indices_per_token = paddle::full(
      {num_experts_per_rank, num_rows}, -1, paddle::DataType::INT32, place);
  auto cumsum_idx_gpu =
      paddle::full({num_experts_per_rank}, 0, paddle::DataType::INT32, place);

  int dequant_scale_size = 1;
  if (moe_quant_type == "w4afp8" && !up_gate_proj_in_scale) {
    dequant_scale_size = moe_topk * num_rows;
  }

  auto dequant_scale =
      GetEmptyTensor({dequant_scale_size}, paddle::DataType::FLOAT32, place);

  switch (input_type) {
    case paddle::DataType::BFLOAT16:
      EPMoeDispatchKernel<paddle::DataType::BFLOAT16>(
          input,
          topk_ids,
          topk_weights,
          num_experts_per_rank_tensor,
          up_gate_proj_in_scale,
          moe_quant_type,
          moe_topk,
          num_rows,
          token_nums_this_rank,
          hidden_size,
          num_experts_per_rank,
          &permute_input,
          &permute_indices_per_token,
          &dst_weights,
          &dst_indices,
          &cumsum_idx_gpu,
          &token_nums_per_expert_cumsum,
          &expert_idx_per_token,
          &dequant_scale);
      break;
    case paddle::DataType::FLOAT16:
      EPMoeDispatchKernel<paddle::DataType::FLOAT16>(
          input,
          topk_ids,
          topk_weights,
          num_experts_per_rank_tensor,
          up_gate_proj_in_scale,
          moe_quant_type,
          moe_topk,
          num_rows,
          token_nums_this_rank,
          hidden_size,
          num_experts_per_rank,
          &permute_input,
          &permute_indices_per_token,
          &dst_weights,
          &dst_indices,
          &cumsum_idx_gpu,
          &token_nums_per_expert_cumsum,
          &expert_idx_per_token,
          &dequant_scale);
      break;
    default:
      PD_THROW("Unsupported data type for EPMoeDispatchKernel");
  }
  return {permute_input,
          permute_indices_per_token,
          token_nums_per_expert_cumsum,
          dst_weights,
          dst_indices,
          cumsum_idx_gpu,
          expert_idx_per_token,
          dequant_scale};
}

std::vector<std::vector<int64_t>> EPMoeExpertDispatchInferShape(
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& topk_ids_shape,
    const std::vector<int64_t>& topk_weights_shape,
    const paddle::optional<std::vector<int64_t>>& up_gate_proj_in_scale_dtype,
    const std::vector<int>& token_nums_per_expert,
    const int token_nums_this_rank) {
  int token_rows = -1;
  int moe_topk = topk_ids_shape[1];
  if (input_shape.size() == 3) {
    token_rows = input_shape[0] * input_shape[1];
  } else {
    token_rows = input_shape[0];
  }
  const int expert_num = token_nums_per_expert.size();  // 本地专家个数
  const int num_rows = token_rows;
  const int hidden_size = input_shape[input_shape.size() - 1];

  return {{token_nums_this_rank, hidden_size},
          {expert_num, num_rows},
          {expert_num},
          {token_nums_this_rank},
          {num_rows, expert_num},
          {expert_num},
          {token_nums_this_rank},  // dst_idx per expert
          {token_nums_this_rank}};
}

std::vector<paddle::DataType> EPMoeExpertDispatchInferDtype(
    const paddle::DataType& input_dtype,
    const paddle::DataType& topk_ids_dtype,
    const paddle::DataType& topk_weights_dtype,
    const std::vector<int>& token_nums_per_expert,
    const int token_nums_this_rank,
    const std::string& moe_quant_type) {
  return {moe_quant_type == "w4a8" ? paddle::DataType::INT8 : input_dtype,
          paddle::DataType::INT32,
          paddle::DataType::INT64,
          paddle::DataType::FLOAT32,
          paddle::DataType::INT32,
          paddle::DataType::INT32,
          paddle::DataType::INT64,
          paddle::DataType::FLOAT32};
}

PD_BUILD_STATIC_OP(ep_moe_expert_dispatch)
    .Inputs({"input",
             "topk_ids",
             "topk_weights",
             paddle::Optional("up_gate_proj_in_scale")})
    .Outputs({"permute_input",
              "permute_indices_per_token",
              "token_nums_per_expert_cumsum",
              "dst_weights",
              "dst_indices",
              "cumsum_idx_gpu",
              "expert_idx_per_token",
              "dequant_scale"})
    .Attrs({"token_nums_per_expert: std::vector<int>",
            "token_nums_this_rank: int",
            "moe_quant_type: std::string"})
    .SetKernelFn(PD_KERNEL(EPMoeExpertDispatch))
    .SetInferShapeFn(PD_INFER_SHAPE(EPMoeExpertDispatchInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(EPMoeExpertDispatchInferDtype));

template <typename T, int NUM_EXPERTS_PER_RANK = 8>
__global__ void permute_x_fp8_kernel(
    const T* src_x,
    const float* scale,
    const int64_t* topk_idx,
    const float* topk_weights,
    const int* token_nums_per_expert,
    const int* token_nums_per_expert_padded,
    const int moe_topk,
    const int num_rows,
    const int token_nums_this_rank,
    const int token_nums_this_rank_padded,
    const int64_t hidden_size,
    T* permute_x,  // [token_nums_this_rank, hidden_size]
    float* permute_scale,
    int* permute_indices_per_token,  // [moe_topk, num_rows]
    float* dst_weights,              // [token_nums_this_rank]
    int* dst_indices,
    int* cumsum_idx_gpu,
    int64_t* token_nums_per_expert_cumsum,
    int64_t* token_nums_per_expert_padded_cumsum,
    int* m_indices) {  // [num_rows, moe_topk]
  const int64_t src_token_idx = blockIdx.x;
  const int tid = threadIdx.x;
  constexpr int vec_size = sizeof(int4) / sizeof(T);
  constexpr int scale_vec_size = sizeof(int4) / sizeof(float);
  __shared__ int write_idx;  // cumsum start idx
  __shared__ int token_nums_per_expert_cum[NUM_EXPERTS_PER_RANK];
  if (tid == 0) {
    int sum_now = 0;
    int sum_now_padded = 0;
    for (int i = 0; i < NUM_EXPERTS_PER_RANK; i++) {
      sum_now += token_nums_per_expert[i];
      sum_now_padded += token_nums_per_expert_padded[i];
      token_nums_per_expert_cum[i] = sum_now_padded;
      if (blockIdx.x == 0) {
        token_nums_per_expert_cumsum[i] = sum_now;
        token_nums_per_expert_padded_cumsum[i] = sum_now_padded;
      }
    }
  }
  __syncthreads();
  const int hidden_size_int4 = hidden_size / vec_size;
  const int hidden_size_scale = hidden_size / 128;
  const int hidden_size_scale_int4 = hidden_size_scale / scale_vec_size;
  const int token_nums_feed_to_ffn =
      token_nums_per_expert_cum[NUM_EXPERTS_PER_RANK - 1];
  // prmt
  for (int64_t s_token_idx = src_token_idx;
       s_token_idx < token_nums_feed_to_ffn;
       s_token_idx += gridDim.x) {
    // the m_indices[s_token_idx] must be a value `i` in [0,
    // NUM_EXPERTS_PER_RANK) here we parallel wo find the `i` we want.
    for (int i = threadIdx.x; i < NUM_EXPERTS_PER_RANK; i += blockDim.x) {
      const int start_idx = i == 0 ? 0 : token_nums_per_expert_cum[i - 1];
      const int end_idx = token_nums_per_expert_cum[i];
      if (s_token_idx >= start_idx && s_token_idx < end_idx) {
        if ((s_token_idx - start_idx) < token_nums_per_expert[i])
          m_indices[s_token_idx] = i;
        break;
      }
    }

    if (s_token_idx < num_rows) {
      const int64_t* topk_idx_now = topk_idx + s_token_idx * moe_topk;
#pragma unroll
      for (int expert_idx = 0; expert_idx < moe_topk; expert_idx++) {
        int expert_now = static_cast<int>(topk_idx_now[expert_idx]);
        if (expert_now == -1) continue;
        const int dst_chunk_start_idx =
            expert_now == 0 ? 0 : token_nums_per_expert_cum[expert_now - 1];
        if (tid == 0) {
          const int offset_now = atomicAdd(cumsum_idx_gpu + expert_now, 1);
          write_idx = offset_now;
        }
        __syncthreads();
        const int token_offset_now = write_idx;
        const int64_t dst_token_idx = dst_chunk_start_idx + token_offset_now;
        permute_indices_per_token[expert_now * num_rows + s_token_idx] =
            dst_token_idx;
        dst_weights[dst_token_idx] =
            topk_weights[s_token_idx * moe_topk + expert_idx];
        // m_indices[dst_token_idx] = expert_now; // not need?
        dst_indices[s_token_idx * NUM_EXPERTS_PER_RANK + expert_now] =
            expert_now;
        // cp x
        for (int64_t v_id = tid; v_id < hidden_size_int4; v_id += blockDim.x) {
          *(reinterpret_cast<int4*>(permute_x + dst_token_idx * hidden_size) +
            v_id) = *(reinterpret_cast<const int4*>(src_x +
                                                    s_token_idx * hidden_size) +
                      v_id);
        }
        // cp scale
        for (int v_id = tid; v_id < hidden_size_scale_int4;
             v_id += blockDim.x) {
          *(reinterpret_cast<int4*>(permute_scale +
                                    dst_token_idx * hidden_size_scale) +
            v_id) =
              *(reinterpret_cast<const int4*>(scale +
                                              s_token_idx * hidden_size_scale) +
                v_id);
        }
      }
    }
  }
}

void EPMoeDispatchFP8Kernel(const paddle::Tensor& input,
                            const paddle::Tensor& scale,
                            const paddle::Tensor& topk_ids,
                            const paddle::Tensor& topk_weights,
                            const paddle::Tensor& token_nums_per_expert,
                            const paddle::Tensor& token_nums_per_expert_padded,
                            const int moe_topk,
                            const int num_rows,
                            const int token_nums_this_rank,
                            const int token_nums_this_rank_padded,
                            const int hidden_size,
                            const int num_experts_per_rank,
                            paddle::Tensor* permute_input,
                            paddle::Tensor* permute_scale,
                            paddle::Tensor* permute_indices_per_token,
                            paddle::Tensor* dst_weights,
                            paddle::Tensor* dst_indices,
                            paddle::Tensor* cumsum_idx_gpu,
                            paddle::Tensor* token_nums_per_expert_cumsum,
                            paddle::Tensor* token_nums_per_expert_padded_cumsum,
                            paddle::Tensor* m_indices) {
  auto stream = input.stream();
  auto place = input.place();
  // const int gridx = min(132 * 8, num_rows);
  const int gridx = 132 * 8;
  DISPATCH_NUM_EXPERTS_PER_RANK(
      num_experts_per_rank,
      NUM_EXPERTS_PER_RANK,
      permute_x_fp8_kernel<phi::dtype::float8_e4m3fn, NUM_EXPERTS_PER_RANK>
      <<<gridx, 512, 0, stream>>>(
          input.data<phi::dtype::float8_e4m3fn>(),
          scale.data<float>(),
          topk_ids.data<int64_t>(),
          topk_weights.data<float>(),
          token_nums_per_expert.data<int>(),
          token_nums_per_expert_padded.data<int>(),
          moe_topk,
          num_rows,
          token_nums_this_rank,
          token_nums_this_rank_padded,
          hidden_size,
          permute_input->data<phi::dtype::float8_e4m3fn>(),
          permute_scale->data<float>(),
          permute_indices_per_token->data<int>(),
          dst_weights->data<float>(),
          dst_indices->data<int>(),
          cumsum_idx_gpu->data<int>(),
          token_nums_per_expert_cumsum->data<int64_t>(),
          token_nums_per_expert_padded_cumsum->data<int64_t>(),
          m_indices->data<int>());)
}

std::vector<paddle::Tensor> EPMoeExpertDispatchFP8(
    const paddle::Tensor& input,
    const paddle::Tensor& scale,
    const paddle::Tensor& topk_ids,
    const paddle::Tensor& topk_weights,
    const paddle::Tensor& num_experts_per_rank_tensor,
    const paddle::Tensor& num_experts_per_rank_padded_tensor,
    const bool use_in_ep,
    const int token_nums_this_rank_padded) {
  const auto input_type = input.dtype();
  const int moe_topk = topk_ids.dims()[1];
  auto place = input.place();
  int token_rows = 0;
  auto input_dims = input.dims();

  if (input_dims.size() == 3) {
    token_rows = input_dims[0] * input_dims[1];
  } else {
    token_rows = input_dims[0];
  }
  const int num_rows = token_rows;
  const int hidden_size = input.dims()[input_dims.size() - 1];
  const int num_experts_per_rank = num_experts_per_rank_tensor.dims()[0];

  int32_t token_nums_feed_to_ffn =
      use_in_ep ? token_nums_this_rank_padded
                : token_rows * moe_topk + num_experts_per_rank * (128 - 1);

  auto permute_input =
      GetEmptyTensor({token_nums_feed_to_ffn, hidden_size}, input_type, place);
  auto permute_scale =
      GetEmptyTensor({token_nums_feed_to_ffn, hidden_size / 128},
                     paddle::DataType::FLOAT32,
                     place);

  auto m_indices = paddle::full(
      {token_nums_feed_to_ffn}, -1, paddle::DataType::INT32, place);
  auto token_nums_per_expert_cumsum =
      GetEmptyTensor({num_experts_per_rank}, paddle::DataType::INT64, place);
  auto token_nums_per_expert_padded_cumsum =
      GetEmptyTensor({num_experts_per_rank}, paddle::DataType::INT64, place);
  auto dst_weights = GetEmptyTensor(
      {token_nums_feed_to_ffn}, paddle::DataType::FLOAT32, place);
  auto dst_indices = GetEmptyTensor(
      {num_rows, num_experts_per_rank}, paddle::DataType::INT32, place);
  auto permute_indices_per_token = paddle::full(
      {num_experts_per_rank, num_rows}, -1, paddle::DataType::INT32, place);
  auto cumsum_idx_gpu =
      paddle::full({num_experts_per_rank}, 0, paddle::DataType::INT32, place);

  EPMoeDispatchFP8Kernel(input,
                         scale,
                         topk_ids,
                         topk_weights,
                         num_experts_per_rank_tensor,
                         num_experts_per_rank_padded_tensor,
                         moe_topk,
                         num_rows,
                         -1,
                         -1,
                         hidden_size,
                         num_experts_per_rank,
                         &permute_input,
                         &permute_scale,
                         &permute_indices_per_token,
                         &dst_weights,
                         &dst_indices,
                         &cumsum_idx_gpu,
                         &token_nums_per_expert_cumsum,
                         &token_nums_per_expert_padded_cumsum,
                         &m_indices);
  return {permute_input,
          permute_scale,
          permute_indices_per_token,
          token_nums_per_expert_cumsum,
          token_nums_per_expert_padded_cumsum,
          dst_weights,
          dst_indices,
          cumsum_idx_gpu,
          m_indices};
}

PD_BUILD_STATIC_OP(ep_moe_expert_dispatch_fp8)
    .Inputs({"input",
             "scale",
             "topk_ids",
             "topk_weights",
             "num_experts_per_rank_tensor",
             "num_experts_per_rank_padded_tensor"})
    .Outputs({"permute_input",
              "permute_scale",
              "permute_indices_per_token",
              "token_nums_per_expert_cumsum",
              "token_nums_per_expert_padded_cumsum",
              "dst_weights",
              "dst_indices",
              "cumsum_idx_gpu",
              "m_indices"})
    .Attrs({"use_in_ep:bool", "token_nums_this_rank_padded:int"})
    .SetKernelFn(PD_KERNEL(EPMoeExpertDispatchFP8));
