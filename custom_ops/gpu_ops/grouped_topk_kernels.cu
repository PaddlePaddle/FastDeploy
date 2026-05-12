
// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include <algorithm>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda/std/limits>

#include "helper.h"

namespace cg = cooperative_groups;

constexpr unsigned FUSED_FULL_WARP_MASK = 0xffffffff;

template <typename T_OUT, typename T_IN>
__device__ inline T_OUT cuda_cast(T_IN val) {
  return val;
}

template <>
__device__ inline float cuda_cast<float, __nv_bfloat16>(__nv_bfloat16 val) {
  return __bfloat162float(val);
}

template <>
__device__ inline __nv_bfloat16 cuda_cast<__nv_bfloat16, float>(float val) {
  return __float2bfloat16(val);
}

template <>
__device__ inline float cuda_cast<float, __half>(__half val) {
  return __half2float(val);
}

template <>
__device__ inline __half cuda_cast<__half, float>(float val) {
  return __float2half(val);
}

// Numerically stable sigmoid via tanh: σ(x) = 0.5 * tanh(0.5*x) + 0.5
template <typename T>
__device__ __forceinline__ T sigmoid_device(T x) {
  float xf = cuda_cast<float, T>(x);
  return cuda_cast<T, float>(0.5f * tanhf(0.5f * xf) + 0.5f);
}

// Sigmoid matching fused_cast_sigmoid_bias: 1 / (1 + exp(-x)).
// Must use the same formula to get bit-identical results when comparing
// against the fused_cast_sigmoid_bias + noaux_tc path.
template <typename InT>
__device__ __forceinline__ float sigmoid_to_float(InT x) {
  float xf = cuda_cast<float, InT>(x);
  return 1.0f / (1.0f + expf(-xf));
}

template <typename T>
__device__ inline T neg_inf() {
  return cuda_cast<T, float>(-cuda::std::numeric_limits<float>::infinity());
}

template <typename T>
__device__ inline bool is_finite_val(T val) {
#if (__CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100 >= 120800)
  return cuda::std::isfinite(val);
#else
  return isfinite(cuda_cast<float, T>(val));
#endif
}

namespace warp_topk_fused {

template <int size, typename T>
__host__ __device__ constexpr T round_up_to_multiple_of(T len) {
  if (len == 0) return 0;
  return ((len - 1) / size + 1) * size;
}

template <typename T>
constexpr __host__ __device__ bool isPowerOf2(T v) {
  return (v && !(v & (v - 1)));
}

template <bool greater, typename T>
__forceinline__ __device__ bool is_better_than(T val, T baseline) {
  return (val > baseline && greater) || (val < baseline && !greater);
}

template <bool greater, typename T, typename idxT>
__forceinline__ __device__ bool is_better_than(T val,
                                               T baseline,
                                               idxT index,
                                               idxT baseline_index) {
  bool res = (val > baseline && greater) || (val < baseline && !greater);
  if (val == baseline)
    res = (index < baseline_index && greater) ||
          (index < baseline_index && !greater);
  return res;
}

template <int size,
          bool ascending,
          bool reverse,
          typename T,
          typename idxT,
          bool is_stable>
struct BitonicMerge {
  __device__ static void merge(T* __restrict__ val_arr,
                               idxT* __restrict__ idx_arr) {
    static_assert(isPowerOf2(size));
    static_assert(size >= 2 * WARP_SIZE);
    constexpr int arr_len = size / WARP_SIZE;
    constexpr int stride = arr_len / 2;
    for (int i = 0; i < stride; ++i) {
      int const other_i = i + stride;
      T& val = val_arr[i];
      T& other_val = val_arr[other_i];
      bool is_better;
      if constexpr (is_stable)
        is_better = is_better_than<ascending>(
            val, other_val, idx_arr[i], idx_arr[other_i]);
      else
        is_better = is_better_than<ascending>(val, other_val);
      if (is_better) {
        T tmp = val;
        val = other_val;
        other_val = tmp;
        idxT tmp2 = idx_arr[i];
        idx_arr[i] = idx_arr[other_i];
        idx_arr[other_i] = tmp2;
      }
    }
    BitonicMerge<size / 2, ascending, reverse, T, idxT, is_stable>::merge(
        val_arr, idx_arr);
    BitonicMerge<size / 2, ascending, reverse, T, idxT, is_stable>::merge(
        val_arr + arr_len / 2, idx_arr + arr_len / 2);
  }
};

template <int size, bool ascending, typename T, typename idxT, bool is_stable>
struct BitonicSort {
  __device__ static void sort(T* __restrict__ val_arr,
                              idxT* __restrict__ idx_arr) {
    static_assert(isPowerOf2(size));
    static_assert(size >= 2 * WARP_SIZE);
    constexpr int arr_len = size / WARP_SIZE;
    BitonicSort<size / 2, true, T, idxT, is_stable>::sort(val_arr, idx_arr);
    BitonicSort<size / 2, false, T, idxT, is_stable>::sort(
        val_arr + arr_len / 2, idx_arr + arr_len / 2);
    BitonicMerge<size, ascending, ascending, T, idxT, is_stable>::merge(
        val_arr, idx_arr);
  }
};

template <bool ascending, typename T, typename idxT, bool is_stable>
struct BitonicSort<32, ascending, T, idxT, is_stable> {
  __device__ static void sort(T* __restrict__ val_arr,
                              idxT* __restrict__ idx_arr) {
    int const lane = threadIdx.x % WARP_SIZE;
    for (int stage = 0; stage < 4; ++stage) {
      for (int stride = (1 << stage); stride > 0; stride /= 2) {
        bool reverse = (lane >> stage) & 2;
        bool is_second = lane & stride;
        T other = __shfl_xor_sync(FUSED_FULL_WARP_MASK, *val_arr, stride);
        idxT other_idx =
            __shfl_xor_sync(FUSED_FULL_WARP_MASK, *idx_arr, stride);
        bool is_better;
        if constexpr (is_stable) {
          if constexpr (ascending)
            is_better = ((*val_arr > other) ||
                         ((*val_arr == other) && (*idx_arr < other_idx))) !=
                        (reverse != is_second);
          else
            is_better = ((*val_arr > other) ||
                         ((*val_arr == other) && (*idx_arr > other_idx))) !=
                        (reverse != is_second);
        } else {
          is_better = (*val_arr != other &&
                       (*val_arr > other) != (reverse != is_second));
        }
        if (is_better) {
          *val_arr = other;
          *idx_arr = other_idx;
        }
      }
    }
    BitonicMerge<32, ascending, ascending, T, idxT, is_stable>::merge(val_arr,
                                                                      idx_arr);
  }
};

template <bool ascending,
          bool reverse,
          typename T,
          typename idxT,
          bool is_stable>
struct BitonicMerge<32, ascending, reverse, T, idxT, is_stable> {
  __device__ static void merge(T* __restrict__ val_arr,
                               idxT* __restrict__ idx_arr) {
    int const lane = threadIdx.x % WARP_SIZE;
    for (int stride = WARP_SIZE / 2; stride > 0; stride /= 2) {
      bool is_second = lane & stride;
      T& val = *val_arr;
      T other = __shfl_xor_sync(FUSED_FULL_WARP_MASK, val, stride);
      idxT& idx = *idx_arr;
      idxT other_idx = __shfl_xor_sync(FUSED_FULL_WARP_MASK, idx, stride);
      bool is_better;
      if constexpr (is_stable) {
        if constexpr (ascending)
          is_better = ((*val_arr > other) ||
                       ((*val_arr == other) && (*idx_arr < other_idx))) ==
                      (reverse != is_second);
        else
          is_better = ((*val_arr > other) ||
                       ((*val_arr == other) && (*idx_arr > other_idx))) ==
                      (reverse != is_second);
      } else {
        is_better =
            (val != other && ((val > other) == (ascending != is_second)));
      }
      if (is_better) {
        val = other;
        idx = other_idx;
      }
    }
  }
};

template <int capacity, bool greater, typename T, typename idxT, bool is_stable>
class WarpSort {
 public:
  __device__ WarpSort(idxT k, T dummy)
      : lane_(threadIdx.x % WARP_SIZE), k_(k), dummy_(dummy) {
    static_assert(capacity >= WARP_SIZE && isPowerOf2(capacity));
    for (int i = 0; i < max_arr_len_; ++i) {
      val_arr_[i] = dummy_;
      idx_arr_[i] = 0;
    }
  }

  __device__ __forceinline__ idxT get_idx(int i = 0) const {
    return idx_arr_[i];
  }
  __device__ __forceinline__ T get_val(int i = 0) const { return val_arr_[i]; }

 protected:
  static constexpr int max_arr_len_ = capacity / WARP_SIZE;
  T val_arr_[max_arr_len_];
  idxT idx_arr_[max_arr_len_];
  int const lane_;
  idxT const k_;
  T const dummy_;
};

// WarpSelect WITHOUT __syncthreads() in done() — safe when only one warp is
// active.
template <int capacity, bool greater, typename T, typename idxT, bool is_stable>
class WarpSelect : public WarpSort<capacity, greater, T, idxT, is_stable> {
 public:
  __device__ WarpSelect(idxT k, T dummy)
      : WarpSort<capacity, greater, T, idxT, is_stable>(k, dummy),
        k_th_(dummy),
        k_th_idx_(0),
        k_th_lane_((k - 1) % WARP_SIZE) {
    extern __shared__ char smem_buf[];
    int const num_of_warp = blockDim.x / WARP_SIZE;
    int const warp_id = threadIdx.x / WARP_SIZE;
    val_smem_ = reinterpret_cast<T*>(smem_buf);
    val_smem_ += warp_id * WARP_SIZE;
    idx_smem_ = reinterpret_cast<idxT*>(
        smem_buf +
        round_up_to_multiple_of<256>(num_of_warp * sizeof(T) * WARP_SIZE));
    idx_smem_ += warp_id * WARP_SIZE;
  }

  __device__ void add(T val, idxT idx) {
    bool do_add;
    if constexpr (is_stable)
      do_add = is_better_than<greater>(val, k_th_, idx, k_th_idx_);
    else
      do_add = is_better_than<greater>(val, k_th_);

    uint32_t mask = __ballot_sync(FUSED_FULL_WARP_MASK, do_add);
    if (mask == 0) return;

    int pos = smem_buf_len_ + __popc(mask & ((0x1u << lane_) - 1));
    if (do_add && pos < WARP_SIZE) {
      val_smem_[pos] = val;
      idx_smem_[pos] = idx;
      do_add = false;
    }
    smem_buf_len_ += __popc(mask);
    if (smem_buf_len_ >= WARP_SIZE) {
      __syncwarp();
      merge_buf_(val_smem_[lane_], idx_smem_[lane_]);
      smem_buf_len_ -= WARP_SIZE;
    }
    if (do_add) {
      pos -= WARP_SIZE;
      val_smem_[pos] = val;
      idx_smem_[pos] = idx;
    }
    __syncwarp();
  }

  // NOTE: no __syncthreads() here — callers must sync externally if needed.
  __device__ void done() {
    if (smem_buf_len_) {
      T val = (lane_ < smem_buf_len_) ? val_smem_[lane_] : dummy_;
      idxT idx = (lane_ < smem_buf_len_) ? idx_smem_[lane_] : 0;
      merge_buf_(val, idx);
    }
  }

 private:
  __device__ void set_k_th_() {
    k_th_ = __shfl_sync(
        FUSED_FULL_WARP_MASK, val_arr_[max_arr_len_ - 1], k_th_lane_);
    if constexpr (is_stable)
      k_th_idx_ = __shfl_sync(
          FUSED_FULL_WARP_MASK, idx_arr_[max_arr_len_ - 1], k_th_lane_);
  }

  __device__ void merge_buf_(T val, idxT idx) {
    BitonicSort<WARP_SIZE, greater, T, idxT, is_stable>::sort(&val, &idx);
    T& old = val_arr_[max_arr_len_ - 1];
    bool is_better;
    if constexpr (is_stable)
      is_better =
          is_better_than<greater>(val, old, idx, idx_arr_[max_arr_len_ - 1]);
    else
      is_better = is_better_than<greater>(val, old);
    if (is_better) {
      old = val;
      idx_arr_[max_arr_len_ - 1] = idx;
    }
    BitonicMerge<capacity, greater, !greater, T, idxT, is_stable>::merge(
        val_arr_, idx_arr_);
    set_k_th_();
  }

  using WarpSort<capacity, greater, T, idxT, is_stable>::max_arr_len_;
  using WarpSort<capacity, greater, T, idxT, is_stable>::val_arr_;
  using WarpSort<capacity, greater, T, idxT, is_stable>::idx_arr_;
  using WarpSort<capacity, greater, T, idxT, is_stable>::lane_;
  using WarpSort<capacity, greater, T, idxT, is_stable>::k_;
  using WarpSort<capacity, greater, T, idxT, is_stable>::dummy_;

  T* val_smem_;
  idxT* idx_smem_;
  int smem_buf_len_ = 0;
  T k_th_;
  idxT k_th_idx_;
  int const k_th_lane_;
};

}  // namespace warp_topk_fused

// ---------------------------------------------------------------------------
// Fused kernel: group-score computation + group selection + expert topk
//               + sparse scores write-back, in one kernel launch.
//
// gridDim  = num_tokens   (one block per token)
// blockDim = n_group * WARP_SIZE   (one warp per group)
// ---------------------------------------------------------------------------
template <typename InT, typename IdxT>
__global__ void grouped_topk_fused_kernel(
    float* scores,  // output: sparse routing weights [num_tokens, num_experts]
    float* topk_values,  // output: topk routing weights   [num_tokens, topk]
    IdxT* topk_indices,  // output: topk expert indices     [num_tokens, topk]
    InT const* gating_output,              // input:  raw logits (float or bf16)
                                           // [num_tokens, num_experts]
    float const* e_score_correction_bias,  // input:  bias [num_experts]
    int64_t const num_tokens,
    int64_t const num_experts,
    int64_t const n_group,
    int64_t const topk_group,
    int64_t const topk,
    bool const renormalize,
    double routed_scaling_factor) {
  int32_t const token_id = static_cast<int32_t>(blockIdx.x);
  if (token_id >= static_cast<int32_t>(num_tokens)) return;

  int32_t const warp_id = threadIdx.x / WARP_SIZE;
  int32_t const lane_id = threadIdx.x % WARP_SIZE;
  int32_t const n_group_i32 = static_cast<int32_t>(n_group);
  int32_t const topk_group_i32 = static_cast<int32_t>(topk_group);
  int32_t const topk_i32 = static_cast<int32_t>(topk);
  int32_t const num_warps = blockDim.x / WARP_SIZE;

  if (warp_id >= n_group_i32 || num_warps < n_group_i32) return;

  int32_t const num_experts_per_group =
      static_cast<int32_t>(num_experts) / n_group_i32;
  int32_t const align_epg = warp_topk_fused::round_up_to_multiple_of<WARP_SIZE>(
      num_experts_per_group);

  InT const* gate_token = gating_output + (int64_t)token_id * num_experts;
  float* scores_token = scores + (int64_t)token_id * num_experts;

  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<32> tile = cg::tiled_partition<32>(block);

  // smem layout: [val_staging (256B-aligned) | idx_staging | (16B pad) |
  // s_group_scores]
  extern __shared__ char smem_buf[];
  size_t const val_aligned = warp_topk_fused::round_up_to_multiple_of<256>(
      static_cast<size_t>(num_warps) * WARP_SIZE * sizeof(float));
  size_t const idx_bytes =
      static_cast<size_t>(num_warps) * WARP_SIZE * sizeof(int32_t);
  uintptr_t ptr =
      (reinterpret_cast<uintptr_t>(smem_buf + val_aligned + idx_bytes) + 15) &
      ~static_cast<uintptr_t>(15);
  float* s_group_scores = reinterpret_cast<float*>(ptr);
  float* s_topk_value =
      reinterpret_cast<float*>(smem_buf);  // val_staging (256B-aligned)

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  // ------------------------------------------------------------------
  // Phase 1 (all warps): compute group score = top2 sum of (gate + bias)
  // ------------------------------------------------------------------
  {
    int32_t const offset = warp_id * num_experts_per_group;
    InT const* gate_g = gate_token + offset;
    float const* bias_g = e_score_correction_bias + offset;

    float largest = neg_inf<float>();
    float second_largest = neg_inf<float>();

    if (num_experts_per_group > WARP_SIZE) {
      for (int i = lane_id; i < num_experts_per_group; i += WARP_SIZE) {
        float val = sigmoid_to_float(gate_g[i]) + bias_g[i];
        if (val > largest) {
          second_largest = largest;
          largest = val;
        } else if (val > second_largest) {
          second_largest = val;
        }
      }
    } else {
      for (int i = lane_id; i < num_experts_per_group; i += WARP_SIZE)
        largest = sigmoid_to_float(gate_g[i]) + bias_g[i];
    }
    __syncwarp();
    float max1 = cg::reduce(tile, largest, cg::greater<float>());
    float max2 = max1;
    int cnt = __popc(__ballot_sync(FUSED_FULL_WARP_MASK, largest == max1));
    if (cnt == 1) {
      largest = (largest == max1) ? second_largest : largest;
      max2 = cg::reduce(tile, largest, cg::greater<float>());
    }
    if (lane_id == 0) s_group_scores[warp_id] = max1 + max2;
  }

  __syncthreads();  // __syncwarp() maybe better?

  // ------------------------------------------------------------------
  // Phase 2 (warp0 only): group selection → expert selection → output
  // ------------------------------------------------------------------
  if (warp_id != 0) return;

  float value = neg_inf<float>();
  float topk_group_value = neg_inf<float>();
  int32_t num_equalto_topkth_group;
  if (token_id < num_tokens) {
    int32_t want_neg_inf_num = WARP_SIZE - n_group + topk_group;
    if (lane_id < n_group && (isfinite(s_group_scores[lane_id]))) {
      value = s_group_scores[lane_id];
    }

    int neg_inf_num = WARP_SIZE - n_group;
    int last_neg_inf_num = 0;
    // Use loop to find the largset top_group
    while (neg_inf_num < want_neg_inf_num) {
      __syncwarp();  // Ensure all threads have valid data before reduction
      topk_group_value = cg::reduce(tile, value, cg::greater<float>());
      if (value == topk_group_value) {
        value = neg_inf<float>();
      }
      last_neg_inf_num = neg_inf_num;

      neg_inf_num = __popc(
          __ballot_sync(FUSED_FULL_WARP_MASK, (value == neg_inf<float>())));
    }
    // There is a possible case:
    // may have many different group holding the same score!
    // but we only accept some of them!
    num_equalto_topkth_group = want_neg_inf_num - last_neg_inf_num;
  }
  __syncwarp();

  warp_topk_fused::WarpSelect</*capability*/ WARP_SIZE,
                              /*greater*/ true,
                              float,
                              int32_t,
                              /* is_stable */ true>
      queue((int32_t)topk, neg_inf<float>());
  int count_equalto_topkth_group = 0;
  bool if_proceed_next_topk = (topk_group_value != neg_inf<float>());
  if (token_id < num_tokens && if_proceed_next_topk) {
    for (int i_group = 0; i_group < n_group; i_group++) {
      if ((s_group_scores[i_group] > topk_group_value) ||
          ((s_group_scores[i_group] == topk_group_value) &&
           (count_equalto_topkth_group < num_equalto_topkth_group))) {
        int32_t offset = i_group * num_experts_per_group;
        for (int32_t i = lane_id; i < align_epg; i += WARP_SIZE) {
          float candidates = neg_inf<float>();
          if (i < num_experts_per_group) {
            float biased = sigmoid_to_float(gate_token[offset + i]) +
                           e_score_correction_bias[offset + i];
            if (is_finite_val(biased)) candidates = biased;
          }
          queue.add(candidates, offset + i);
        }
        if (s_group_scores[i_group] == topk_group_value) {
          count_equalto_topkth_group++;
        }
      }
    }
    queue.done();
    __syncwarp();
  }

  float topk_sum = 1e-20;
  if (token_id < num_tokens && if_proceed_next_topk) {
    for (int i = lane_id;
         i < warp_topk_fused::round_up_to_multiple_of<WARP_SIZE>(topk);
         i += WARP_SIZE) {
      int32_t idx = i / WARP_SIZE;
      float value =
          i < topk ? sigmoid_to_float(gate_token[queue.get_idx(idx)]) : 0.0f;
      if (i < topk) {
        s_topk_value[i] = value;
      }
      topk_sum += cg::reduce(tile, value, cg::plus<float>());
    }
  }
  __syncwarp();

  if (token_id < num_tokens && if_proceed_next_topk) {
    for (int i = lane_id; i < num_experts; i += WARP_SIZE) {
      scores_token[i] = 0;
    }
  }
  __syncwarp();

  topk_values += (int64_t)token_id * topk;
  topk_indices += (int64_t)token_id * topk;
  if (token_id < num_tokens) {
    if (if_proceed_next_topk) {
      for (int i = lane_id; i < topk; i += WARP_SIZE) {
        float value;
        if (renormalize) {
          value = s_topk_value[i] / topk_sum * routed_scaling_factor;
        } else {
          value = s_topk_value[i] * routed_scaling_factor;
        }
        int32_t idx = i / WARP_SIZE;  // topk may be bigger than WARP_SIZE
        scores_token[queue.get_idx(idx)] = value;
        topk_indices[i] = queue.get_idx(idx);
        topk_values[i] = value;
      }
    } else {
      for (int i = lane_id; i < topk; i += WARP_SIZE) {
        int32_t idx = i / WARP_SIZE;
        topk_indices[i] = queue.get_idx(idx);
        topk_values[i] = static_cast<float>(1.0f / topk);
      }
    }
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

// ---------------------------------------------------------------------------
// Launch wrapper
// ---------------------------------------------------------------------------
template <typename InT, typename IdxT>
void invokeFusedNoAuxTc(InT* gating_output,
                        float* e_score_correction_bias,
                        float* scores,
                        float* topk_values,
                        IdxT* topk_indices,
                        int64_t const num_tokens,
                        int64_t const num_experts,
                        int64_t const n_group,
                        int64_t const topk_group,
                        int64_t const topk,
                        bool const renormalize,
                        double const routed_scaling_factor,
                        cudaStream_t const stream) {
  auto* kernel = &grouped_topk_fused_kernel<InT, IdxT>;

  // blockDim = n_group * WARP_SIZE  (one warp per group)
  int32_t const num_warps = static_cast<int32_t>(n_group);

  // smem = WarpSelect staging (float) + 16B pad + group_scores buffer (float)
  size_t const val_aligned = warp_topk_fused::round_up_to_multiple_of<256>(
      static_cast<size_t>(num_warps) * WARP_SIZE * sizeof(float));
  size_t const idx_bytes =
      static_cast<size_t>(num_warps) * WARP_SIZE * sizeof(int32_t);
  size_t const extra_bytes = 16 + static_cast<size_t>(n_group) * sizeof(float);
  size_t const smem_bytes = val_aligned + idx_bytes + extra_bytes;

  cudaLaunchConfig_t config;
  config.gridDim = static_cast<uint32_t>(num_tokens);
  config.blockDim = static_cast<uint32_t>(n_group) * WARP_SIZE;
  config.dynamicSmemBytes = smem_bytes;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = false;
  config.numAttrs = 1;
  config.attrs = attrs;

  cudaLaunchKernelEx(&config,
                     kernel,
                     scores,
                     topk_values,
                     topk_indices,
                     gating_output,
                     e_score_correction_bias,
                     num_tokens,
                     num_experts,
                     n_group,
                     topk_group,
                     topk,
                     renormalize,
                     routed_scaling_factor);
}

#define INSTANTIATE_FUSED_NOAUX_TC(InT, IdxT)  \
  template void invokeFusedNoAuxTc<InT, IdxT>( \
      InT * gating_output,                     \
      float* e_score_correction_bias,          \
      float* scores,                           \
      float* topk_values,                      \
      IdxT* topk_indices,                      \
      int64_t const num_tokens,                \
      int64_t const num_experts,               \
      int64_t const n_group,                   \
      int64_t const topk_group,                \
      int64_t const topk,                      \
      bool const renormalize,                  \
      double const routed_scaling_factor,      \
      cudaStream_t const stream);

INSTANTIATE_FUSED_NOAUX_TC(float, int64_t);
INSTANTIATE_FUSED_NOAUX_TC(__nv_bfloat16, int64_t);
INSTANTIATE_FUSED_NOAUX_TC(__half, int64_t);

// ---------------------------------------------------------------------------
// Paddle op wrapper
// ---------------------------------------------------------------------------
std::vector<paddle::Tensor> grouped_topk(
    paddle::Tensor& gating_output,
    paddle::Tensor& e_score_correction_bias,
    int n_group,
    int topk_group,
    int topk,
    bool renormalize,
    float routed_scaling_factor) {
  auto input_shape = gating_output.shape();
  PD_CHECK(input_shape.size() == 2);
  int64_t num_tokens = input_shape[0];
  int64_t num_experts = input_shape[1];
  auto place = gating_output.place();
  PD_CHECK(n_group <= 32, "grouped_topk: n_group must be <= 32");
  PD_CHECK(topk <= 32, "grouped_topk: topk must be <= WARP_SIZE (32)");

  // Outputs are always float32 regardless of input dtype
  auto scores = paddle::empty(
      {num_tokens, num_experts}, paddle::DataType::FLOAT32, place);
  auto topk_values =
      paddle::empty({num_tokens, topk}, paddle::DataType::FLOAT32, place);
  auto topk_indices =
      paddle::empty({num_tokens, topk}, paddle::DataType::INT64, place);

  auto stream = gating_output.stream();
  auto dtype = gating_output.dtype();

  float* scores_ptr = reinterpret_cast<float*>(scores.data<float>());
  float* topk_values_ptr = reinterpret_cast<float*>(topk_values.data<float>());
  int64_t* topk_idx_ptr =
      reinterpret_cast<int64_t*>(topk_indices.data<int64_t>());
  float* bias_ptr =
      reinterpret_cast<float*>(e_score_correction_bias.data<float>());

  if (dtype == paddle::DataType::BFLOAT16) {
    invokeFusedNoAuxTc<__nv_bfloat16, int64_t>(
        reinterpret_cast<__nv_bfloat16*>(
            gating_output.data<paddle::bfloat16>()),
        bias_ptr,
        scores_ptr,
        topk_values_ptr,
        topk_idx_ptr,
        num_tokens,
        num_experts,
        static_cast<int64_t>(n_group),
        static_cast<int64_t>(topk_group),
        static_cast<int64_t>(topk),
        renormalize,
        static_cast<double>(routed_scaling_factor),
        stream);
  } else if (dtype == paddle::DataType::FLOAT16) {
    invokeFusedNoAuxTc<__half, int64_t>(
        reinterpret_cast<__half*>(gating_output.data<paddle::float16>()),
        bias_ptr,
        scores_ptr,
        topk_values_ptr,
        topk_idx_ptr,
        num_tokens,
        num_experts,
        static_cast<int64_t>(n_group),
        static_cast<int64_t>(topk_group),
        static_cast<int64_t>(topk),
        renormalize,
        static_cast<double>(routed_scaling_factor),
        stream);
  } else {
    PD_CHECK(
        dtype == paddle::DataType::FLOAT32,
        "grouped_topk: gating_output must be float32, float16, or bfloat16");
    invokeFusedNoAuxTc<float, int64_t>(
        reinterpret_cast<float*>(gating_output.data<float>()),
        bias_ptr,
        scores_ptr,
        topk_values_ptr,
        topk_idx_ptr,
        num_tokens,
        num_experts,
        static_cast<int64_t>(n_group),
        static_cast<int64_t>(topk_group),
        static_cast<int64_t>(topk),
        renormalize,
        static_cast<double>(routed_scaling_factor),
        stream);
  }

  return {scores, topk_values, topk_indices};
}

std::vector<paddle::DataType> GroupedTopkInferDtype(
    const paddle::DataType& /*gating_output_dtype*/,
    const paddle::DataType& /*e_score_correction_bias_dtype*/) {
  // Outputs are always float32: cast is fused into the kernel.
  return {paddle::DataType::FLOAT32,
          paddle::DataType::FLOAT32,
          paddle::DataType::INT64};
}

std::vector<std::vector<int64_t>> GroupedTopkInferShape(
    const std::vector<int64_t>& gating_output_shape,
    const std::vector<int64_t>&,
    const int topk) {
  auto num_tokens = gating_output_shape[0];
  auto num_experts = gating_output_shape[1];
  return {{num_tokens, num_experts}, {num_tokens, topk}, {num_tokens, topk}};
}

PD_BUILD_STATIC_OP(grouped_topk)
    .Inputs({"gating_output", "e_score_correction_bias"})
    .Outputs({"output_tensor", "topk_values", "topk_indices"})
    .Attrs({"n_group: int",
            "topk_group: int",
            "topk: int",
            "renormalize: bool",
            "routed_scaling_factor: float"})
    .SetKernelFn(PD_KERNEL(grouped_topk))
    .SetInferShapeFn(PD_INFER_SHAPE(GroupedTopkInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(GroupedTopkInferDtype));
