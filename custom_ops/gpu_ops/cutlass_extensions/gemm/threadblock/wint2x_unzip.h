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

#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>

#include <cuda_runtime.h>

#include "cutlass/arch/memory.h"
#include "cutlass/trace.h"
#include "cutlass_extensions/wint_type_traits.h"

namespace cutlass {
namespace gemm {
namespace threadblock {

template <typename T, int N>
using UnzipArray = cutlass::AlignedArray<T, N, (N * cutlass::sizeof_bits<T>::value / 8)>;

template <typename T, WintQuantMethod QuantMethod, int TileRows,
          int TileColumns, int NumThreads = 128>
struct UnzipAndDequantFunctor {
  __device__ void operator()(const T *in_ptr, const T *supper_scale_ptr,
                             T *out_ptr, const int64_t in_stride) {}
};

template <typename T, int TileRows, int TileColumns, int NumThreads>
struct UnzipAndDequantFunctor<T, WintQuantMethod::kWeightOnlyInt25, TileRows,
                              TileColumns, NumThreads> {
  using ZippedT = uint16_t;
  using ScaleComputeT = float;

  static constexpr int32_t kGroupSize = 64;
  static constexpr int32_t kZippedGroupSize = 10;
  static constexpr int32_t kNumPackedValues = 7;

  static constexpr int32_t kWeightMask = 0x7;
  static constexpr int32_t kLocalScaleMask = 0x1FFF;
  static constexpr int32_t kBZP = 4;

  __device__ inline T Compute(int32_t zipped_value, int32_t shift_bit,
                              ScaleComputeT scale) {
    int32_t shifted_value = (zipped_value >> shift_bit) & kWeightMask;
    int32_t value = shifted_value - kBZP;

    ScaleComputeT scaled_value = static_cast<ScaleComputeT>(value) * scale;
    return static_cast<T>(scaled_value);
  }

  __device__ void operator()(const uint16_t *in_ptr, const T *super_scale_ptr,
                             T *out_ptr, const int64_t in_stride) {
    int32_t shift_bits[7] = {13, 11, 9, 6, 4, 2, 0};

    int tid = threadIdx.x;

#pragma unroll
    for (int col = tid; col < TileColumns; col += NumThreads) {
      ScaleComputeT super_scale =
          static_cast<ScaleComputeT>(super_scale_ptr[col]);

#pragma unroll
      for (int group_id = 0; group_id < TileRows / 64; ++group_id) {
        // the last row in group
        int zipped_row_last = group_id * 10 + 9;
        int zipped_offset_last = zipped_row_last * in_stride + col;
        int32_t zipped_value_last =
            static_cast<int32_t>(in_ptr[zipped_offset_last]);

        ScaleComputeT local_scale =
            static_cast<ScaleComputeT>(zipped_value_last & kLocalScaleMask);
        ScaleComputeT scale = local_scale * super_scale;

#pragma unroll
        for (int zipped_row_in_group = 0; zipped_row_in_group < 9;
             ++zipped_row_in_group) {
          int zipped_row = group_id * 10 + zipped_row_in_group;
          int zipped_offset = zipped_row * in_stride + col;
          int32_t zipped_value = static_cast<int32_t>(in_ptr[zipped_offset]);

          int row_in_group = group_id * 64 + zipped_row_in_group * 7;

#pragma unroll
          for (int shift_bit_id = 0; shift_bit_id < 7; ++shift_bit_id) {
            int32_t shift_bit = shift_bits[shift_bit_id];
            T value = Compute(zipped_value, shift_bit, scale);
            out_ptr[(row_in_group + shift_bit_id) * TileColumns + col] = value;
          }
        }

        int row_in_group_last = group_id * 64 + 63;
        T value_last = Compute(zipped_value_last, shift_bits[0], scale);
        out_ptr[row_in_group_last * TileColumns + col] = value_last;
      }
    }
    __syncthreads();
  }
};

template <typename T, int TileRows, int TileColumns, int NumThreads>
struct UnzipAndDequantFunctor<T, WintQuantMethod::kWeightOnlyInt2, TileRows,
                              TileColumns, NumThreads> {
  using ZippedT = uint8_t;
  using ScaleComputeT = float;

  static constexpr int32_t kGroupSize = 64;
  static constexpr int32_t kPackNum = 4;
  static constexpr int32_t kWeightMask = 0x3F;
  static constexpr int32_t kLocalScaleMask = 0xF;
  static constexpr int32_t kBZP = 32;

  // weight               [16, N]     uint8_t
  // local_scale          [1, N]      uint8_t
  // code_scale           [N]         float
  // code_zp              [N]         float
  // super_scale          [N]         T

  // code_scale, code_zp and super_scale
  static constexpr int32_t kColumnWiseSmemBytes = (2 * sizeof(float) + sizeof(T)) * TileColumns;
  // zipped weights and local_scale
  static constexpr int32_t kZippedSmemBytes = (TileRows / 4 + (TileRows + 127) / 128) * TileColumns;

  struct Arguments {
    uint8_t *weight_ptr;
    uint8_t *local_scale_ptr;
    float *code_scale_ptr;
    float *code_zp_ptr;
    T *super_scale_ptr;

    __device__ Arguments() : weight_ptr(nullptr), local_scale_ptr(nullptr), code_scale_ptr(nullptr), code_zp_ptr(nullptr), super_scale_ptr(nullptr) {}

    __device__ explicit Arguments(uint8_t *smem_ptr) {
      SetZippedPtrs(smem_ptr);
      SetColumnWisePtrs(smem_ptr + kZippedSmemBytes);
    }

    __device__ Arguments(uint8_t *zipped_smem_ptr, uint8_t *column_wise_smem_ptr) {
      SetZippedPtrs(zipped_smem_ptr);
      SetColumnWisePtrs(column_wise_smem_ptr);
    }

    __device__ void SetZippedPtrs(uint8_t *zipped_smem_ptr) {
      weight_ptr = zipped_smem_ptr;
      local_scale_ptr = zipped_smem_ptr + (TileRows / 4) * TileColumns;
    }

    __device__ void SetColumnWisePtrs(uint8_t *column_wise_smem_ptr) {
      code_scale_ptr = reinterpret_cast<float *>(column_wise_smem_ptr);
      code_zp_ptr = reinterpret_cast<float *>(column_wise_smem_ptr + sizeof(float) * TileColumns);
      super_scale_ptr = reinterpret_cast<T *>(column_wise_smem_ptr + 2 * sizeof(float) * TileColumns);
    }
  };

  __device__ void Load(const uint8_t *g_weight_ptr, const uint8_t *g_local_scale_ptr,
                       const float *g_code_scale_ptr, const float *g_code_zp_ptr,
                       const T *g_super_scale_ptr,
                       Arguments *args, const int64_t in_stride, bool need_preload) {
    int tid = threadIdx.x;

#pragma unroll
    for (int col = tid; col < TileColumns; col += NumThreads) {
      if (need_preload) {
        if (g_super_scale_ptr) {
          args->super_scale_ptr[col] = g_super_scale_ptr[col];
        } else {
          args->super_scale_ptr[col] = static_cast<T>(1);
        }

        args->code_scale_ptr[col] = g_code_scale_ptr[col];
        args->code_zp_ptr[col] = g_code_zp_ptr[col];
      }

#pragma unroll
      for (int ls_row_id = 0; ls_row_id < TileRows / 128; ++ls_row_id) {
        int local_scale_offset = ls_row_id * in_stride + col;
        args->local_scale_ptr[ls_row_id * TileColumns + col] = g_local_scale_ptr[local_scale_offset];
      }

#pragma unroll
      for (int zipped_row = 0; zipped_row < TileRows / 4; ++zipped_row) {
        int s_zipped_offset = zipped_row * TileColumns + col;
        int g_zipped_offset = zipped_row * 4 * in_stride + col;

        args->weight_ptr[s_zipped_offset] = g_weight_ptr[g_zipped_offset];
      }
    }
    __syncthreads();
  }

  __device__ void LoadAsync(const uint8_t *g_weight_ptr,
                            const uint8_t *g_local_scale_ptr,
                            const float *g_code_scale_ptr,
                            const float *g_code_zp_ptr,
                            const T *g_super_scale_ptr,
                            Arguments *args, const int64_t in_stride, bool need_preload) {
    int tid = threadIdx.x;

    constexpr int kBytesPerThread = 16; // 16B per thread

    constexpr int weight_size = TileRows / 4 * TileColumns;
    constexpr int local_scale_size = (TileRows + 127) / 128 * TileColumns;
    constexpr int code_scale_size = sizeof(float) * TileColumns;
    constexpr int code_zp_size = sizeof(float) * TileColumns;
    constexpr int super_scale_size = sizeof(T) * TileColumns;

    constexpr int total_size = weight_size + local_scale_size + code_scale_size + code_zp_size + super_scale_size;
    constexpr int total_tasks = total_size / kBytesPerThread;

    constexpr int cur_num_threads = total_tasks / ((total_tasks + NumThreads - 1) / NumThreads);

    constexpr int weight_threads = weight_size * cur_num_threads / total_size;
    constexpr int local_scale_threads = local_scale_size * cur_num_threads / total_size;
    constexpr int code_scale_threads = code_scale_size * cur_num_threads / total_size;
    constexpr int code_zp_threads = code_zp_size * cur_num_threads / total_size;
    constexpr int super_scale_threads = super_scale_size * cur_num_threads / total_size;

    static_assert(TileColumns % weight_threads == 0,
                  "TileColumns must be divisible by weight_threads to ensure correct thread mapping.");

    static_assert(TileColumns % local_scale_threads == 0,
                  "TileColumns must be divisible by local_scale_threads to ensure correct thread mapping.");

    if (tid < weight_threads) {
      constexpr int weight_per_thread_size = weight_size / weight_threads;
      constexpr int kIterations = (weight_per_thread_size + kBytesPerThread - 1) / kBytesPerThread;

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kIterations; ++i) {
          int z_offset = (tid * weight_per_thread_size + i * kBytesPerThread);
          int g_offset = z_offset / TileColumns * in_stride + z_offset % TileColumns;
          cutlass::arch::cp_async<kBytesPerThread, cutlass::arch::CacheOperation::Global>(
              args->weight_ptr + z_offset, g_weight_ptr + g_offset, true);
      }
    } else if (tid < weight_threads + local_scale_threads) {
      constexpr int start_thread_id = weight_threads;
      constexpr int local_scale_per_thread_size = local_scale_size / local_scale_threads;
      constexpr int kIterations = (local_scale_per_thread_size + kBytesPerThread - 1) / kBytesPerThread;

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kIterations; ++i) {
          int z_offset = (tid - start_thread_id) * local_scale_per_thread_size + i * kBytesPerThread;
          int g_offset = z_offset / TileColumns * in_stride + z_offset % TileColumns;
          cutlass::arch::cp_async<kBytesPerThread, cutlass::arch::CacheOperation::Global>(
              args->local_scale_ptr + z_offset, g_local_scale_ptr + g_offset, true);
      }
    } else if (need_preload) {
      if (tid < weight_threads + local_scale_threads + code_scale_threads) {
        constexpr int start_thread_id = weight_threads + local_scale_threads;
        constexpr int code_scale_per_thread_size = code_scale_size / code_scale_threads;
        constexpr int kIterations = (code_scale_per_thread_size + kBytesPerThread - 1) / kBytesPerThread;

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kIterations; ++i) {
          int offset = ((tid - start_thread_id) * code_scale_per_thread_size + i * kBytesPerThread) / sizeof(float);
          cutlass::arch::cp_async<kBytesPerThread, cutlass::arch::CacheOperation::Global>(
              args->code_scale_ptr + offset, g_code_scale_ptr + offset, true);
        }
      } else if (tid < weight_threads + local_scale_threads + code_scale_threads + code_zp_threads) {
        constexpr int start_thread_id = weight_threads + local_scale_threads + code_scale_threads;
        constexpr int code_zp_per_thread_size = code_zp_size / code_zp_threads;
        constexpr int kIterations = (code_zp_per_thread_size + kBytesPerThread - 1) / kBytesPerThread;

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kIterations; ++i) {
          int offset = ((tid - start_thread_id) * code_zp_per_thread_size + i * kBytesPerThread) / sizeof(float);
          cutlass::arch::cp_async<kBytesPerThread, cutlass::arch::CacheOperation::Global>(
              args->code_zp_ptr + offset, g_code_zp_ptr + offset, true);
        }
      } else if (tid < weight_threads + local_scale_threads + code_scale_threads + code_zp_threads + super_scale_threads) {
        if (g_super_scale_ptr) {
          constexpr int start_thread_id = weight_threads + local_scale_threads + code_scale_threads + code_zp_threads;
          constexpr int super_scale_per_thread_size = super_scale_size / super_scale_threads;
          constexpr int kIterations = (super_scale_per_thread_size + kBytesPerThread - 1) / kBytesPerThread;

          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < kIterations; ++i) {
            int offset = ((tid - start_thread_id) * super_scale_per_thread_size + i * kBytesPerThread) / sizeof(T);
            cutlass::arch::cp_async<kBytesPerThread, cutlass::arch::CacheOperation::Global>(
                args->super_scale_ptr + offset, g_super_scale_ptr + offset, true);
          }
        }
      }
    }
  }

  __device__ void Compute(const Arguments &args, T *out_ptr,
                          const int64_t block_start_row) {
    int32_t shift_bits[4] = {9, 6, 3, 0};

    int tid = threadIdx.x;

#pragma unroll
    for (int col = tid; col < TileColumns; col += NumThreads) {
      ScaleComputeT super_scale =
          static_cast<ScaleComputeT>(args.super_scale_ptr[col]);
      ScaleComputeT code_scale =
          static_cast<ScaleComputeT>(args.code_scale_ptr[col]);
      ScaleComputeT code_zp = static_cast<ScaleComputeT>(args.code_zp_ptr[col]);

#pragma unroll
      for (int group_id = 0; group_id < TileRows / 64; ++group_id) {
        int local_scale_offset = (group_id / 2) * TileColumns + col;
        int32_t local_scale =
            static_cast<int32_t>(args.local_scale_ptr[local_scale_offset]);

        ScaleComputeT zipped_value[16];

#pragma unroll
        for (int zipped_row = 0; zipped_row < 16; ++zipped_row) {
          int zipped_offset = (group_id * 16 + zipped_row) * TileColumns + col;
          zipped_value[zipped_row] =
              static_cast<ScaleComputeT>(args.weight_ptr[zipped_offset]);
        }

        int local_scale_shift = ((block_start_row / 64 + group_id + 1) & 1) * 4;
        int32_t shifted_local_scale =
            (local_scale >> local_scale_shift) & kLocalScaleMask;
        ScaleComputeT scale =
            static_cast<ScaleComputeT>(shifted_local_scale) * super_scale;

#pragma unroll
        for (int zipped_row = 0; zipped_row < 16; ++zipped_row) {
          int32_t decode_value =
              static_cast<int32_t>(floor(zipped_value[zipped_row] * code_scale + code_zp +
                                         static_cast<ScaleComputeT>(0.5)));

          int row = group_id * 64 + zipped_row * 4;

#pragma unroll
          for (int shift_bit_id = 0; shift_bit_id < 4; ++shift_bit_id) {
            int32_t shift_bit = shift_bits[shift_bit_id];
            int32_t shifted_value = (decode_value >> shift_bit) & kWeightMask;

            ScaleComputeT value =
                static_cast<ScaleComputeT>(shifted_value - kBZP);
            out_ptr[(row + shift_bit_id) * TileColumns + col] =
                static_cast<T>(scale * value);
          }
        }
      }
    }
    __syncthreads();
  }

  __device__ void ComputeVectorized(const Arguments &args, T *out_ptr,
                                    const int64_t block_start_row) {
    constexpr int kNumWeightsPerThread = TileRows * TileColumns / (4 * NumThreads);
    constexpr int N = (kNumWeightsPerThread >= 32) ? 4 : 2;
    constexpr int RowStride = NumThreads * N / TileColumns;
    constexpr int kNumIters = kNumWeightsPerThread / N;

    static_assert(N * NumThreads >= TileColumns, "N * NumThreads should be no less than TileColumns.");

    constexpr ScaleComputeT decode_value_zp = static_cast<ScaleComputeT>(0.5);

    int tid = threadIdx.x;
    int begin_col_id = (tid * N) % TileColumns;
    int begin_row_id = (tid * N) / TileColumns;

    static_assert(TileRows <= 128, "TileRows is expected to no more than 128.");

    UnzipArray<uint8_t, N> local_scales =
        *reinterpret_cast<const UnzipArray<uint8_t, N> *>(args.local_scale_ptr + begin_col_id);

    UnzipArray<uint8_t, N> zipped_values[2];
    int zipped_offset = begin_row_id * TileColumns + begin_col_id;
    zipped_values[0] =
        *reinterpret_cast<const UnzipArray<uint8_t, N> *>(args.weight_ptr + zipped_offset);

    UnzipArray<T, N> super_scales =
        *reinterpret_cast<const UnzipArray<T, N> *>(args.super_scale_ptr + begin_col_id);
    UnzipArray<float, N> code_scales =
        *reinterpret_cast<const UnzipArray<float, N> *>(args.code_scale_ptr + begin_col_id);
    UnzipArray<float, N> code_zps =
        *reinterpret_cast<const UnzipArray<float, N> *>(args.code_zp_ptr + begin_col_id);

    // special for TileRows = 64
    int local_scale_shift = (((block_start_row / 64) + 1) & 1) * 4;
    UnzipArray<ScaleComputeT, N> scales;

#pragma unroll
    for (int i = 0; i < N; ++i) {
      int32_t shifted_local_scale =
          (static_cast<int32_t>(local_scales[i]) >> local_scale_shift) & kLocalScaleMask;
      scales[i] =
            static_cast<ScaleComputeT>(shifted_local_scale) * static_cast<ScaleComputeT>(super_scales[i]);
    }

#pragma unroll
    for (int iter_id = 0; iter_id < kNumIters; ++iter_id) {
      int zipped_row = begin_row_id + iter_id * RowStride;
      int row = zipped_row * 4;

      if (iter_id < kNumIters - 1) {
        int zipped_offset = (zipped_row + RowStride) * TileColumns + begin_col_id;
        zipped_values[(iter_id + 1) & 1] =
            *reinterpret_cast<const UnzipArray<uint8_t, N> *>(args.weight_ptr + zipped_offset);
      }

      UnzipArray<T, N> outs[4];

#pragma unroll
      for (int i = 0; i < N; ++i) {
        int32_t decode_value =
            static_cast<int32_t>(floor(static_cast<ScaleComputeT>(zipped_values[iter_id & 1][i]) * code_scales[i]
                                       + code_zps[i] + decode_value_zp));

        ScaleComputeT value_3 = static_cast<ScaleComputeT>((decode_value & kWeightMask) - kBZP);
        decode_value >>= 3;
        ScaleComputeT value_2 = static_cast<ScaleComputeT>((decode_value & kWeightMask) - kBZP);
        decode_value >>= 3;
        ScaleComputeT value_1 = static_cast<ScaleComputeT>((decode_value & kWeightMask) - kBZP);
        decode_value >>= 3;
        ScaleComputeT value_0 = static_cast<ScaleComputeT>((decode_value & kWeightMask) - kBZP);
        outs[0][i] = static_cast<T>(scales[i] * value_0);
        outs[1][i] = static_cast<T>(scales[i] * value_1);
        outs[2][i] = static_cast<T>(scales[i] * value_2);
        outs[3][i] = static_cast<T>(scales[i] * value_3);
      }

#pragma unroll
      for (int shift_bit_id = 0; shift_bit_id < 4; ++shift_bit_id) {
        UnzipArray<T, N> *tmp_out_ptr = reinterpret_cast<UnzipArray<T, N> *>(
            out_ptr + (row + shift_bit_id) * TileColumns + begin_col_id);
        *tmp_out_ptr = outs[shift_bit_id];
      }
    }
    __syncthreads();
  }
};

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass
