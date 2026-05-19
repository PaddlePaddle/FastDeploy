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

#include "helper.h"

#include <cstdint>

constexpr int BLOCK_THREADS = 512;

template <typename T,
          typename ScaleT,
          int VecSize,
          int TOP_K,
          bool SWIZZLE_SCALE>
__global__ void PrefillPermuteToMaskedGemmKernel(
    T* __restrict__ permute_x,
    ScaleT* __restrict__ permute_scale,
    int32_t* __restrict__ permuted_indice_map,
    int32_t* __restrict__ token_nums_per_expert,
    const T* __restrict__ x,
    const ScaleT* __restrict__ scale,
    const int64_t* __restrict__ topk_ids,
    const int num_tokens,
    const int hidden,
    const int hidden_scale,
    const int max_tokens_per_expert) {
  __shared__ int32_t smem_offset;
  __shared__ int64_t smem_topk_ids[TOP_K];

  const int tidx = threadIdx.x;
  const int x_num_vecs = hidden / VecSize;
  // Pre-compute swizzle constants outside the slot loop (compile-time dead if
  // !SWIZZLE_SCALE)
  const int scale_bytes_per_token =
      hidden_scale * static_cast<int>(sizeof(ScaleT));
  const int m_tiles = max_tokens_per_expert / 128;
  const int k_tiles = scale_bytes_per_token / 4;

  for (int token_idx = blockIdx.x; token_idx < num_tokens;
       token_idx += gridDim.x) {
    if (tidx < TOP_K) {
      smem_topk_ids[tidx] = topk_ids[token_idx * TOP_K + tidx];
    }
    __syncthreads();
    bool should_break = true;
    for (int slot = 0; slot < TOP_K; slot++) {
      int64_t expert_idx = smem_topk_ids[slot];
      if (expert_idx != -1) {
        should_break = false;
        if (tidx == 0) {
          smem_offset = atomicAdd(&token_nums_per_expert[expert_idx], 1);
          permuted_indice_map[token_idx * TOP_K + slot] = static_cast<int32_t>(
              expert_idx * max_tokens_per_expert + smem_offset);
        }
        __syncthreads();

        int offset = smem_offset;

        // Vectorized copy of x[token_idx, :] -> permute_x[expert_idx, offset,
        // :]
        const T* src_x = x + static_cast<int64_t>(token_idx) * hidden;
        T* dst_x =
            permute_x +
            static_cast<int64_t>(expert_idx) * max_tokens_per_expert * hidden +
            static_cast<int64_t>(offset) * hidden;

        AlignedVector<T, VecSize> vec_x;
        for (int v = tidx; v < x_num_vecs; v += BLOCK_THREADS) {
          Load<T, VecSize>(src_x + v * VecSize, &vec_x);
          Store<T, VecSize>(vec_x, dst_x + v * VecSize);
        }

        const ScaleT* src_scale =
            scale + static_cast<int64_t>(token_idx) * hidden_scale;

        if constexpr (SWIZZLE_SCALE) {
          // Directly write packed FP8 scale bytes into the swizzled layout used
          // by flashinfer cutedsl: [E, M/128, K/4, 32, 4, 4]. The tensor is
          // exposed to Paddle as packed float32 [E, M, K/4].
          const uint8_t* src_scale_bytes =
              reinterpret_cast<const uint8_t*>(src_scale);
          uint8_t* dst_scale_bytes = reinterpret_cast<uint8_t*>(permute_scale);
          const int rm = offset >> 7;
          const int m_in = offset & 127;
          const int m_in2 = m_in >> 5;
          const int m_in3 = m_in & 31;

          for (int s = tidx; s < scale_bytes_per_token; s += BLOCK_THREADS) {
            const int rk = s >> 2;
            const int k_in = s & 3;
            const int64_t dst_idx =
                (((((static_cast<int64_t>(expert_idx) * m_tiles + rm) *
                        k_tiles +
                    rk) *
                       32 +
                   m_in3) *
                      4 +
                  m_in2) *
                     4 +
                 k_in);
            dst_scale_bytes[dst_idx] = src_scale_bytes[s];
          }
        } else {
          // Copy scale[token_idx, :] -> permute_scale with transposed layout.
          // Physical layout is [E, S, M], accessed as [E, M, S] via strides
          // [S*M, 1, M].
          ScaleT* dst_scale_base = permute_scale +
                                   static_cast<int64_t>(expert_idx) *
                                       hidden_scale * max_tokens_per_expert +
                                   offset;

          for (int s = tidx; s < hidden_scale; s += BLOCK_THREADS) {
            dst_scale_base[static_cast<int64_t>(s) * max_tokens_per_expert] =
                src_scale[s];
          }
        }

        __syncthreads();
      }
    }
    if (should_break) {
      break;
    }
  }
}

template <paddle::DataType D, paddle::DataType ScaleD, int TOP_K>
std::vector<paddle::Tensor> PrefillPermuteToMaskedGemmDispatch(
    const paddle::Tensor& x,
    const paddle::Tensor& scale,
    const paddle::Tensor& topk_ids,
    const int num_local_experts,
    const int max_token_num,
    const bool make_scale_interleaved) {
  typedef PDTraits<D> traits_;
  typedef PDTraits<ScaleD> scale_traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t;
  typedef typename scale_traits_::DataType ScaleDataType_;
  typedef typename scale_traits_::data_t scale_data_t;

  const int num_tokens = x.shape()[0];
  const int hidden = x.shape()[1];
  const int hidden_scale = scale.shape()[1];
  const int topk = topk_ids.shape()[1];

  auto place = x.place();
  auto stream = x.stream();

  auto permute_x = GetEmptyTensor(
      {num_local_experts, max_token_num, hidden}, x.dtype(), place);

  paddle::Tensor permute_scale;
  if (make_scale_interleaved) {
    const int scale_bytes_per_token =
        hidden_scale * static_cast<int>(sizeof(ScaleDataType_));
    PD_CHECK(
        max_token_num % 128 == 0,
        "make_scale_interleaved requires max_token_num to be divisible by 128");
    PD_CHECK(scale_bytes_per_token % 4 == 0,
             "make_scale_interleaved requires the unpacked FP8 scale dimension "
             "to be "
             "divisible by 4");
    permute_scale = GetEmptyTensor(
        {num_local_experts, max_token_num, hidden_scale}, ScaleD, place);
  } else {
    permute_scale =
        GetEmptyTensor({num_local_experts, max_token_num, hidden_scale},
                       {static_cast<int64_t>(hidden_scale) * max_token_num,
                        1,
                        static_cast<int64_t>(max_token_num)},
                       ScaleD,
                       place);
  }

  auto permuted_indice_map =
      GetEmptyTensor({num_tokens, topk}, paddle::DataType::INT32, place);
  auto token_nums_per_expert =
      GetEmptyTensor({num_local_experts, 1}, paddle::DataType::INT32, place);

  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaMemsetAsync(token_nums_per_expert.data<int32_t>(),
                      0,
                      num_local_experts * sizeof(int32_t),
                      stream));
  // memset 0xFF for int32 produces -1
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaMemsetAsync(permuted_indice_map.data<int32_t>(),
                      0xFF,
                      num_tokens * topk * sizeof(int32_t),
                      stream));

  constexpr int VecSize = 16 / sizeof(DataType_);

  int dev;
  cudaGetDevice(&dev);
  int sm_count;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
  int num_blocks = sm_count * 2;

#define LAUNCH_PREFILL_PERMUTE(SWIZZLE)                           \
  PrefillPermuteToMaskedGemmKernel<DataType_,                     \
                                   ScaleDataType_,                \
                                   VecSize,                       \
                                   TOP_K,                         \
                                   SWIZZLE>                       \
      <<<num_blocks, BLOCK_THREADS, 0, stream>>>(                 \
          reinterpret_cast<DataType_*>(permute_x.data<data_t>()), \
          reinterpret_cast<ScaleDataType_*>(                      \
              permute_scale.template data<scale_data_t>()),       \
          permuted_indice_map.data<int32_t>(),                    \
          token_nums_per_expert.data<int32_t>(),                  \
          reinterpret_cast<const DataType_*>(x.data<data_t>()),   \
          reinterpret_cast<const ScaleDataType_*>(                \
              scale.template data<scale_data_t>()),               \
          topk_ids.data<int64_t>(),                               \
          num_tokens,                                             \
          hidden,                                                 \
          hidden_scale,                                           \
          max_token_num)

  if (make_scale_interleaved) {
    LAUNCH_PREFILL_PERMUTE(true);
  } else {
    LAUNCH_PREFILL_PERMUTE(false);
  }

#undef LAUNCH_PREFILL_PERMUTE

  return {permute_x, permute_scale, permuted_indice_map, token_nums_per_expert};
}

std::vector<paddle::Tensor> PrefillPermuteToMaskedGemm(
    const paddle::Tensor& x,
    const paddle::Tensor& scale,
    const paddle::Tensor& topk_ids,
    const int num_local_experts,
    const int max_token_num,
    const bool make_scale_interleaved) {
  if (make_scale_interleaved) {
    PD_CHECK(
        x.dtype() == paddle::DataType::UINT8 &&
            scale.dtype() == paddle::DataType::FLOAT32,
        "make_scale_interleaved=true is only valid for UINT8 x + FLOAT32 scale "
        "(FP4 comm quant path)");
  }
  const int topk = topk_ids.shape()[1];

#define DISPATCH_TOPK(DTYPE, SCALE_DTYPE, TOPK_VAL)                          \
  case TOPK_VAL:                                                             \
    return PrefillPermuteToMaskedGemmDispatch<DTYPE, SCALE_DTYPE, TOPK_VAL>( \
        x,                                                                   \
        scale,                                                               \
        topk_ids,                                                            \
        num_local_experts,                                                   \
        max_token_num,                                                       \
        make_scale_interleaved);

  switch (x.dtype()) {
    case paddle::DataType::FLOAT8_E4M3FN: {
      switch (scale.dtype()) {
        case paddle::DataType::FLOAT32: {
          switch (topk) {
            DISPATCH_TOPK(
                paddle::DataType::FLOAT8_E4M3FN, paddle::DataType::FLOAT32, 4)
            DISPATCH_TOPK(
                paddle::DataType::FLOAT8_E4M3FN, paddle::DataType::FLOAT32, 8)
            default:
              PD_THROW("Unsupported topk value, must be 4 or 8");
          }
        }
        case paddle::DataType::INT32: {
          switch (topk) {
            DISPATCH_TOPK(
                paddle::DataType::FLOAT8_E4M3FN, paddle::DataType::INT32, 4)
            DISPATCH_TOPK(
                paddle::DataType::FLOAT8_E4M3FN, paddle::DataType::INT32, 8)
            default:
              PD_THROW("Unsupported topk value, must be 4 or 8");
          }
        }
      }
    }
    case paddle::DataType::UINT8: {
      switch (scale.dtype()) {
        case paddle::DataType::FLOAT32: {
          switch (topk) {
            DISPATCH_TOPK(paddle::DataType::UINT8, paddle::DataType::FLOAT32, 4)
            DISPATCH_TOPK(paddle::DataType::UINT8, paddle::DataType::FLOAT32, 6)
            DISPATCH_TOPK(paddle::DataType::UINT8, paddle::DataType::FLOAT32, 8)
            default:
              PD_THROW("Unsupported topk value, must be 4 or 6 or 8");
          }
        }
        default:
          PD_THROW("Unsupported scale dtype for UINT8 x, must be float32");
      }
    }
    case paddle::DataType::BFLOAT16: {
      switch (scale.dtype()) {
        case paddle::DataType::FLOAT32: {
          switch (topk) {
            DISPATCH_TOPK(
                paddle::DataType::BFLOAT16, paddle::DataType::FLOAT32, 4)
            DISPATCH_TOPK(
                paddle::DataType::BFLOAT16, paddle::DataType::FLOAT32, 6)
            DISPATCH_TOPK(
                paddle::DataType::BFLOAT16, paddle::DataType::FLOAT32, 8)
            default:
              PD_THROW("Unsupported topk value, must be 4 or 6 or 8");
          }
        }
        case paddle::DataType::INT32: {
          switch (topk) {
            DISPATCH_TOPK(
                paddle::DataType::BFLOAT16, paddle::DataType::INT32, 4)
            DISPATCH_TOPK(
                paddle::DataType::BFLOAT16, paddle::DataType::INT32, 8)
            default:
              PD_THROW("Unsupported topk value, must be 4 or 8");
          }
        }
        case paddle::DataType::UINT8: {
          switch (topk) {
            DISPATCH_TOPK(
                paddle::DataType::BFLOAT16, paddle::DataType::UINT8, 4)
            DISPATCH_TOPK(
                paddle::DataType::BFLOAT16, paddle::DataType::UINT8, 8)
            default:
              PD_THROW("Unsupported topk value, must be 4 or 8");
          }
        }
      }
    }
    default:
      PD_THROW("Unsupported dtype, must be uint8, float8_e4m3fn or bfloat16");
  }

#undef DISPATCH_TOPK
}

std::vector<std::vector<int64_t>> PrefillPermuteToMaskedGemmInferShape(
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& scale_shape,
    const std::vector<int64_t>& topk_ids_shape,
    const int num_local_experts,
    const int max_token_num,
    const bool make_scale_interleaved) {
  int64_t num_tokens = x_shape[0];
  int64_t hidden = x_shape[1];
  int64_t hidden_scale = scale_shape[1];
  int64_t topk = topk_ids_shape[1];

  return {
      {num_local_experts, max_token_num, hidden},
      {num_local_experts, max_token_num, hidden_scale},
      {num_tokens, topk},
      {num_local_experts, 1},
  };
}

std::vector<paddle::DataType> PrefillPermuteToMaskedGemmInferDtype(
    const paddle::DataType& x_dtype,
    const paddle::DataType& scale_dtype,
    const paddle::DataType& topk_ids_dtype,
    const int num_local_experts,
    const int max_token_num,
    const bool make_scale_interleaved) {
  return {
      x_dtype, scale_dtype, paddle::DataType::INT32, paddle::DataType::INT32};
}

PD_BUILD_STATIC_OP(prefill_permute_to_masked_gemm)
    .Inputs({"x", "scale", "topk_ids"})
    .Outputs({"permute_x",
              "permute_scale",
              "permuted_indice_map",
              "token_nums_per_expert"})
    .Attrs({"num_local_experts: int",
            "max_token_num: int",
            "make_scale_interleaved: bool"})
    .SetKernelFn(PD_KERNEL(PrefillPermuteToMaskedGemm))
    .SetInferShapeFn(PD_INFER_SHAPE(PrefillPermuteToMaskedGemmInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(PrefillPermuteToMaskedGemmInferDtype));
