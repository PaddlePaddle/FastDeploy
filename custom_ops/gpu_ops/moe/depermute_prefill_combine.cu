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

constexpr int DEPERMUTE_BLOCK_THREADS = 512;

// Depermute and combine expert outputs back to token-major layout.
//
// For each output token, this kernel:
//   1. Loads indice_map and topk_weights from shared memory
//   2. For each valid expert slot (indice >= 0), reads the expert output row,
//      scales by topk_weight, and accumulates in float32
//   3. Writes the combined result back in the original dtype
//
// x:            [num_experts, max_tokens_per_expert, hidden] - expert outputs
// indice_map:   [num_worst_tokens, topk] int32 - flat indices (expert_idx * M +
// offset) topk_weights: [num_worst_tokens, topk] float32 - combination weights
// depermuted_x: [num_worst_tokens, hidden] - output (same dtype as x)

template <typename T, int VecSize, int TOP_K>
__global__ void DepermutePrefillCombineKernel(
    T* __restrict__ depermuted_x,
    const T* __restrict__ x,
    const int32_t* __restrict__ indice_map,
    const float* __restrict__ topk_weights,
    const int num_worst_tokens,
    const int hidden,
    const int max_tokens_per_expert) {
  __shared__ int32_t smem_indices[TOP_K];
  __shared__ float smem_weights[TOP_K];

  const int tidx = threadIdx.x;
  const int num_vecs = hidden / VecSize;

  for (int token_idx = blockIdx.x; token_idx < num_worst_tokens;
       token_idx += gridDim.x) {
    // Thread 0 loads indice_map, thread 32 loads topk_weights
    if (tidx < TOP_K) {
      smem_indices[tidx] = indice_map[token_idx * TOP_K + tidx];
    }
    if (tidx >= TOP_K && tidx < 2 * TOP_K) {
      int k = tidx - TOP_K;
      smem_weights[k] = topk_weights[token_idx * TOP_K + k];
    }
    __syncthreads();

    // Check if any expert slot is valid
    bool need_store = false;
#pragma unroll
    for (int k = 0; k < TOP_K; k++) {
      if (smem_indices[k] >= 0) {
        need_store = true;
        break;
      }
    }

    if (need_store) {
      // Each thread processes a subset of hidden vectors
      for (int v = tidx; v < num_vecs; v += DEPERMUTE_BLOCK_THREADS) {
        // Initialize accumulator in float32
        float acc[VecSize];
#pragma unroll
        for (int i = 0; i < VecSize; i++) {
          acc[i] = 0.0f;
        }

        // Accumulate weighted contributions from each expert
        for (int k = 0; k < TOP_K; k++) {
          int32_t indice = smem_indices[k];
          if (indice >= 0) {
            float weight = smem_weights[k];
            int64_t expert_idx =
                static_cast<int64_t>(indice) / max_tokens_per_expert;
            int64_t offset =
                static_cast<int64_t>(indice) % max_tokens_per_expert;

            const T* src = x + expert_idx * max_tokens_per_expert * hidden +
                           offset * hidden;

            AlignedVector<T, VecSize> vec;
            Load<T, VecSize>(src + v * VecSize, &vec);

#pragma unroll
            for (int i = 0; i < VecSize; i++) {
              acc[i] += static_cast<float>(vec[i]) * weight;
            }
          }
        }

        // Cast back and store
        AlignedVector<T, VecSize> out_vec;
#pragma unroll
        for (int i = 0; i < VecSize; i++) {
          out_vec[i] = static_cast<T>(acc[i]);
        }
        Store<T, VecSize>(out_vec,
                          depermuted_x +
                              static_cast<int64_t>(token_idx) * hidden +
                              v * VecSize);
      }
    }

    __syncthreads();
  }
}

template <paddle::DataType D, int TOP_K>
std::vector<paddle::Tensor> DepermutePrefillCombineDispatch(
    const paddle::Tensor& x,
    const paddle::Tensor& indice_map,
    const paddle::Tensor& topk_weights,
    const int num_worst_tokens) {
  typedef PDTraits<D> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t;

  const int hidden = x.shape()[2];
  const int max_tokens_per_expert = x.shape()[1];

  auto place = x.place();
  auto stream = x.stream();

  auto depermuted_x =
      GetEmptyTensor({num_worst_tokens, hidden}, x.dtype(), place);

  constexpr int VecSize = 16 / sizeof(DataType_);

  int dev;
  cudaGetDevice(&dev);
  int sm_count;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
  int num_blocks = min(sm_count * 2, num_worst_tokens);

  DepermutePrefillCombineKernel<DataType_, VecSize, TOP_K>
      <<<num_blocks, DEPERMUTE_BLOCK_THREADS, 0, stream>>>(
          reinterpret_cast<DataType_*>(depermuted_x.data<data_t>()),
          reinterpret_cast<const DataType_*>(x.data<data_t>()),
          indice_map.data<int32_t>(),
          topk_weights.data<float>(),
          num_worst_tokens,
          hidden,
          max_tokens_per_expert);

  return {depermuted_x};
}

std::vector<paddle::Tensor> DepermutePrefillCombine(
    const paddle::Tensor& x,
    const paddle::Tensor& indice_map,
    const paddle::Tensor& topk_weights,
    const int num_worst_tokens) {
  const int topk = indice_map.shape()[1];

#define DISPATCH_TOPK(DTYPE, TOPK_VAL)                       \
  case TOPK_VAL:                                             \
    return DepermutePrefillCombineDispatch<DTYPE, TOPK_VAL>( \
        x, indice_map, topk_weights, num_worst_tokens);

  switch (x.dtype()) {
    case paddle::DataType::FLOAT8_E4M3FN: {
      switch (topk) {
        DISPATCH_TOPK(paddle::DataType::FLOAT8_E4M3FN, 4)
        DISPATCH_TOPK(paddle::DataType::FLOAT8_E4M3FN, 6)
        DISPATCH_TOPK(paddle::DataType::FLOAT8_E4M3FN, 8)
        default:
          PD_THROW("Unsupported topk value, must be 4, 6 or 8");
      }
    }
    case paddle::DataType::BFLOAT16: {
      switch (topk) {
        DISPATCH_TOPK(paddle::DataType::BFLOAT16, 4)
        DISPATCH_TOPK(paddle::DataType::BFLOAT16, 6)
        DISPATCH_TOPK(paddle::DataType::BFLOAT16, 8)
        default:
          PD_THROW("Unsupported topk value, must be 4, 6 or 8");
      }
    }
    default:
      PD_THROW("Unsupported dtype, must be float8_e4m3fn or bfloat16");
  }

#undef DISPATCH_TOPK
}

std::vector<std::vector<int64_t>> DepermutePrefillCombineInferShape(
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& indice_map_shape,
    const std::vector<int64_t>& topk_weights_shape,
    const int num_worst_tokens) {
  int64_t hidden = x_shape[2];
  return {{num_worst_tokens, hidden}};
}

std::vector<paddle::DataType> DepermutePrefillCombineInferDtype(
    const paddle::DataType& x_dtype,
    const paddle::DataType& indice_map_dtype,
    const paddle::DataType& topk_weights_dtype,
    const int num_worst_tokens) {
  return {x_dtype};
}

PD_BUILD_STATIC_OP(depermute_prefill_combine)
    .Inputs({"x", "indice_map", "topk_weights"})
    .Outputs({"depermuted_x"})
    .Attrs({"num_worst_tokens: int"})
    .SetKernelFn(PD_KERNEL(DepermutePrefillCombine))
    .SetInferShapeFn(PD_INFER_SHAPE(DepermutePrefillCombineInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(DepermutePrefillCombineInferDtype));
