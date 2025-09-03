// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <algorithm>
#include "paddle/extension.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

constexpr int kBlockSize = 256;
constexpr int kNumWaves = 16;

#ifdef PADDLE_WITH_HIP
inline hipError_t GetNumBlocks(int64_t n, int *num_blocks) {
    int dev;
    {
        hipError_t err = hipGetDevice(&dev);
        if (err != hipSuccess) {
            return err;
        }
    }
    int sm_count;
    {
        hipError_t err = hipDeviceGetAttribute(
            &sm_count, hipDeviceAttributeMultiprocessorCount, dev);
        if (err != hipSuccess) {
            return err;
        }
    }
    int tpm;
    {
        hipError_t err = hipDeviceGetAttribute(
            &tpm, hipDeviceAttributeMaxThreadsPerMultiProcessor, dev);
        if (err != hipSuccess) {
            return err;
        }
    }
    *num_blocks = std::max<int>(
        1,
        std::min<int64_t>((n + kBlockSize - 1) / kBlockSize,
                          sm_count * tpm / kBlockSize * kNumWaves));
    return hipSuccess;
}
#else
inline cudaError_t GetNumBlocks(int64_t n, int *num_blocks) {
    int dev;
    {
        cudaError_t err = cudaGetDevice(&dev);
        if (err != cudaSuccess) {
            return err;
        }
    }
    int sm_count;
    {
        cudaError_t err = cudaDeviceGetAttribute(
            &sm_count, cudaDevAttrMultiProcessorCount, dev);
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
    *num_blocks = std::max<int>(
        1,
        std::min<int64_t>((n + kBlockSize - 1) / kBlockSize,
                          sm_count * tpm / kBlockSize * kNumWaves));
    return cudaSuccess;
}
#endif
template <paddle::DataType D>
class PDTraits;

template <>
class PDTraits<paddle::DataType::FLOAT32> {
    public:
    typedef float DataType;
    typedef float data_t;
};

template <>
class PDTraits<paddle::DataType::FLOAT16> {
    public:
    typedef half DataType;
    typedef paddle::float16 data_t;
};

/*
Position_ids: bsz, max_seq_length
*/

template <typename T, int N>
struct GetPackType {
    using type =
        typename std::aligned_storage<N * sizeof(T), N * sizeof(T)>::type;
};

template <typename T, int N>
using PackType = typename GetPackType<T, N>::type;

template <typename T, int N>
union Pack {
    static_assert(sizeof(PackType<T, N>) == sizeof(T) * N, "");
    __device__ Pack() {
        // do nothing
    }
    PackType<T, N> storage;
    T elem[N];
};

__global__ __launch_bounds__(kBlockSize) void fused_get_rotary_embedding(
    const float *position_ids,
    const int32_t bsz,
    const int32_t max_seq_length,
    const int32_t max_position_seq_length,
    const int32_t head_dim,
    const int32_t prompt_num,
    const float inv_head_dim,
    const int32_t elem_cnt,
    float *rope_embedding) {
    /*
    In Naive implementation, it will stacks [freqs, freqs]
    And actually, each threads can process 1 values, and store continuous 2 same
    values. So here We construct a Pack to store 2 values.
    */
    constexpr int PackSize = 2;
    Pack<float, PackSize> SinStorePack{};
    Pack<float, PackSize> CosStorePack{};

    const int half_head_dim = head_dim / PackSize;
    const int32_t global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int idx = global_thread_idx, step = blockDim.x * gridDim.x;
         idx < elem_cnt;
         idx += step) {
        const int32_t bsz_seq_idx = idx / half_head_dim;
        const int32_t bsz_idx = bsz_seq_idx / max_seq_length;
        const int32_t seq_idx = bsz_seq_idx % max_seq_length;
        const int64_t position_offset =
            bsz_idx * max_position_seq_length + seq_idx + prompt_num;
        const int32_t half_head_idx = (idx % half_head_dim) * PackSize;
        const float exponent_factor =
            -static_cast<float>(half_head_idx) *
            inv_head_dim;  // * inv_head_dim equals to / head_dim.
        const float inv_freq_val = powf(10000.0f, exponent_factor);
        const float freqs_val = position_ids[position_offset] * inv_freq_val;
        const float cos_embedding_val = cos(freqs_val);
        const float sin_embedding_val = sin(freqs_val);

/*
Since After stack, the continuous 2 elements value is same.
So here each threads store 2 computed embedding value.
*/
#pragma unroll
        for (int unroll_idx = 0; unroll_idx < PackSize; unroll_idx++) {
            CosStorePack.elem[unroll_idx] = cos_embedding_val;
            SinStorePack.elem[unroll_idx] = sin_embedding_val;
        }

        const int32_t cos_offset = bsz_seq_idx * head_dim + half_head_idx;
        const int32_t sin_offset = bsz * max_seq_length * head_dim + cos_offset;
        *(reinterpret_cast<PackType<float, PackSize> *>(
            rope_embedding + cos_offset)) = CosStorePack.storage;
        *(reinterpret_cast<PackType<float, PackSize> *>(
            rope_embedding + sin_offset)) = SinStorePack.storage;
    }
}

std::vector<paddle::Tensor> GetRoPE(const paddle::Tensor &input_ids,
                                    const paddle::Tensor &position_ids,
                                    const paddle::Tensor &head_dim_shape_tensor,
                                    int prompt_num) {
    const int64_t batch_size = input_ids.shape()[0];
    const int64_t max_seq_length = input_ids.shape()[1];
    const int64_t max_position_seq_length = position_ids.shape()[1];
    const int64_t head_dim = head_dim_shape_tensor.shape()[0];
    const float inv_head_dim = 1.0f / static_cast<float>(head_dim);

    auto cu_stream = position_ids.stream();

    auto rotary_embedding =
        paddle::empty({2, batch_size, 1, max_seq_length, head_dim},
                      paddle::DataType::FLOAT32,
                      position_ids.place());

    assert(head_dim % 2 == 0);
    const int32_t elem_cnt = batch_size * max_seq_length * head_dim / 2;
    int32_t grid_size = 1;
    GetNumBlocks(elem_cnt, &grid_size);

    fused_get_rotary_embedding<<<grid_size, kBlockSize, 0, cu_stream>>>(
        position_ids.data<float>(),
        batch_size,
        max_seq_length,
        max_position_seq_length,
        head_dim,
        prompt_num,
        inv_head_dim,
        elem_cnt,
        reinterpret_cast<float *>(rotary_embedding.data<float>()));
    return {rotary_embedding};
}

std::vector<std::vector<int64_t>> GetRoPEInferShape(
    const std::vector<int64_t> &input_ids_shape,
    const std::vector<int64_t> &position_ids_shape,
    const std::vector<int64_t> &head_dim_shape_tensor_shape) {
    const int64_t batch_size = position_ids_shape[0];
    const int64_t max_seq_length = input_ids_shape[1];
    const int64_t head_dim = head_dim_shape_tensor_shape[0];
    std::vector<int64_t> out_shape = {
        2, batch_size, 1, max_seq_length, head_dim};
    return {out_shape};
}

std::vector<paddle::DataType> GetRoPEInferDtype(
    const paddle::DataType &input_ids_dtype,
    const paddle::DataType &position_ids_dtype,
    const paddle::DataType &head_dim_shape_tensor_dtype) {
    // RoPE output dtype is Float.
    return {paddle::DataType::FLOAT32};
}

PD_BUILD_STATIC_OP(fused_get_rotary_embedding)
    .Inputs({"input_ids", "position_ids", "head_dim_shape_tensor"})
    .Outputs({"rotary_embedding"})
    .Attrs({"prompt_num: int"})
    .SetKernelFn(PD_KERNEL(GetRoPE))
    .SetInferShapeFn(PD_INFER_SHAPE(GetRoPEInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(GetRoPEInferDtype));
