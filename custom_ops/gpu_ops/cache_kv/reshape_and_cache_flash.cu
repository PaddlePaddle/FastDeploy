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

#include "paddle/extension.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/core/memory/memcpy.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

#ifdef PADDLE_WITH_HIP
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
namespace cub = hipcub;
#else
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#endif

#include <cute/tensor.hpp>

// CuTe-based implementation of reshape_and_cache_flash with:
//   1. Compile-time head_dim template (64 / 128) for loop unrolling
//      and CuTe static-shape optimisations.
//   2. Dynamic FP8 E4M3 quantisation: per-head absmax is computed
//      on-the-fly; the FP8 cache AND per-head dequant scale are both
//      written by the kernel.
//
// Scale convention:
//   quantise  : fp8  = float_val / scale
//   dequantise: float = fp8_val  * scale
//   where scale = absmax / kFp8E4M3Max
//
// Two kernels are provided:
//   - reshape_and_cache_flash_cute_kernel     (non-FP8, direct copy)
//   - reshape_and_cache_flash_cute_fp8_kernel (dynamic FP8 quantisation)

using namespace cute;

// FP8 E4M3 representable maximum
static constexpr float kFp8E4M3Max = 448.0f;

// ═══════════════════════════════════════════════════════════════════
//  Block-wide absmax reduction  (BlockSize must be 32-aligned pow2)
// ═══════════════════════════════════════════════════════════════════
template <int BlockSize>
__device__ __forceinline__ float block_reduce_absmax(float val) {
  static_assert(BlockSize >= 32 && (BlockSize & (BlockSize - 1)) == 0,
                "BlockSize must be a power-of-two >= 32");
  constexpr int kNumWarps = BlockSize / 32;

  __shared__ float smem[kNumWarps];

  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;

  // Intra-warp max
  float v = fabsf(val);
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFF, v, offset));
  }

  // One value per warp → shared memory
  if (lane == 0) smem[warp_id] = v;
  __syncthreads();

  // First warp reduces across warps
  if (warp_id == 0) {
    v = (lane < kNumWarps) ? smem[lane] : 0.f;
#pragma unroll
    for (int offset = kNumWarps / 2; offset > 0; offset >>= 1) {
      v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFF, v, offset));
    }
    if (lane == 0) smem[0] = v;
  }
  __syncthreads();

  return smem[0];
}

// ═══════════════════════════════════════════════════════════════════
//  128-bit vectorised 1D copy (same type, no conversion)
// ═══════════════════════════════════════════════════════════════════
template <typename T>
__device__ __forceinline__ void vec128_copy_1d(const T* __restrict__ src,
                                               T* __restrict__ dst,
                                               int n_elems,
                                               int tid,
                                               int stride) {
  constexpr int kVecElems = 16 / sizeof(T);  // bf16→8, fp32→4

  auto gSrc = make_tensor(make_gmem_ptr(src), make_shape(n_elems));
  auto gDst = make_tensor(make_gmem_ptr(dst), make_shape(n_elems));

  // recast to uint4 (128 bits) for coalesced 16-byte transactions
  auto gSrcV = recast<uint4>(gSrc);
  auto gDstV = recast<uint4>(gDst);
  const int n_vecs = size(gSrcV);

  for (int i = tid; i < n_vecs; i += stride) {
    gDstV(i) = gSrcV(i);
  }
  // scalar tail (< kVecElems elements)
  const int tail_start = n_vecs * kVecElems;
  for (int i = tid + tail_start; i < n_elems; i += stride) {
    gDst(i) = gSrc(i);
  }
}

// ═══════════════════════════════════════════════════════════════════
//  Kernel 1 — Non-FP8 direct copy  (NHD / HND layouts)
// ═══════════════════════════════════════════════════════════════════
//
//  Grid  : (num_tokens)
//  Block : min(num_heads * head_dim, 512)
//
//  Two runtime paths inside the kernel:
//    A) NHD contiguous (head_stride == head_dim) → flat 128-bit copy
//    B) HND strided    (head_stride > head_dim)  → warp-per-head copy
//
template <int head_dim, typename scalar_t>
__global__ void reshape_and_cache_flash_cute_kernel(
    const scalar_t* __restrict__ key,    // [num_tokens, num_heads, head_dim]
    const scalar_t* __restrict__ value,  // [num_tokens, num_heads, head_dim]
    scalar_t* __restrict__ key_cache,    // [num_blocks, ...]
    scalar_t* __restrict__ value_cache,  // [num_blocks, ...]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int64_t block_stride,                // key_cache.strides()[0]
    const int64_t page_stride,                 // key_cache.strides()[1]
    const int64_t head_stride,                 // key_cache.strides()[2]
    const int64_t key_stride,                  // key.strides()[0]
    const int64_t value_stride,                // value.strides()[0]
    const int num_heads,
    const int block_size) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) return;

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;

  const scalar_t* key_src = key + token_idx * key_stride;
  const scalar_t* value_src = value + token_idx * value_stride;
  scalar_t* key_dst =
      key_cache + block_idx * block_stride + block_offset * page_stride;
  scalar_t* value_dst =
      value_cache + block_idx * block_stride + block_offset * page_stride;

  // ── Path A: NHD contiguous ──
  // head_stride == head_dim means all heads are packed together.
  if (head_stride == head_dim) {
    const int n_elems = num_heads * head_dim;
    vec128_copy_1d(key_src, key_dst, n_elems, threadIdx.x, blockDim.x);
    vec128_copy_1d(value_src, value_dst, n_elems, threadIdx.x, blockDim.x);
    return;
  }

  // ── Path B: HND (heads are strided) ──
  // CuTe 2-D tensors: (num_heads, head_dim)
  //   src  stride = (head_dim,     1)   ← contiguous row-major
  //   dst  stride = (head_stride,  1)   ← heads may be non-contiguous
  auto gK_src =
      make_tensor(make_gmem_ptr(key_src),
                  make_layout(make_shape(num_heads, Int<head_dim>{}),
                              make_stride(Int<head_dim>{}, Int<1>{})));
  auto gV_src =
      make_tensor(make_gmem_ptr(value_src),
                  make_layout(make_shape(num_heads, Int<head_dim>{}),
                              make_stride(Int<head_dim>{}, Int<1>{})));
  auto gK_dst = make_tensor(make_gmem_ptr(key_dst),
                            make_layout(make_shape(num_heads, Int<head_dim>{}),
                                        make_stride(head_stride, Int<1>{})));
  auto gV_dst = make_tensor(make_gmem_ptr(value_dst),
                            make_layout(make_shape(num_heads, Int<head_dim>{}),
                                        make_stride(head_stride, Int<1>{})));

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int num_warps = blockDim.x >> 5;

  for (int h = warp_id; h < num_heads; h += num_warps) {
    // CuTe slice → 1-D [head_dim] view; compiler sees Int<head_dim>
    auto k_src_h = gK_src(h, _);
    auto k_dst_h = gK_dst(h, _);
    auto v_src_h = gV_src(h, _);
    auto v_dst_h = gV_dst(h, _);

    // head_dim is compile-time → loop is fully unrolled
    // head_dim=128, 32 lanes → 4 iterations each
#pragma unroll
    for (int i = lane; i < head_dim; i += 32) {
      k_dst_h(i) = k_src_h(i);
    }
#pragma unroll
    for (int i = lane; i < head_dim; i += 32) {
      v_dst_h(i) = v_src_h(i);
    }
  }
}

// ═══════════════════════════════════════════════════════════════════
//  Kernel 2 — Dynamic FP8 E4M3 quantisation (head-wise scale)
// ═══════════════════════════════════════════════════════════════════
//
//  Grid  : (num_tokens, num_heads)   — one block per (token, head)
//  Block : (head_dim)                — one thread per element
//
//  Per block:
//    1.  Load head_dim elements of K (and V) from input
//    2.  Block-reduce to find absmax
//    3.  scale = absmax / 448
//    4.  fp8_val = __nv_cvt_float_to_fp8(float_val / scale)
//    5.  Store FP8 data → cache,  store scale → scale buffer
//
template <int head_dim, typename scalar_t>
__global__ void reshape_and_cache_flash_cute_fp8_kernel(
    const scalar_t* __restrict__ key,    // [num_tokens, num_heads, head_dim]
    const scalar_t* __restrict__ value,  // [num_tokens, num_heads, head_dim]
    uint8_t* __restrict__ key_cache,     // [num_blocks, ...] fp8 e4m3
    uint8_t* __restrict__ value_cache,   // same
    float* __restrict__ k_scale_cache,   // [num_blocks, block_size, num_heads]
    float* __restrict__ v_scale_cache,   // same
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int64_t key_stride,                  // key.strides()[0]
    const int64_t value_stride,                // value.strides()[0]
    const int64_t cache_block_stride,          // cache.strides()[0] in uint8_t
    const int64_t cache_page_stride,   // cache.strides()[1]  (token-in-block)
    const int64_t cache_head_stride,   // cache.strides()[2]  (head)
    const int64_t scale_block_stride,  // scale.strides()[0]
    const int64_t scale_page_stride,   // scale.strides()[1]
    const int num_heads,
    const int block_size) {
  static_assert(head_dim == 64 || head_dim == 128,
                "Only head_dim 64 and 128 are supported");

  const int token_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int tid = threadIdx.x;  // 0 .. head_dim-1

  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) return;

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;

  // ── CuTe source tensors: 1-D [head_dim] per (token, head) ──
  auto gK_src = make_tensor(
      make_gmem_ptr(key + token_idx * key_stride + head_idx * head_dim),
      make_shape(Int<head_dim>{}));
  auto gV_src = make_tensor(
      make_gmem_ptr(value + token_idx * value_stride + head_idx * head_dim),
      make_shape(Int<head_dim>{}));

  // ── CuTe dest tensors: 1-D [head_dim] in FP8 cache ──
  const int64_t cache_base = block_idx * cache_block_stride +
                             block_offset * cache_page_stride +
                             head_idx * cache_head_stride;
  auto gK_dst = make_tensor(make_gmem_ptr(key_cache + cache_base),
                            make_shape(Int<head_dim>{}));
  auto gV_dst = make_tensor(make_gmem_ptr(value_cache + cache_base),
                            make_shape(Int<head_dim>{}));

  // ── Scale position ──
  const int64_t scale_pos = block_idx * scale_block_stride +
                            block_offset * scale_page_stride + head_idx;

  // ────────────────── K: load → absmax → quantise → store ──────────────────
  float k_val = static_cast<float>(gK_src(tid));
  float k_absmax = block_reduce_absmax<head_dim>(k_val);
  float k_scale = fmaxf(k_absmax / kFp8E4M3Max, 1e-12f);
  float k_q = k_val / k_scale;

#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
  gK_dst(tid) = __nv_cvt_float_to_fp8(k_q, __NV_SATFINITE, __NV_E4M3);
#else
  gK_dst(tid) = 0;
#endif
#else
  gK_dst(tid) = 0;
#endif
  if (tid == 0) {
    k_scale_cache[scale_pos] = k_scale;
  }

  // ────────────────── V: same flow (reuses smem after sync) ────────────────
  __syncthreads();

  float v_val = static_cast<float>(gV_src(tid));
  float v_absmax = block_reduce_absmax<head_dim>(v_val);
  float v_scale = fmaxf(v_absmax / kFp8E4M3Max, 1e-12f);
  float v_q = v_val / v_scale;

#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
  gV_dst(tid) = __nv_cvt_float_to_fp8(v_q, __NV_SATFINITE, __NV_E4M3);
#else
  gV_dst(tid) = 0;
#endif
#else
  gV_dst(tid) = 0;
#endif
  if (tid == 0) {
    v_scale_cache[scale_pos] = v_scale;
  }
}

// ═══════════════════════════════════════════════════════════════════
//  Paddle Custom Op Implementation
// ═══════════════════════════════════════════════════════════════════

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

template <>
class PDTraits<paddle::DataType::BFLOAT16> {
 public:
#ifdef PADDLE_WITH_HIP
  typedef hip_bfloat16 DataType;
#else
  typedef __nv_bfloat16 DataType;
#endif
  typedef paddle::bfloat16 data_t;
};

// FP8 E4M3 KV cache data type enum
enum class Fp8KVCacheDataType {
  kAuto,     // Use non-FP8 (direct copy)
  kFp8E4M3,  // Use dynamic FP8 E4M3 quantization
};

// Get FP8 KV cache data type from string
inline Fp8KVCacheDataType get_fp8_kv_cache_data_type(
    const std::string& dtype_str) {
  if (dtype_str == "auto" || dtype_str.empty()) {
    return Fp8KVCacheDataType::kAuto;
  } else if (dtype_str == "fp8_e4m3") {
    return Fp8KVCacheDataType::kFp8E4M3;
  } else {
    PD_THROW("Unsupported kv_cache_dtype: ", dtype_str);
    return Fp8KVCacheDataType::kAuto;
  }
}

// ═══════════════════════════════════════════════════════════════════
//  Kernel launch function for non-FP8 path
// ═══════════════════════════════════════════════════════════════════
template <typename scalar_t>
void LaunchNonFP8Kernel(const scalar_t* key,
                        const scalar_t* value,
                        scalar_t* key_cache,
                        scalar_t* value_cache,
                        const int64_t* slot_mapping,
                        const int64_t block_stride,
                        const int64_t page_stride,
                        const int64_t head_stride,
                        const int64_t key_stride,
                        const int64_t value_stride,
                        const int num_tokens,
                        const int num_heads,
                        const int head_dim,
                        const int block_size,
                        const gpuStream_t& stream) {
  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_dim, 512));

  if (head_dim == 64) {
    reshape_and_cache_flash_cute_kernel<64, scalar_t>
        <<<grid, block, 0, stream>>>(key,
                                     value,
                                     key_cache,
                                     value_cache,
                                     slot_mapping,
                                     block_stride,
                                     page_stride,
                                     head_stride,
                                     key_stride,
                                     value_stride,
                                     num_heads,
                                     block_size);
  } else if (head_dim == 128) {
    reshape_and_cache_flash_cute_kernel<128, scalar_t>
        <<<grid, block, 0, stream>>>(key,
                                     value,
                                     key_cache,
                                     value_cache,
                                     slot_mapping,
                                     block_stride,
                                     page_stride,
                                     head_stride,
                                     key_stride,
                                     value_stride,
                                     num_heads,
                                     block_size);
  } else {
    PD_THROW(
        "Unsupported head_dim: ", head_dim, ". Only 64 and 128 are supported.");
  }
}

// ═══════════════════════════════════════════════════════════════════
//  Kernel launch function for FP8 path
// ═══════════════════════════════════════════════════════════════════
template <typename scalar_t>
void LaunchFP8Kernel(const scalar_t* key,
                     const scalar_t* value,
                     uint8_t* key_cache,
                     uint8_t* value_cache,
                     float* k_scale_cache,
                     float* v_scale_cache,
                     const int64_t* slot_mapping,
                     const int64_t key_stride,
                     const int64_t value_stride,
                     const int64_t cache_block_stride,
                     const int64_t cache_page_stride,
                     const int64_t cache_head_stride,
                     const int64_t scale_block_stride,
                     const int64_t scale_page_stride,
                     const int num_tokens,
                     const int num_heads,
                     const int head_dim,
                     const int block_size,
                     const gpuStream_t& stream) {
  dim3 grid(num_tokens, num_heads);
  dim3 block(head_dim);

  if (head_dim == 64) {
    reshape_and_cache_flash_cute_fp8_kernel<64, scalar_t>
        <<<grid, block, 0, stream>>>(key,
                                     value,
                                     key_cache,
                                     value_cache,
                                     k_scale_cache,
                                     v_scale_cache,
                                     slot_mapping,
                                     key_stride,
                                     value_stride,
                                     cache_block_stride,
                                     cache_page_stride,
                                     cache_head_stride,
                                     scale_block_stride,
                                     scale_page_stride,
                                     num_heads,
                                     block_size);
  } else if (head_dim == 128) {
    reshape_and_cache_flash_cute_fp8_kernel<128, scalar_t>
        <<<grid, block, 0, stream>>>(key,
                                     value,
                                     key_cache,
                                     value_cache,
                                     k_scale_cache,
                                     v_scale_cache,
                                     slot_mapping,
                                     key_stride,
                                     value_stride,
                                     cache_block_stride,
                                     cache_page_stride,
                                     cache_head_stride,
                                     scale_block_stride,
                                     scale_page_stride,
                                     num_heads,
                                     block_size);
  } else {
    PD_THROW(
        "Unsupported head_dim: ", head_dim, ". Only 64 and 128 are supported.");
  }
}

// ═══════════════════════════════════════════════════════════════════
//  Host function: dispatch based on data type
// ═══════════════════════════════════════════════════════════════════
template <paddle::DataType D>
std::vector<paddle::Tensor> ReshapeAndCacheFlash(
    const paddle::Tensor& key,
    const paddle::Tensor& value,
    const paddle::Tensor& key_cache,
    const paddle::Tensor& value_cache,
    const paddle::Tensor& slot_mapping,
    const paddle::Tensor& k_scale,
    const paddle::Tensor& v_scale,
    const std::string& kv_cache_dtype) {
  typedef PDTraits<D> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t;

  // Get tensor shapes and strides
  auto key_shape = key.shape();
  auto key_cache_shape = key_cache.shape();

  const int num_tokens = key_shape[0];
  const int num_heads = key_shape[1];
  const int head_dim = key_shape[2];
  const int block_size = key_cache_shape[1];

  // Get CUDA stream
  auto stream = key.stream();

  // Get data pointers
  const data_t* key_ptr = key.data<data_t>();
  const data_t* value_ptr = value.data<data_t>();
  const int64_t* slot_mapping_ptr = slot_mapping.data<int64_t>();

  // Get FP8 KV cache data type
  Fp8KVCacheDataType kv_dt = get_fp8_kv_cache_data_type(kv_cache_dtype);

  if (kv_dt == Fp8KVCacheDataType::kAuto) {
    // ════════════ Non-FP8 path ════════════
    const int64_t key_stride = key.strides()[0];
    const int64_t value_stride = value.strides()[0];
    const int64_t block_stride = key_cache.strides()[0];
    const int64_t page_stride = key_cache.strides()[1];
    const int64_t head_stride = key_cache.strides()[2];

    data_t* key_cache_ptr = const_cast<data_t*>(key_cache.data<data_t>());
    data_t* value_cache_ptr = const_cast<data_t*>(value_cache.data<data_t>());

    LaunchNonFP8Kernel<DataType_>(reinterpret_cast<const DataType_*>(key_ptr),
                                  reinterpret_cast<const DataType_*>(value_ptr),
                                  reinterpret_cast<DataType_*>(key_cache_ptr),
                                  reinterpret_cast<DataType_*>(value_cache_ptr),
                                  slot_mapping_ptr,
                                  block_stride,
                                  page_stride,
                                  head_stride,
                                  key_stride,
                                  value_stride,
                                  num_tokens,
                                  num_heads,
                                  head_dim,
                                  block_size,
                                  stream);

  } else if (kv_dt == Fp8KVCacheDataType::kFp8E4M3) {
    // ════════════ Dynamic FP8 path ════════════
    PD_CHECK(k_scale.dtype() == paddle::DataType::FLOAT32,
             "k_scale must be float32 for dynamic FP8 quantisation");
    PD_CHECK(v_scale.dtype() == paddle::DataType::FLOAT32,
             "v_scale must be float32 for dynamic FP8 quantisation");

    const int64_t key_stride = key.strides()[0];
    const int64_t value_stride = value.strides()[0];
    const int64_t cache_block_stride = key_cache.strides()[0];
    const int64_t cache_page_stride = key_cache.strides()[1];
    const int64_t cache_head_stride = key_cache.strides()[2];
    const int64_t scale_block_stride = k_scale.strides()[0];
    const int64_t scale_page_stride = k_scale.strides()[1];

    const uint8_t* key_cache_ptr = key_cache.data<uint8_t>();
    const uint8_t* value_cache_ptr = value_cache.data<uint8_t>();
    float* k_scale_ptr = const_cast<float*>(k_scale.data<float>());
    float* v_scale_ptr = const_cast<float*>(v_scale.data<float>());

    LaunchFP8Kernel<DataType_>(reinterpret_cast<const DataType_*>(key_ptr),
                               reinterpret_cast<const DataType_*>(value_ptr),
                               const_cast<uint8_t*>(key_cache_ptr),
                               const_cast<uint8_t*>(value_cache_ptr),
                               k_scale_ptr,
                               v_scale_ptr,
                               slot_mapping_ptr,
                               key_stride,
                               value_stride,
                               cache_block_stride,
                               cache_page_stride,
                               cache_head_stride,
                               scale_block_stride,
                               scale_page_stride,
                               num_tokens,
                               num_heads,
                               head_dim,
                               block_size,
                               stream);
  } else {
    PD_THROW("Unsupported kv_cache_dtype: ", kv_cache_dtype);
  }

  // Return in-place modified cache tensors
  return {key_cache, value_cache, k_scale, v_scale};
}

// ═══════════════════════════════════════════════════════════════════
//  Dispatcher function: select based on input data type
// ═══════════════════════════════════════════════════════════════════
std::vector<paddle::Tensor> ReshapeAndCacheFlashKernel(
    const paddle::Tensor& key,
    const paddle::Tensor& value,
    const paddle::Tensor& key_cache,
    const paddle::Tensor& value_cache,
    const paddle::Tensor& slot_mapping,
    const paddle::Tensor& k_scale,
    const paddle::Tensor& v_scale,
    const std::string& kv_cache_dtype) {
  switch (key.type()) {
    case paddle::DataType::BFLOAT16: {
      return ReshapeAndCacheFlash<paddle::DataType::BFLOAT16>(key,
                                                              value,
                                                              key_cache,
                                                              value_cache,
                                                              slot_mapping,
                                                              k_scale,
                                                              v_scale,
                                                              kv_cache_dtype);
    }
    case paddle::DataType::FLOAT16: {
      return ReshapeAndCacheFlash<paddle::DataType::FLOAT16>(key,
                                                             value,
                                                             key_cache,
                                                             value_cache,
                                                             slot_mapping,
                                                             k_scale,
                                                             v_scale,
                                                             kv_cache_dtype);
    }
    case paddle::DataType::FLOAT32: {
      return ReshapeAndCacheFlash<paddle::DataType::FLOAT32>(key,
                                                             value,
                                                             key_cache,
                                                             value_cache,
                                                             slot_mapping,
                                                             k_scale,
                                                             v_scale,
                                                             kv_cache_dtype);
    }
    default: {
      PD_THROW("Unsupported input dtype: ",
               key.type(),
               ". Only bfloat16, float16 and float32 are supported.");
    }
  }
}

// ═══════════════════════════════════════════════════════════════════
//  InferShape function
// ═══════════════════════════════════════════════════════════════════
std::vector<std::vector<int64_t>> ReshapeAndCacheFlashInferShape(
    const std::vector<int64_t>& key_shape,
    const std::vector<int64_t>& value_shape,
    const std::vector<int64_t>& key_cache_shape,
    const std::vector<int64_t>& value_cache_shape,
    const std::vector<int64_t>& slot_mapping_shape,
    const std::vector<int64_t>& k_scale_shape,
    const std::vector<int64_t>& v_scale_shape) {
  return {key_cache_shape, value_cache_shape, k_scale_shape, v_scale_shape};
}

// ═══════════════════════════════════════════════════════════════════
//  InferDtype function
// ═══════════════════════════════════════════════════════════════════
std::vector<paddle::DataType> ReshapeAndCacheFlashInferDtype(
    const paddle::DataType& key_dtype,
    const paddle::DataType& value_dtype,
    const paddle::DataType& key_cache_dtype,
    const paddle::DataType& value_cache_dtype,
    const paddle::DataType& slot_mapping_dtype,
    const paddle::DataType& k_scale_dtype,
    const paddle::DataType& v_scale_dtype) {
  return {key_cache_dtype, value_cache_dtype, k_scale_dtype, v_scale_dtype};
}

// ═══════════════════════════════════════════════════════════════════
//  Register the custom op
// ═══════════════════════════════════════════════════════════════════
PD_BUILD_STATIC_OP(reshape_and_cache_flash)
    .Inputs({"key",
             "value",
             "key_cache",
             "value_cache",
             "slot_mapping",
             "k_scale",
             "v_scale"})
    .Outputs({"key_cache_out", "value_cache_out", "k_scale_out", "v_scale_out"})
    .SetInplaceMap({{"key_cache", "key_cache_out"},
                    {"value_cache", "value_cache_out"},
                    {"k_scale", "k_scale_out"},
                    {"v_scale", "v_scale_out"}})
    .Attrs({"kv_cache_dtype: std::string"})
    .SetKernelFn(PD_KERNEL(ReshapeAndCacheFlashKernel))
    .SetInferShapeFn(PD_INFER_SHAPE(ReshapeAndCacheFlashInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(ReshapeAndCacheFlashInferDtype));
