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

/**
 * DeepSeek Kv3.2 (DsKv3.2) Attention WriteCache Implementation
 *
 * This file implements writecache operations for DeepSeek MLA (Multi-head
 * Latent Attention) with FP8 quantization support, migrated from vLLM.
 *
 * Key features:
 * 1. DS MLA FP8 cache format (656 bytes per token):
 *    - 512 bytes: quantized NoPE part (fp8_e4m3)
 *    - 16 bytes: scale factors (4 x float32)
 *    - 128 bytes: RoPE part (64 x bf16, unquantized)
 *
 * 2. Standard MLA cache format (kv_lora_rank + pe_dim elements)
 *
 * 3. Indexer K quantization and cache operations
 */

#include "ds_mla_cache_kernel.cuh"
#include "helper.h"
#include "remote_cache_kv_ipc.h"

//==============================================================================
// DS MLA FP8 WriteCache Implementation
//==============================================================================

/**
 * Prefill stage: Write KV cache with DS MLA FP8 format
 */
template <paddle::DataType T>
std::vector<paddle::Tensor> PrefillDSMLAWriteCacheFP8(
    const AppendAttnMetaData& meta_data,
    const paddle::Tensor& kv_nope,
    const paddle::Tensor& kv_pe,
    const paddle::Tensor& slot_mapping,
    cudaStream_t& stream,
    paddle::Tensor* kv_cache) {
  typedef PDTraits<T> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t;

  auto num_tokens = slot_mapping.dims()[0];
  auto kv_lora_rank = 512;  // DS MLA uses 512
  auto pe_dim = 64;         // DS MLA uses 64
  auto block_size = meta_data.block_size;

  // Entry size for DS MLA FP8: 512 (fp8) + 16 (scales) + 128 (rope bf16) = 656
  // bytes
  const int entry_size = 656;

  // Launch kernel with 96 threads (64 for NoPE, 32 for RoPE)
  dim3 grid(num_tokens);
  dim3 block(96);

  const auto& kv_cache_dims = kv_cache->dims();
  int block_stride = kv_cache->strides()[0];
  int entry_stride = entry_size;
  int kv_c_stride = kv_nope.strides()[0];
  int k_pe_stride = kv_pe.strides()[0];

  ds_mla::concat_and_cache_ds_mla_kernel<DataType_><<<grid, block, 0, stream>>>(
      reinterpret_cast<DataType_*>(const_cast<data_t*>(kv_nope.data<data_t>())),
      reinterpret_cast<DataType_*>(const_cast<data_t*>(kv_pe.data<data_t>())),
      reinterpret_cast<uint8_t*>(kv_cache->data<uint8_t>()),
      slot_mapping.data<int64_t>(),
      block_stride,
      entry_stride,
      kv_c_stride,
      k_pe_stride,
      kv_lora_rank,
      pe_dim,
      block_size);
  return {};
}

/**
 * Decode stage: Write KV cache with DS MLA FP8 format
 */
template <paddle::DataType T>
std::vector<paddle::Tensor> DecodeDSMLAWriteCacheFP8(
    const AppendAttnMetaData& meta_data,
    const paddle::Tensor& kv_nope,
    const paddle::Tensor& kv_pe,
    const paddle::Tensor& slot_mapping,
    cudaStream_t& stream,
    paddle::Tensor* kv_cache) {
  typedef PDTraits<T> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t;

  auto num_tokens = slot_mapping.dims()[0];
  auto kv_lora_rank = 512;
  auto pe_dim = 64;
  auto block_size = meta_data.block_size;
  const int entry_size = 656;

  dim3 grid(num_tokens);
  dim3 block(96);

  const auto& kv_cache_dims = kv_cache->dims();
  int block_stride = kv_cache->strides()[0];
  int entry_stride = entry_size;
  int kv_c_stride = kv_nope.strides()[0];
  int k_pe_stride = kv_pe.strides()[0];

  ds_mla::concat_and_cache_ds_mla_kernel<DataType_><<<grid, block, 0, stream>>>(
      reinterpret_cast<DataType_*>(const_cast<data_t*>(kv_nope.data<data_t>())),
      reinterpret_cast<DataType_*>(const_cast<data_t*>(kv_pe.data<data_t>())),
      reinterpret_cast<uint8_t*>(kv_cache->data<uint8_t>()),
      slot_mapping.data<int64_t>(),
      block_stride,
      entry_stride,
      kv_c_stride,
      k_pe_stride,
      kv_lora_rank,
      pe_dim,
      block_size);

  return {};
}

//==============================================================================
// Standard MLA WriteCache Implementation
//==============================================================================

/**
 * Prefill stage: Write KV cache with standard MLA format
 */
template <paddle::DataType T>
std::vector<paddle::Tensor> PrefillDSMLAWriteCache(
    const AppendAttnMetaData& meta_data,
    const paddle::Tensor& kv_nope,
    const paddle::Tensor& kv_pe,
    const paddle::Tensor& slot_mapping,
    const float* scale,
    cudaStream_t& stream,
    paddle::Tensor* kv_cache) {
  typedef PDTraits<T> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t;

  auto num_tokens = slot_mapping.dims()[0];
  auto kv_lora_rank = meta_data.head_dims_v;
  auto pe_dim = meta_data.head_dims - meta_data.head_dims_v;
  auto block_size = meta_data.block_size;

  const auto& kv_cache_dims = kv_cache->dims();
  int block_stride = kv_cache->strides()[0];
  int entry_stride = kv_cache->strides()[1];
  int kv_c_stride = kv_nope.strides()[0];
  int k_pe_stride = kv_pe.strides()[0];

  dim3 grid(num_tokens);
  dim3 block(std::min(kv_lora_rank, 512));

  ds_mla::concat_and_cache_mla_kernel<DataType_, DataType_>
      <<<grid, block, 0, stream>>>(
          reinterpret_cast<DataType_*>(
              const_cast<data_t*>(kv_nope.data<data_t>())),
          reinterpret_cast<DataType_*>(
              const_cast<data_t*>(kv_pe.data<data_t>())),
          reinterpret_cast<DataType_*>(kv_cache->data<data_t>()),
          slot_mapping.data<int64_t>(),
          block_stride,
          entry_stride,
          kv_c_stride,
          k_pe_stride,
          kv_lora_rank,
          pe_dim,
          block_size,
          scale);

  return {};
}

//==============================================================================
// Indexer K Quantization and Cache Operations
//==============================================================================

/**
 * Quantize K tensor to FP8 and write to cache
 */
template <paddle::DataType T>
std::vector<paddle::Tensor> IndexerKQuantAndCache(
    const paddle::Tensor& k,
    const paddle::Tensor& slot_mapping,
    const int head_dim,
    const int quant_block_size,
    const int cache_block_size,
    const int cache_stride,
    const bool use_ue8m0,
    cudaStream_t& stream,
    paddle::Tensor* kv_cache) {
  typedef PDTraits<T> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t;

  int num_tokens = k.dims()[0];

  constexpr int vec_size = 4;
  dim3 grid(num_tokens,
            (head_dim + quant_block_size * vec_size - 1) /
                (quant_block_size * vec_size));
  dim3 block(32, vec_size);

  ds_mla::indexer_k_quant_and_cache_kernel<DataType_>
      <<<grid, block, 0, stream>>>(
          reinterpret_cast<DataType_*>(const_cast<data_t*>(k.data<data_t>())),
          reinterpret_cast<uint8_t*>(kv_cache->data<uint8_t>()),
          slot_mapping.data<int64_t>(),
          head_dim,
          quant_block_size,
          cache_block_size,
          cache_stride,
          use_ue8m0);

  return {};
}

/**
 * Gather K from quantized cache
 */
void CpGatherIndexerKQuantCache(const paddle::Tensor& kv_cache,
                                paddle::Tensor& dst_k,
                                paddle::Tensor& dst_scale,
                                const paddle::Tensor& block_table,
                                const paddle::Tensor& cu_seq_lens,
                                cudaStream_t& stream) {
  int batch_size = block_table.dims()[0];
  int num_tokens = dst_k.dims()[0];
  int head_dim = dst_k.dims()[1];
  int quant_block_size = head_dim * 4 / dst_scale.dims()[1];

  constexpr int vec_size = 16;

#define CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(BLOCK_Y_SIZE)                  \
  ds_mla::cp_gather_indexer_k_quant_cache_kernel<BLOCK_Y_SIZE>              \
      <<<dim3((num_tokens + BLOCK_Y_SIZE - 1) / BLOCK_Y_SIZE,               \
              (head_dim + 8 * vec_size - 1) / (8 * vec_size)),              \
         dim3(8, BLOCK_Y_SIZE),                                             \
         0,                                                                 \
         stream>>>(reinterpret_cast<const char*>(kv_cache.data<uint8_t>()), \
                   reinterpret_cast<char*>(dst_k.data<uint8_t>()),          \
                   reinterpret_cast<char*>(dst_scale.data<float>()),        \
                   block_table.data<int>(),                                 \
                   cu_seq_lens.data<int>(),                                 \
                   batch_size,                                              \
                   dst_k.strides()[0],                                      \
                   dst_k.dims()[1],                                         \
                   kv_cache.strides()[0],                                   \
                   kv_cache.strides()[1],                                   \
                   kv_cache.dims()[1],                                      \
                   block_table.dims()[1],                                   \
                   num_tokens,                                              \
                   quant_block_size);

  if (num_tokens < 32) {
    CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(1);
  } else if (num_tokens < 64) {
    CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(2);
  } else if (num_tokens < 128) {
    CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(4);
  } else if (num_tokens < 256) {
    CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(8);
  } else if (num_tokens < 512) {
    CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(16);
  } else {
    CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(32);
  }

#undef CALL_CP_GATHER_INDEXER_K_QUANT_CACHE
}

//==============================================================================
// Kernel Entry Points
//==============================================================================

/**
 * DS MLA WriteCache entry point - supports both FP8 and standard formats
 */
std::vector<paddle::Tensor> DSMLAWriteCacheKernel(
    const paddle::Tensor& kv_nope,
    const paddle::Tensor& kv_pe,
    const paddle::Tensor& kv_cache,
    const paddle::Tensor& slot_mapping,
    const paddle::optional<paddle::Tensor>& scale,
    const std::string& cache_quant_type_str,
    const bool is_prefill) {
  cudaStream_t stream = kv_pe.stream();
  AppendAttnMetaData meta_data;

  const auto& kv_nope_dims = kv_nope.dims();
  const auto& kv_pe_dims = kv_pe.dims();
  const auto& kv_cache_dims = kv_cache.dims();

  meta_data.kv_num_heads = kv_cache_dims[1];
  const auto nope_size =
      kv_nope_dims[kv_nope_dims.size() - 1] / meta_data.kv_num_heads;
  meta_data.token_nums = kv_nope_dims[0];
  meta_data.head_dims = kv_cache_dims[3];
  meta_data.head_dims_v = nope_size;
  meta_data.block_size = kv_cache_dims[2];

  const float* scale_ptr = scale ? scale.get().data<float>() : nullptr;

  if (cache_quant_type_str == "fp8_ds_mla") {
    // FP8 DS MLA format
    switch (kv_pe.dtype()) {
      case paddle::DataType::BFLOAT16: {
        if (is_prefill) {
          return PrefillDSMLAWriteCacheFP8<paddle::DataType::BFLOAT16>(
              meta_data,
              kv_nope,
              kv_pe,
              slot_mapping,
              stream,
              const_cast<paddle::Tensor*>(&kv_cache));
        } else {
          return DecodeDSMLAWriteCacheFP8<paddle::DataType::BFLOAT16>(
              meta_data,
              kv_nope,
              kv_pe,
              slot_mapping,
              stream,
              const_cast<paddle::Tensor*>(&kv_cache));
        }
      }
      case paddle::DataType::FLOAT16: {
        if (is_prefill) {
          return PrefillDSMLAWriteCacheFP8<paddle::DataType::FLOAT16>(
              meta_data,
              kv_nope,
              kv_pe,
              slot_mapping,
              stream,
              const_cast<paddle::Tensor*>(&kv_cache));
        } else {
          return DecodeDSMLAWriteCacheFP8<paddle::DataType::FLOAT16>(
              meta_data,
              kv_nope,
              kv_pe,
              slot_mapping,
              stream,
              const_cast<paddle::Tensor*>(&kv_cache));
        }
      }
      default:
        PD_THROW("Unsupported dtype for DS MLA FP8 cache");
    }
  } else {
    // Standard MLA format (auto/bf16/fp16)
    switch (kv_pe.dtype()) {
      case paddle::DataType::BFLOAT16: {
        return PrefillDSMLAWriteCache<paddle::DataType::BFLOAT16>(
            meta_data,
            kv_nope,
            kv_pe,
            slot_mapping,
            scale_ptr,
            stream,
            const_cast<paddle::Tensor*>(&kv_cache));
      }
      case paddle::DataType::FLOAT16: {
        return PrefillDSMLAWriteCache<paddle::DataType::FLOAT16>(
            meta_data,
            kv_nope,
            kv_pe,
            slot_mapping,
            scale_ptr,
            stream,
            const_cast<paddle::Tensor*>(&kv_cache));
      }
      default:
        PD_THROW("Unsupported dtype for DS MLA cache");
    }
  }
  return {};
}

/**
 * Indexer K Quant and Cache entry point
 */
std::vector<paddle::Tensor> IndexerKQuantAndCacheKernel(
    const paddle::Tensor& k,
    const paddle::Tensor& kv_cache,
    const paddle::Tensor& slot_mapping,
    const int64_t quant_block_size,
    const std::string& scale_fmt) {
  cudaStream_t stream = k.stream();
  int num_tokens = k.dims()[0];
  int head_dim = k.dims()[1];
  int cache_block_size = kv_cache.dims()[1];
  int cache_stride = kv_cache.dims()[2];
  bool use_ue8m0 = scale_fmt == "ue8m0";

  switch (k.dtype()) {
    case paddle::DataType::BFLOAT16: {
      return IndexerKQuantAndCache<paddle::DataType::BFLOAT16>(
          k,
          slot_mapping,
          head_dim,
          quant_block_size,
          cache_block_size,
          cache_stride,
          use_ue8m0,
          stream,
          const_cast<paddle::Tensor*>(&kv_cache));
    }
    case paddle::DataType::FLOAT16: {
      return IndexerKQuantAndCache<paddle::DataType::FLOAT16>(
          k,
          slot_mapping,
          head_dim,
          quant_block_size,
          cache_block_size,
          cache_stride,
          use_ue8m0,
          stream,
          const_cast<paddle::Tensor*>(&kv_cache));
    }
    default:
      PD_THROW("Unsupported dtype for Indexer K Quant");
  }
  return {};
}

/**
 * Gather Indexer K from Quant Cache entry point
 */
std::vector<paddle::Tensor> CpGatherIndexerKQuantCacheKernel(
    const paddle::Tensor& kv_cache,
    paddle::Tensor& dst_k,
    paddle::Tensor& dst_scale,
    const paddle::Tensor& block_table,
    const paddle::Tensor& cu_seq_lens) {
  cudaStream_t stream = kv_cache.stream();
  CpGatherIndexerKQuantCache(
      kv_cache, dst_k, dst_scale, block_table, cu_seq_lens, stream);
  return {};
}

//==============================================================================
// Paddle Custom Operator Registration
//==============================================================================

PD_BUILD_STATIC_OP(ds_mla_write_cache)
    .Inputs({"kv_nope",
             "kv_pe",
             "kv_cache",
             "slot_mapping",
             paddle::Optional("scale")})
    .Outputs({"kv_cache_out"})
    .SetInplaceMap({{"kv_cache", "kv_cache_out"}})
    .Attrs({"cache_quant_type_str: std::string", "is_prefill: bool"})
    .SetKernelFn(PD_KERNEL(DSMLAWriteCacheKernel));

PD_BUILD_STATIC_OP(indexer_k_quant_and_cache)
    .Inputs({"k", "kv_cache", "slot_mapping"})
    .Outputs({"kv_cache_out"})
    .SetInplaceMap({{"kv_cache", "kv_cache_out"}})
    .Attrs({"quant_block_size: int64_t", "scale_fmt: std::string"})
    .SetKernelFn(PD_KERNEL(IndexerKQuantAndCacheKernel));

PD_BUILD_STATIC_OP(cp_gather_indexer_k_quant_cache)
    .Inputs({"kv_cache", "dst_k", "dst_scale", "block_table", "cu_seq_lens"})
    .Outputs({"dst_k_out", "dst_scale_out"})
    .SetInplaceMap({{"dst_k", "dst_k_out"}, {"dst_scale", "dst_scale_out"}})
    .SetKernelFn(PD_KERNEL(CpGatherIndexerKQuantCacheKernel));
