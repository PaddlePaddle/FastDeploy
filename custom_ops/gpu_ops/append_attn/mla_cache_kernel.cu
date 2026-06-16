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
#pragma once

#include "mla_cache_kernel.cuh"
#include "helper.h"
#include "remote_cache_kv_ipc.h"

template <paddle::DataType T>
std::vector<paddle::Tensor> MLAWriteCache(
    const paddle::Tensor& kv_nope,
    const paddle::Tensor& kv_pe,
    const paddle::Tensor& seq_lens,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& batch_id_per_token,
    const paddle::Tensor& cu_seqlens_q,
    const paddle::Tensor& block_tables,
    const paddle::Tensor& slot_mapping,
    const paddle::optional<paddle::Tensor>& kv_signal_data,
    cudaStream_t& stream,
    const std::string& cache_quant_type_str,
    paddle::Tensor* kv_cache) {
  typedef PDTraits<T> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t;

  const auto& kv_nope_dims = kv_nope.dims();
  const auto& kv_cache_dims = (*kv_cache).dims();
  auto max_blocks_per_seq = block_tables.dims()[1];
  auto num_tokens = kv_nope_dims[0];
  auto block_size = kv_cache_dims[2];
  auto kv_num_heads = kv_cache_dims[1];
  auto nope_size = 
    kv_nope_dims[kv_nope_dims.size() - 1] / kv_num_heads;
  auto all_size = kv_cache_dims[3];
  int pe_size = all_size - nope_size;  
  const uint32_t elem_nums = num_tokens * kv_num_heads * all_size;
  constexpr int PackSize = 16 / sizeof(DataType_);
  const int pack_num = elem_nums / PackSize;
  const int blocksize = 128;
  int grid_size = 1;
  GetNumBlocks<128>(pack_num, &grid_size);

  if (cache_quant_type_str == "cache_fp8") {
    using CT = __nv_fp8_e4m3;
    prefill_absorb_cache_kernel<DataType_, PackSize, CT>
        <<<grid_size, blocksize, 0, stream>>>(
            reinterpret_cast<DataType_*>(
                const_cast<data_t*>(kv_nope.data<data_t>())),
            reinterpret_cast<DataType_*>(
                const_cast<data_t*>(kv_pe.data<data_t>())),
            reinterpret_cast<CT*>(kv_cache->data<uint8_t>()),
            block_tables.data<int>(),
            slot_mapping.data<int64_t>(),
            batch_id_per_token.data<int>(),
            cu_seqlens_q.data<int>(),
            seq_lens.data<int>(),
            seq_lens_decoder.data<int>(),
            max_blocks_per_seq,
            kv_num_heads,
            nope_size,
            pe_size,
            block_size,
            elem_nums);
  } else if (cache_quant_type_str == "none") {
    prefill_absorb_cache_kernel<DataType_, PackSize, DataType_>
        <<<grid_size, blocksize, 0, stream>>>(
            reinterpret_cast<DataType_*>(
                const_cast<data_t*>(kv_nope.data<data_t>())),
            reinterpret_cast<DataType_*>(
                const_cast<data_t*>(kv_pe.data<data_t>())),
            reinterpret_cast<DataType_*>(kv_cache->data<data_t>()),
            block_tables.data<int>(),
            slot_mapping.data<int64_t>(),
            batch_id_per_token.data<int>(),
            cu_seqlens_q.data<int>(),
            seq_lens.data<int>(),
            seq_lens_decoder.data<int>(),
            max_blocks_per_seq,
            kv_num_heads,
            nope_size,
            pe_size,
            block_size,
            elem_nums);
  } else {
    PD_THROW("Unsupported cache_quant_type_str type: %s.",
             cache_quant_type_str.c_str());
  }

  const char* fmt_write_cache_completed_signal_str =
      std::getenv("FLAGS_fmt_write_cache_completed_signal");
  const char* FLAGS_use_pd_disaggregation_per_chunk =
      std::getenv("FLAGS_use_pd_disaggregation_per_chunk");

  if (fmt_write_cache_completed_signal_str &&
      (std::strcmp(fmt_write_cache_completed_signal_str, "true") == 0 ||
       std::strcmp(fmt_write_cache_completed_signal_str, "1") == 0)) {
    if (FLAGS_use_pd_disaggregation_per_chunk &&
        (std::strcmp(FLAGS_use_pd_disaggregation_per_chunk, "true") == 0 ||
         std::strcmp(FLAGS_use_pd_disaggregation_per_chunk, "1") == 0)) {
      cudaLaunchHostFunc(
          stream,
          &(RemoteCacheKvIpc::
                save_cache_kv_complete_signal_layerwise_per_query),
          (void*)nullptr);
    } else {
      if (kv_signal_data) {
        cudaLaunchHostFunc(
            stream,
            &RemoteCacheKvIpc::save_cache_kv_complete_signal_layerwise,
            (void*)(const_cast<int64_t*>(
                kv_signal_data.get().data<int64_t>())));
      }
    }
  }
  return {};
}

std::vector<paddle::Tensor> MLAWriteCacheKernel(
    const paddle::Tensor& kv_nope,
    const paddle::Tensor& kv_pe,
    const paddle::Tensor& kv_cache,
    const paddle::Tensor& seq_lens,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& batch_id_per_token,
    const paddle::Tensor& cu_seqlens_q,
    const paddle::Tensor& block_tables,
    const paddle::Tensor& slot_mapping,
    const paddle::optional<paddle::Tensor>& kv_signal_data,
    const std::string& cache_quant_type_str) {
  cudaStream_t stream = kv_pe.stream();
  switch (kv_pe.dtype()) {
    case paddle::DataType::BFLOAT16: {
      return MLAWriteCache<paddle::DataType::BFLOAT16>(
          kv_nope,
          kv_pe,
          seq_lens,
          seq_lens_decoder,
          batch_id_per_token,
          cu_seqlens_q,
          block_tables,
          slot_mapping,
          kv_signal_data,
          stream,
          cache_quant_type_str,
          const_cast<paddle::Tensor*>(&kv_cache));
    }
    case paddle::DataType::FLOAT16: {
      return MLAWriteCache<paddle::DataType::FLOAT16>(
          kv_nope,
          kv_pe,
          seq_lens,
          seq_lens_decoder,
          batch_id_per_token,
          cu_seqlens_q,
          block_tables,
          slot_mapping,
          kv_signal_data,
          stream,
          cache_quant_type_str,
          const_cast<paddle::Tensor*>(&kv_cache));
    }
  }
  return {};
}

PD_BUILD_STATIC_OP(mla_write_cache)
    .Inputs({"kv_nope",
             "kv_pe",
             "kv_cache",
             "seq_lens",
             "seq_lens_decoder",
             "batch_id_per_token",
             "cu_seqlens_q",
             "block_tables",
             "slot_mapping",
             paddle::Optional("kv_signal_data")})
    .Outputs({"kv_cache_out"})
    .SetInplaceMap({{"kv_cache", "kv_cache_out"}})
    .Attrs({"cache_quant_type_str: std::string"})
    .SetKernelFn(PD_KERNEL(MLAWriteCacheKernel));
