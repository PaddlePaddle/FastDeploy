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

#include "helper.h"
#include "append_attn/utils.cuh"
// #include "gqa_decoder_flash_rewrite_cache.h"

// template <typename T, int VecSize = 8>
__global__ void pack_qk_write_cache(
    const __nv_bfloat16 * q,
    const __nv_bfloat16 * k,
    __nv_bfloat16 * out_q,
    __nv_bfloat16 * cache_k,
    const int *block_tables,
    const int *batch_id_per_token,
    const int *seq_lens_encoder,
    const int *seq_lens_decoder,
    const int *cu_seqlens_q,
    const int *cu_seqlens_k,
    const int q_share_heads,
    const int k_share_tokens,
    const int elem_cnt,
    const int block_num_pre_batch,
    const int q_heads,
    const int kv_heads,
    const int head_dim,
    const int block_size){
   
    // using LoadT = AlignedVector<T, VecSize>;
    // LoadT src_vec;
    

    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int step = gridDim.x * blockDim.x;
    const int head_dim_vector8 = head_dim / 8; // = 16
    // __shared__  __nv_bfloat16 q_shared_group[kv_heads * head_dim];
    // const int num_block_pre_batch = block_num_pre_batch;

    for (int64_t linear_index = global_id; linear_index < (elem_cnt/8); linear_index += step) {
        const int bid = linear_index / 64; 
        const int tid = linear_index % 64; //thread层只保留一个warp,剩下划到block_id上
        const int k_head_id = tid/16;   
        const int v8tid = tid%16;    //一个warp的thread，分为两个half warp，每16个线程负责一个head_dim
        




        int batch_id      = bid / (indexer_top_k/2 * kv_heads);
        int indexer_kv_id = bid % (indexer_top_k/2 * kv_heads) / kv_heads * 2; //每个warp读两个token，因为token是连续存储的
        int head_id       = bid % (indexer_top_k/2 * kv_heads) % kv_heads;

        if (indexer_kv_id >= indexer_top_k || batch_id >= bsz || head_id >= kv_heads) return;
        if (seq_lens_decoder[batch_id] <= 0) return;
        if (indexer_kv_id+dim_id > (seq_lens_decoder[batch_id]+1)) return;

        const int q_token_start_id = cu_seqlens_q[batch_id];
        const int total_seq = cu_seqlens_q[bsz];
        
        // 从token_sparse_index中取到当前Q选中的kv-cache的总 ID；
        // [head,q_token,k_len]
        // const int tmp = head_id*(total_seq*indexer_top_k) + q_token_start_id*indexer_top_k + indexer_kv_id + dim_id;
        const int tmp = q_token_start_id*kv_heads*indexer_top_k + head_id*indexer_top_k + indexer_kv_id + dim_id;
        const int cache_kv_id = token_sparse_index[tmp];
        if (cache_kv_id ==-1) return;

        //定位该kv-cache的总 ID 在cache_k/v的哪个block_id中,并进一步得知在该block_id中的具体位置
        const int block_id = block_tables[batch_id*block_num_pre_batch + cache_kv_id / block_size];
        if (block_id == -1) return;
        const int cache_kv_id_inblock = cache_kv_id % block_size;
        
        const int src_id = block_id*kv_heads*block_size*head_dim_vector8 + head_id*block_size*head_dim_vector8 + cache_kv_id_inblock*head_dim_vector8 + v8tid;
        // const int dst_id = batch_id*indexer_top_k*kv_heads*head_dim_vector8 + (indexer_kv_id + dim_id) *kv_heads*head_dim_vector8 + head_id*head_dim_vector8 + v8tid;
        const int dst_id = (num_block_pre_batch*batch_id + ((indexer_kv_id+dim_id)/block_size))*kv_heads*block_size*head_dim_vector8 + head_id*block_size*head_dim_vector8 + ((indexer_kv_id+dim_id)%block_size)*head_dim_vector8 + v8tid;
        

        const float4* cache_k_8 =  reinterpret_cast<const float4*>(cache_k);
        const float4* cache_v_8 =  reinterpret_cast<const float4*>(cache_v);
        float4* key_new_8 = reinterpret_cast<float4*>(key_new);
        float4* value_new_8 = reinterpret_cast<float4*>(value_new);
        key_new_8[dst_id] = cache_k_8[src_id];
        value_new_8[dst_id] = cache_v_8[src_id];
        

    }
}


void IndexerPackQKwriteCache(
    const paddle::Tensor& q,
    const paddle::Tensor& k,
    paddle::Tensor& out_q,
    paddle::Tensor& cache_k,
    const paddle::Tensor& block_tables,
    const paddle::Tensor& batch_id_per_token,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& cu_seqlens_q,
    const paddle::Tensor& cu_seqlens_k,
    const int q_share_heads,
    const int k_share_tokens,
){
    
    auto stream = q.stream();
    const int block_num_pre_batch = block_tables.dims()[1];
    auto num_tokens = q.dims()[0];
    const int batch = block_tables.dims()[0];
    const int q_heads = q.dims()[1];
    const int kv_heads = k.dims()[1];
    const int head_dim = q.dims()[3];
    
    const int elem_cnt = div(num_tokens,k_share_tokens) * kv_heads * head_dim; //5242880 /8 = 655360
    const int block_size = 64;
    int grid_size = num_tokens;//indexer_top_k;
    constexpr int PackSize = 8; //16 / sizeof(T);
    const int pack_num = elem_cnt / PackSize;

    dim3 block_dim(64);// 64 = 2*32 = 4*16 
                       // ～ 4 heads *（16*8）dim

    pack_qk_write_cache<<<grid_size,block_dim,0,stream>>>(
        reinterpret_cast<const __nv_bfloat16 *>(q.data<phi::dtype::bfloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(k.data<phi::dtype::bfloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(out_q.data<phi::dtype::bfloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(cache_k.data<phi::dtype::bfloat16>()),
        block_tables.data<int32_t>(),
        batch_id_per_token.data<int32_t>(),
        seq_lens_encoder.data<int32_t>(),
        seq_lens_decoder.data<int32_t>(),
        cu_seqlens_q.data<int32_t>(),
        cu_seqlens_k.data<int32_t>(),
        q_share_heads,
        k_share_tokens,
        elem_cnt,
        block_num_pre_batch,
        q_heads,
        kv_heads,
        head_dim,
        block_size);

}

// PD_BUILD_STATIC_OP(gqa_decoder_flash_rewrite_cache)
//     .Inputs({"cache_k",
//              "cache_v",
//              "key_new",
//              "value_new",
//              "token_sparse_index",
//              "block_tables",
//              "seq_lens_decoder",
//              "cu_seqlens_q"})
//     .SetKernelFn(PD_KERNEL(GQAFlashRewriteCache));

