// // Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
// //
// // Licensed under the Apache License, Version 2.0 (the "License");
// // you may not use this file except in compliance with the License.
// // You may obtain a copy of the License at
// //
// //     http://www.apache.org/licenses/LICENSE-2.0
// //
// // Unless required by applicable law or agreed to in writing, software
// // distributed under the License is distributed on an "AS IS" BASIS,
// // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// // See the License for the specific language governing permissions and
// // limitations under the License.

// #include "append_attn/encoder_write_cache_with_rope_impl.cuh"
// #include "helper.h"
// #include "paddle/extension.h"
// #include "paddle/phi/backends/context_pool.h"
// #include "paddle/phi/core/memory/memcpy.h"
// #include "remote_cache_kv_ipc.h"

// template <typename T, int VecSize = 1>
// __global__ void GQAVariableLengthRotarySplitKernel(
//     const T *qkv,
//     const float *cos_emb,
//     const float *sin_emb,
//     const float *q_norm_weight,
//     const float *k_norm_weight,
//     const int *batch_id_per_token,
//     const int *cu_seqlens_q,
//     const int *seq_lens,
//     const int *seq_lens_decoder,
//     const int *cu_seqlens_k,
//     T *qkv_out,
//     T *q,
//     T *k,
//     T *v,
//     const int64_t elem_cnt,
//     const int q_num_head,
//     const int kv_num_head,
//     const int seq_len,
//     const int last_dim,
//     const bool rope_3d,
//     const float rms_norm_eps) {
//   using LoadT = AlignedVector<T, VecSize>;
//   constexpr int HalfVecSize = VecSize / 2;
//   using LoadEmbT = AlignedVector<float, HalfVecSize>;
//   using LoadFloat = AlignedVector<float, VecSize>;
//   LoadT src_vec;
//   LoadEmbT cos_emb_vec;
//   LoadEmbT sin_emb_vec;
//   LoadFloat tmp_vec;
//   LoadFloat q_norm_vec, k_norm_vec;
//   int64_t global_warp_idx = blockDim.y * blockIdx.x + threadIdx.y;
//   int64_t all_warp_num = gridDim.x * blockDim.y;
//   const int half_lastdim = last_dim / 2;
//   const int offset =
//       (q_num_head + kv_num_head * 2) * last_dim;  // for all q,k,v
//   const int all_head_num = elem_cnt / last_dim;
//   for (int gloabl_hi = global_warp_idx; gloabl_hi < all_head_num;
//        gloabl_hi += all_warp_num) {
//     int64_t linear_index =
//         gloabl_hi * last_dim + threadIdx.x * VecSize;  // 全局index
//     const int token_idx =
//         linear_index / offset;  // token id(第几个token,不分qkv)
//     const int ori_bi = batch_id_per_token[token_idx];  // 第几个batch
//     if (seq_lens[ori_bi] == 0) continue;
//     const int bias = linear_index % offset;
//     const int hi = bias / last_dim;
//     const int h_bias = bias % last_dim;

//     const int ori_seq_id =
//         (token_idx - cu_seqlens_q[ori_bi]) +
//         seq_lens_decoder
//             [ori_bi];  // 在当前seq中的id(拼接了seq到一个batch的情况下有效)
//     const int64_t emb_idx =
//         ori_seq_id * half_lastdim + h_bias / 2;  // embedding的id
//     const int64_t base_idx =
//         token_idx * (q_num_head + 2 * kv_num_head) * last_dim + hi * last_dim +
//         h_bias;
//     Load<T, VecSize>(&qkv[base_idx], &src_vec);
//     const int kv_write_idx = cu_seqlens_k[ori_bi] + ori_seq_id;
//     int64_t base_split_idx;
//     T *out_p = nullptr;
//     if (hi < q_num_head) {
//       base_split_idx =
//           token_idx * q_num_head * last_dim + hi * last_dim + h_bias;
//       out_p = q;
//     } else if (hi < q_num_head + kv_num_head) {
//       base_split_idx = kv_write_idx * kv_num_head * last_dim +
//                        (hi - q_num_head) * last_dim + h_bias;
//       out_p = k;
//     } else {
//       out_p = v;
//       base_split_idx = kv_write_idx * kv_num_head * last_dim +
//                        (hi - q_num_head - kv_num_head) * last_dim + h_bias;
//     }

//     // TODO check this correct or not
//     int64_t new_emb_idx =
//         rope_3d ? emb_idx + ori_bi * last_dim * seq_len : emb_idx;
//     float thread_m2 = 0.0f;
//     float warp_m2 = 0.0f;

//     if (q_norm_weight && k_norm_weight) {
//       if (hi < q_num_head + kv_num_head) {  // only q and k need rope
//         Load<float, HalfVecSize>(&cos_emb[new_emb_idx], &cos_emb_vec);
//         Load<float, HalfVecSize>(&sin_emb[new_emb_idx], &sin_emb_vec);
// #pragma unroll
//         for (int i = 0; i < HalfVecSize; i++) {
//           const float input_left = static_cast<float>(src_vec[2 * i]);
//           const float input_right = static_cast<float>(src_vec[2 * i + 1]);
//           const float cos_tmp = cos_emb_vec[i];
//           const float sin_tmp = sin_emb_vec[i];
//           float tmp1 = input_left * cos_tmp - input_right * sin_tmp;
//           float tmp2 = input_right * cos_tmp + input_left * sin_tmp;
//           tmp_vec[2 * i] = tmp1;
//           tmp_vec[2 * i + 1] = tmp2;
//           thread_m2 += tmp1 * tmp1 + tmp2 * tmp2;
//         }
//       }
//       WelfordWarpAllReduce<float, 32>(thread_m2, &warp_m2);  // 单个head的标准差

//       if (hi < q_num_head + kv_num_head) {  // only q and k need norm
//         float row_variance = max(warp_m2 / last_dim, 0.0f);
//         float row_inv_var = Rsqrt(row_variance + rms_norm_eps);
//         if (hi < q_num_head) {
//           Load<float, VecSize>(&q_norm_weight[threadIdx.x * VecSize],
//                                &q_norm_vec);
// #pragma unroll
//           for (int i = 0; i < VecSize; i++) {
//             src_vec[i] =
//                 static_cast<T>(tmp_vec[i] * row_inv_var * q_norm_vec[i]);
//           }
//         } else {
//           Load<float, VecSize>(&k_norm_weight[threadIdx.x * VecSize],
//                                &k_norm_vec);
//           for (int i = 0; i < VecSize; i++) {
//             src_vec[i] =
//                 static_cast<T>(tmp_vec[i] * row_inv_var * k_norm_vec[i]);
//           }
//         }
//       }
//     } else {
//       if (hi < q_num_head + kv_num_head) {
//         Load<float, HalfVecSize>(&cos_emb[new_emb_idx], &cos_emb_vec);
//         Load<float, HalfVecSize>(&sin_emb[new_emb_idx], &sin_emb_vec);
// #pragma unroll
//         for (int i = 0; i < HalfVecSize; i++) {
//           const float input_left = static_cast<float>(src_vec[2 * i]);
//           const float input_right = static_cast<float>(src_vec[2 * i + 1]);
//           const float cos_tmp = cos_emb_vec[i];
//           const float sin_tmp = sin_emb_vec[i];
//           src_vec[2 * i] =
//               static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
//           src_vec[2 * i + 1] =
//               static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
//         }
//       }
//     }
//     Store<T, VecSize>(src_vec, &qkv_out[base_idx]);
//     Store<T, VecSize>(src_vec, &out_p[base_split_idx]);
//   }
// }

// template <typename T, uint32_t HEAD_DIM>
// void gqa_rotary_qk_split_variable(
//     T *qkv_out,  // [token_num, 3, num_head, dim_head]
//     T *q,
//     T *k,
//     T *v,
//     const T *qkv_input,
//     const float *rotary_emb,  // [2, 1, 1, seq_len, dim_head / 2]
//     const float *q_norm_weight,
//     const float *k_norm_weight,
//     const int *batch_id_per_token,
//     const int *seq_lens_encoder,
//     const int *seq_lens_decoder,
//     const int *cu_seqlens_q,
//     const int *cu_seqlens_k,
//     const int token_num,
//     const int num_heads,
//     const int kv_num_heads,
//     const int seq_len,
//     const int input_output_len,
//     const int dim_head,
//     const bool rope_3d,
//     const float rms_norm_eps,
//     const cudaStream_t &stream) {
// //   assert(dim_head == 128 && "dim_head must be 128");
//   int64_t elem_nums = token_num * (num_heads + 2 * kv_num_heads) * dim_head;

// //   constexpr int HEAD_DIM = 128;
//   constexpr int PackSize = HEAD_DIM / kWarpSize;
//   const int pack_num = elem_nums / PackSize;
//   const int blocksize = 128;
//   int grid_size = 1;
//   GetNumBlocks<128>(pack_num, &grid_size);
//   dim3 block_size(kWarpSize, blocksize / kWarpSize);

//   const float *cos_emb = rotary_emb;
//   const float *sin_emb = rotary_emb + input_output_len * dim_head / 2;
//   launchWithPdlWhenEnabled(GQAVariableLengthRotarySplitKernel<T, PackSize>,
//                            grid_size,
//                            block_size,
//                            0,
//                            stream,
//                            qkv_input,
//                            cos_emb,
//                            sin_emb,
//                            q_norm_weight,
//                            k_norm_weight,
//                            batch_id_per_token,
//                            cu_seqlens_q,
//                            seq_lens_encoder,
//                            seq_lens_decoder,
//                            cu_seqlens_k,
//                            qkv_out,
//                            q,
//                            k,
//                            v,
//                            elem_nums,
//                            num_heads,
//                            kv_num_heads,
//                            seq_len,
//                            dim_head,
//                            rope_3d,
//                            rms_norm_eps);
// }

// template <typename T,
//           typename CacheT,
//           uint32_t HEAD_DIM,
//           uint32_t BLOCK_SIZE,
//           uint32_t NUM_WARPS = 4>
// __global__ void append_cache_kv_c16(const T *__restrict__ cache_k,
//                                     const T *__restrict__ cache_v,
//                                     T *__restrict__ k_out,
//                                     T *__restrict__ v_out,
//                                     const int *__restrict__ seq_lens_this_time,
//                                     const int *__restrict__ seq_lens_decoder,
//                                     const int *__restrict__ cu_seqlens_k,
//                                     const int *__restrict__ block_tables,
//                                     const int *batch_ids,
//                                     const int *tile_ids_per_batch,
//                                     const int max_blocks_per_seq,
//                                     const int kv_num_heads) {
//   // start_kv_idx: start kv_idx current block
//   // batch_id：block's batch_id
//   // TODO: 1.scale preload 2.frag_dq_T reuse 3.pipeline 4.store aligned 5.cacheT
//   // with template（int8/fp8)
//   const uint32_t tile_idx = blockIdx.x, kv_head_idx = blockIdx.z;
//   const uint32_t tid = threadIdx.x, wid = threadIdx.y;

//   const uint32_t batch_id = batch_ids[tile_idx];
//   const uint32_t start_kv_idx = tile_ids_per_batch[tile_idx] * BLOCK_SIZE;
//   const uint32_t end_idx = seq_lens_decoder[batch_id] - start_kv_idx;
//   if (seq_lens_this_time[batch_id] <= 0) {
//     return;
//   }

//   const int *cur_block_table = block_tables + batch_id * max_blocks_per_seq;
//   uint32_t block_id = cur_block_table[start_kv_idx / BLOCK_SIZE];
//   // cache_kv idx
//   uint32_t kv_h_stride = BLOCK_SIZE * HEAD_DIM;
//   uint32_t block_stride = kv_num_heads * kv_h_stride;
//   const CacheT *cur_cache_k =
//       cache_k + block_id * block_stride + kv_head_idx * kv_h_stride;
//   const CacheT *cur_cache_v =
//       cache_v + block_id * block_stride + kv_head_idx * kv_h_stride;

//   // k_out v_out idx
//   uint32_t kv_t_stride = kv_num_heads * HEAD_DIM;
//   T *k_write_ptr =
//       k_out + (cu_seqlens_k[batch_id] + start_kv_idx) * kv_t_stride;
//   T *v_write_ptr =
//       v_out + (cu_seqlens_k[batch_id] + start_kv_idx) * kv_t_stride;

//   uint32_t kv_frag[4];
//   T *frag_dq_T = reinterpret_cast<T *>(kv_frag);

//   constexpr uint32_t num_vecs_per_head =
//       HEAD_DIM / num_elems_per_128b<CacheT>();
//   constexpr uint32_t inv_kv_stride = 8 / num_vecs_per_head;

//   extern __shared__ uint8_t smem[];
//   smem_t k_smem(smem);
//   uint32_t k_smem_offset_w =
//       smem_t::get_permuted_offset<num_vecs_per_head, inv_kv_stride>(
//           wid * 4 + tid / 8, tid % 8);  // 4 * 4 per warp

//   uint32_t k_smem_offset_r =
//       smem_t::get_permuted_offset<num_vecs_per_head, inv_kv_stride>(
//           wid * 16 + 8 * (tid / 16) + tid % 8, (tid % 16) / 8);

//   uint32_t k_read_idx =
//       (wid * 4 + tid / 8) * HEAD_DIM + tid % 8 * num_elems_per_128b<CacheT>();

//   // load k_smem 64 rows 128 cols
//   for (int fz = 0; fz < 4;
//        fz++) {  // 4 rows pre warp once, 16 rows all 4 warps once, need 4 iter
//     for (int fy = 0; fy < 2; fy++) {  // 8 * 128b = 64 * bf16 once, need 2 iter
//       k_smem.load_128b_async<SharedMemFillMode::kNoFill>(
//           k_smem_offset_w, cur_cache_k + k_read_idx, end_idx > 0);
//       k_smem_offset_w = k_smem.advance_offset_by_column<8, num_vecs_per_head>(
//           k_smem_offset_w, fy);
//       k_read_idx += 8 * num_elems_per_128b<CacheT>();
//     }
//     k_smem_offset_w =
//         k_smem.advance_offset_by_row<4 * NUM_WARPS, num_vecs_per_head>(
//             k_smem_offset_w) -
//         16;
//     k_read_idx += 4 * NUM_WARPS * HEAD_DIM - 16 * num_elems_per_128b<CacheT>();
//   }
//   commit_group();
//   wait_group<0>();
//   __syncthreads();

//   // deal k_smem 64 rows 128 cols
//   for (int fz = 0; fz < 1;
//        fz++) {  // 16 rows pre warp once, 64 rows all 4 warps once, need 1 iter
//     uint32_t row_idx = wid * 16 + tid / 4;
//     for (int fy = 0; fy < 8; fy++) {  // 2 * 128b = 16 * bf16 once, need 8 iter
//       uint32_t col_idx = fy * 16 + tid % 4 * 2;
//       k_smem.ldmatrix_m8n8x4(k_smem_offset_r, kv_frag);
//       // layout
//       /***
//         r0c0,r0c1, r0c8,r0c9
//         r8c0,r8c1, r8c8,r8c9
//       ***/
//       T *k_tile_ptr0 = k_write_ptr + row_idx * kv_t_stride +
//                        kv_head_idx * HEAD_DIM + col_idx;
//       T *k_tile_ptr1 = k_tile_ptr0 + 8 * kv_t_stride;

//       if (row_idx < end_idx) {
//         k_tile_ptr0[0] = frag_dq_T[0];
//         k_tile_ptr0[1] = frag_dq_T[1];
//         k_tile_ptr0[8] = frag_dq_T[2];
//         k_tile_ptr0[9] = frag_dq_T[3];
//       }

//       if (row_idx + 8 < end_idx) {
//         k_tile_ptr1[0] = frag_dq_T[4];
//         k_tile_ptr1[1] = frag_dq_T[5];
//         k_tile_ptr1[8] = frag_dq_T[6];
//         k_tile_ptr1[9] = frag_dq_T[7];
//       }
//       k_smem_offset_r = k_smem.advance_offset_by_column<2, num_vecs_per_head>(
//           k_smem_offset_r, fy);
//     }
//     k_smem_offset_r =
//         k_smem.advance_offset_by_row<16 * NUM_WARPS, num_vecs_per_head>(
//             k_smem_offset_r) -
//         16;
//   }

//   // ================v================
//   smem_t v_smem(smem + BLOCK_SIZE * HEAD_DIM * sizeof(CacheT));
//   uint32_t v_smem_offset_w =
//       smem_t::get_permuted_offset<num_vecs_per_head, inv_kv_stride>(
//           wid * 4 + tid / 8, tid % 8);  // 4 * 4 per warp
//   uint32_t v_smem_offset_r =
//       smem_t::get_permuted_offset<num_vecs_per_head, inv_kv_stride>(
//           wid * 16 + 8 * (tid / 16) + tid % 8, (tid % 16) / 8);

//   uint32_t v_read_idx =
//       (wid * 4 + tid / 8) * HEAD_DIM + tid % 8 * num_elems_per_128b<CacheT>();

//   // load v_smem 64 rows 128 cols
//   for (int fz = 0; fz < 4; fz++) {    // // 4 rows pre warp once, 16 rows all 4
//                                       // warps once, need 4 iter
//     for (int fy = 0; fy < 2; fy++) {  // 8 * 128b = 64 * bf16 once, need 2 iter
//       v_smem.load_128b_async<SharedMemFillMode::kNoFill>(
//           v_smem_offset_w, cur_cache_v + v_read_idx, end_idx > 0);
//       v_smem_offset_w = v_smem.advance_offset_by_column<8, num_vecs_per_head>(
//           v_smem_offset_w, fy);
//       v_read_idx += 8 * num_elems_per_128b<CacheT>();
//     }
//     v_smem_offset_w =
//         v_smem.advance_offset_by_row<4 * NUM_WARPS, num_vecs_per_head>(
//             v_smem_offset_w) -
//         16;
//     v_read_idx += 4 * NUM_WARPS * HEAD_DIM - 16 * num_elems_per_128b<CacheT>();
//   }
//   commit_group();
//   wait_group<0>();
//   __syncthreads();

//   // deal v_smem 64 rows 128 cols
//   for (int fz = 0; fz < 1;
//        fz++) {  //  16 rows pre warp once, 64 rows all 4 warps once, need 1 iter
//     uint32_t row_idx = wid * 16 + tid / 4;
//     for (int fy = 0; fy < 8; fy++) {  // 2 * 128b = 16 * bf16 once, need 8 iter
//       uint32_t col_idx = fy * 16 + tid % 4 * 2;
//       v_smem.ldmatrix_m8n8x4(v_smem_offset_r, kv_frag);
//       // layout
//       /***
//         r0c0,r0c1, r0c8,r0c9
//         r8c0,r8c1, r8c8,r8c9
//       ***/
//       T *v_tile_ptr0 = v_write_ptr + row_idx * kv_t_stride +
//                        kv_head_idx * HEAD_DIM + col_idx;
//       T *v_tile_ptr1 = v_tile_ptr0 + 8 * kv_t_stride;

//       if (row_idx < end_idx) {
//         v_tile_ptr0[0] = frag_dq_T[0];
//         v_tile_ptr0[1] = frag_dq_T[1];
//         v_tile_ptr0[8] = frag_dq_T[2];
//         v_tile_ptr0[9] = frag_dq_T[3];
//       }

//       if (row_idx + 8 < end_idx) {
//         v_tile_ptr1[0] = frag_dq_T[4];
//         v_tile_ptr1[1] = frag_dq_T[5];
//         v_tile_ptr1[8] = frag_dq_T[6];
//         v_tile_ptr1[9] = frag_dq_T[7];
//       }
//       v_smem_offset_r = v_smem.advance_offset_by_column<2, num_vecs_per_head>(
//           v_smem_offset_r, fy);
//     }
//     v_smem_offset_r =
//         v_smem.advance_offset_by_row<16 * NUM_WARPS, num_vecs_per_head>(
//             v_smem_offset_r) -
//         16;
//   }
// }


// template <typename T, uint32_t HEAD_DIM, uint32_t BLOCK_SIZE>
// void AppendCacheKV(const paddle::Tensor &cache_k,
//                    const paddle::Tensor &cache_v,
//                    const paddle::Tensor &cache_k_dequant_scales,
//                    const paddle::Tensor &cache_v_dequant_scales,
//                    const paddle::Tensor &cache_k_zp,
//                    const paddle::Tensor &cache_v_zp,
//                    const paddle::Tensor &seq_lens_this_time,
//                    const paddle::Tensor &seq_lens_decoder,
//                    const paddle::Tensor &cu_seqlens_k,
//                    const paddle::Tensor &block_tables,
//                    const paddle::Tensor &cache_batch_ids,
//                    const paddle::Tensor &cache_tile_ids_per_batch,
//                    const paddle::Tensor &cache_num_blocks_x,
//                    const int max_blocks_per_seq,
//                    const int kv_num_heads,
//                    const std::string &cache_quant_type,
//                    paddle::Tensor *k_out,
//                    paddle::Tensor *v_out,
//                    const cudaStream_t &stream) {
//   using NV_TYPE = typename cascade_attn_type_traits<T>::type;
//   constexpr int NUM_WARPS = 4;
//   int block_num = cache_num_blocks_x.data<int>()[0];
//   dim3 grids(block_num, 1, kv_num_heads);
//   dim3 blocks(32, NUM_WARPS);
//   if (cache_quant_type == "none") {
//     const uint32_t smem_size = BLOCK_SIZE * HEAD_DIM * sizeof(T) * 2;
//     auto kernel_func =
//         append_cache_kv_c16<NV_TYPE, NV_TYPE, HEAD_DIM, BLOCK_SIZE, NUM_WARPS>;

//     if (smem_size >= 48 * 1024) {
//       cudaFuncSetAttribute(
//           kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
//     }
//     launchWithPdlWhenEnabled(
//         kernel_func,
//         grids,
//         blocks,
//         smem_size,
//         stream,
//         reinterpret_cast<NV_TYPE *>(const_cast<T *>(cache_k.data<T>())),
//         reinterpret_cast<NV_TYPE *>(const_cast<T *>(cache_v.data<T>())),
//         reinterpret_cast<NV_TYPE *>(k_out->data<T>()),
//         reinterpret_cast<NV_TYPE *>(v_out->data<T>()),
//         seq_lens_this_time.data<int>(),
//         seq_lens_decoder.data<int>(),
//         cu_seqlens_k.data<int>(),
//         block_tables.data<int>(),
//         cache_batch_ids.data<int>(),
//         cache_tile_ids_per_batch.data<int>(),
//         max_blocks_per_seq,
//         kv_num_heads);
//   } else {
//     PADDLE_THROW("%s mode isn't implemented yet", cache_quant_type.c_str());
//   }
// }

// std::vector<paddle::Tensor> IndexerGQARopeWriteCacheKernel(
//     const paddle::Tensor &qkv,
//     const paddle::Tensor &key_cache,
//     const paddle::Tensor &value_cache,
//     const paddle::Tensor &cu_seqlens_q,
//     const paddle::Tensor &cu_seqlens_k,
//     const paddle::Tensor &rotary_embs,
//     const paddle::Tensor &seq_lens_this_time,
//     const paddle::Tensor &seq_lens_encoder,
//     const paddle::Tensor &seq_lens_decoder,
//     const paddle::Tensor &batch_id_per_token,
//     const paddle::Tensor &block_tables,
//     const paddle::Tensor &kv_batch_ids,
//     const paddle::Tensor &kv_tile_ids,
//     const paddle::Tensor &kv_num_blocks,
//     const paddle::Tensor &cache_batch_ids,
//     const paddle::Tensor &cache_tile_ids,
//     const paddle::Tensor &cache_num_blocks,
//     const paddle::optional<paddle::Tensor> &q_norm_weight,
//     const paddle::optional<paddle::Tensor> &k_norm_weight,
//     const paddle::optional<paddle::Tensor> &cache_k_quant_scales,
//     const paddle::optional<paddle::Tensor> &cache_v_quant_scales,
//     const paddle::optional<paddle::Tensor> &cache_k_dequant_scales,
//     const paddle::optional<paddle::Tensor> &cache_v_dequant_scales,
//     const paddle::optional<paddle::Tensor> &cache_k_zp,
//     const paddle::optional<paddle::Tensor> &cache_v_zp,
//     const paddle::optional<paddle::Tensor> &kv_signal_data,
//     const int kv_token_num,
//     const int max_seq_len,
//     const float rms_norm_eps,
//     const std::string &cache_quant_type,
//     const bool rope_3d) {
//   typedef PDTraits<paddle::DataType::BFLOAT16> traits_;
//   typedef typename traits_::DataType DataType_;
//   typedef typename traits_::data_t data_t;

//   const int kv_num_blocks_data = kv_num_blocks.data<int>()[0];
//   const auto &qkv_dims = qkv.dims();
//   const auto &key_cache_dims = key_cache.dims();
//   const int token_num = qkv_dims[0];
//   const int max_blocks_per_seq = block_tables.dims()[1];
//   const int block_size = key_cache.dims()[2];
//   const int batch_size = seq_lens_this_time.dims()[0];
//   const int kv_num_heads = key_cache_dims[1];
//   const int head_dim = cache_quant_type == "cache_int4_zp"
//                            ? key_cache_dims[3] * 2
//                            : key_cache_dims[3];
//   const int num_heads =
//       qkv_dims[qkv_dims.size() - 1] / head_dim - 2 * kv_num_heads;
//   const float softmax_scale = 1.f / sqrt(head_dim);

//   AppendAttnMetaData meta_data;
//   meta_data.token_nums = token_num;
//   meta_data.kv_num_heads = kv_num_heads;
//   meta_data.head_dims = head_dim;
//   meta_data.q_num_heads = num_heads;
//   meta_data.max_blocks_per_seq = max_blocks_per_seq;
//   meta_data.block_size = block_size;
//   meta_data.batch_size = seq_lens_this_time.dims()[0];

//   phi::GPUContext *dev_ctx = static_cast<phi::GPUContext *>(
//       phi::DeviceContextPool::Instance().Get(qkv.place()));

//   auto stream = qkv.stream();
//   paddle::Tensor qkv_out = GetEmptyTensor(qkv.dims(), qkv.dtype(), qkv.place());
//   paddle::Tensor q = GetEmptyTensor(
//       {token_num, num_heads, head_dim}, qkv.dtype(), qkv.place());
//   paddle::Tensor k = GetEmptyTensor(
//       {kv_token_num, kv_num_heads, head_dim}, qkv.dtype(), qkv.place());
//   paddle::Tensor v = GetEmptyTensor(
//       {kv_token_num, kv_num_heads, head_dim}, qkv.dtype(), qkv.place());

//   if(meta_data.head_dims ==128)
//   {
//     // rope
//     gqa_rotary_qk_split_variable<data_t,128>(
//         qkv_out.data<data_t>(),
//         q.data<data_t>(),
//         k.data<data_t>(),
//         v.data<data_t>(),
//         qkv.data<data_t>(),
//         rotary_embs.data<float>(),
//         q_norm_weight ? q_norm_weight.get().data<float>() : nullptr,
//         k_norm_weight ? k_norm_weight.get().data<float>() : nullptr,
//         batch_id_per_token.data<int>(),
//         seq_lens_encoder.data<int>(),
//         seq_lens_decoder.data<int>(),
//         cu_seqlens_q.data<int>(),
//         cu_seqlens_k.data<int>(),
//         token_num,
//         num_heads,
//         kv_num_heads,
//         max_seq_len,
//         rope_3d ? rotary_embs.dims()[3] : rotary_embs.dims()[2],
//         head_dim,
//         rope_3d,
//         rms_norm_eps,
//         stream);
//   }else if(meta_data.head_dims ==64)
//   {
//     gqa_rotary_qk_split_variable<data_t,64>(
//         qkv_out.data<data_t>(),
//         q.data<data_t>(),
//         k.data<data_t>(),
//         v.data<data_t>(),
//         qkv.data<data_t>(),
//         rotary_embs.data<float>(),
//         q_norm_weight ? q_norm_weight.get().data<float>() : nullptr,
//         k_norm_weight ? k_norm_weight.get().data<float>() : nullptr,
//         batch_id_per_token.data<int>(),
//         seq_lens_encoder.data<int>(),
//         seq_lens_decoder.data<int>(),
//         cu_seqlens_q.data<int>(),
//         cu_seqlens_k.data<int>(),
//         token_num,
//         num_heads,
//         kv_num_heads,
//         max_seq_len,
//         rope_3d ? rotary_embs.dims()[3] : rotary_embs.dims()[2],
//         head_dim,
//         rope_3d,
//         rms_norm_eps,
//         stream);
//   }else{
//     PADDLE_THROW("only support head_dim as 64 or 128");
//   }

//   if (token_num < kv_token_num) {
//     if(meta_data.head_dims ==128){
//          AppendCacheKV<data_t, 128, 64>(key_cache,
//                                    value_cache,
//                                    cache_k_dequant_scales.get(),
//                                    cache_v_dequant_scales.get(),
//                                    cache_k_zp.get(),
//                                    cache_v_zp.get(),
//                                    seq_lens_this_time,
//                                    seq_lens_decoder,
//                                    cu_seqlens_k,
//                                    block_tables,
//                                    cache_batch_ids,
//                                    cache_tile_ids,
//                                    cache_num_blocks,
//                                    max_blocks_per_seq,
//                                    kv_num_heads,
//                                    cache_quant_type,
//                                    &k,
//                                    &v,
//                                    stream);
//     }else if(meta_data.head_dims ==64){
//             AppendCacheKV<data_t, 64, 64>(key_cache,
//                                    value_cache,
//                                    cache_k_dequant_scales.get(),
//                                    cache_v_dequant_scales.get(),
//                                    cache_k_zp.get(),
//                                    cache_v_zp.get(),
//                                    seq_lens_this_time,
//                                    seq_lens_decoder,
//                                    cu_seqlens_k,
//                                    block_tables,
//                                    cache_batch_ids,
//                                    cache_tile_ids,
//                                    cache_num_blocks,
//                                    max_blocks_per_seq,
//                                    kv_num_heads,
//                                    cache_quant_type,
//                                    &k,
//                                    &v,
//                                    stream);
//     }else{
//       PADDLE_THROW("only support head_dim as 64 or 128");
//     }

//   }
//   // write cache
//   if (cache_quant_type == "none") {
//     CascadeAppendWriteCacheKVQKV<data_t>(
//         meta_data,
//         qkv_out,
//         block_tables,
//         batch_id_per_token,
//         cu_seqlens_q,
//         seq_lens_encoder,
//         seq_lens_decoder,
//         max_seq_len,
//         stream,
//         const_cast<paddle::Tensor *>(&key_cache),
//         const_cast<paddle::Tensor *>(&value_cache));
//   } else {
//     PD_THROW(
//         "cache_quant_type_str should be one of [none, cache_int8, cache_fp8, "
//         "cache_int4_zp]");
//   }
//   const char *fmt_write_cache_completed_signal_str =
//       std::getenv("FLAGS_fmt_write_cache_completed_signal");
//   const char *FLAGS_use_pd_disaggregation_per_chunk =
//       std::getenv("FLAGS_use_pd_disaggregation_per_chunk");
//   if (fmt_write_cache_completed_signal_str &&
//       (std::strcmp(fmt_write_cache_completed_signal_str, "true") == 0 ||
//        std::strcmp(fmt_write_cache_completed_signal_str, "1") == 0)) {
//     if (FLAGS_use_pd_disaggregation_per_chunk &&
//         (std::strcmp(FLAGS_use_pd_disaggregation_per_chunk, "true") == 0 ||
//          std::strcmp(FLAGS_use_pd_disaggregation_per_chunk, "1") == 0)) {
//       cudaLaunchHostFunc(
//           qkv.stream(),
//           &(RemoteCacheKvIpc::
//                 save_cache_kv_complete_signal_layerwise_per_query),
//           (void *)nullptr);
//     } else {
//       if (kv_signal_data) {
//         cudaLaunchHostFunc(
//             qkv.stream(),
//             &RemoteCacheKvIpc::save_cache_kv_complete_signal_layerwise,
//             (void *)(const_cast<int64_t *>(
//                 kv_signal_data.get().data<int64_t>())));
//       }
//     }
//   }
//   return {q, k, v, qkv_out};
// }

// PD_BUILD_STATIC_OP(indexer_gqa_rope_write_cache)
//     .Inputs({"qkv",
//              "key_cache",
//              "value_cache",
//              "cu_seqlens_q",
//              "cu_seqlens_k",
//              "rotary_embs",
//              "seq_lens_this_time",
//              "seq_lens_encoder",
//              "seq_lens_decoder",
//              "batch_id_per_token",
//              "block_tables",
//              "kv_batch_ids",
//              "kv_tile_ids_per_batch",
//              "kv_num_blocks",
//              "cache_batch_ids",
//              "cache_tile_ids_per_batch",
//              "cache_num_blocks",
//              paddle::Optional("q_norm_weight"),
//              paddle::Optional("k_norm_weight"),
//              paddle::Optional("cache_k_quant_scales"),
//              paddle::Optional("cache_v_quant_scales"),
//              paddle::Optional("cache_k_dequant_scales"),
//              paddle::Optional("cache_v_dequant_scales"),
//              paddle::Optional("cache_k_zp"),
//              paddle::Optional("cache_v_zp"),
//              paddle::Optional("kv_signal_data")})
//     .Outputs({"q", "k", "v", "qkv_out", "key_cache_out", "value_cache_out"})
//     .SetInplaceMap({{"key_cache", "key_cache_out"},
//                     {"value_cache", "value_cache_out"}})
//     .Attrs({"kv_token_num: int",
//             "max_seq_len: int",
//             "rms_norm_eps: float",
//             "cache_quant_type: std::string",
//             "rope_3d: bool"})
//     .SetKernelFn(PD_KERNEL(IndexerGQARopeWriteCacheKernel));
