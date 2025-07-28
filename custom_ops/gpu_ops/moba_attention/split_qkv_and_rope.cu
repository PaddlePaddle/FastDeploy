#include "paddle/extension.h"
#include "moba_attention_utils.hpp"
#include "moba_attention.h"

namespace moba {
template <typename input_type, int moba_block_size, int kBlockM, int kMaxN, int tokens_per_block, bool need_k_mean>
__global__ void fused_block_mean_and_rope_kernel(
        const input_type *qkv_input,
        const input_type *qkv_bias,
        input_type *k_gate_mean,
        input_type *q_input,
        input_type *k_input,
        input_type *v_input,
        const float *rope_sin_cos,
        const int *seq_len_encoder,
        const int *seq_len_decoder,
        const int *cu_seq_q,
        const int *cu_seq_k,
        const int max_seq_q,
        const int max_seq_k,
        const int head_num,
        const int kv_head_num,
        const int max_input_length) {

    constexpr int kPackSize = 16 / sizeof(input_type);
    constexpr int kHeadDim = 128;

    using src_type = Vec<input_type, kPackSize>;

    using rope_type = Vec<float, kPackSize / 2>;
    using pack_half = std::conditional_t<std::is_same<input_type, cutlass::half_t>::value, __half2, nv_bfloat162>;

    __align__(16) __shared__ input_type local_sum_mem[128 / 32 * kHeadDim];

    const int bidb = blockIdx.x;
    const int bidh = blockIdx.y;
    const int bidt_q = blockIdx.z * tokens_per_block;
    const int bidt_v = blockIdx.z * tokens_per_block;
    const int bidt_k = need_k_mean ? blockIdx.z * moba_block_size : blockIdx.z * tokens_per_block;
    const int tidx = threadIdx.x;
    const int lane_id = tidx % 32;
    const int warp_id = tidx / 32;
    const int seq_len = seq_len_encoder[bidb];
    const int seq_len_start = seq_len_decoder[bidb];

    if (seq_len == 0) {
        return;
    }

    const int all_head_num = head_num + 2 * kv_head_num;
    const int hidden = all_head_num * kHeadDim;

    const int row_idx = tidx / (kHeadDim / kPackSize);
    const int col_idx = tidx % (kHeadDim / kPackSize);

    const int bias_idx = bidh * kHeadDim + col_idx * kPackSize;

    src_type src, src_bias;
    rope_type sin, cos;

    const bool need_add_bias = qkv_bias != nullptr;

    if (need_add_bias) {
        src_bias.load_from(qkv_bias + bias_idx);
    }

    if (bidh < head_num) {
        const int cur_token = bidt_q + row_idx;
        const float * cos_rope = rope_sin_cos + (cur_token + seq_len_start) * (kHeadDim / 2) + col_idx * (kPackSize / 2);
        const float * sin_rope = cos_rope + max_input_length * (kHeadDim / 2);

        if (cur_token < seq_len) {
            src.load_from(qkv_input + cu_seq_q[bidb] * hidden + bias_idx + cur_token * hidden);

            if (need_add_bias) {
                src.add(src_bias);
            }

            sin.load_from(sin_rope);
            cos.load_from(cos_rope);
            apply_rotary_embedding<input_type, kPackSize>(src, cos, sin);

            src.store_to(q_input + (cu_seq_q[bidb] + cur_token) * head_num * kHeadDim + bias_idx);
        }
    } else if (bidh < head_num + kv_head_num) {
        if constexpr (!need_k_mean) {
            const int cur_token = bidt_k + row_idx;
            const float * cos_rope = rope_sin_cos + (cur_token + seq_len_start) * (kHeadDim / 2) + col_idx * (kPackSize / 2);
            const float * sin_rope = cos_rope + max_input_length * (kHeadDim / 2);

            if (cur_token < seq_len) {
                src.load_from(qkv_input + cu_seq_q[bidb] * hidden + bias_idx + cur_token * hidden);

                if (need_add_bias) {
                    src.add(src_bias);
                }

                sin.load_from(sin_rope);
                cos.load_from(cos_rope);
                apply_rotary_embedding<input_type, kPackSize>(src, cos, sin);

                src.store_to(k_input + (cu_seq_k[bidb] + cur_token) * head_num * kHeadDim + bias_idx- head_num * kHeadDim);
            }
        } else {
            if (bidt_k >= seq_len) {
                return;
            }

            src_type local_sum;
            local_sum.set_zero();

            const input_type* qkv = qkv_input + cu_seq_q[bidb] * hidden + bias_idx;

            for (int i = 0; i < moba_block_size; i += tokens_per_block) {
                const int cur_token = bidt_k + i + row_idx;
                if (cur_token < seq_len) {
                    src.load_from(qkv + cur_token * hidden);

                    if (need_add_bias) {
                        src.add(src_bias);
                    }
                    const float * cos_rope = rope_sin_cos + (cur_token + seq_len_start) * (kHeadDim / 2) + col_idx * (kPackSize / 2);
                    const float * sin_rope = cos_rope + max_input_length * (kHeadDim / 2);
                    sin.load_from(sin_rope);
                    cos.load_from(cos_rope);

                    apply_rotary_embedding<input_type, kPackSize>(src, cos, sin);

                    src.store_to(k_input + (cu_seq_k[bidb] + cur_token) * kv_head_num * kHeadDim + bias_idx - head_num * kHeadDim);

                    local_sum.add(src);
                }
            }

            src_type neighbor;

            #pragma unroll
            for (int i = 0; i < kPackSize; i+=2) {
                *reinterpret_cast<int32_t*>(neighbor.data.elt + i) = __shfl_down_sync(0xffffffff, *reinterpret_cast<int32_t*>(local_sum.data.elt + i), 16);
            }

            local_sum.add(neighbor);

            if (lane_id < 16) {
                local_sum.store_to(local_sum_mem + warp_id * kHeadDim + lane_id * kPackSize);
            }

            __syncthreads();

            pack_half * local_sum_mem_half = reinterpret_cast<pack_half*>(local_sum_mem);

            pack_half local_sum_half = local_sum_mem_half[tidx];


            if (tidx < kHeadDim / 2) {

                #pragma unroll
                for (int i = 1; i < 4; i++) {
                    local_sum_half += local_sum_mem_half[tidx + i * (kHeadDim / 2)];
                }

                float inv_tokens_sum = fdividef(1.0f, min(seq_len - bidt_k, moba_block_size));

                local_sum_half *= float_2_half2<input_type>(inv_tokens_sum);

                const int store_mean_idx = ((bidb * kMaxN + blockIdx.z + seq_len_start / moba_block_size) * kv_head_num * kHeadDim + (bidh - head_num) * kHeadDim) / 2 + tidx;

                reinterpret_cast<pack_half*>(k_gate_mean)[store_mean_idx] = local_sum_half;
            }
        }
    } else {
        const int cur_token = bidt_v + row_idx;

        if (cur_token < seq_len) {
            src.load_from(qkv_input + cu_seq_q[bidb] * hidden + bias_idx + cur_token * hidden);
            if (need_add_bias) {
                src.add(src_bias);
            }

            src.store_to(v_input + (cu_seq_k[bidb] + cur_token) * kv_head_num * kHeadDim + bias_idx - (head_num + kv_head_num) * kHeadDim);
        }
    }
}

template <typename input_type, int moba_block_size, int kBlockM, int kMaxN>
void fused_block_mean_and_rope(
        const input_type *qkv_input,
        const input_type *qkv_bias,
        input_type *k_gate_mean,
        input_type *q_input,
        input_type *k_input,
        input_type *v_input,
        const float *rope_sin_cos,
        const int *seq_len_encoder,
        const int *seq_len_decoder,
        const int *cu_seq_q,
        const int *cu_seq_k,
        const int max_seq_q,
        const int max_seq_k,
        const int head_num,
        const int kv_head_num,
        const int bsz,
        const int max_input_length,
        cudaStream_t stream) {

    static_assert(moba_block_size >= 64, "moba_block_size must be at least 64");
    constexpr int kPackSize = 16 / sizeof(input_type);
    constexpr int kHeadDim = 128;
    constexpr int kThreads = 128;
    constexpr int tokens_per_block = kThreads / (kHeadDim / kPackSize);
    dim3 grid_dims;
    grid_dims.x = bsz;
    grid_dims.y = head_num + 2 * kv_head_num;
    grid_dims.z = (max_seq_q + tokens_per_block - 1) / tokens_per_block;

    if (k_gate_mean != nullptr) {
        fused_block_mean_and_rope_kernel<input_type, moba_block_size, kBlockM, kMaxN, tokens_per_block, true>
        <<<grid_dims, kThreads, 0, stream>>>(
            qkv_input,
            qkv_bias,
            k_gate_mean,
            q_input,
            k_input,
            v_input,
            rope_sin_cos,
            seq_len_encoder,
            seq_len_decoder,
            cu_seq_q,
            cu_seq_k,
            max_seq_q,
            max_seq_k,
            head_num,
            kv_head_num,
            max_input_length);
    } else {
        fused_block_mean_and_rope_kernel<input_type, moba_block_size, kBlockM, kMaxN, tokens_per_block, false>
        <<<grid_dims, kThreads, 0, stream>>>(
            qkv_input,
            qkv_bias,
            k_gate_mean,
            q_input,
            k_input,
            v_input,
            rope_sin_cos,
            seq_len_encoder,
            seq_len_decoder,
            cu_seq_q,
            cu_seq_k,
            max_seq_q,
            max_seq_k,
            head_num,
            kv_head_num,
            max_input_length);
    }
}

void FusedBlockMeanAndRope(
        const paddle::Tensor& qkv_out,
        const paddle::Tensor& k_block_means,
        const paddle::Tensor& q_input,
        const paddle::Tensor& k_input,
        const paddle::Tensor& v_input,
        const paddle::Tensor& rotary_embs,
        const paddle::Tensor& seq_len_encoder,
        const paddle::Tensor& seq_len_decoder,
        const paddle::Tensor& cu_seq_q,
        const paddle::Tensor& cu_seq_k,
        const paddle::optional<paddle::Tensor>& qkv_bias,
        const int head_num,
        const int kv_head_num,
        const int head_dim,
        const int max_input_length,
        const int max_seq_q,
        const int max_seq_k,
        const std::string &cache_quant_type_str) {

    constexpr int kBlockM = 128;
    constexpr int kBlockN = 128;
    constexpr int kMobaBlockSize = 128;
    constexpr int kMaxN = 1024;

    if (k_input.dtype() == paddle::DataType::FLOAT16) {
        using T = phi::dtype::float16;
        using cute_type = typename cuteType<T>::type;
        fused_block_mean_and_rope<cute_type, kMobaBlockSize, kBlockM, kMaxN>(
            reinterpret_cast<cute_type *>(const_cast<T*>(qkv_out.data<T>())),
            qkv_bias ? reinterpret_cast<cute_type *>(const_cast<T*>(qkv_bias.get().data<T>())) : nullptr,
            reinterpret_cast<cute_type *>(const_cast<T*>(k_block_means.data<T>())),
            reinterpret_cast<cute_type*>(const_cast<T*>(q_input.data<T>())),
            reinterpret_cast<cute_type*>(const_cast<T*>(k_input.data<T>())),
            reinterpret_cast<cute_type*>(const_cast<T*>(v_input.data<T>())),
            rotary_embs.data<float>(),
            seq_len_encoder.data<int>(),
            seq_len_decoder.data<int>(),
            cu_seq_q.data<int>(),
            cu_seq_k.data<int>(),
            max_seq_q,
            max_seq_k,
            head_num,
            kv_head_num,
            seq_len_encoder.dims()[0],
            max_input_length,
            qkv_out.stream());
    } else if (k_input.dtype() == paddle::DataType::BFLOAT16) {
        using T = phi::dtype::bfloat16;
        using cute_type = typename cuteType<T>::type;
        fused_block_mean_and_rope<cute_type, kMobaBlockSize, kBlockM, kMaxN>(
            reinterpret_cast<cute_type *>(const_cast<T*>(qkv_out.data<T>())),
            qkv_bias ? reinterpret_cast<cute_type *>(const_cast<T*>(qkv_bias.get().data<T>())) : nullptr,
            reinterpret_cast<cute_type *>(const_cast<T*>(k_block_means.data<T>())),
            reinterpret_cast<cute_type*>(const_cast<T*>(q_input.data<T>())),
            reinterpret_cast<cute_type*>(const_cast<T*>(k_input.data<T>())),
            reinterpret_cast<cute_type*>(const_cast<T*>(v_input.data<T>())),
            rotary_embs.data<float>(),
            seq_len_encoder.data<int>(),
            seq_len_decoder.data<int>(),
            cu_seq_q.data<int>(),
            cu_seq_k.data<int>(),
            max_seq_q,
            max_seq_k,
            head_num,
            kv_head_num,
            seq_len_encoder.dims()[0],
            max_input_length,
            qkv_out.stream());
    }
}


template <typename T, int kBlockSize, int kHeadDim>
__global__ void get_kv_from_cache_c16_kernel(
        T * k_input,
        T * v_input,
        const int * seq_len_encoder,
        const int * seq_len_decoder,
        const int * cu_seq_k,
        const T * cache_k,
        const T * cache_v,
        const int * block_tables,
        const int kv_head_num,
        const int head_dim,
        const int batch_size,
        const int max_input_length,
        const int max_blocks_per_seq) {

    const int block_idx = blockIdx.x;
    int bidh = blockIdx.y;
    const int bidb = blockIdx.z;
    const int seq_len = seq_len_decoder[bidb] + seq_len_encoder[bidb];
    const int tidx = threadIdx.x;
    const int base_token_idx = block_idx * kBlockSize;

    if (base_token_idx >= seq_len || seq_len_encoder[bidb] == 0) {
        return;
    }

    constexpr int kPackSize = 16 / sizeof(T);

    const int row_idx = tidx / (kHeadDim / kPackSize);
    const int col_idx = tidx % (kHeadDim / kPackSize) * kPackSize;
    const int physical_block_number = block_tables[bidb * max_blocks_per_seq + block_idx];


    const int ramian_tokens = seq_len - base_token_idx;

    if (bidh < kv_head_num) {
        const int cache_offset = physical_block_number * kv_head_num * kBlockSize * kHeadDim + bidh * kBlockSize * kHeadDim + col_idx;
        const int base_store_idx = (base_token_idx + cu_seq_k[bidb]) * kv_head_num * kHeadDim + bidh * kHeadDim + col_idx;
        #pragma unroll
        for (int i = row_idx; i < kBlockSize; i += 128 / (kHeadDim / kPackSize)) {
            if (i < ramian_tokens) {
                *reinterpret_cast<float4*>(k_input + base_store_idx + i * kv_head_num * kHeadDim) = *reinterpret_cast<const float4*>(cache_k + cache_offset + i * kHeadDim);
            }
        }
    } else {
        bidh -= kv_head_num;
        const int cache_offset = physical_block_number * kv_head_num * kBlockSize * kHeadDim + bidh * kBlockSize * kHeadDim + col_idx;
        const int base_store_idx = (base_token_idx + cu_seq_k[bidb]) * kv_head_num * kHeadDim + bidh * kHeadDim + col_idx;
        #pragma unroll
        for (int i = row_idx; i < kBlockSize; i += 128 / (kHeadDim / kPackSize)) {
            if (i < ramian_tokens) {
                *reinterpret_cast<float4*>(v_input + base_store_idx + i * kv_head_num * kHeadDim) = *reinterpret_cast<const float4*>(cache_v + cache_offset + i * kHeadDim);
            }
        }
    }
}

template <typename T>
void get_kv_from_cache(
        T * k_input,
        T * v_input,
        const int * seq_len_encoder,
        const int * seq_len_decoder,
        const int * cu_seq_k,
        const void * cache_k,
        const void * cache_v,
        const int * block_tables,
        const T * cache_k_dequant_scale,
        const T * cache_v_dequant_scale,
        const T * cache_k_zero_points,
        const T * cache_v_zero_points,
        const int kv_head_num,
        const int head_dim,
        const int max_seq_k,
        const int batch_size,
        const int max_input_length,
        const int max_blocks_per_seq,
        const std::string &cache_quant_type_str,
        cudaStream_t stream) {

    constexpr int kThreads = 128;
    constexpr int kHeadDim = 128;
    assert(kHeadDim == head_dim);
    constexpr int kBlockSize = 64;
    if (cache_quant_type_str == "none") {
        dim3 grid_dims;
        grid_dims.x = (max_seq_k + kBlockSize - 1) / kBlockSize;
        grid_dims.y = kv_head_num * 2;
        grid_dims.z = batch_size;
        get_kv_from_cache_c16_kernel<T, kBlockSize, kHeadDim><<<grid_dims, kThreads, 0, stream>>>(
            k_input,
            v_input,
            seq_len_encoder,
            seq_len_decoder,
            cu_seq_k,
            reinterpret_cast<const T*>(cache_k),
            reinterpret_cast<const T*>(cache_v),
            block_tables,
            kv_head_num,
            head_dim,
            batch_size,
            max_input_length,
            max_blocks_per_seq);
    }
    // cudaDeviceSynchronize();
    //     auto err = cudaGetLastError();
    //     std::cout << "debug get_kv_from_cache err = " << err << ", str = " << cudaGetErrorString(err) << std::endl;
}

void GetKVFromCache(
        const paddle::Tensor& k_input,
        const paddle::Tensor& v_input,
        const paddle::Tensor& cu_seq_k,
        const paddle::Tensor& seq_len_encoder,
        const paddle::Tensor& seq_len_decoder,
        const paddle::Tensor& cache_k,
        const paddle::Tensor& cache_v,
        const paddle::Tensor& block_tables,
        const paddle::optional<paddle::Tensor>& cache_k_dequant_scale,
        const paddle::optional<paddle::Tensor>& cache_v_dequant_scale,
        const paddle::optional<paddle::Tensor>& cache_k_zero_points,
        const paddle::optional<paddle::Tensor>& cache_v_zero_points,
        const int head_num,
        const int kv_head_num,
        const int head_dim,
        const int max_input_length,
        const int max_seq_k,
        const std::string &cache_quant_type_str) {

    if (k_input.dtype() == paddle::DataType::FLOAT16) {
        using T = phi::dtype::float16;
        using cute_type = typename cuteType<T>::type;
        get_kv_from_cache<cute_type>(
            reinterpret_cast<cute_type*>(const_cast<T*>(k_input.data<T>())),
            reinterpret_cast<cute_type*>(const_cast<T*>(v_input.data<T>())),
            seq_len_encoder.data<int>(),
            seq_len_decoder.data<int>(),
            cu_seq_k.data<int>(),
            cache_k.data(),
            cache_v.data(),
            block_tables.data<int>(),
            cache_k_dequant_scale ? reinterpret_cast<cute_type*>(const_cast<T*>(cache_k_dequant_scale.get().data<T>())) : nullptr,
            cache_v_dequant_scale ? reinterpret_cast<cute_type*>(const_cast<T*>(cache_v_dequant_scale.get().data<T>())) : nullptr,
            cache_k_zero_points ? reinterpret_cast<cute_type*>(const_cast<T*>(cache_k_zero_points.get().data<T>())) : nullptr,
            cache_v_zero_points ? reinterpret_cast<cute_type*>(const_cast<T*>(cache_v_zero_points.get().data<T>())) : nullptr,
            kv_head_num,
            head_dim,
            max_seq_k,
            seq_len_encoder.dims()[0],
            max_input_length,
            block_tables.dims()[1],
            cache_quant_type_str,
            k_input.stream());
    } else if (k_input.dtype() == paddle::DataType::BFLOAT16) {
        using T = phi::dtype::bfloat16;
        using cute_type = typename cuteType<T>::type;
        get_kv_from_cache<cute_type>(
            reinterpret_cast<cute_type*>(const_cast<T*>(k_input.data<T>())),
            reinterpret_cast<cute_type*>(const_cast<T*>(v_input.data<T>())),
            seq_len_encoder.data<int>(),
            seq_len_decoder.data<int>(),
            cu_seq_k.data<int>(),
            cache_k.data(),
            cache_v.data(),
            block_tables.data<int>(),
            cache_k_dequant_scale ? reinterpret_cast<cute_type*>(const_cast<T*>(cache_k_dequant_scale.get().data<T>())) : nullptr,
            cache_v_dequant_scale ? reinterpret_cast<cute_type*>(const_cast<T*>(cache_v_dequant_scale.get().data<T>())) : nullptr,
            cache_k_zero_points ? reinterpret_cast<cute_type*>(const_cast<T*>(cache_k_zero_points.get().data<T>())) : nullptr,
            cache_v_zero_points ? reinterpret_cast<cute_type*>(const_cast<T*>(cache_v_zero_points.get().data<T>())) : nullptr,
            kv_head_num,
            head_dim,
            max_seq_k,
            seq_len_encoder.dims()[0],
            max_input_length,
            block_tables.dims()[1],
            cache_quant_type_str,
            k_input.stream());
    }
}

__global__ void get_cur_cu_seq_len_k_kernel(
        const int* __restrict__ seq_lens_encoder,
        const int* __restrict__ seq_lens_decoder,
        const int* __restrict__ seq_lens_this_time,
        int* __restrict__ cu_seqlens_k,
        int* __restrict__ cu_seq_q_pack,
        int* __restrict__ q_pack_tokens,
        const int pack_size,
        const int bsz) {

    int total_tokens = 0;
    cu_seqlens_k[0] = 0;
    cu_seq_q_pack[0] = 0;

    for (uint32_t bid = 0; bid < bsz; bid++) {
        int cache_len = seq_lens_decoder[bid];
        const int q_len = seq_lens_encoder[bid];
        if (q_len <= 0) {
            cache_len = 0;
        }
        total_tokens += (cache_len + q_len);
        cu_seqlens_k[bid + 1] = total_tokens;
        cu_seq_q_pack[bid + 1] = cu_seq_q_pack[bid] + (q_len + pack_size -1) / pack_size * pack_size;
    }
    q_pack_tokens[0] = cu_seq_q_pack[bsz];
}

std::vector<paddle::Tensor> GetCurCuSeqLenk(
        const paddle::Tensor& seq_lens_encoder,
        const paddle::Tensor& seq_lens_decoder,
        const paddle::Tensor& seq_lens_this_time,
        const int pack_size) {
    auto stream = seq_lens_decoder.stream();
    auto place = seq_lens_decoder.place();
    int bsz = seq_lens_this_time.shape()[0];

    paddle::Tensor cu_seq_q_pack = paddle::empty({bsz + 1}, paddle::DataType::INT32, place);
    paddle::Tensor cu_seqlens_k = paddle::empty({bsz + 1}, paddle::DataType::INT32, place);
    paddle::Tensor q_pack_tokens = paddle::empty({1}, paddle::DataType::INT32, place);

    get_cur_cu_seq_len_k_kernel<<<1, 1, 0, stream>>>(
        seq_lens_encoder.data<int>(),
        seq_lens_decoder.data<int>(),
        seq_lens_this_time.data<int>(),
        cu_seqlens_k.data<int>(),
        cu_seq_q_pack.data<int>(),
        q_pack_tokens.data<int>(),
        pack_size,
        bsz
    );

    auto q_pack_tokens_cpu = q_pack_tokens.copy_to(paddle::CPUPlace(), true);
    return {cu_seq_q_pack, cu_seqlens_k, q_pack_tokens_cpu};
}


template <typename T, int moba_block_size, int kHeadDim, int kMaxN>
__global__ void moba_mlp_einsum_kernel(
        const T * src_data,
        const T * weight_data,
        const int * seq_lens_encoder,
        const int * seq_lens_decoder,
        const int * cu_seq_k,
        T * dst_data,
        const int head_num) {

    constexpr int kPackSize = 16 / sizeof(T);
    const int block_idx = blockIdx.x;
    const int bidh = blockIdx.y;
    const int bidb = blockIdx.z;
    const int tidx = threadIdx.x;
    const int lane_id = tidx % 32;
    const int warp_id = tidx / 32;

    __align__(16) __shared__ T local_sum_mem[128 / 32 * kHeadDim];

    const int seq_len_encoder = seq_lens_encoder[bidb];
    const int seq_len_decoder = seq_len_encoder + seq_lens_decoder[bidb];

    const int seq_len_this_block = seq_len_decoder - block_idx * moba_block_size;

    if (seq_len_encoder == 0 || seq_len_this_block <= 0) {
        return;
    }


    using SrcType = Vec<T, kPackSize>;

    constexpr int tidx_per_row = kHeadDim / kPackSize;

    const int row_idx = tidx / tidx_per_row;
    const int col_idx = tidx % tidx_per_row * kPackSize;

    const int src_base_idx = cu_seq_k[bidb] * head_num * kHeadDim + block_idx * moba_block_size * head_num * kHeadDim + bidh * kHeadDim + row_idx * head_num * kHeadDim + col_idx;
    const int weight_base_idx = bidh * kHeadDim * moba_block_size + row_idx * kHeadDim + col_idx;

    constexpr int step = 128 / tidx_per_row;

    SrcType sums, src, weight;

    sums.set_zero();

    for (int i = 0; i < moba_block_size; i += step) {
        if (i >= seq_len_this_block) {
            break;
        }
        src.load_from(src_data + src_base_idx + i * head_num * kHeadDim);
        weight.load_from(weight_data + weight_base_idx + i * kHeadDim);
        sums.fma(src, weight);
    }

    SrcType neighbor;

    #pragma unroll
    for (int i = 0; i < kPackSize; i+=2) {
        *reinterpret_cast<int32_t*>(neighbor.data.elt + i) = __shfl_down_sync(0xffffffff, *reinterpret_cast<int32_t*>(sums.data.elt + i), 16);
    }

    sums.add(neighbor);

    if (lane_id < 16) {
        sums.store_to(local_sum_mem + warp_id * kHeadDim + lane_id * kPackSize);
    }

    __syncthreads();
    using pack_half = std::conditional_t<std::is_same<T, phi::dtype::float16>::value, __half2, nv_bfloat162>;
    pack_half * local_sum_mem_half = reinterpret_cast<pack_half*>(local_sum_mem);

    if (tidx < kHeadDim / 2) {
        pack_half local_sum_half = local_sum_mem_half[tidx];
        #pragma unroll
        for (int i = 1; i < 4; i++) {
            local_sum_half += local_sum_mem_half[tidx + i * (kHeadDim / 2)];
        }
        local_sum_mem_half[tidx] = local_sum_half;
    }

    __syncthreads();

    const int store_row_id = tidx / (kHeadDim / kPackSize);
    const int store_col_id = tidx % (kHeadDim / kPackSize) * kPackSize;

    sums.load_from(local_sum_mem + store_col_id);

    const int base_store_idx = bidb * kMaxN * head_num * kHeadDim + (block_idx * (moba_block_size / 128) + store_row_id) * head_num * kHeadDim + bidh * kHeadDim + store_col_id;

    sums.store_to(dst_data + base_store_idx);
}


template <typename T, int kHeadDim, int kMaxN>
void moba_mlp_einsum(
        const T * src_data,
        const T * weight_data,
        const int * seq_lens_encoder,
        const int * seq_lens_decoder,
        const int * cu_seq_k,
        T * dst_data,
        const int moba_block_size,
        const int max_seq_len,
        const int head_num,
        const int batch_size,
        cudaStream_t stream) {

    dim3 grid_dims;
    grid_dims.x = (max_seq_len + moba_block_size - 1) / moba_block_size;
    grid_dims.y = head_num;
    grid_dims.z = batch_size;

    if (moba_block_size == 1024) {
        moba_mlp_einsum_kernel<T, 1024, kHeadDim, kMaxN><<<grid_dims, 128, 0, stream>>>(
            src_data,
            weight_data,
            seq_lens_encoder,
            seq_lens_decoder,
            cu_seq_k,
            dst_data,
            head_num);
    } else {
        PADDLE_THROW(phi::errors::Unimplemented(
            "MobaMlpEinsum not implemented for moba_block_size = %d", moba_block_size));
    }

}


std::vector<paddle::Tensor> MobaMlpEinsum(
        const paddle::Tensor& k_input,
        const paddle::Tensor& attn_gate_weight,
        const paddle::Tensor& seq_lens_encoder,
        const paddle::Tensor& seq_lens_decoder,
        const paddle::Tensor& cu_seq_k,
        const int max_seq_len,
        const int kv_head_num) {

    const int kHeadDim = 128;
    const int kMaxN = 1024;
    const int moba_block_size = attn_gate_weight.dims()[1];
    const int batch_size = seq_lens_encoder.dims()[0];
    paddle::Tensor k_gate_weight = paddle::zeros({batch_size, kMaxN, kv_head_num, kHeadDim}, k_input.dtype(), k_input.place());

    if (k_input.dtype() == paddle::DataType::FLOAT16) {
        using T = phi::dtype::float16;
        moba_mlp_einsum<T, kHeadDim, kMaxN>(
            const_cast<T*>(k_input.data<T>()),
            const_cast<T*>(attn_gate_weight.data<T>()),
            const_cast<int*>(seq_lens_encoder.data<int>()),
            const_cast<int*>(seq_lens_decoder.data<int>()),
            const_cast<int*>(cu_seq_k.data<int>()),
            k_gate_weight.data<T>(),
            moba_block_size,
            max_seq_len,
            kv_head_num,
            batch_size,
            k_input.stream()
        );
    } else if (k_input.dtype() == paddle::DataType::BFLOAT16) {
        using T = phi::dtype::bfloat16;
        moba_mlp_einsum<T, kHeadDim, kMaxN>(
            const_cast<T*>(k_input.data<T>()),
            const_cast<T*>(attn_gate_weight.data<T>()),
            const_cast<int*>(seq_lens_encoder.data<int>()),
            const_cast<int*>(seq_lens_decoder.data<int>()),
            const_cast<int*>(cu_seq_k.data<int>()),
            k_gate_weight.data<T>(),
            moba_block_size,
            max_seq_len,
            kv_head_num,
            batch_size,
            k_input.stream()
        );
    }
    return {k_gate_weight};
}


}

PD_BUILD_OP(fused_block_mean_and_rope)
    .Inputs({
        "qkv_out",
        "k_block_means",
        "q_input",
        "k_input",
        "v_input",
        "rotary_embs",
        "seq_len_encoder",
        "seq_len_decoder",
        "cu_seq_q",
        "cu_seq_k",
        paddle::Optional("qkv_bias")})
    .Attrs({
        "head_num: int",
        "kv_head_num: int",
        "head_dim: int",
        "max_input_length: int",
        "max_seq_q: int",
        "max_seq_k: int",
        "cache_quant_type_str: std::string"})
    .Outputs({"q_input_out", "k_input_out", "v_input_out", "k_block_means_out"})
    .SetInplaceMap({{"q_input", "q_input_out"},
                    {"k_input", "k_input_out"},
                    {"v_input", "v_input_out"},
                    {"k_block_means", "k_block_means_out"}})
    .SetKernelFn(PD_KERNEL(moba::FusedBlockMeanAndRope));

PD_BUILD_OP(get_kv_from_cache)
    .Inputs({
        "k_input",
        "v_input",
        "cu_seq_k",
        "seq_len_encoder",
        "seq_len_decoder",
        "cache_k",
        "cache_v",
        "block_tables",
        paddle::Optional("cache_k_dequant_scale"),
        paddle::Optional("cache_v_dequant_scale"),
        paddle::Optional("cache_k_zero_points"),
        paddle::Optional("cache_v_zero_points")})
    .Attrs({
        "head_num: int",
        "kv_head_num: int",
        "head_dim: int",
        "max_input_length: int",
        "max_seq_k: int",
        "cache_quant_type_str: std::string"})
    .Outputs({"k_input_out", "v_input_out"})
    .SetInplaceMap({{"k_input", "k_input_out"},
                    {"v_input", "v_input_out"}})
    .SetKernelFn(PD_KERNEL(moba::GetKVFromCache));

PD_BUILD_OP(get_cur_cu_seq_len_k)
    .Inputs({
            "seq_lens_encoder",
            "seq_lens_decoder",
            "seq_lens_this_time"})
    .Attrs({
        "pack_size: int"})
    .Outputs({"cu_seq_q_pack", "cu_seqlens_k", "q_pack_tokens"})
    .SetKernelFn(PD_KERNEL(moba::GetCurCuSeqLenk));


PD_BUILD_OP(moba_mlp_einsum)
    .Inputs({
        "k_input",
        "attn_gate_weight",
        "seq_lens_encoder",
        "seq_lens_decoder",
        "cu_seq_k"})
    .Attrs({
        "max_seq_len: int",
        "kv_head_num: int"})
    .Outputs({"k_gate"})
    .SetKernelFn(PD_KERNEL(moba::MobaMlpEinsum));
