#include "encoder_write_cache_with_rope_impl.cuh"
#include "helper.h"
#include "paddle/extension.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/core/memory/memcpy.h"
#include "remote_cache_kv_ipc.h"

template <typename T, int VecSize = 1>
__global__ void GQAVariableLengthRotarySplitKernel_Qwen3(
    const T *qkv,
    const float *cos_emb,
    const float *sin_emb,
    const int *batch_id_per_token,
    const int *cu_seqlens_q,
    const int *seq_lens_encoder,
    const int *seq_lens_decoder,
    const int *cu_seqlens_k,
    T *qkv_out,
    T *q,
    T *k,
    T *v,
    const int64_t elem_cnt,
    const int q_num_head,
    const int kv_num_head,
    const int max_model_len,
    const int head_dim) {
  using LoadT = AlignedVector<T, VecSize>;
  using LoadEmbT = AlignedVector<float, VecSize>;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;

  const int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int offset = (q_num_head + kv_num_head * 2) * (head_dim / 2);
  const int64_t loop_times = elem_cnt / 2;

  for (int64_t linear_index = global_thread_idx * VecSize;
       linear_index < loop_times;
       linear_index += gridDim.x * blockDim.x * VecSize) {
    const int token_idx = linear_index / offset;

    const int ori_bi = batch_id_per_token[token_idx];  // 第几个batch

    int cache_kv_len = seq_lens_decoder[ori_bi];
    // 这里其实是不需要处理的，但是由于FA3的bug，所以必须！
    if (seq_lens_encoder[ori_bi] == 0) cache_kv_len = 0;

    const int bias = linear_index % offset;
    const int hi = bias / (head_dim / 2);
    const int h_bias = bias % (head_dim / 2);
    // we should handle token_idx, hi 头 的 h_bias 部分！

    const int ori_seq_id =
        (token_idx - cu_seqlens_q[ori_bi]) +
        cache_kv_len;  // 在当前seq中的id(拼接了seq到一个batch的情况下有效)

    const int half_headdim = head_dim / 2;
    const int64_t emb_idx = ori_seq_id * head_dim + h_bias;  // embedding的id

    const int64_t read_idx =
        token_idx * (q_num_head + 2 * kv_num_head) * head_dim + hi * head_dim +
        h_bias;

    LoadT src_vec0;
    LoadT src_vec1;

    Load<T, VecSize>(&qkv[read_idx], &src_vec0);
    Load<T, VecSize>(&qkv[read_idx + 64], &src_vec1);

    const int kv_write_idx = cu_seqlens_k[ori_bi] + ori_seq_id;
    int64_t base_split_idx;
    T *out_p = nullptr;
    if (hi < q_num_head) {
      base_split_idx =
          token_idx * q_num_head * head_dim + hi * head_dim + h_bias;
      out_p = q;
    } else if (hi < q_num_head + kv_num_head) {
      base_split_idx = kv_write_idx * kv_num_head * head_dim +
                       (hi - q_num_head) * head_dim + h_bias;
      out_p = k;
    } else {
      out_p = v;
      base_split_idx = kv_write_idx * kv_num_head * head_dim +
                       (hi - q_num_head - kv_num_head) * head_dim + h_bias;
    }

    // TODO check this correct or not
    int64_t new_emb_idx = emb_idx;

    if (hi < q_num_head + kv_num_head) {
      Load<float, VecSize>(&cos_emb[new_emb_idx], &cos_emb_vec);
      Load<float, VecSize>(&sin_emb[new_emb_idx], &sin_emb_vec);
#pragma unroll
      for (int i = 0; i < VecSize; i++) {
        float input_left = static_cast<float>(src_vec0[i]);
        float input_right = static_cast<float>(src_vec1[i]);

        const float cos_tmp = cos_emb_vec[i];
        const float sin_tmp = sin_emb_vec[i];
        src_vec0[i] =
            static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
        src_vec1[i] =
            static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
      }
    }
    Store<T, VecSize>(src_vec0, &qkv_out[read_idx]);
    Store<T, VecSize>(src_vec0, &out_p[base_split_idx]);
    Store<T, VecSize>(src_vec1, &qkv_out[read_idx + 64]);
    Store<T, VecSize>(src_vec1, &out_p[base_split_idx + 64]);
  }
}

template <typename T>
void gqa_rotary_qk_split_variable_qwen3(T *qkv_out,
                                        T *q,
                                        T *k,
                                        T *v,
                                        const T *qkv_input,
                                        const float *rotary_emb,
                                        const int *batch_id_per_token,
                                        const int *seq_lens_encoder,
                                        const int *seq_lens_decoder,
                                        const int *cu_seqlens_q,
                                        const int *cu_seqlens_k,
                                        const int token_num,
                                        const int num_heads,
                                        const int kv_num_heads,
                                        const int max_model_len,
                                        const int head_dim,
                                        const cudaStream_t &stream) {
  assert(head_dim == 128 && "head_dim must be 128");

  int64_t elem_nums = token_num * (num_heads + 2 * kv_num_heads) * head_dim;

  constexpr int HEAD_DIM = 128;
  constexpr int PackSize = 8;
  const int pack_num = elem_nums / PackSize;
  const int blocksize = 128;
  int grid_size = 1;
  GetNumBlocks<128>(pack_num, &grid_size);
  dim3 block_size(128);

  const float *cos_emb = rotary_emb;
  const float *sin_emb = rotary_emb + max_model_len * head_dim;
  launchWithPdlWhenEnabled(
      GQAVariableLengthRotarySplitKernel_Qwen3<T, PackSize>,
      grid_size,
      block_size,
      0,
      stream,
      qkv_input,
      cos_emb,
      sin_emb,
      batch_id_per_token,
      cu_seqlens_q,
      seq_lens_encoder,
      seq_lens_decoder,
      cu_seqlens_k,
      qkv_out,
      q,
      k,
      v,
      elem_nums,
      num_heads,
      kv_num_heads,
      max_model_len,
      head_dim);
}
