#include "encoder_write_cache_with_rope_impl.cuh"
#include "helper.h"
#include "paddle/extension.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/core/memory/memcpy.h"
#include "remote_cache_kv_ipc.h"

// head_dim dispatch for RoPE/KV-cache write: supports 128 (Qwen3) and 256
// (Qwen3.5). Kept separate from DISPATCH_HEAD_DIM in utils.cuh, which only
// covers 64/128 and is shared with attention kernels not validated for
// head_dim=256.
#define DISPATCH_GQA_ROPE_HEAD_DIM(head_dim, HEAD_DIM, ...)             \
  switch (head_dim) {                                                   \
    case 128: {                                                         \
      constexpr uint32_t HEAD_DIM = 128;                                \
      __VA_ARGS__                                                       \
      break;                                                            \
    }                                                                   \
    case 256: {                                                         \
      constexpr uint32_t HEAD_DIM = 256;                                \
      __VA_ARGS__                                                       \
      break;                                                            \
    }                                                                   \
    default:                                                            \
      PADDLE_THROW("unsupported head_dim: %d for gqa_rope_write_cache", \
                   head_dim);                                           \
  }

template <typename T, int VecSize = 1, bool EnforceFmulRN = false>
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
    const int head_dim,
    const bool rope_3d) {
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
    int64_t new_emb_idx =
        rope_3d ? emb_idx + ori_bi * 2 * max_model_len * head_dim : emb_idx;

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
            static_cast<T>(fmul_func<EnforceFmulRN>(input_left, cos_tmp) -
                           fmul_func<EnforceFmulRN>(input_right, sin_tmp));
        src_vec1[i] =
            static_cast<T>(fmul_func<EnforceFmulRN>(input_right, cos_tmp) +
                           fmul_func<EnforceFmulRN>(input_left, sin_tmp));
      }
    }
    Store<T, VecSize>(src_vec0, &qkv_out[read_idx]);
    Store<T, VecSize>(src_vec0, &out_p[base_split_idx]);
    Store<T, VecSize>(src_vec1, &qkv_out[read_idx + 64]);
    Store<T, VecSize>(src_vec1, &out_p[base_split_idx + 64]);
  }
}

// Qwen3.5 neox partial rotary kernel (head_dim=256, warp-based).
//
// Applies non-interleaved (neox) partial RoPE to Q and K, then splits QKV.
// Only [0, rotary_dim) participates in rotation; [rotary_dim, head_dim) is
// passed through unchanged.
//
// Rotation formula (same as Python rotate_half):
//   left  [0, half_rotary_dim):   out = q[h]*cos[h] - q[h+half]*sin[h]
//   right [half_rotary_dim, rotary_dim): out = q[h]*cos[h-half] +
//   q[h-half]*sin[h-half] pass  [rotary_dim, head_dim): out = q[h]
//
// Warp layout: each warp (32 threads) owns one head; VecSize=PackSize=8
// covers all 256 elements per head (256/32=8 elements per thread).
//
// rotary_emb layout (from QwenRotaryEmbedding):
//   shape = (2, 1, max_seq_len, 1, rotary_dim)
//   cos_emb = rotary_emb[0],  sin_emb starts at offset max_seq_len*rotary_dim
//   emb_idx = ori_seq_id * rotary_dim + h_bias  (h_bias < rotary_dim)
template <typename T, int VecSize = 1, bool EnforceFmulRN = false>
__global__ void GQAVariableLengthNeoxPartialRotarySplitKernel_Qwen3_5(
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
    const int head_dim,
    const int rotary_dim) {
  using LoadT = AlignedVector<T, VecSize>;
  using LoadEmbT = AlignedVector<float, VecSize>;

  // src_vec: elements at h_bias; src_vec_pair: partner at h_bias ±
  // half_rotary_dim
  LoadT src_vec;
  LoadT src_vec_pair;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;

  // warp index: each warp processes one head across [0, head_dim)
  int64_t global_warp_idx = blockDim.y * blockIdx.x + threadIdx.y;
  int64_t all_warp_num = gridDim.x * blockDim.y;

  // midpoint of the rotary range; pairs are (h, h ± half_rotary_dim)
  const int half_rotary_dim = rotary_dim / 2;

  // total elements per token across all Q+K+V heads
  const int offset = (q_num_head + kv_num_head * 2) * head_dim;

  // total heads to process = warp workload
  const int all_head_num = elem_cnt / head_dim;

  for (int global_hi = global_warp_idx; global_hi < all_head_num;
       global_hi += all_warp_num) {
    // threadIdx.x * VecSize: this thread's h_bias offset within the head
    int64_t linear_index = global_hi * head_dim + threadIdx.x * VecSize;

    // token index in the flattened token stream
    const int token_idx = linear_index / offset;

    // -1 means padding token; skip to avoid out-of-bounds access
    const int ori_bi = batch_id_per_token[token_idx];
    if (ori_bi == -1) continue;

    // FA3 bug workaround: force kv_len=0 when this batch has no encoder tokens
    int cache_kv_len = seq_lens_decoder[ori_bi];
    if (seq_lens_encoder[ori_bi] == 0) cache_kv_len = 0;

    const int bias = linear_index % offset;

    // head index and intra-head column offset
    const int hi = bias / head_dim;
    const int h_bias = bias % head_dim;

    // position within its sequence, accounting for cached KV offset
    const int ori_seq_id = (token_idx - cu_seqlens_q[ori_bi]) + cache_kv_len;

    // global read address for this thread's elements
    const int64_t base_idx =
        token_idx * (q_num_head + 2 * kv_num_head) * head_dim + hi * head_dim +
        h_bias;

    // load VecSize elements at h_bias
    Load<T, VecSize>(&qkv[base_idx], &src_vec);

    // write offset into paged KV cache for K/V heads
    const int kv_write_idx = cu_seqlens_k[ori_bi] + ori_seq_id;

    // write address in split q/k/v buffers
    int64_t base_split_idx;
    T *out_p = nullptr;
    if (hi < q_num_head) {
      // Q head: sequential layout
      base_split_idx =
          token_idx * q_num_head * head_dim + hi * head_dim + h_bias;
      out_p = q;
    } else if (hi < q_num_head + kv_num_head) {
      // K head: paged cache offset
      base_split_idx = kv_write_idx * kv_num_head * head_dim +
                       (hi - q_num_head) * head_dim + h_bias;
      out_p = k;
    } else {
      // V head: paged cache offset
      out_p = v;
      base_split_idx = kv_write_idx * kv_num_head * head_dim +
                       (hi - q_num_head - kv_num_head) * head_dim + h_bias;
    }

    // only Q and K rotate; V passes through unchanged
    if (hi < q_num_head + kv_num_head) {
      if (h_bias < half_rotary_dim) {
        Load<T, VecSize>(&qkv[base_idx + half_rotary_dim], &src_vec_pair);

        const int64_t emb_idx = ori_seq_id * rotary_dim + h_bias;
        Load<float, VecSize>(&cos_emb[emb_idx], &cos_emb_vec);
        Load<float, VecSize>(&sin_emb[emb_idx], &sin_emb_vec);

#pragma unroll
        for (int i = 0; i < VecSize; i++) {
          const float x_l = static_cast<float>(src_vec[i]);  // q[h]
          const float x_r =
              static_cast<float>(src_vec_pair[i]);  // q[h + half_rotary_dim]
          src_vec[i] =
              static_cast<T>(fmul_func<EnforceFmulRN>(x_l, cos_emb_vec[i]) -
                             fmul_func<EnforceFmulRN>(x_r, sin_emb_vec[i]));
        }

      } else if (h_bias < rotary_dim) {
        Load<T, VecSize>(&qkv[base_idx - half_rotary_dim], &src_vec_pair);

        const int64_t emb_idx =
            ori_seq_id * rotary_dim + h_bias - half_rotary_dim;
        Load<float, VecSize>(&cos_emb[emb_idx], &cos_emb_vec);
        Load<float, VecSize>(&sin_emb[emb_idx], &sin_emb_vec);

#pragma unroll
        for (int i = 0; i < VecSize; i++) {
          const float x_r = static_cast<float>(src_vec[i]);
          const float x_l = static_cast<float>(src_vec_pair[i]);
          src_vec[i] =
              static_cast<T>(fmul_func<EnforceFmulRN>(x_r, cos_emb_vec[i]) +
                             fmul_func<EnforceFmulRN>(x_l, sin_emb_vec[i]));
        }
      }
      // h_bias ∈ [rotary_dim, head_dim)：pass-through
    }

    Store<T, VecSize>(src_vec, &qkv_out[base_idx]);
    Store<T, VecSize>(src_vec, &out_p[base_split_idx]);
  }
}

// Launcher for GQAVariableLengthNeoxPartialRotarySplitKernel_Qwen3_5
// (head_dim=256). Differs from the GLM/head_dim=128 variant in: PackSize=8,
// sin_emb offset uses rotary_dim (not head_dim), and a 2D block where each warp
// owns one full head.
template <typename T, bool EnforceFmulRN = false>
void gqa_neox_partial_rotary_qk_split_variable_qwen3_5(
    T *qkv_out,
    T *q,
    T *k,
    T *v,
    const T *qkv_input,
    const float *rotary_emb,  // [cos: max_model_len*rotary_dim][sin:
                              // max_model_len*rotary_dim]
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
    const int rotary_dim,
    const cudaStream_t &stream) {
  PADDLE_ENFORCE_EQ(head_dim, 256, "head_dim must be 256");
  PADDLE_ENFORCE_LE(rotary_dim, head_dim, "rotary_dim must be <= head_dim");

  int64_t elem_nums = token_num * (num_heads + 2 * kv_num_heads) * head_dim;

  // PackSize=8: each of 32 warp threads handles 8 elements, covering all 256
  // per head.
  constexpr int HEAD_DIM = 256;
  constexpr int PackSize = HEAD_DIM / kWarpSize;  // = 8
  PADDLE_ENFORCE_EQ(rotary_dim / 2 % PackSize,
                    0,
                    "half rotary_dim must be divisible by PackSize");

  // One warp per head; 128-thread block = 4 warps.
  const int all_head_num = elem_nums / HEAD_DIM;
  const int blocksize = 128;
  int grid_size = 1;
  GetNumBlocks<128>(all_head_num, &grid_size);
  // 2D block: x=warp lanes (kWarpSize), y=warps per block.
  dim3 block_size(kWarpSize, blocksize / kWarpSize);

  // sin_emb follows cos_emb; split at max_model_len*rotary_dim (not head_dim)
  // because partial rotary only stores rotary_dim entries per position.
  const float *cos_emb = rotary_emb;
  const float *sin_emb = rotary_emb + max_model_len * rotary_dim;

  launchWithPdlWhenEnabled(
      GQAVariableLengthNeoxPartialRotarySplitKernel_Qwen3_5<T,
                                                            PackSize,
                                                            EnforceFmulRN>,
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
      head_dim,
      rotary_dim);
}

template <typename T, bool EnforceFmulRN = false>
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
                                        const bool rope_3d,
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
      GQAVariableLengthRotarySplitKernel_Qwen3<T, PackSize, EnforceFmulRN>,
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
      head_dim,
      rope_3d);
}
