// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Paddle port of mha_varlen_fwd_block from
// Block-Sparse-Attention/csrc/block_sparse_attn/flash_api.cpp (Tri Dao / J. Guo).
// Forward-only (inference). Backward kernels are not compiled.
//
// Note: the directory ./src is a symlink to
//   <repo_root>/Block-Sparse-Attention/csrc/block_sparse_attn/src
// so the headers below resolve through the symlink without copying source.

#include "paddle/extension.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <vector>
#include <limits>
#include <optional>

#include <cutlass/numeric_types.h>

#include "src/namespace_config.h"
#include "src/hardware_info.h"
#include "src/flash.h"
#include "src/static_switch.h"

#define BSA_CHECK_GPU(x) PD_CHECK((x).is_gpu(), #x " must be on GPU")
#define BSA_CHECK_DTYPE(x, dt) PD_CHECK((x).dtype() == dt, #x " has wrong dtype")

namespace FLASH_NAMESPACE {

static constexpr int SPARSE_SIZE = 128;

// Forward kernel template, instantiated by src/flash_fwd_block_hdim*_*_sm80.cu.
template <typename elem_type, int kHeadDim, bool Is_causal>
void run_mha_fwd_block_(Flash_fwd_params& params, cudaStream_t stream);

static void set_params_fprop(Flash_fwd_params& params,
                             size_t b, size_t seqlen_q, size_t seqlen_k,
                             size_t seqlen_q_rounded, size_t seqlen_k_rounded,
                             size_t h, size_t h_k, size_t d, size_t d_rounded,
                             const paddle::Tensor& q,
                             const paddle::Tensor& k,
                             const paddle::Tensor& v,
                             paddle::Tensor& out,
                             void* cu_seqlens_q_d,
                             void* cu_seqlens_k_d,
                             void* p_d,
                             void* softmax_lse_d,
                             float p_dropout,
                             float softmax_scale,
                             int window_size_left,
                             int window_size_right) {
  params = {};
  params.is_bf16 = (q.dtype() == paddle::DataType::BFLOAT16);

  // Pointers
  params.q_ptr = const_cast<void*>(q.data());
  params.k_ptr = const_cast<void*>(k.data());
  params.v_ptr = const_cast<void*>(v.data());
  params.o_ptr = out.data();

  // Strides (in elements). Varlen layout [total_seq, num_heads, head_size]
  // with last-dim contiguous. Caller must guarantee contiguity.
  params.q_row_stride  = static_cast<int64_t>(h)   * d;
  params.k_row_stride  = static_cast<int64_t>(h_k) * d;
  params.v_row_stride  = static_cast<int64_t>(h_k) * d;
  params.q_head_stride = d;
  params.k_head_stride = d;
  params.v_head_stride = d;
  params.o_row_stride  = static_cast<int64_t>(h) * d;
  params.o_head_stride = d;

  params.cu_seqlens_q = static_cast<int*>(cu_seqlens_q_d);
  params.cu_seqlens_k = static_cast<int*>(cu_seqlens_k_d);
  params.seqused_k    = nullptr;

  params.p_ptr = p_d;
  params.softmax_lse_ptr = softmax_lse_d;

  params.b   = b;
  params.h   = h;
  params.h_k = h_k;
  params.h_h_k_ratio  = h / h_k;
  params.seqlen_q     = seqlen_q;
  params.seqlen_k     = seqlen_k;
  params.seqlen_q_rounded = seqlen_q_rounded;
  params.seqlen_k_rounded = seqlen_k_rounded;
  params.d            = d;
  params.d_rounded    = d_rounded;

  params.scale_softmax      = softmax_scale;
  params.scale_softmax_log2 = softmax_scale * static_cast<float>(M_LOG2E);

  params.p_dropout              = 1.f - p_dropout;
  params.p_dropout_in_uint8_t   = static_cast<uint8_t>(std::floor(params.p_dropout * 255.0f));
  params.rp_dropout             = 1.f / params.p_dropout;
  params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
  PD_CHECK(p_dropout < 1.f, "p_dropout must be < 1");

  params.is_causal = (window_size_left < 0) && (window_size_right == 0);
  if (window_size_left  < 0 && window_size_right >= 0) window_size_left  = static_cast<int>(seqlen_k);
  if (window_size_left >= 0 && window_size_right  < 0) window_size_right = static_cast<int>(seqlen_k);
  params.window_size_left  = window_size_left;
  params.window_size_right = window_size_right;

  params.is_seqlens_k_cumulative = true;
}

static void run_mha_fwd_block(Flash_fwd_params& params, cudaStream_t stream) {
  FP16_SWITCH(!params.is_bf16, [&] {
    HEADDIM_SWITCH(params.d, [&] {
      BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        run_mha_fwd_block_<elem_type, kHeadDim, Is_causal>(params, stream);
      });
    });
  });
}

std::vector<paddle::Tensor> BlockSparseAttnFwd(
    const paddle::Tensor& q,                 // [total_q, num_heads,   head_size]
    const paddle::Tensor& k,                 // [total_k, num_heads_k, head_size]
    const paddle::Tensor& v,                 // [total_k, num_heads_k, head_size]
    const paddle::Tensor& cu_seqlens_q,      // int32 [b+1]
    const paddle::Tensor& cu_seqlens_k,      // int32 [b+1]
    const paddle::Tensor& head_mask_type,    // int32 [num_heads]
    const paddle::optional<paddle::Tensor>& streaming_info,   // int32 [num_heads*2]
    const paddle::optional<paddle::Tensor>& base_blockmask,   // int32 [b, n_bs_h, sm/m, sk/n]
    int max_seqlen_q,
    int max_seqlen_k,
    float p_dropout,
    float softmax_scale,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    int m_block_dim,
    int n_block_dim,
    bool exact_streaming,
    bool return_softmax) {
  BSA_CHECK_GPU(q); BSA_CHECK_GPU(k); BSA_CHECK_GPU(v);
  BSA_CHECK_GPU(cu_seqlens_q); BSA_CHECK_GPU(cu_seqlens_k);
  BSA_CHECK_GPU(head_mask_type);

  auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
  PD_CHECK(cc_major >= 8, "BlockSparseAttention requires Ampere (sm80) or newer");

  const auto q_dtype = q.dtype();
  PD_CHECK(q_dtype == paddle::DataType::FLOAT16 || q_dtype == paddle::DataType::BFLOAT16,
           "BlockSparseAttention only supports fp16/bf16");
  BSA_CHECK_DTYPE(k, q_dtype);
  BSA_CHECK_DTYPE(v, q_dtype);
  BSA_CHECK_DTYPE(cu_seqlens_q, paddle::DataType::INT32);
  BSA_CHECK_DTYPE(cu_seqlens_k, paddle::DataType::INT32);
  BSA_CHECK_DTYPE(head_mask_type, paddle::DataType::INT32);

  const bool has_blockmask = base_blockmask.is_initialized();
  const bool has_streaming = streaming_info.is_initialized();
  if (has_blockmask) {
    BSA_CHECK_GPU(base_blockmask.get());
    BSA_CHECK_DTYPE(base_blockmask.get(), paddle::DataType::INT32);
    PD_CHECK(m_block_dim % SPARSE_SIZE == 0, "m_block_dim must be a multiple of 128");
    PD_CHECK(n_block_dim % SPARSE_SIZE == 0, "n_block_dim must be a multiple of 128");
  }
  if (has_streaming) {
    BSA_CHECK_GPU(streaming_info.get());
    BSA_CHECK_DTYPE(streaming_info.get(), paddle::DataType::INT32);
    PD_CHECK(m_block_dim % SPARSE_SIZE == 0, "m_block_dim must be a multiple of 128");
    PD_CHECK(n_block_dim % SPARSE_SIZE == 0, "n_block_dim must be a multiple of 128");
  }

  const auto& q_shape = q.shape();
  const auto& k_shape = k.shape();
  PD_CHECK(q_shape.size() == 3, "q must be 3D [total_q, num_heads, head_size]");
  PD_CHECK(k_shape.size() == 3, "k must be 3D [total_k, num_heads_k, head_size]");

  const int total_q     = static_cast<int>(q_shape[0]);
  const int num_heads   = static_cast<int>(q_shape[1]);
  const int head_size   = static_cast<int>(q_shape[2]);
  const int total_k     = static_cast<int>(k_shape[0]);
  const int num_heads_k = static_cast<int>(k_shape[1]);
  const int batch_size  = static_cast<int>(cu_seqlens_q.shape()[0]) - 1;

  PD_CHECK(batch_size > 0, "batch size must be positive");
  PD_CHECK(head_size <= 256, "head_size > 256 is not supported");
  PD_CHECK(head_size % 8 == 0, "head_size must be a multiple of 8");
  PD_CHECK(num_heads % num_heads_k == 0, "num_heads must be divisible by num_heads_k");

  if (window_size_left  >= max_seqlen_k) window_size_left  = -1;
  if (window_size_right >= max_seqlen_k) window_size_right = -1;
  if (is_causal) window_size_right = 0;

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int head_size_rounded = round_multiple(head_size, head_size <= 128 ? 32 : 64);
  const int seqlen_q_rounded  = round_multiple(max_seqlen_q, SPARSE_SIZE);
  const int seqlen_k_rounded  = round_multiple(max_seqlen_k, SPARSE_SIZE);

  auto out         = paddle::empty(q_shape, q_dtype, q.place());
  auto softmax_lse = paddle::empty({batch_size, num_heads, max_seqlen_q},
                                   paddle::DataType::FLOAT32, q.place());

  Flash_fwd_params params;
  set_params_fprop(params,
                   batch_size, max_seqlen_q, max_seqlen_k,
                   seqlen_q_rounded, seqlen_k_rounded,
                   num_heads, num_heads_k, head_size, head_size_rounded,
                   q, k, v, out,
                   const_cast<void*>(cu_seqlens_q.data()),
                   const_cast<void*>(cu_seqlens_k.data()),
                   /*p_d=*/nullptr,
                   softmax_lse.data(),
                   p_dropout, softmax_scale,
                   is_causal ? -1 : window_size_left,
                   is_causal ?  0 : window_size_right);
  params.total_q = total_q;

  params.head_mask_type =
      static_cast<int*>(const_cast<void*>(head_mask_type.data()));

  if (has_blockmask) {
    params.blockmask =
        static_cast<int*>(const_cast<void*>(base_blockmask.get().data()));
    params.m_block_dim           = m_block_dim;
    params.n_block_dim           = n_block_dim;
    params.num_blocksparse_heads = static_cast<int>(base_blockmask.get().shape()[1]);
  } else {
    params.blockmask = nullptr;
  }

  if (has_streaming) {
    params.streaming_info =
        static_cast<int*>(const_cast<void*>(streaming_info.get().data()));
    params.is_exact_streaming = exact_streaming;
    params.m_block_dim = m_block_dim;
    params.n_block_dim = n_block_dim;
  } else {
    params.streaming_info     = nullptr;
    params.is_exact_streaming = false;
  }

  // Inference: dropout disabled. Provide a dummy non-null rng_state buffer
  // because the kernel writes into it unconditionally.
  static thread_local uint64_t dummy_rng_state[2] = {0, 0};
  params.rng_state = dummy_rng_state;

  cudaStream_t stream = q.stream();
  if (max_seqlen_k > 0) {
    run_mha_fwd_block(params, stream);
  } else {
    // Manual dtype-size lookup (avoid paddle::experimental::SizeOf which
    // is not available in all Paddle versions).
    size_t elem_size =
        (q_dtype == paddle::DataType::FLOAT16 ||
         q_dtype == paddle::DataType::BFLOAT16) ? 2 : 4;
    cudaMemsetAsync(out.data(), 0, out.numel() * elem_size, stream);
  }

  return {out, softmax_lse};
}

}  // namespace FLASH_NAMESPACE

// ---- Paddle op registration --------------------------------------------------
std::vector<std::vector<int64_t>> BlockSparseAttnFwdInferShape(
    const std::vector<int64_t>& q_shape,
    const std::vector<int64_t>& /*k*/,
    const std::vector<int64_t>& /*v*/,
    const std::vector<int64_t>& cu_q,
    const std::vector<int64_t>& /*cu_k*/,
    const std::vector<int64_t>& /*hmt*/) {
  const int64_t b = (cu_q.empty() ? 1 : cu_q[0] - 1);
  const int64_t h = q_shape.size() >= 2 ? q_shape[1] : 1;
  std::vector<int64_t> lse_shape{b, h, /*seqlen_q*/ 1};
  return {q_shape, lse_shape};
}

std::vector<paddle::DataType> BlockSparseAttnFwdInferDtype(
    const paddle::DataType& q_dtype,
    const paddle::DataType& /*k*/,
    const paddle::DataType& /*v*/,
    const paddle::DataType& /*cu_q*/,
    const paddle::DataType& /*cu_k*/,
    const paddle::DataType& /*hmt*/) {
  return {q_dtype, paddle::DataType::FLOAT32};
}

PD_BUILD_OP(block_sparse_attn_fwd)
    .Inputs({"q", "k", "v",
             "cu_seqlens_q", "cu_seqlens_k", "head_mask_type",
             paddle::Optional("streaming_info"),
             paddle::Optional("base_blockmask")})
    .Outputs({"out", "softmax_lse"})
    .Attrs({"max_seqlen_q: int",
            "max_seqlen_k: int",
            "p_dropout: float",
            "softmax_scale: float",
            "is_causal: bool",
            "window_size_left: int",
            "window_size_right: int",
            "m_block_dim: int",
            "n_block_dim: int",
            "exact_streaming: bool",
            "return_softmax: bool"})
    .SetKernelFn(PD_KERNEL(FLASH_NAMESPACE::BlockSparseAttnFwd))
    .SetInferShapeFn(PD_INFER_SHAPE(BlockSparseAttnFwdInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(BlockSparseAttnFwdInferDtype));
