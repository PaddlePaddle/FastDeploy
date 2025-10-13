// adapted from: https://github.com/vllm-project/vllm/blob/118ff921118cc81061a2af865a1e13840ceb6792/csrc/quantization/cutlass_w8a8/c3x/cutlass_gemm_caller.cuh


#include "quantization/common.cuh"

// adapted from: https://github.com/sgl-project/sglang/blob/v0.5.2rc2/sgl-kernel/csrc/gemm/per_token_quant_fp8.cu

// ---------------------------------------------------------------------------
// 1. Warp‑local, no shared memory
//    • One warp handles one token.
//    • Eight tokens per 256‑thread CTA.
// ---------------------------------------------------------------------------
template <typename T, typename DST_DTYPE, int kTokensPerCTA = 8, int kVecSize = 16>
__global__ void per_token_quant_fp8_kernel(
    const T* __restrict__ input,
    DST_DTYPE* __restrict__ output_q,
    float* __restrict__ output_s,
    const float scale_ub,
    const int64_t hidden_size,
    const int64_t num_tokens) {
  const int warp_id = threadIdx.x / WARP_SIZE;        // 0‑7  (8 warps)
  const int lane_id = threadIdx.x & (WARP_SIZE - 1);  // 0‑31
  const int token_id = blockIdx.x * kTokensPerCTA + warp_id;
  if (token_id >= num_tokens) return;

  // Global tensors for this token
  const T* token_input = input + token_id * hidden_size;
  DST_DTYPE* token_output = output_q + token_id * hidden_size;
  float* token_scale = output_s + token_id;

  //
  // Pass-1: Perform a warp reduce to find the max_value of a token's hidden_size
  //
  float max_value = 0.f;
  using vec_t = AlignedVector<T, kVecSize>;
  const int32_t num_vec_elems = hidden_size / kVecSize;

  for (int32_t i = lane_id; i < num_vec_elems; i += WARP_SIZE) {
    vec_t input_vec;
    Load(token_input + i * kVecSize, &input_vec);

#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      max_value = fmaxf(max_value, fabsf(static_cast<float>(input_vec[j])));
    }
  }

  float warp_max = warpReduceMax(max_value);
  if (scale_ub > 0){
    warp_max = fminf(warp_max, scale_ub);
  }
  float scale;
  scale = warp_max / FP8_E4M3_MAX;
  // Broadcast scale
  if (lane_id == 0) {
    token_scale[0] = scale;
  }
  float scale_inv = (scale == 0.f) ? 0.f : 1.0f / scale;

  //
  // Pass-2: quantize and write back
  //
  for (int i = lane_id; i < num_vec_elems; i += WARP_SIZE) {
    vec_t input_vec;
    Load(token_input + i * kVecSize, &input_vec);
    DST_DTYPE output_arr[kVecSize];
#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      float val = static_cast<float>(input_vec[j]) * scale_inv;
      val = fmaxf(fminf(val, FP8_E4M3_MAX), -FP8_E4M3_MAX);
      output_arr[j] = static_cast<DST_DTYPE>(val);
    }
    if constexpr (kVecSize == 16) {
      *(uint4*)(token_output + i * kVecSize) = *(uint4*)output_arr;
    } else {
      // Use element-wise copy for vector size 8 to ensure correctness
      for (int k = 0; k < kVecSize; ++k) {
        token_output[i * kVecSize + k] = output_arr[k];
      }
    }
  }
}

// ---------------------------------------------------------------------------
// 2.  Baseline kernel (1 token / CTA, CUB block reduce)
// ---------------------------------------------------------------------------
template <typename T, typename DST_DTYPE, int kVecSize = 16>
__global__ void per_token_quant_fp8_small_batch_kernel(
    const T* __restrict__ input,
    DST_DTYPE* __restrict__ output_q,
    float* __restrict__ output_s,
    const float scale_ub,
    const int64_t hidden_size,
    const int64_t num_tokens) {
  const int token_idx = blockIdx.x;
  if (token_idx >= num_tokens) return;

  const int tid = threadIdx.x;
  const int block_dim = blockDim.x;

  const T* token_input = input + token_idx * hidden_size;
  DST_DTYPE* token_output = output_q + token_idx * hidden_size;

  float max_value = 0.0f;

  // Use template parameter for vector size
  using vec_t = AlignedVector<T, kVecSize>;
  const int32_t num_vec_elems = hidden_size / kVecSize;

  // Find max using vectorized loads
  for (int32_t i = tid; i < num_vec_elems; i += block_dim) {
    vec_t input_vec;
    Load(token_input + i * kVecSize, &input_vec);

#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      float val = static_cast<float>(input_vec[j]);
      max_value = fmaxf(max_value, fabsf(val));
    }
  }

  max_value = blockReduceMax(max_value);
  if (scale_ub > 0){
    max_value = fminf(max_value, scale_ub);
  }
  __shared__ float scale;
  if (tid == 0) {
    scale = max_value / FP8_E4M3_MAX;
    output_s[token_idx] = scale;
  }
  __syncthreads();

  const float scale_inv = 1.0f / scale;

  // Quantize using vectorized loads
  for (int32_t i = tid; i < num_vec_elems; i += block_dim) {
    vec_t input_vec;
    Load(token_input + i * kVecSize, &input_vec);

    DST_DTYPE output_arr[kVecSize];
#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      float val = fmaxf(fminf(static_cast<float>(input_vec[j]) * scale_inv, FP8_E4M3_MAX), -FP8_E4M3_MAX);
      output_arr[j] = static_cast<DST_DTYPE>(val);
    }

    if constexpr (kVecSize == 16) {
      *(uint4*)(token_output + i * kVecSize) = *(uint4*)output_arr;
    } else {
      // Use element-wise copy for vector size 8 to ensure correctness
      for (int k = 0; k < kVecSize; ++k) {
        token_output[i * kVecSize + k] = output_arr[k];
      }
    }
  }
}

namespace fastdeploy {

template <typename scalar_t, typename fp8_type>
__global__ void scaled_fp8_quant_kernel(fp8_type *__restrict__ out,
                                        const scalar_t *__restrict__ input,
                                        const float *__restrict__ scale,
                                        int64_t num_elems) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  // Invert the scale so that we can use multiplications to avoid expensive
  // division.
  const float inverted_scale = 1.0f / (*scale);
  scaled_fp8_conversion_vec<scalar_t, true>(
      out, input, inverted_scale, num_elems, tid, blockDim.x * gridDim.x);
}

template <typename scalar_t, typename fp8_type>
__global__ void dynamic_per_token_scaled_fp8_quant_kernel(
    fp8_type *__restrict__ out, float *__restrict__ scale,
    scalar_t const *__restrict__ input, float scale_ub, const int hidden_size) {
  int const tid = threadIdx.x;
  int const token_idx = blockIdx.x;

  // Use int64 to avoid overflowing an int32 when calculating this offset
  int64_t offset = static_cast<int64_t>(token_idx) * hidden_size;
  scalar_t const *__restrict__ token_input = &input[offset];
  fp8_type *__restrict__ token_output = &out[offset];

  // For vectorization, token_input and token_output pointers need to be
  // aligned at 8-byte and 4-byte addresses respectively.
  bool const can_vectorize = hidden_size % 4 == 0;

  float absmax_val = 0.0f;
  if (can_vectorize) {
    absmax_val = thread_max_vec(token_input, hidden_size, tid, blockDim.x);
  } else {
    for (int i = tid; i < hidden_size; i += blockDim.x) {
      float const x = static_cast<float>(token_input[i]);
      absmax_val = max(absmax_val, fabs(x));
    }
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStorage;
  float const block_absmax_val_maybe =
      BlockReduce(reduceStorage).Reduce(absmax_val, cub::Max{}, blockDim.x);
  __shared__ float token_scale;
  if (tid == 0) {
    if (scale_ub > 0) {
      token_scale = min(block_absmax_val_maybe, scale_ub);
    } else {
      token_scale = block_absmax_val_maybe;
    }
    // token scale computation
    // token_scale = max(token_scale / 448.f,
    //                   min_scaling_factor<fp8_type>::val());
    token_scale = token_scale / 448.f;
    scale[token_idx] = token_scale;
  }
  __syncthreads();

  // Note that we don't use inverted scales so we can match FBGemm impl.
  if (can_vectorize) {
    scaled_fp8_conversion_vec<scalar_t, false>(
        token_output, token_input, token_scale, hidden_size, tid, blockDim.x);
  } else {
    for (int i = tid; i < hidden_size; i += blockDim.x) {
      token_output[i] = scaled_fp8_conversion<false, fp8_type>(
          static_cast<float>(token_input[i]), token_scale);
    }
  }
}

} // namespace fastdeploy

void StaticScaledFp8Quant(paddle::Tensor &out,         // [..., d]
                          paddle::Tensor const &input, // [..., d]
                          paddle::Tensor const &scale) // [1]
{
  PD_CHECK(out.dtype() == paddle::DataType::FLOAT8_E4M3FN);
  using fp8_t = phi::dtype::float8_e4m3fn;
  auto rank = input.dims().size();
  int64_t num_tokens = input.numel() / input.dims()[rank - 1];
  int64_t num_elems = input.numel();
  dim3 grid(num_tokens);
  dim3 block(1024);

  cudaStream_t stream = input.stream();

  switch (input.dtype()) {
  case paddle::DataType::FLOAT32: {
    using scalar_t = float;
    fastdeploy::scaled_fp8_quant_kernel<scalar_t, fp8_t>
        <<<grid, block, 0, stream>>>(out.data<fp8_t>(), input.data<scalar_t>(),
                                     scale.data<float>(), num_elems);
    break;
  }
  case paddle::DataType::FLOAT16: {
    using scalar_t = phi::dtype::float16;
    fastdeploy::scaled_fp8_quant_kernel<scalar_t, fp8_t>
        <<<grid, block, 0, stream>>>(out.data<fp8_t>(), input.data<scalar_t>(),
                                     scale.data<float>(), num_elems);
    break;
  }
  case paddle::DataType::BFLOAT16: {
    using scalar_t = phi::dtype::bfloat16;
    fastdeploy::scaled_fp8_quant_kernel<scalar_t, fp8_t>
        <<<grid, block, 0, stream>>>(out.data<fp8_t>(), input.data<scalar_t>(),
                                     scale.data<float>(), num_elems);
    break;
  }
  default:
    PD_THROW("Only supported attr of input type in [fp32, fp16, bf16].");
  }
}

void DynamicScaledFp8Quant(paddle::Tensor &out,         // [..., d]
                           paddle::Tensor const &input, // [..., d]
                           paddle::Tensor &scale)       // [1]
{
  PD_CHECK(out.dtype() == paddle::DataType::FLOAT8_E4M3FN);
  using fp8_t = phi::dtype::float8_e4m3fn;
  auto rank = input.dims().size();
  int64_t num_tokens = input.numel() / input.dims()[rank - 1];
  int64_t num_elems = input.numel();
  dim3 grid(num_tokens);
  dim3 block(1024);

  cudaStream_t stream = input.stream();

  switch (input.dtype()) {
  case paddle::DataType::FLOAT32: {
    using scalar_t = float;
    fastdeploy::segmented_max_reduction<scalar_t, fp8_t>
        <<<grid, block, 0, stream>>>(scale.data<float>(),
                                     input.data<scalar_t>(), num_elems);
    fastdeploy::scaled_fp8_quant_kernel<scalar_t, fp8_t>
        <<<grid, block, 0, stream>>>(out.data<fp8_t>(), input.data<scalar_t>(),
                                     scale.data<float>(), num_elems);
    break;
  }
  case paddle::DataType::FLOAT16: {
    using scalar_t = phi::dtype::float16;
    fastdeploy::segmented_max_reduction<scalar_t, fp8_t>
        <<<grid, block, 0, stream>>>(scale.data<float>(),
                                     input.data<scalar_t>(), num_elems);
    fastdeploy::scaled_fp8_quant_kernel<scalar_t, fp8_t>
        <<<grid, block, 0, stream>>>(out.data<fp8_t>(), input.data<scalar_t>(),
                                     scale.data<float>(), num_elems);
    break;
  }
  case paddle::DataType::BFLOAT16: {
    using scalar_t = phi::dtype::bfloat16;
    fastdeploy::segmented_max_reduction<scalar_t, fp8_t>
        <<<grid, block, 0, stream>>>(scale.data<float>(),
                                     input.data<scalar_t>(), num_elems);
    fastdeploy::scaled_fp8_quant_kernel<scalar_t, fp8_t>
        <<<grid, block, 0, stream>>>(out.data<fp8_t>(), input.data<scalar_t>(),
                                     scale.data<float>(), num_elems);
    break;
  }
  default:
    PD_THROW("Only supported attr of input type in [fp32, fp16, bf16].");
  }
}

void DynamicPerTokenScaledFp8Quant(paddle::Tensor &out,         // [..., d]
                                   paddle::Tensor const &input, // [..., d]
                                   paddle::Tensor &scales, float scale_ub) {
  PD_CHECK(input.is_contiguous());
  PD_CHECK(out.is_contiguous());
  PD_CHECK(out.dtype() == paddle::DataType::FLOAT8_E4M3FN);
  using fp8_t = phi::dtype::float8_e4m3fn;
  auto rank = input.dims().size();
  int const hidden_size = input.dims()[rank - 1];
  int const num_tokens = input.numel() / hidden_size;
  cudaStream_t stream = input.stream();

  if (hidden_size % 8 == 0){
    int device = 0;
    cudaGetDevice(&device);
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    const int TOKENS_PER_CTA = 8;
    const bool use_warp_kernel = (num_tokens >= sm_count * 2 * TOKENS_PER_CTA);
    const bool use_vec16 = (hidden_size % 16 == 0);
    DISPATCH_FLOAT_FP6_DTYPE(input.dtype(), scalar_t, {
      if (use_warp_kernel) {
        // -------- warp‑local ---------------------------------------------------
        constexpr int THREADS = TOKENS_PER_CTA * WARP_SIZE;  // 256
        dim3 grid((num_tokens + TOKENS_PER_CTA - 1) / TOKENS_PER_CTA);
        dim3 block(THREADS);

        if (use_vec16) {
          per_token_quant_fp8_kernel<scalar_t, __nv_fp8_e4m3, TOKENS_PER_CTA, 16><<<grid, block, 0, stream>>>(
              reinterpret_cast<const scalar_t*>(input.data<scalar_t>()),
              reinterpret_cast<__nv_fp8_e4m3*>(out.data<fp8_t>()),
              reinterpret_cast<float*>(scales.data<float>()),
              scale_ub,
              hidden_size,
              num_tokens);
        } else {
          per_token_quant_fp8_kernel<scalar_t, __nv_fp8_e4m3, TOKENS_PER_CTA, 8><<<grid, block, 0, stream>>>(
              reinterpret_cast<const scalar_t*>(input.data<scalar_t>()),
              reinterpret_cast<__nv_fp8_e4m3*>(out.data<fp8_t>()),
              reinterpret_cast<float*>(scales.data<float>()),
              scale_ub,
              hidden_size,
              num_tokens);
        }
      } else {
        // -------- baseline -----------------------------------------------------
        constexpr int THREADS = 256;
        dim3 grid(num_tokens);
        dim3 block(THREADS);

        if (use_vec16) {
          per_token_quant_fp8_small_batch_kernel<scalar_t, __nv_fp8_e4m3, 16><<<grid, block, 0, stream>>>(
              reinterpret_cast<const scalar_t*>(input.data<scalar_t>()),
              reinterpret_cast<__nv_fp8_e4m3*>(out.data<fp8_t>()),
              reinterpret_cast<float*>(scales.data<float>()),
              scale_ub,
              hidden_size,
              num_tokens);
        } else {
          per_token_quant_fp8_small_batch_kernel<scalar_t, __nv_fp8_e4m3, 8><<<grid, block, 0, stream>>>(
              reinterpret_cast<const scalar_t*>(input.data<scalar_t>()),
              reinterpret_cast<__nv_fp8_e4m3*>(out.data<fp8_t>()),
              reinterpret_cast<float*>(scales.data<float>()),
              scale_ub,
              hidden_size,
              num_tokens);
        }
      }
    });
    return;
  }

  dim3 const grid(num_tokens);
  dim3 const block(std::min(hidden_size, 1024));

  DISPATCH_FLOAT_FP6_DTYPE(input.dtype(), scalar_t, {
    fastdeploy::dynamic_per_token_scaled_fp8_quant_kernel<scalar_t, fp8_t>
        <<<grid, block, 0, stream>>>(out.data<fp8_t>(), scales.data<float>(),
                                     input.data<scalar_t>(), scale_ub,
                                     hidden_size);
  });

}

PD_BUILD_STATIC_OP(static_scaled_fp8_quant)
    .Inputs({"out", "input", "scale"})
    .Outputs({"out_q"})
    .SetInplaceMap({{"out", "out_q"}})
    .SetKernelFn(PD_KERNEL(StaticScaledFp8Quant));

PD_BUILD_STATIC_OP(dynamic_scaled_fp8_quant)
    .Inputs({"out", "input", "scale"})
    .Outputs({"out_q", "out_scale"})
    .SetInplaceMap({{"out", "out_q"},
                    {"scale", "out_scale"}})
    .SetKernelFn(PD_KERNEL(DynamicScaledFp8Quant));

PD_BUILD_STATIC_OP(dynamic_per_token_scaled_fp8_quant)
    .Inputs({"out", "input", "scale"})
    .Attrs({"scale_ub: float"})
    .Outputs({"out_q"})
    .SetInplaceMap({{"out", "out_q"}})
    .SetKernelFn(PD_KERNEL(DynamicPerTokenScaledFp8Quant));
