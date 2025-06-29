// adapted from: https://github.com/vllm-project/vllm/blob/118ff921118cc81061a2af865a1e13840ceb6792/csrc/quantization/cutlass_w8a8/c3x/cutlass_gemm_caller.cuh


#include "quantization/common.cuh"

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
  dim3 const grid(num_tokens);
  dim3 const block(std::min(hidden_size, 1024));

  cudaStream_t stream = input.stream();

  switch (input.dtype()) {
  case paddle::DataType::FLOAT32: {
    using scalar_t = float;
    fastdeploy::dynamic_per_token_scaled_fp8_quant_kernel<scalar_t, fp8_t>
        <<<grid, block, 0, stream>>>(out.data<fp8_t>(), scales.data<float>(),
                                     input.data<scalar_t>(), scale_ub,
                                     hidden_size);
    break;
  }
  case paddle::DataType::FLOAT16: {
    using scalar_t = phi::dtype::float16;
    fastdeploy::dynamic_per_token_scaled_fp8_quant_kernel<scalar_t, fp8_t>
        <<<grid, block, 0, stream>>>(out.data<fp8_t>(), scales.data<float>(),
                                     input.data<scalar_t>(), scale_ub,
                                     hidden_size);
    break;
  }
  case paddle::DataType::BFLOAT16: {
    using scalar_t = phi::dtype::bfloat16;
    fastdeploy::dynamic_per_token_scaled_fp8_quant_kernel<scalar_t, fp8_t>
        <<<grid, block, 0, stream>>>(out.data<fp8_t>(), scales.data<float>(),
                                     input.data<scalar_t>(), scale_ub,
                                     hidden_size);
    break;
  }
  default:
    PD_THROW("Only supported attr of input type in [fp32, fp16, bf16].");
  }
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
