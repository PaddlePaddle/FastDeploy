#include "norm.cuh"

#define CHECK_CUDA(x) PD_CHECK(x.is_gpu(), #x " must be a CUDA tensor")

#define CHECK_LAST_DIM_CONTIGUOUS(x)       \
  do {                                     \
    auto n_dim = x.dims().size();          \
    PD_CHECK(x.strides()[n_dim - 1] == 1); \
  } while (0)

#define CHECK_LAST_DIM_CONTIGUOUS_INPUT(x) \
  CHECK_CUDA(x);                           \
  CHECK_LAST_DIM_CONTIGUOUS(x)

#define CHECK_DEVICE(x, y) \
  PD_CHECK(x.place() == y.place(), #x " and " #y " must be on the same device");

#define CHECK_DIM(d, x) \
  PD_CHECK(x.dims().size() == d, #x " dims must equal " #d);

void rmsnorm(paddle::Tensor &output,
             paddle::Tensor &input,
             paddle::Tensor &weight,
             float eps,
             bool enable_pdl) {
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(input);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(output);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(weight);
  CHECK_DEVICE(input, weight);
  CHECK_DIM(1, weight);  // weight: (hidden_size)
  auto input_ndim = input.dims().size();
  if (input_ndim == 2) {
    // Normal RMSNorm: [batch_size, hidden_size]
    // Use CTA parallelization for better parallelism
    CHECK_DIM(2, output);
    PD_CHECK(input.dims()[1] == weight.dims()[0]);
    unsigned int batch_size = input.dims()[0];
    unsigned int hidden_size = input.dims()[1];
    PD_CHECK(output.dims()[0] == batch_size);
    PD_CHECK(output.dims()[1] == hidden_size);
    const cudaStream_t stream = input.stream();

    DISPATCH_FP16_DTYPE(input.dtype(), scalar_t, {
      fastdeploy::norm::RMSNorm(
          reinterpret_cast<scalar_t *>(input.data<scalar_t>()),
          reinterpret_cast<scalar_t *>(weight.data<scalar_t>()),
          reinterpret_cast<scalar_t *>(output.data<scalar_t>()),
          batch_size,
          hidden_size,
          input.strides()[0],
          output.strides()[0],
          eps,
          enable_pdl,
          stream);
    });
  } else if (input_ndim == 3) {
    // QK RMSNorm: [batch_size, num_heads, head_dim]
    // Use warp-level parallization
    CHECK_DIM(3, output);  // output: (batch_size, num_heads, hidden_size)
    PD_CHECK(input.dims()[2], weight.dims()[0]);
    unsigned int batch_size = input.dims()[0];
    unsigned int num_heads = input.dims()[1];
    unsigned int hidden_size = input.dims()[2];
    PD_CHECK(output.dims()[0], batch_size);
    PD_CHECK(output.dims()[1], num_heads);
    PD_CHECK(output.dims()[2], hidden_size);

    const cudaStream_t stream = input.stream();
    DISPATCH_FP16_DTYPE(input.dtype(), scalar_t, {
      fastdeploy::norm::QKRMSNorm(
          reinterpret_cast<scalar_t *>(input.data<scalar_t>()),
          reinterpret_cast<scalar_t *>(weight.data<scalar_t>()),
          reinterpret_cast<scalar_t *>(output.data<scalar_t>()),
          batch_size,
          num_heads,
          hidden_size,
          input.strides()[0],
          input.strides()[1],
          output.strides()[0],
          output.strides()[1],
          eps,
          enable_pdl,
          stream);
    });
  } else {
    PD_CHECK(false, "Unsupported input dimension: " + input_ndim);
  }
}

void fused_add_rmsnorm(paddle::Tensor &input,
                       paddle::Tensor &residual,
                       paddle::Tensor &weight,
                       float eps,
                       bool enable_pdl) {
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(input);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(residual);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(weight);
  CHECK_DEVICE(input, residual);
  CHECK_DEVICE(input, weight);
  CHECK_DIM(2, input);     // input: (batch_size, hidden_size)
  CHECK_DIM(2, residual);  // residual: (batch_size, hidden_size)
  CHECK_DIM(1, weight);    // weight: (hidden_size)
  unsigned int batch_size = input.dims()[0];
  unsigned int hidden_size = input.dims()[1];
  PD_CHECK(residual.dims()[0] == batch_size);
  PD_CHECK(residual.dims()[1] == hidden_size);
  PD_CHECK(weight.dims()[0] == hidden_size);
  const cudaStream_t stream = input.stream();

  DISPATCH_FP16_DTYPE(input.dtype(), scalar_t, {
    fastdeploy::norm::FusedAddRMSNorm(
        reinterpret_cast<scalar_t *>(input.data<scalar_t>()),
        reinterpret_cast<scalar_t *>(residual.data<scalar_t>()),
        reinterpret_cast<scalar_t *>(weight.data<scalar_t>()),
        batch_size,
        hidden_size,
        input.strides()[0],
        residual.strides()[0],
        eps,
        enable_pdl,
        stream);
  });
}

PD_BUILD_STATIC_OP(rmsnorm)
    .Inputs({"output", "input", "weight"})
    .Attrs({"eps: float", "enable_pdl:bool"})
    .Outputs({"out"})
    .SetInplaceMap({{"output", "out"}})
    .SetKernelFn(PD_KERNEL(rmsnorm));

PD_BUILD_STATIC_OP(fused_add_rmsnorm)
    .Inputs({"input", "residual", "weight"})
    .Attrs({"eps: float", "enable_pdl:bool"})
    .Outputs({"out_input", "out_residual"})
    .SetInplaceMap({{"input", "out_input"}, {"residual", "out_residual"}})
    .SetKernelFn(PD_KERNEL(fused_add_rmsnorm));
