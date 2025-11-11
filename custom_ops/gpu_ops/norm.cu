// adapted from:
// https://github.com/flashinfer-ai/flashinfer/blob/0e48aaf941a6b05f6557c9c9f606884f826afedd/csrc/norm.cu
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

std::vector<paddle::Tensor> fused_add_rmsnorm(paddle::Tensor& input,
                                              paddle::Tensor& residual,
                                              paddle::Tensor& weight,
                                              const float eps,
                                              const bool enable_pdl) {
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
        reinterpret_cast<scalar_t*>(input.data<scalar_t>()),
        reinterpret_cast<scalar_t*>(residual.data<scalar_t>()),
        reinterpret_cast<scalar_t*>(weight.data<scalar_t>()),
        batch_size,
        hidden_size,
        input.strides()[0],
        residual.strides()[0],
        eps,
        enable_pdl,
        stream);
  });
  return {input, residual};
}

std::vector<paddle::DataType> FusedAddRMSNormTcInferDtype(
    const paddle::DataType& input_dtype,
    const paddle::DataType& residual_dtype,
    const paddle::DataType& weight_dtype,
    const float eps,
    const bool enable_pdl) {
  return {input_dtype, residual_dtype};
}

std::vector<std::vector<int64_t>> FusedAddRMSNormTcInferShape(
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& residual_shape,
    const std::vector<int64_t>& weight_shape,
    const float eps,
    const bool enable_pdl) {
  return {input_shape, residual_shape};
}

PD_BUILD_STATIC_OP(fused_add_rmsnorm)
    .Inputs({"input", "residual", "weight"})
    .Attrs({"eps: float", "enable_pdl:bool"})
    .Outputs({"out_input", "out_residual"})
    .SetInplaceMap({{"input", "out_input"}, {"residual", "out_residual"}})
    .SetKernelFn(PD_KERNEL(fused_add_rmsnorm))
    .SetInferShapeFn(PD_INFER_SHAPE(FusedAddRMSNormTcInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FusedAddRMSNormTcInferDtype));
