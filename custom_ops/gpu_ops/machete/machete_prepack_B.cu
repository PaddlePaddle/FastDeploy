#include "machete_mm_launcher.cuh"
#include "machete_prepack_launcher.cuh"
// #include "core/scalar_type.hpp"

// #include "core/registration.h"

paddle::Tensor prepack_B(
    paddle::Tensor const& B, paddle::DataType const& a_type, int64_t b_type_id,
    std::string const& maybe_group_scales_type_str) {
  machete::ScalarType const b_type = machete::ScalarType::from_id(b_type_id);
  std::optional<paddle::DataType> maybe_group_scales_type;
  if (maybe_group_scales_type_str == "float16") {
    maybe_group_scales_type = paddle::DataType::FLOAT16;
  }
  else if (maybe_group_scales_type_str == "bfloat16") {
    maybe_group_scales_type = paddle::DataType::BFLOAT16;
  }
  else if (maybe_group_scales_type_str == "float32") {
    maybe_group_scales_type = paddle::DataType::FLOAT32;
  }
  else if (maybe_group_scales_type_str == "") {
    maybe_group_scales_type = std::nullopt;
  }
  else {
    PADDLE_ENFORCE(false, "maybe_group_scales_type_str not supported!");
  }
  return machete::prepack_B_dispatch(
      {.B = B,
       .a_type = a_type,
       .b_type = b_type,
       .maybe_group_scales_type = maybe_group_scales_type});
}

std::vector<paddle::Tensor> MachetePrepackBKernel(
    paddle::Tensor const& B, std::string const& a_type_str, std::string const& b_type_str,
    std::string const& maybe_group_scales_type_str) {

  machete::ScalarTypeId b_type_id;
  paddle::DataType a_type, maybe_group_scales_type;

  if (b_type_str == "uint4") {
    b_type_id = machete::kU4.id();
  } else if (b_type_str == "uint4b8") {
    b_type_id = machete::kU4B8.id();
  } else {
    PADDLE_ENFORCE(false, "b_type_str not supported!");
  }

  if (a_type_str == "float16") {
    a_type = paddle::DataType::FLOAT16;
  }
  else if (a_type_str == "bfloat16") {
    a_type = paddle::DataType::BFLOAT16;
  }
  else {
    PADDLE_ENFORCE(false, "a_type_str not supported!");
  }
  auto Bt = paddle::experimental::transpose(B, {1, 0});
  // printf("MachetePrepackBKernel B.shape: %d, %d\n", Bt.shape()[0], Bt.shape()[1]);
  // printf("MachetePrepackBKernel B.strides: %d, %d\n", Bt.strides()[0], Bt.strides()[1]);

  paddle::Tensor B_prepacked = prepack_B(Bt, a_type, b_type_id, maybe_group_scales_type_str);
  return {B_prepacked};

}

// std::vector<std::vector<int64_t>> MachetePrepackBInferShape(
//     std::vector<int64_t> const& B_shape, paddle::DataType const& a_type, std::string b_type_str,
//     std::optional<paddle::DataType> const& maybe_group_scales_type) {
//   return {B_shape};
// }

// std::vector<paddle::DataType> MachetePrepackBInferDtype(
//     paddle::DataType const& B_type, paddle::DataType const& a_type, std::string b_type_str,
//     std::optional<paddle::DataType> const& maybe_group_scales_type) {
//   return {B_type};
// }

PD_BUILD_STATIC_OP(machete_prepack_B)
    .Inputs({"B"})
    .Outputs({"B_prepacked"})
    .Attrs({"a_type_str:std::string", "b_type_str:std::string", "maybe_group_scales_type_str:std::string"})
    .SetKernelFn(PD_KERNEL(MachetePrepackBKernel));
    // .SetInferShapeFn(PD_INFER_SHAPE(MachetePrepackBInferShape))
    // .SetInferDtypeFn(PD_INFER_DTYPE(MachetePrepackBInferDtype));
