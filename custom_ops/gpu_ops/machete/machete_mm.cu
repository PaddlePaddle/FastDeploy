#include "machete_mm_launcher.cuh"
#include "machete_prepack_launcher.cuh"
// #include "core/scalar_type.hpp"

// #include "core/registration.h"

template <typename T>
std::optional<T> ConvertToStdOptional(const paddle::optional<T>& paddle_opt) {
    return paddle_opt ? std::optional<T>(paddle_opt.get()) : std::nullopt;
}

paddle::Tensor mm(paddle::Tensor const& A, paddle::Tensor const& B,
                 int64_t b_type_id,
                 std::optional<paddle::DataType> const& maybe_out_type,
                 std::optional<paddle::Tensor> const& maybe_group_scales,
                 std::optional<paddle::Tensor> const& maybe_group_zeros,
                 int64_t maybe_group_size,
                 std::optional<paddle::Tensor> const& maybe_channel_scales,
                 std::optional<paddle::Tensor> const& maybe_token_scales,
                 std::string maybe_schedule) {
  machete::ScalarType const b_type = machete::ScalarType::from_id(b_type_id);
  std::optional<int64_t> maybe_group_size_opt;
  std::optional<std::string> maybe_schedule_opt;
  // if (maybe_group_size == -1) {
  //   maybe_group_size_opt = std::nullopt;
  // }
  if (maybe_schedule == "") {
    maybe_schedule_opt = std::nullopt;
  }
  return machete::mm_dispatch({.A = A,
                      .B = B,
                      .b_type = b_type,
                      .maybe_out_type = maybe_out_type,
                      .maybe_group_scales = maybe_group_scales,
                      .maybe_group_zeros = maybe_group_zeros,
                      .maybe_group_size = maybe_group_size_opt,
                      .maybe_channel_scales = maybe_channel_scales,
                      .maybe_token_scales = maybe_token_scales,
                      .maybe_schedule = maybe_schedule_opt});
}

std::vector<paddle::Tensor> MacheteMMKernel(
    paddle::Tensor const& A, paddle::Tensor const& B,
    paddle::optional<paddle::Tensor> const& maybe_group_scales,
    paddle::optional<paddle::Tensor> const& maybe_group_zeros,
    paddle::optional<paddle::Tensor> const& maybe_channel_scales,
    paddle::optional<paddle::Tensor> const& maybe_token_scales,
    std::string const& b_type_str,
    std::string const& maybe_out_type_str,
    int64_t const& maybe_group_size,
    std::string const& maybe_schedule
  ) {

  machete::ScalarTypeId b_type_id;
  paddle::DataType maybe_out_type;
  if (b_type_str == "uint4") {
    b_type_id = machete::kU4.id();
  } else if (b_type_str == "uint4b8") {
    b_type_id = machete::kU4B8.id();
  } else {
    PADDLE_ENFORCE(false, "b_type_str not supported!");
  }
  if (maybe_out_type_str == "float16") {
    maybe_out_type = paddle::DataType::FLOAT16;
  } else if (maybe_out_type_str == "bfloat16") {
    maybe_out_type = paddle::DataType::BFLOAT16;
  } else {
    maybe_out_type = A.dtype();
  }
  auto out = mm(A, B, b_type_id, maybe_out_type,
                ConvertToStdOptional<paddle::Tensor>(maybe_group_scales),
                ConvertToStdOptional<paddle::Tensor>(maybe_group_zeros),
                maybe_group_size,
                ConvertToStdOptional<paddle::Tensor>(maybe_channel_scales),
                ConvertToStdOptional<paddle::Tensor>(maybe_token_scales),
                maybe_schedule);
  return {out};
}

// PD_BUILD_STATIC_OP(machete_mm)
//     .Inputs({"A", "B",
//             paddle::Optional("maybe_group_scales"),
//             paddle::Optional("maybe_group_zeros"),
//             paddle::Optional("maybe_channel_scales"),
//             paddle::Optional("maybe_token_scales")})
//     .Outputs({"out"})
//     .Attrs({"b_type_str:std::string", "maybe_out_type_str:std::string", "maybe_group_size:int64_t", "maybe_schedule:std::string"})
//     .SetKernelFn(PD_KERNEL(MacheteMMKernel));
