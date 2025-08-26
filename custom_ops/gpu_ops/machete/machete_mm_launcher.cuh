#pragma once

#include <Python.h>

#include "machete_mm_kernel.cuh"
#include "utils/paddle_utils.hpp"
#include "utils/scalar_type.h"

namespace machete {

struct MMArgs {
  paddle::Tensor const& A;
  paddle::Tensor const& B;
  machete::ScalarType const& b_type;
  std::optional<paddle::DataType> const& maybe_out_type;
  std::optional<paddle::Tensor> const& maybe_group_scales;
  std::optional<paddle::Tensor> const& maybe_group_zeros;
  std::optional<int64_t> maybe_group_size;
  std::optional<paddle::Tensor> const& maybe_channel_scales;
  std::optional<paddle::Tensor> const& maybe_token_scales;
  std::optional<std::string> maybe_schedule;
};

struct SupportedSchedulesArgs {
  paddle::DataType a_type;
  machete::ScalarType b_type;
  std::optional<paddle::DataType> maybe_group_scales_type;
  std::optional<paddle::DataType> maybe_group_zeros_type;
  std::optional<paddle::DataType> maybe_channel_scales_type;
  std::optional<paddle::DataType> maybe_token_scales_type;
  std::optional<paddle::DataType> maybe_out_type;
};

paddle::Tensor mm_dispatch(MMArgs args);

std::vector<std::string> supported_schedules_dispatch(
    SupportedSchedulesArgs args);

template <typename MacheteKernel>
paddle::Tensor run_impl(MMArgs args) {
  // const at::cuda::OptionalCUDAGuard device_guard(device_of(args.A));

  // auto device = args.A.device();
  // auto stream = at::cuda::getCurrentCUDAStream(device.index());
  auto place = args.A.place();
  cudaStream_t stream = args.A.stream();

  int M = args.A.shape()[0];
  int N = args.B.shape()[1];
  int K = args.A.shape()[1];

  // Allocate output
  paddle::Tensor D = paddle::empty(
      {M, N},
      equivalent_scalar_type_v<typename MacheteKernel::ElementD>,
      place);

  auto arguments = MacheteKernel::create_arguments(
      stream,  //
      args.A, args.B, D, args.maybe_group_scales, args.maybe_group_zeros,
      args.maybe_group_size, args.maybe_channel_scales,
      args.maybe_token_scales);
  PD_CHECK(MacheteKernel::can_implement(arguments),
              "Machete kernel cannot be run with these arguments");

  size_t workspace_size = MacheteKernel::get_workspace_size(arguments);
  int S = static_cast<int>(workspace_size);
  // phi::Allocator* allocator = paddle::GetAllocator(place);
  // auto workspace = allocator->Allocate(workspace_size);
  // MacheteKernel::run(arguments, workspace->ptr(), stream);
  // paddle::Tensor workspace = paddle::empty({S}, paddle::DataType::UINT8, place);
  paddle::Tensor workspace = GetEmptyTensor({S}, paddle::DataType::UINT8, place);
  MacheteKernel::run(arguments, workspace.data(), stream);

  return D;
};

};  // namespace machete
