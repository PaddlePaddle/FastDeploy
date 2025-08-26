#pragma once

#include "machete_prepack_kernel.cuh"
#include "utils/paddle_utils.hpp"
#include "utils/scalar_type.h"

namespace machete {

struct PrepackBArgs {
  paddle::Tensor const& B;
  paddle::DataType a_type;
  machete::ScalarType b_type;
  std::optional<paddle::DataType> maybe_group_scales_type;
};

template <typename PrepackedLayoutB>
paddle::Tensor prepack_impl(paddle::Tensor const B) {
  // const at::cuda::OptionalCUDAGuard device_guard(device_of(B));
  using ElementB = typename PrepackedLayoutB::ElementB;
  using PPBlockShape_NK = typename PrepackedLayoutB::PPBlockShape_NK;

  // auto device = B.device();
  // auto stream = at::cuda::getCurrentCUDAStream(device.index());
  cudaStream_t stream = B.stream();
  auto B_ptr = static_cast<ElementB const*>(B.data());
  // elements per storage item for B
  auto eles_per_storage =
      (SizeOf(B.dtype()) * 8) / cute::sizeof_bits_v<ElementB>;

  // paddle B passed in is/should be (packed_K,N), the kernel expects (N,K,L) (to
  // match cutlass using (N,K,L) for B), so we transpose B to (N,packed_K,L)
  // auto Bt_packed = B.transpose();
  auto Bt_packed = paddle::experimental::transpose(B, {1, 0});

  PD_CHECK(
      (B.shape()[0] * eles_per_storage) % size<1>(PPBlockShape_NK{}) == 0,
      "B.shape[0] (in terms of unpacked elements) must be a multiple of ",
      size<1>(PPBlockShape_NK{}));
  PD_CHECK(B.shape()[1] % size<0>(PPBlockShape_NK{}) == 0,
              "B.shape[1] must be a multiple of ", size<0>(PPBlockShape_NK{}));

  using StrideB = cutlass::detail::TagToStrideB_t<cutlass::layout::ColumnMajor>;
  auto const l_Bt_packed = make_cute_layout<StrideB>(Bt_packed, "B");
  // auto const l_Bt_packed = make_cute_layout<StrideB>(B, "B");

  // convert (N,packed_K,L) layout to (N,K,L) layout
  //  in effect we want to do: blocked_product(layout_Bt_packed,
  //      make_ordered_layout(make_shape(_1{}, eles_per_storage, _1{}),
  //                          Step<_1, _0, _2>{}));
  // but blocked_product does not support dynamic strides so we implement the
  // equivalent manually,
  //   new_shape = (N, packed_K, L) * (1, eles_per_storage, 1) -> (N, K, L)
  //   new_stride = (s0, s1, s2) * (eles_per_storage, 1, eles_per_storage)
  //                 when s1 == 1
  PD_CHECK(stride<1>(l_Bt_packed) == 1, "stride<1>(l_Bt_packed) must be 1");
  // clang-format off
  auto const layout_Bt = make_layout(
      transform_with_idx(l_Bt_packed.shape(), [&](auto ele, auto idx) {
        return idx == 1 ? ele * eles_per_storage : ele;
      }),
      transform_with_idx(l_Bt_packed.stride(), [&](auto ele, auto idx) {
        return idx != 1 ? ele * eles_per_storage : ele;
      }));
  // clang-format on

  // Allocate output
  paddle::Tensor D = paddle::empty_like(B);

  prepack_B_template<PrepackedLayoutB>(
      stream, B_ptr, layout_Bt, static_cast<ElementB*>(D.data()));

  return D;
};

paddle::Tensor prepack_B_dispatch(PrepackBArgs args);

};  // namespace machete
