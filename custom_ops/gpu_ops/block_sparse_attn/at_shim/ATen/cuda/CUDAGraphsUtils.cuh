// Stub for ATen/cuda/CUDAGraphsUtils.cuh
// Provides at::cuda::philox::unpack with the correct *signature*. Inference
// never enters the dropout path (p_dropout==0), so the returned values are
// irrelevant — they just need to compile.
#pragma once

#include <cstdint>
#include <tuple>

namespace at { namespace cuda { namespace philox {

// Templated to accept whatever PhiloxCudaState definition wins at the call
// site (paddle compat layer may define its own).
template <typename T>
__host__ __device__ inline std::tuple<uint64_t, uint64_t>
unpack(const T& /*arg*/) {
  return std::make_tuple(uint64_t(0), uint64_t(0));
}

}}}  // namespace at::cuda::philox
