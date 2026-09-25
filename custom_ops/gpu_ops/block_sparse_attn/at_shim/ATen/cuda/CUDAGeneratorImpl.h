// Stub for ATen/cuda/CUDAGeneratorImpl.h
// Purpose: satisfy Block-Sparse-Attention's `flash.h` which only needs
// the type `at::PhiloxCudaState`. Dropout RNG is never used in inference
// (p_dropout==0), so any non-degenerate definition is fine; we only
// need it to compile.
#pragma once
#include <cstdint>

namespace at {

struct PhiloxCudaState {
  uint64_t seed_ = 0;
  uint64_t offset_ = 0;
  bool captured_ = false;
  // dummy fields to mimic upstream layout
  struct Payload { uint64_t* ptr = nullptr; uint64_t val = 0; };
  Payload seed{};
  Payload offset{};
  uint64_t offset_intragraph_ = 0;

  PhiloxCudaState() = default;
  PhiloxCudaState(uint64_t s, uint64_t o) : seed_(s), offset_(o) {}
};

}  // namespace at
