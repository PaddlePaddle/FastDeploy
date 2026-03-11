// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

// CCCL 3.0 compatibility header for CUDA 13.0+
// In CCCL 3.0, cub::Sum, cub::Max, cub::Min are removed from the cub namespace.
// This header provides compatible implementations that work with both old and
// new versions.

// Include cub headers based on platform
#ifdef PADDLE_WITH_HIP
#include <hipcub/hipcub.hpp>
#else
#include <cub/cub.cuh>
#endif

// Detect CUDA 13.0+ (CCCL 3.0)
// __CUDACC_VER_MAJOR__ >= 13 indicates CUDA 13.0 or later
#if defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ >= 13
#define FD_CCCL_V3 1
#endif

namespace fd_cub_compat {

// ============================================================================
// Sum, Max, Min functors
// ============================================================================

#ifdef FD_CCCL_V3
// CUDA 13.0+ (CCCL 3.0): Use custom implementations since cub::Sum/Max/Min are
// removed

/// Functor for computing the sum of two values
struct Sum {
  /// Apply the sum operation
  template <typename T>
  __host__ __device__ __forceinline__ T operator()(const T &a,
                                                   const T &b) const {
    return a + b;
  }
};

/// Functor for computing the maximum of two values
struct Max {
  /// Apply the max operation
  template <typename T>
  __host__ __device__ __forceinline__ T operator()(const T &a,
                                                   const T &b) const {
    return (b > a) ? b : a;
  }
};

/// Functor for computing the minimum of two values
struct Min {
  /// Apply the min operation
  template <typename T>
  __host__ __device__ __forceinline__ T operator()(const T &a,
                                                   const T &b) const {
    return (b < a) ? b : a;
  }
};

#else
// CUDA 12.x and earlier: Use native cub implementations

#ifdef PADDLE_WITH_HIP
using Sum = hipcub::Sum;
using Max = hipcub::Max;
using Min = hipcub::Min;
#else
using Sum = cub::Sum;
using Max = cub::Max;
using Min = cub::Min;
#endif

#endif  // FD_CCCL_V3

// ============================================================================
// ArgMax, ArgMin functors
// These are also removed in CCCL 3.0
// ============================================================================

#ifdef FD_CCCL_V3
// CUDA 13.0+ (CCCL 3.0): Use custom implementations since cub::ArgMax/ArgMin
// are removed

/// Functor for computing the ArgMax of two values (for cub::BlockReduce with
/// KeyValuePair) Returns the key-value pair with the larger value
struct ArgMax {
  /// Apply ArgMax operation (returns pair with max value and its key/index)
  template <typename KeyValuePair>
  __host__ __device__ __forceinline__ KeyValuePair
  operator()(const KeyValuePair &a, const KeyValuePair &b) const {
    return (b.value > a.value) ? b : a;
  }
};

/// Functor for computing the ArgMin of two values (for cub::BlockReduce with
/// KeyValuePair) Returns the key-value pair with the smaller value
struct ArgMin {
  /// Apply ArgMin operation (returns pair with min value and its key/index)
  template <typename KeyValuePair>
  __host__ __device__ __forceinline__ KeyValuePair
  operator()(const KeyValuePair &a, const KeyValuePair &b) const {
    return (b.value < a.value) ? b : a;
  }
};

#else
// CUDA 12.x and earlier: Use native cub implementations

#ifdef PADDLE_WITH_HIP
using ArgMax = hipcub::ArgMax;
using ArgMin = hipcub::ArgMin;
#else
// For older CUDA versions, wrap the native cub::ArgMax/ArgMin
struct ArgMax {
  template <typename KeyValuePair>
  __host__ __device__ __forceinline__ KeyValuePair
  operator()(const KeyValuePair &a, const KeyValuePair &b) const {
    cub::ArgMax argmax;
    return argmax(a, b);
  }
};

struct ArgMin {
  template <typename KeyValuePair>
  __host__ __device__ __forceinline__ KeyValuePair
  operator()(const KeyValuePair &a, const KeyValuePair &b) const {
    cub::ArgMin argmin;
    return argmin(a, b);
  }
};

#endif  // PADDLE_WITH_HIP

#endif  // FD_CCCL_V3

}  // namespace fd_cub_compat
