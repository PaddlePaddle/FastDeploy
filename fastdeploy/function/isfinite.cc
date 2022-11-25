// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "fastdeploy/function/isfinite.h"
#include "fastdeploy/core/float16.h"
#include <algorithm>
#include <type_traits>

namespace fastdeploy {
namespace function {

template <typename T, class Enable = void> struct IsNanFunctor {
  bool operator()(const T& a) const { return std::isnan(a); }
};

template <typename T>
struct IsNanFunctor<T,
                    typename std::enable_if<std::is_integral<T>::value>::type> {
  bool operator()(const T& a) const { return false; }
};

template <> struct IsNanFunctor<fastdeploy::float16, void> {
  bool operator()(const fastdeploy::float16& a) const {
    return fastdeploy::isnan(a);
  }
};

template <typename T, class Enable = void> struct IsInfFunctor {
  bool operator()(const T& a) const { return std::isinf(a); }
};

template <typename T>
struct IsInfFunctor<T,
                    typename std::enable_if<std::is_integral<T>::value>::type> {
  bool operator()(const T& a) const { return false; }
};

template <> struct IsInfFunctor<fastdeploy::float16, void> {
  bool operator()(const fastdeploy::float16& a) const {
    return fastdeploy::isinf(a);
  }
};

template <typename T, class Enable = void> struct IsFiniteFunctor {
  bool operator()(const T& a) const { return std::isfinite(a); }
};

template <typename T>
struct IsFiniteFunctor<
    T, typename std::enable_if<std::is_integral<T>::value>::type> {
  bool operator()(const T& a) const { return true; }
};

template <> struct IsFiniteFunctor<fastdeploy::float16, void> {
  bool operator()(const fastdeploy::float16& a) const {
    return fastdeploy::isfinite(a);
  }
};

#define DEFINE_ISFINITE_KERNEL(isfinite_kernel, functor)                       \
  template <typename T>                                                        \
  void isfinite_kernel(const FDTensor& x, FDTensor* out) {                     \
    out->Allocate(x.Shape(), FDDataType::BOOL);                                \
    functor<T> unary_func;                                                     \
    bool* out_ptr = reinterpret_cast<bool*>(out->Data());                      \
    const T* input_ptr = reinterpret_cast<const T*>(x.Data());                 \
    std::transform(input_ptr, input_ptr + x.Numel(), out_ptr, unary_func);     \
  }

DEFINE_ISFINITE_KERNEL(IsNanKernel, IsNanFunctor)
DEFINE_ISFINITE_KERNEL(IsInfKernel, IsInfFunctor)
DEFINE_ISFINITE_KERNEL(IsFiniteKernel, IsFiniteFunctor)
#undef DEFINE_ISFINITE_KERNEL

void IsNan(const FDTensor& x, FDTensor* out) {
  FD_VISIT_FLOAT_TYPES(x.dtype, "IsNanKernel",
                       ([&] { IsNanKernel<data_t>(x, out); }));
}

void IsInf(const FDTensor& x, FDTensor* out) {
  FD_VISIT_FLOAT_TYPES(x.dtype, "IsInfKernel",
                       ([&] { IsInfKernel<data_t>(x, out); }));
}

void IsFinite(const FDTensor& x, FDTensor* out) {
  FD_VISIT_FLOAT_TYPES(x.dtype, "IsFiniteKernel",
                       ([&] { IsFiniteKernel<data_t>(x, out); }));
}

}  // namespace function
}  // namespace fastdeploy