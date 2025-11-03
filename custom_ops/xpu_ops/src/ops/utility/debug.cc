// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "ops/utility/debug.h"
#include <stdarg.h>
#include <cmath>  // for std::sqrt
#include <cstring>
#include <memory>
#include <numeric>  // for std::accumulate
#include <sstream>
#include <utility>
#include <vector>
#include "paddle/phi/common/float16.h"
#include "xpu/internal/infra_op.h"

namespace paddle {

std::string string_format(const std::string fmt_str, ...) {
  // Reserve two times as much as the length of the fmt_str
  int final_n, n = (static_cast<int>(fmt_str.size())) * 2;
  std::unique_ptr<char[]> formatted;
  va_list ap;
  while (1) {
    formatted.reset(new char[n]);
    // Wrap the plain char array into the unique_ptr
    std::strcpy(&formatted[0], fmt_str.c_str());  // NOLINT
    va_start(ap, fmt_str);
    final_n = vsnprintf(&formatted[0], n, fmt_str.c_str(), ap);
    va_end(ap);
    if (final_n < 0 || final_n >= n)
      n += std::abs(final_n - n + 1);
    else
      break;
  }
  return std::string(formatted.get());
}

std::string shape_to_string(const std::vector<int64_t>& shape) {
  std::ostringstream os;
  auto rank = shape.size();
  if (rank > 0) {
    os << shape[0];
    for (size_t i = 1; i < rank; i++) {
      os << ", " << shape[i];
    }
  }
  return os.str();
}

template <typename T>
float cal_mean(const std::vector<T>& data) {
  return std::accumulate(data.begin(), data.end(), 0.f) /
         static_cast<float>(data.size());
}

template <typename T>
float cal_std(const std::vector<T>& data) {
  float mean = cal_mean(data);
  float variance = std::accumulate(data.begin(),
                                   data.end(),
                                   0.0,
                                   [mean](T acc, T val) {
                                     return acc + (val - mean) * (val - mean);
                                   }) /
                   data.size();
  return std::sqrt(variance);
}

template <typename T>
void DebugPrintXPUTensor(const phi::XPUContext* xpu_ctx,
                         const paddle::Tensor& input,
                         std::string tag,
                         int len) {
  const T* input_data_ptr = input.data<T>();
  std::vector<T> input_data(len);
  xpu::do_device2host(
      xpu_ctx->x_context(), input_data_ptr, input_data.data(), len);
  for (int i = 0; i < len; ++i) {
    std::cout << "DebugPrintXPUTensor " << tag << ", data: " << input_data[i]
              << std::endl;
  }

  std::cout << "DebugPrintXPUTensor " << tag
            << ", mean: " << cal_mean(input_data) << std::endl;
  std::cout << "DebugPrintXPUTensor " << tag << ", std: " << cal_std(input_data)
            << std::endl;
}

template <typename T>
void DebugPrintXPUTensorv2(const paddle::Tensor& input,
                           std::string tag,
                           int len) {
  auto input_cpu = input.copy_to(phi::CPUPlace(), false);
  std::ostringstream os;

  const T* input_data = input_cpu.data<T>();
  for (int i = 0; i < len; ++i) {
    os << input_data[i] << ", ";
  }
  std::cout << "DebugPrintXPUTensorv2 " << tag << ", data: " << os.str()
            << std::endl;
}

template <>
void DebugPrintXPUTensorv2<paddle::float16>(const paddle::Tensor& input,
                                            std::string tag,
                                            int len) {
  auto input_cpu = input.copy_to(phi::CPUPlace(), false);
  std::ostringstream os;

  const paddle::float16* input_data = input_cpu.data<paddle::float16>();
  for (int i = 0; i < len; ++i) {
    os << static_cast<float>(input_data[i]) << ", ";
  }
  std::cout << "DebugPrintXPUTensorv2 " << tag << ", data: " << os.str()
            << std::endl;
}

template <>
void DebugPrintXPUTensorv2<paddle::bfloat16>(const paddle::Tensor& input,
                                             std::string tag,
                                             int len) {
  auto input_cpu = input.copy_to(phi::CPUPlace(), false);
  std::ostringstream os;

  const paddle::bfloat16* input_data = input_cpu.data<paddle::bfloat16>();
  for (int i = 0; i < len; ++i) {
    os << static_cast<float>(input_data[i]) << ", ";
  }
  std::cout << "DebugPrintXPUTensorv2 " << tag << ", data: " << os.str()
            << std::endl;
}

template <>
void DebugPrintXPUTensorv2<int8_t>(const paddle::Tensor& input,
                                   std::string tag,
                                   int len) {
  auto input_cpu = input.copy_to(phi::CPUPlace(), false);

  std::ostringstream os;

  const int8_t* input_data = input_cpu.data<int8_t>();
  for (int i = 0; i < len; ++i) {
    int8_t tmp = input_data[i] >> 4;
    os << (int32_t)tmp << ", ";
  }
  std::cout << "DebugPrintXPUTensorv2 " << tag << ", data: " << os.str()
            << std::endl;
}

#define INSTANTIATE_DEBUGPRINT_XPUTENSOR(Type, FuncName, ...) \
  template void FuncName<Type>(__VA_ARGS__);

#define INSTANTIATE_DEBUGPRINT_XPUTENSOR_V1(Type)                  \
  INSTANTIATE_DEBUGPRINT_XPUTENSOR(Type,                           \
                                   DebugPrintXPUTensor,            \
                                   const phi::XPUContext* xpu_ctx, \
                                   const paddle::Tensor& input,    \
                                   std::string tag,                \
                                   int len)

#define INSTANTIATE_DEBUGPRINT_XPUTENSOR_V2(Type)               \
  INSTANTIATE_DEBUGPRINT_XPUTENSOR(Type,                        \
                                   DebugPrintXPUTensorv2,       \
                                   const paddle::Tensor& input, \
                                   std::string tag,             \
                                   int len)

// do not support bool type now, please use DebugPrintXPUTensorv2<bool>
// INSTANTIATE_DEBUGPRINT_XPUTENSOR_V1(bool)
INSTANTIATE_DEBUGPRINT_XPUTENSOR_V1(float)
INSTANTIATE_DEBUGPRINT_XPUTENSOR_V1(int)
INSTANTIATE_DEBUGPRINT_XPUTENSOR_V1(int64_t)

INSTANTIATE_DEBUGPRINT_XPUTENSOR_V2(int8_t)
INSTANTIATE_DEBUGPRINT_XPUTENSOR_V2(bool)
INSTANTIATE_DEBUGPRINT_XPUTENSOR_V2(int64_t)
INSTANTIATE_DEBUGPRINT_XPUTENSOR_V2(float)
INSTANTIATE_DEBUGPRINT_XPUTENSOR_V2(int)
INSTANTIATE_DEBUGPRINT_XPUTENSOR_V2(paddle::float16)
INSTANTIATE_DEBUGPRINT_XPUTENSOR_V2(paddle::bfloat16)

}  // namespace paddle
