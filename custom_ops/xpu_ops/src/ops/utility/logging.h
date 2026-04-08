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

#pragma once

#include <assert.h>
#include <time.h>
#if !defined(_WIN32)
#include <sys/time.h>
#include <sys/types.h>
#else
#define NOMINMAX  // msvc max/min macro conflict with std::min/max
#include <windows.h>
#undef min
#undef max
extern struct timeval;
static int gettimeofday(struct timeval* tp, void* tzp) {
  LARGE_INTEGER now, freq;
  QueryPerformanceCounter(&now);
  QueryPerformanceFrequency(&freq);
  tp->tv_sec = now.QuadPart / freq.QuadPart;
  tp->tv_usec = (now.QuadPart % freq.QuadPart) * 1000000 / freq.QuadPart;
  return (0);
}
#endif

#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>

// LOG()
#define LOG(status) LOG_##status.stream()
#define LOG_INFO paddle::CustomLogMessage(__FILE__, __FUNCTION__, __LINE__, "I")
#define LOG_ERROR LOG_INFO
#define LOG_WARNING \
  paddle::CustomLogMessage(__FILE__, __FUNCTION__, __LINE__, "W")
#define LOG_FATAL \
  paddle::CustomLogMessageFatal(__FILE__, __FUNCTION__, __LINE__)

// VLOG()
#define VLOG(level) \
  paddle::CustomVLogMessage(__FILE__, __FUNCTION__, __LINE__, level).stream()

namespace paddle {

struct CustomException : public std::exception {
  const std::string exception_prefix = "Custom exception: \n";
  std::string message;
  explicit CustomException(const char* detail) {
    message = exception_prefix + std::string(detail);
  }
  const char* what() const noexcept { return message.c_str(); }
};

class CustomLogMessage {
 public:
  CustomLogMessage(const char* file,
                   const char* func,
                   int lineno,
                   const char* level = "I");
  ~CustomLogMessage();

  std::ostream& stream() { return log_stream_; }

 protected:
  std::stringstream log_stream_;
  std::string level_;

  CustomLogMessage(const CustomLogMessage&) = delete;
  void operator=(const CustomLogMessage&) = delete;
};

class CustomLogMessageFatal : public CustomLogMessage {
 public:
  CustomLogMessageFatal(const char* file,
                        const char* func,
                        int lineno,
                        const char* level = "F")
      : CustomLogMessage(file, func, lineno, level) {}
  ~CustomLogMessageFatal() noexcept(false);
};

class CustomVLogMessage {
 public:
  CustomVLogMessage(const char* file,
                    const char* func,
                    int lineno,
                    const int32_t level_int = 0);
  ~CustomVLogMessage();

  std::ostream& stream() { return log_stream_; }

 protected:
  std::stringstream log_stream_;
  int32_t GLOG_v_int;
  int32_t level_int;

  CustomVLogMessage(const CustomVLogMessage&) = delete;
  void operator=(const CustomVLogMessage&) = delete;
};

}  // namespace paddle
