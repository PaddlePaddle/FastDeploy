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

#pragma once

#include <stdlib.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#if defined(_WIN32)
#ifdef FASTDEPLOY_LIB
#define FASTDEPLOY_DECL __declspec(dllexport)
#else
#define FASTDEPLOY_DECL __declspec(dllimport)
#endif  // FASTDEPLOY_LIB
#else
#define FASTDEPLOY_DECL __attribute__((visibility("default")))
#endif  // _WIN32

namespace fastdeploy {

class FASTDEPLOY_DECL FDLogger {
 public:
  FDLogger() {
    line_ = "";
    prefix_ = "[FastDeploy]";
    verbose_ = true;
  }
  explicit FDLogger(bool verbose, const std::string& prefix = "[FastDeploy]");

  template <typename T>
  FDLogger& operator<<(const T& val) {
    if (!verbose_) {
      return *this;
    }
    std::stringstream ss;
    ss << val;
    line_ += ss.str();
    return *this;
  }
  FDLogger& operator<<(std::ostream& (*os)(std::ostream&));
  ~FDLogger() {
    if (!verbose_ && line_ != "") {
      std::cout << line_ << std::endl;
    }
  }

 private:
  std::string line_;
  std::string prefix_;
  bool verbose_ = true;
};

#ifndef __REL_FILE__
#define __REL_FILE__ __FILE__
#endif

#define FDERROR                                                \
  FDLogger(true, "[ERROR]") << __REL_FILE__ << "(" << __LINE__ \
                            << ")::" << __FUNCTION__ << "\t"

#define FDASSERT(condition, message) \
  if (!(condition)) {                \
    FDERROR << message << std::endl; \
    std::abort();                    \
  }

}  // namespace fastdeploy
