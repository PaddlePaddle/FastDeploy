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

#include <iostream>
#include <sstream>
#include <string>

namespace paddle2onnx {

inline void Assert(bool condition, const std::string& message) {
  if (!condition) {
    fprintf(stderr, "[ERROR] %s\n", message.c_str());
    std::abort();
  }
}

inline const std::string RequireOpset(const int32_t& opset_version) {
  return "Requires the minimal opset version of " +
         std::to_string(opset_version) + ".";
}

class P2OLogger {
 public:
  P2OLogger() {
    line_ = "";
    prefix_ = "[Paddle2ONNX]";
    verbose_ = true;
  }
  explicit P2OLogger(bool verbose,
                     const std::string& prefix = "[Paddle2ONNX]") {
    verbose_ = verbose;
    line_ = "";
    prefix_ = prefix;
  }

  template <typename T>
  P2OLogger& operator<<(const T& val) {
    if (!verbose_) {
      return *this;
    }
    std::stringstream ss;
    ss << val;
    line_ += ss.str();
    return *this;
  }
  P2OLogger& operator<<(std::ostream& (*os)(std::ostream&)) {
    if (!verbose_) {
      return *this;
    }
    std::cout << prefix_ << " " << line_ << std::endl;
    line_ = "";
    return *this;
  }
  ~P2OLogger() {
    if (!verbose_ && line_ != "") {
      std::cout << line_ << std::endl;
    }
  }

 private:
  std::string line_;
  std::string prefix_;
  bool verbose_ = true;
};

}  // namespace paddle2onnx
