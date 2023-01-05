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

#include "fastdeploy/runtime/enum_variables.h"

namespace fastdeploy {
std::ostream& operator<<(std::ostream& out, const Backend& backend) {
  if (backend == Backend::ORT) {
    out << "Backend::ORT";
  } else if (backend == Backend::TRT) {
    out << "Backend::TRT";
  } else if (backend == Backend::PDINFER) {
    out << "Backend::PDINFER";
  } else if (backend == Backend::OPENVINO) {
    out << "Backend::OPENVINO";
  } else if (backend == Backend::RKNPU2) {
    out << "Backend::RKNPU2";
  } else if (backend == Backend::SOPHGOTPU) {
    out << "Backend::SOPHGOTPU";
  } else if (backend == Backend::POROS) {
    out << "Backend::POROS";
  } else if (backend == Backend::LITE) {
    out << "Backend::PDLITE";
  } else {
    out << "UNKNOWN-Backend";
  }
  return out;
}

}  // namespace fastdeploy
