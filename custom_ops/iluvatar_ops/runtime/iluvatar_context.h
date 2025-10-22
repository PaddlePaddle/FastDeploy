// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
#include <ixinfer.h>
#include <iostream>
#include <vector>

#define CUINFER_CHECK(func)                                                \
  do {                                                                     \
    cuinferStatus_t status = (func);                                       \
    if (status != CUINFER_STATUS_SUCCESS) {                                \
      std::cerr << "Error in file " << __FILE__ << " on line " << __LINE__ \
                << ": " << cuinferGetErrorString(status) << std::endl;     \
      throw std::runtime_error("CUINFER_CHECK ERROR");                     \
    }                                                                      \
  } while (0)

namespace iluvatar {

class IluvatarContext {
 public:
  IluvatarContext() = default;
  ~IluvatarContext();

  cuinferHandle_t getIxInferHandle();

 private:
  cuinferHandle_t ixinfer_handle_{nullptr};
};
IluvatarContext* getContextInstance();

}  // namespace iluvatar
