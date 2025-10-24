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

#include "iluvatar_context.h"

#include <memory>
#include <mutex>
namespace iluvatar {
IluvatarContext::~IluvatarContext() {
  if (ixinfer_handle_) {
    cuinferDestroy(ixinfer_handle_);
  }
}
cuinferHandle_t IluvatarContext::getIxInferHandle() {
  if (!ixinfer_handle_) {
    cuinferCreate(&ixinfer_handle_);
  }
  return ixinfer_handle_;
}

IluvatarContext* getContextInstance() {
  static IluvatarContext context;
  return &context;
}
}  // namespace iluvatar
