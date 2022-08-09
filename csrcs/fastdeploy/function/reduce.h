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

#include "fastdeploy/core/fd_tensor.h"

namespace fastdeploy {

FASTDEPLOY_DECL void Max(const FDTensor& x, FDTensor* out,
                         const std::vector<int64_t>& dims,
                         bool keep_dim = false, bool reduce_all = false);

FASTDEPLOY_DECL void Min(const FDTensor& x, FDTensor* out,
                         const std::vector<int64_t>& dims,
                         bool keep_dim = false, bool reduce_all = false);

FASTDEPLOY_DECL void Sum(const FDTensor& x, FDTensor* out,
                         const std::vector<int64_t>& dims,
                         bool keep_dim = false, bool reduce_all = false);

FASTDEPLOY_DECL void All(const FDTensor& x, FDTensor* out,
                         const std::vector<int64_t>& dims,
                         bool keep_dim = false, bool reduce_all = false);

FASTDEPLOY_DECL void Any(const FDTensor& x, FDTensor* out,
                         const std::vector<int64_t>& dims,
                         bool keep_dim = false, bool reduce_all = false);

FASTDEPLOY_DECL void Mean(const FDTensor& x, FDTensor* out,
                          const std::vector<int64_t>& dims,
                          bool keep_dim = false, bool reduce_all = false);

FASTDEPLOY_DECL void Prod(const FDTensor& x, FDTensor* out,
                          const std::vector<int64_t>& dims,
                          bool keep_dim = false, bool reduce_all = false);

}  // namespace fastdeploy
