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

#include "app/base_app.h"
#include "fastdeploy/utils/utils.h"
#include "fastdeploy/core/fd_tensor.h"

#include <memory>

namespace fastdeploy {
namespace streamer {

/*! @brief FDStreamer class, user inferfaces for FastDeploy Streamer
 */
class FASTDEPLOY_DECL FDStreamer {
 public:
  /** \brief Init FD streamer
   *
   * \param[in] config_file config file path
   * \return true if the streamer is initialized, otherwise false
   */
  bool Init(const std::string& config_file);

  bool Run();

  bool RunAsync();

  void SetupCallback();

  bool TryPullFrame(FDTensor& tensor, int timeout_ms = 1);

  bool Destroyed() {
    return app_->Destroyed();
  }

 private:
  std::unique_ptr<BaseApp> app_;
};
}  // namespace streamer
}  // namespace fastdeploy
