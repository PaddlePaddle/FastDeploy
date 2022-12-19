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

#include <gst/gst.h>
#include <gst/app/gstappsink.h>

namespace fastdeploy {
namespace streamer {

/*! @brief VideoDecoderApp class
 */
class FASTDEPLOY_DECL VideoDecoderApp : public BaseApp {
 public:
  explicit VideoDecoderApp(AppConfig& app_config) : BaseApp(app_config) {}

  bool Init(const std::string& config_file);

  bool TryPullFrame(FDTensor& tensor, int timeout_ms);

 private:
  void GetAppsinkFromPipeline();
  GstAppSink* appsink_;
};
}  // namespace streamer
}  // namespace fastdeploy
