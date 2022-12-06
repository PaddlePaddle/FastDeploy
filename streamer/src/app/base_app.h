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

#include "fastdeploy/utils/utils.h"
#include "deepstream/perf.h"

#include <gst/gst.h>

namespace fastdeploy {
namespace streamer {

enum AppType {
  VIDEO_ANALYTICS,  ///< Video analytics app
};

struct AppConfig {
  AppType type;
  bool enable_perf_measurement = false;
  int perf_interval_sec = 5;
};

/*! @brief Base App class
 */
class BaseApp {
 public:
  BaseApp() {}
  explicit BaseApp(AppConfig& app_config) {
    app_config_ = app_config;
  }
  virtual ~BaseApp() = default;

  virtual bool Init() = 0;

  virtual bool Run() = 0;

  GstElement* GetPipeline() {
    return pipeline_;
  }

  void SetupPerfMeasurement();

 protected:
  AppConfig app_config_;
  GstElement* pipeline_;
  NvDsAppPerfStructInt perf_struct_;
};
}  // namespace streamer
}  // namespace fastdeploy
