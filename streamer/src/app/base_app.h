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
#include "gstreamer/types.h"

#include <gst/gst.h>
#include <future>  // NOLINT

namespace fastdeploy {
namespace streamer {

enum AppType {
  VIDEO_ANALYTICS,  ///< Video analytics app
  VIDEO_DECODER,  ///< Video decoder app
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

  virtual bool Init(const std::string& config_file);

  bool Run();

  bool RunAsync();

  void Destroy();

  void SetupPerfMeasurement();

  AppConfig* GetAppConfig() {
    return &app_config_;
  }

  GstElement* GetPipeline() {
    return pipeline_;
  }

  GMainLoop* GetLoop() {
    return loop_;
  }

  guint GetBusId() {
    return bus_watch_id_;
  }

  bool Destroyed() {
    return destroyed_;
  }

 protected:
  AppConfig app_config_;
  GstElement* pipeline_;
  GMainLoop* loop_;
  guint bus_watch_id_;
  PerfContext perf_ctx_;
  std::future<void> future_;
  bool destroyed_ = false;
};
}  // namespace streamer
}  // namespace fastdeploy
