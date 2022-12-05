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

#include "fd_streamer.h"
#include "app/yaml_parser.h"
#include "app/video_analytics.h"
#include "fastdeploy/utils/unique_ptr.h"

namespace fastdeploy {
namespace streamer {

bool FDStreamer::Init(const std::string& config_file) {
  AppConfig app_config;
  YamlParser parser(config_file);
  parser.ParseAppConfg(app_config);
  if (app_config.type == AppType::VIDEO_ANALYTICS) {
    app_ = utils::make_unique<VideoAnalyticsApp>();
    auto casted_app = dynamic_cast<VideoAnalyticsApp*>(app_.get());
    casted_app->Init();
    parser.BuildPipelineFromConfig(casted_app->GetPipeline());
  }
  return true;
}

bool FDStreamer::Run() {
  return app_->Run();
}

}  // namespace streamer
}  // namespace fastdeploy
