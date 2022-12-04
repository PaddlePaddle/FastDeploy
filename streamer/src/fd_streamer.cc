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

namespace fastdeploy {
namespace streamer {

bool FDStreamer::Init(const std::string& config_file) {
  AppConfig app_config;
  YamlParser parser(config_file);
  parser.ParseAppConfg(app_config);
  if (app_config.type == AppType::VIDEO_ANALYTICS) {
    auto p = new VideoAnalyticsApp();
    p->Init();
    parser.BuildPipelineFromConfig(p->GetPipeline());
    app = static_cast<void*>(p);
  }
  return true;
}

bool FDStreamer::Run() {
  // /* Set the pipeline to "playing" state */
  // // g_print("Now playing: %s\n", argv[1]);
  // gst_element_set_state(pipeline_, GST_STATE_PLAYING);

  // /* Wait till pipeline encounters an error or EOS */
  // g_print("Running...\n");
  // g_main_loop_run(loop_);

  // /* Out of the main loop, clean up nicely */
  // g_print("Returned, stopping playback\n");
  // gst_element_set_state(pipeline_, GST_STATE_NULL);
  // g_print("Deleting pipeline\n");
  // gst_object_unref(GST_OBJECT(pipeline_));
  // g_source_remove(bus_watch_id_);
  // g_main_loop_unref(loop_);
  return true;
}

}  // namespace streamer
}  // namespace fastdeploy
