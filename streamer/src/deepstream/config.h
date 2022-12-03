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

namespace fastdeploy {
namespace streamer {

struct FASTDEPLOY_DECL NvUriSrcBinConfig {
  int gpu_id = 0;
  int file_loop = false;
  std::string uri;
  int source_id = -1;
};

struct FASTDEPLOY_DECL NvUriSrcBinListConfig {
  int gpu_id = 0;
  int file_loop = false;
  std::vector<NvUriSrcBinConfig> src_bins;
};

struct FASTDEPLOY_DECL NvStreamMuxConfig {
  int gpu_id = 0;
  int batch_size = 0;
  int width = 0;
  int height = 0;
  int batched_push_timeout = -1;
};

struct FASTDEPLOY_DECL NvInferConfig {
  int gpu_id = 0;
  std::string config_file_path;
};

struct FASTDEPLOY_DECL NvTrackerConfig {
  int gpu_id = 0;
  int tracker_width = 640;
  int tracker_height = 384;
  std::string ll_lib_file;
  std::string ll_config_file;
  bool enable_batch_process = true;
};

struct FASTDEPLOY_DECL NvMultiStreamTilerConfig {
  int gpu_id = 0;
  int rows = 0;
  int columns = 0;
};

struct FASTDEPLOY_DECL NvOsdBinConfig {
  int gpu_id = 0;
};

struct FASTDEPLOY_DECL NvVideoEncFileSinkBin {
  int gpu_id = 0;
  int bitrate = 0;
  std::string output_file;
};

}  // namespace streamer
}  // namespace fastdeploy
