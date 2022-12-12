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

#include "app/video_decoder.h"
#include "gstreamer/utils.h"

namespace fastdeploy {
namespace streamer {

bool VideoDecoderApp::Init(const std::string& config_file) {
  FDINFO << "this " << std::endl;
  BaseApp::Init(config_file);
  GetAppsinkFromPipeline();
  return true;
}

bool VideoDecoderApp::TryPullFrame(FDTensor& tensor, int timeout_ms) {
  GstSample* sample = gst_app_sink_try_pull_sample(appsink_,
                                                   timeout_ms * GST_MSECOND);
  if (sample == NULL) {
    return false;
  }
  GstCaps* caps = NULL;
  uint8_t* data = nullptr;
  Frame frame;
  do {
    bool ret = GetFrameFromSample(sample, frame);
    if (!ret) {
      FDERROR << "Failed to get buffer from sample." << std::endl;
      break;
    }
    FDASSERT(frame.device == Device::CPU,
             "Currently, only CPU frame is supported");

    std::vector<int64_t> shape = GetFrameShape(frame);
    tensor.Resize(shape, FDDataType::UINT8, "", frame.device);
    FDTensor::CopyBuffer(tensor.Data(), frame.data, tensor.Nbytes(),
                        tensor.device);
  } while (false);

  if (sample) gst_sample_unref(sample);
  return true;
}

void VideoDecoderApp::GetAppsinkFromPipeline() {
  GstElement* elem = NULL;
  auto elem_names = GetSinkElemNames(GST_BIN(pipeline_));
  for (auto& elem_name : elem_names) {
    std::cout << elem_name << std::endl;
    if (elem_name.find("appsink") != std::string::npos) {
      elem = gst_bin_get_by_name(GST_BIN(pipeline_), elem_name.c_str());
    }
  }
  FDASSERT(elem != NULL, "Can't find a appsink in the pipeline");
  appsink_ = GST_APP_SINK_CAST(elem);
}
}  // namespace streamer
}  // namespace fastdeploy
