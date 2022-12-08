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

#include <gst/app/gstappsink.h>
#include <cuda_runtime_api.h>

namespace fastdeploy {
namespace streamer {

static int count = 0;
static GstFlowReturn NewSampleCallback(GstAppSink* appsink, gpointer data) {
  GstSample* sample = gst_app_sink_pull_sample(appsink);
  if (sample == NULL) {
    FDINFO << "Can't pull sample." << std::endl;
    return GST_FLOW_OK;
  }
  auto obj = reinterpret_cast<VideoDecoderApp*>(data);
  // obj->frame_cnt_++;
  // std::cout << obj->frame_cnt_ << std::endl;

  GstBuffer* buffer = NULL;
  GstMapInfo map;
  const GstStructure* info = NULL;
  GstCaps* caps = NULL;
  int sample_width = 0;
  int sample_height = 0;
  do {
    buffer = gst_sample_get_buffer(sample);
    if (buffer == NULL) {
      FDERROR << "Failed to get buffer from sample." << std::endl;
      break;
    }

    gst_buffer_map(buffer, &map, GST_MAP_READ);

    if (map.data == NULL) {
      FDERROR << "Appsink buffer data is empty." << std::endl;
      break;
    }
 
    caps = gst_sample_get_caps(sample);
    if (caps == NULL) {
      FDERROR << "Failed to get caps from sample." << std::endl;
      break;
    }

    FDINFO << "caps: " << gst_caps_to_string(caps) << std::endl;
 
    info = gst_caps_get_structure(caps, 0);
    if (info == NULL) {
      FDERROR << "Failed to get structure from caps." << std::endl;
      break;
    }

    gst_structure_get_int(info, "width", &sample_width);
    gst_structure_get_int(info, "height", &sample_height);

    FDINFO << "width: " << sample_width << " height: " << sample_height
           << " size: " << map.memory->size << std::endl;

    std::vector<int64_t> shape = {sample_height, sample_width, 3};
    obj->UpdateQueue(map.data, shape);
  } while (false);

  if (buffer) gst_buffer_unmap(buffer, &map);
  if (sample) gst_sample_unref(sample);
  return GST_FLOW_OK;
}

static GstFlowReturn NewPrerollCallback(GstAppSink* appsink, gpointer data) {
  FDINFO << "new preroll callback" << std::endl;
  return GST_FLOW_OK;
}

static void EosCallback(GstAppSink* appsink, gpointer data) {
  FDINFO << "eos callback" << std::endl;
}

void VideoDecoderApp::SetupAppSinkCallback() {
  GstElement* elem = NULL;
  auto elem_names = GetSinkElemNames(GST_BIN(pipeline_));
  for (auto& elem_name : elem_names) {
    std::cout << elem_name << std::endl;
    if (elem_name.find("appsink") != std::string::npos) {
      elem = gst_bin_get_by_name(GST_BIN(pipeline_), elem_name.c_str());
    }
  }
  FDASSERT(elem != NULL, "Can't find a properly sink bin in the pipeline");

  GstAppSink* appsink = GST_APP_SINK_CAST(elem);

  gst_app_sink_set_emit_signals(appsink, TRUE);

  GstAppSinkCallbacks callbacks = {
      EosCallback,
      NewPrerollCallback,
      NewSampleCallback
  };
  gst_app_sink_set_callbacks(appsink, &callbacks,
                             reinterpret_cast<void*>(this), NULL);
  
  ring_buffers_.resize(max_queue_size_);
  // cudaSetDevice(1);
}

bool VideoDecoderApp::PopTensor(FDTensor& tensor) {
  std::lock_guard<std::mutex> guard(queue_mutex_);
  if (tensor_queue_.empty()) return false;
  
  auto fst = tensor_queue_.front();
  tensor.Resize(fst->shape, fst->dtype, "", fst->device);
  FDTensor::CopyBuffer(tensor.Data(), fst->Data(), fst->Nbytes(), fst->device);

  tensor_queue_.pop();
  return true;
}

void VideoDecoderApp::UpdateQueue(uint8_t* data,
                                  const std::vector<int64_t>& shape) {
  std::lock_guard<std::mutex> guard(queue_mutex_);

  auto tensor = &ring_buffers_[frame_cnt_ % ring_buffers_.size()];
  frame_cnt_++;
  std::cout << "frame: " << frame_cnt_ << std::endl;

  tensor->Resize(shape, FDDataType::UINT8, "", Device::CPU);
  FDTensor::CopyBuffer(tensor->Data(), data, tensor->Nbytes(), Device::CPU);

  tensor_queue_.push(tensor);
  if (tensor_queue_.size() == max_queue_size_) {
    // drop the oldest frame
    tensor_queue_.pop();
  }
}

}  // namespace streamer
}  // namespace fastdeploy
