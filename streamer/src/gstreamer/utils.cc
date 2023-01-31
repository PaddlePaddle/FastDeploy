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

#include "gstreamer/utils.h"

namespace fastdeploy {
namespace streamer {
std::string GetElementName(GstElement* elem) {
  gchar* name = gst_element_get_name(elem);
  std::string res(name);
  g_free(name);
  return res;
}

std::vector<std::string> GetSinkElemNames(GstBin* bin) {
  GstIterator *it;
  GValue val = G_VALUE_INIT;
  gboolean done = FALSE;
  std::vector<std::string> names;

  it = gst_bin_iterate_sinks(bin);
  do {
    switch (gst_iterator_next(it, &val)) {
      case GST_ITERATOR_OK: {
        GstElement* sink = static_cast<GstElement*>(g_value_get_object(&val));
        names.push_back(GetElementName(sink));
        g_value_reset(&val);
        break;
      }
      case GST_ITERATOR_RESYNC:
        gst_iterator_resync(it);
        break;
      case GST_ITERATOR_ERROR:
        GST_ERROR("Error iterating over %s's sink elements",
            GST_ELEMENT_NAME(bin));
      case GST_ITERATOR_DONE:
        g_value_unset(&val);
        done = TRUE;
        break;
    }
  } while (!done);

  gst_iterator_free(it);
  return names;
}

GstElement* CreatePipeline(const std::string& pipeline_desc) {
  GError *error = NULL;
  FDINFO << "Trying to launch pipeline: " << pipeline_desc << std::endl;
  GstElement* pipeline = gst_parse_launch(pipeline_desc.c_str(), &error);
  FDASSERT(pipeline != NULL, "Failed parse pipeline, error: %s",
           error->message);
  return pipeline;
}

std::vector<int64_t> GetFrameShape(const Frame& frame) {
  if (frame.format == PixelFormat::I420) {
    return { frame.height * 3 / 2, frame.width, 1 };
  } else if (frame.format == PixelFormat::BGR) {
    return { frame.height, frame.width, 3 };
  } else {
    FDASSERT(false, "Unsupported format: %d.", frame.format);
  }
}

PixelFormat GetPixelFormat(const std::string& format) {
  if (format == "I420") {
    return PixelFormat::I420;
  } else if (format == "BGR") {
    return PixelFormat::BGR;
  } else {
    FDASSERT(false, "Unsupported format: %s.", format.c_str());
  }
}

void GetFrameInfo(GstCaps* caps, Frame& frame) {
  const GstStructure* struc = gst_caps_get_structure(caps, 0);
  std::string name = gst_structure_get_name(struc);

  if (name.rfind("video", 0) != 0) {
    FDASSERT(false, "GetFrameInfo only support video caps.");
  }

  GstCapsFeatures* features = gst_caps_get_features(caps, 0);
  if (gst_caps_features_contains(features, "memory:NVMM")) {
    frame.device = Device::GPU;
  } else {
    frame.device = Device::CPU;
  }
  gst_structure_get_int(struc, "width", &frame.width);
  gst_structure_get_int(struc, "height", &frame.height);
  std::string format_str = gst_structure_get_string(struc, "format");
  frame.format = GetPixelFormat(format_str);
}

bool GetFrameFromSample(GstSample* sample, Frame& frame) {
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
    frame.data = map.data;
    GetFrameInfo(caps, frame);
  } while (false);
  if (buffer) gst_buffer_unmap(buffer, &map);
  return true;
}

}  // namespace streamer
}  // namespace fastdeploy
