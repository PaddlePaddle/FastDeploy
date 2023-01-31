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

#include "fastdeploy/core/fd_type.h"
#include "fastdeploy/runtime/enum_variables.h"
#include "fastdeploy/utils/perf.h"
#include <gst/gst.h>

namespace fastdeploy {
namespace streamer {
enum PixelFormat {
  I420,
  BGR
};

struct Frame {
  int width;
  int height;
  PixelFormat format;
  uint8_t* data = nullptr;
  Device device = Device::CPU;
};

struct PerfResult {
  double fps = 0.0;
  double fps_avg = 0.0;
};

typedef void (*PerfCallback)(gpointer ctx, PerfResult* str);

struct PerfContext {
  gulong measurement_interval_ms;
  gulong perf_measurement_timeout_id;
  bool stop;
  gpointer user_data;
  GMutex lock;
  PerfCallback callback;
  GstPad* sink_bin_pad;
  gulong fps_measure_probe_id;
  uint64_t buffer_cnt = 0;
  uint64_t total_buffer_cnt = 0;
  TimeCounter tc;
  TimeCounter total_tc;
  double total_played_duration = 0.0;
  bool first_buffer_arrived = false;
};

}  // namespace streamer
}  // namespace fastdeploy
