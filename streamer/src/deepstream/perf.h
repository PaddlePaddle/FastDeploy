/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#pragma once

#include <gst/gst.h>

#define MAX_SOURCE_BINS 1024

namespace fastdeploy {
namespace streamer {

struct NvDsAppPerfStruct {
  gdouble fps[MAX_SOURCE_BINS];
  gdouble fps_avg[MAX_SOURCE_BINS];
  guint num_instances;
};

typedef void (*perf_callback)(gpointer ctx, NvDsAppPerfStruct* str);

struct NvDsInstancePerfStruct {
  guint buffer_cnt;
  guint total_buffer_cnt;
  struct timeval total_fps_time;
  struct timeval start_fps_time;
  struct timeval last_fps_time;
  struct timeval last_sample_fps_time;
};

struct NvDsAppPerfStructInt {
  gulong measurement_interval_ms;
  gulong perf_measurement_timeout_id;
  guint num_instances;
  gboolean stop;
  gpointer context;
  GMutex struct_lock;
  perf_callback callback;
  GstPad *sink_bin_pad;
  gulong fps_measure_probe_id;
  NvDsInstancePerfStruct instance_str[MAX_SOURCE_BINS];
  guint dewarper_surfaces_per_frame;
};

gboolean enable_perf_measurement(NvDsAppPerfStructInt *str,
    GstPad *sink_bin_pad, guint num_sources, gulong interval_sec,
    guint num_surfaces_per_frame, perf_callback callback);

void pause_perf_measurement(NvDsAppPerfStructInt *str);
void resume_perf_measurement(NvDsAppPerfStructInt *str);

}  // namespace streamer
}  // namespace fastdeploy
