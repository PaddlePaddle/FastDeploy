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

#include "gstreamer/perf.h"
#include <iostream>

namespace fastdeploy {
namespace streamer {

static GstPadProbeReturn SinkBinBufProbe(GstPad* pad, GstPadProbeInfo* info, gpointer u_data) {
  PerfContext* ctx = (PerfContext*)u_data;
  if (ctx->stop) return GST_PAD_PROBE_OK;

  g_mutex_lock(&ctx->lock);
  if (ctx->buffer_cnt == 0) {
    ctx->tc.Start();
  }
  ctx->tc.End();
  ctx->buffer_cnt++;

  if (!ctx->first_buffer_arrived) {
    ctx->total_tc.Start();
    ctx->first_buffer_arrived = true;
    std::cout << "The first buffer after resuming arrives." << std::endl;
  }
  ctx->total_tc.End();
  ctx->total_buffer_cnt++;
  g_mutex_unlock(&ctx->lock);
  return GST_PAD_PROBE_OK;
}

static gboolean PerfMeasurementCallback(gpointer data) {
  PerfContext* ctx = (PerfContext*)data;
  PerfResult perf_result;

  g_mutex_lock(&ctx->lock);
  if (ctx->stop) {
    g_mutex_unlock(&ctx->lock);
    return FALSE;
  }

  if (ctx->buffer_cnt < 10) {
    g_mutex_unlock(&ctx->lock);
    return TRUE;
  }

  double duration = ctx->tc.Duration();
  double total_duration = ctx->total_tc.Duration() + ctx->total_played_duration;

  perf_result.fps = ctx->buffer_cnt / duration;
  perf_result.fps_avg = ctx->total_buffer_cnt / total_duration;
  ctx->buffer_cnt = 0;
  g_mutex_unlock(&ctx->lock);

  ctx->callback(ctx->user_data, &perf_result);
  return TRUE;
}

void PausePerfMeasurement(PerfContext* ctx) {
  g_mutex_lock(&ctx->lock);
  ctx->stop = true;
  ctx->total_played_duration += ctx->total_tc.Duration();
  g_mutex_unlock(&ctx->lock);
}

void ResumePerfMeasurement(PerfContext* ctx) {
  g_mutex_lock(&ctx->lock);
  if (!ctx->stop) {
    g_mutex_unlock(&ctx->lock);
    return;
  }

  ctx->stop = false;
  ctx->buffer_cnt = 0;
  ctx->first_buffer_arrived = false;
  if (!ctx->perf_measurement_timeout_id) {
    ctx->perf_measurement_timeout_id = g_timeout_add(
        ctx->measurement_interval_ms, PerfMeasurementCallback, ctx);
  }
  g_mutex_unlock(&ctx->lock);
}

bool EnablePerfMeasurement(PerfContext* ctx, GstPad* sink_bin_pad,
    gulong interval_sec, PerfCallback callback) {
  if (!callback) {
    return false;
  }

  g_mutex_init(&ctx->lock);
  ctx->perf_measurement_timeout_id = 0;
  ctx->measurement_interval_ms = interval_sec * 1000;
  ctx->callback = callback;
  ctx->stop = TRUE;
  ctx->sink_bin_pad = sink_bin_pad;
  ctx->fps_measure_probe_id = gst_pad_add_probe(sink_bin_pad,
      GST_PAD_PROBE_TYPE_BUFFER, SinkBinBufProbe, ctx, NULL);

  ResumePerfMeasurement(ctx);
  return true;
}

}  // namespace streamer
}  // namespace fastdeploy
