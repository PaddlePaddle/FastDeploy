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

#include "app/base_app.h"
#include "gstreamer/utils.h"
#include "app/yaml_parser.h"

namespace fastdeploy {
namespace streamer {

static GMutex fps_lock;

static gboolean bus_watch_callback(GstBus* bus, GstMessage* msg, gpointer data) {
  GMainLoop* loop = (GMainLoop*)data;
  switch (GST_MESSAGE_TYPE(msg)) {
    case GST_MESSAGE_EOS:
      g_print("End of stream\n");
      g_main_loop_quit(loop);
      break;
    case GST_MESSAGE_ERROR: {
      gchar* debug;
      GError* error;
      gst_message_parse_error(msg, &error, &debug);
      g_printerr("ERROR from element %s: %s\n",
          GST_OBJECT_NAME(msg->src), error->message);
      if (debug)
        g_printerr("Error details: %s\n", debug);
      g_free(debug);
      g_error_free(error);
      g_main_loop_quit(loop);
      break;
    }
    default:
      break;
  }
  return TRUE;
}

static void perf_cb(gpointer context, NvDsAppPerfStruct* str) {
  guint numf = str->num_instances;

  g_mutex_lock(&fps_lock);
  for (int i = 0; i < numf; i++) {
    std::cout << "Instance: " << i << ", FPS: " << str->fps[i]
              << ", total avg.: " << str->fps_avg[i] << std::endl;
  }
  g_mutex_unlock(&fps_lock);
}

bool BaseApp::Init(const std::string& config_file) {
  gst_init(NULL, NULL);
  loop_ = g_main_loop_new(NULL, FALSE);

  YamlParser parser(config_file);
  pipeline_ = parser.BuildPipelineFromConfig();

  GstBus* bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline_));
  bus_watch_id_ = gst_bus_add_watch(bus, bus_watch_callback, loop_);
  gst_object_unref(bus);

  SetupPerfMeasurement();
  return true;
}

bool BaseApp::Run() {
  gst_element_set_state(pipeline_, GST_STATE_PLAYING);

  /* Wait till pipeline encounters an error or EOS */
  g_print("Running...\n");
  g_main_loop_run(loop_);

  g_print("Returned, stopping playback\n");
  gst_element_set_state(pipeline_, GST_STATE_NULL);
  g_print("Deleting pipeline\n");
  gst_object_unref(GST_OBJECT(pipeline_));
  g_source_remove(bus_watch_id_);
  g_main_loop_unref(loop_);
  return true;
}

static void MainLoopThread(BaseApp* app) {
  g_main_loop_run(app->GetLoop());

  g_print("Returned, stopping playback\n");
  gst_element_set_state(app->GetPipeline(), GST_STATE_NULL);
  g_print("Deleting pipeline\n");
  gst_object_unref(GST_OBJECT(app->GetPipeline()));
  g_source_remove(app->GetBusId());
  g_main_loop_unref(app->GetLoop());
}

bool BaseApp::RunAsync() {
  gst_element_set_state(pipeline_, GST_STATE_PLAYING);
  g_print("Running Asynchronous...\n");
  std::thread t(MainLoopThread, this);
  thread_ = std::move(t);
  return true;
}

void BaseApp::SetupPerfMeasurement() {
  if (!app_config_.enable_perf_measurement) return;

  GstElement* elem = NULL;
  auto elem_names = GetSinkElemNames(GST_BIN(pipeline_));
  for (auto& elem_name : elem_names) {
    std::cout << elem_name << std::endl;
    if (elem_name.find("nvvideoencfilesinkbin") != std::string::npos) {
      elem = gst_bin_get_by_name(GST_BIN(pipeline_), elem_name.c_str());
    } else if (elem_name.find("appsink") != std::string::npos) {
      elem = gst_bin_get_by_name(GST_BIN(pipeline_), elem_name.c_str());
    }
  }
  FDASSERT(elem != NULL, "Can't find a properly sink bin in the pipeline");

  GstPad* perf_pad = gst_element_get_static_pad(elem, "sink");
  FDASSERT(perf_pad != NULL, "Unable to get sink pad");

  perf_struct_.context = nullptr;
  enable_perf_measurement(&perf_struct_, perf_pad, 1,
                          (gulong)(app_config_.perf_interval_sec), 1, perf_cb);

  gst_object_unref(perf_pad);
}
}  // namespace streamer
}  // namespace fastdeploy
