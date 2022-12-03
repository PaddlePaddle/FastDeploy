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

#include "gstreamer/fd_source_bin.h"

#include "deepstream/config.h"

#include "yaml-cpp/yaml.h"

namespace fastdeploy {
namespace streamer {

std::vector<GstElement*> source_bins;
std::vector<GstElement*> unlinked_elems;

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

void LinkElements() {
  std::cout << "size: " << unlinked_elems.size() << std::endl;
  for (size_t i = 1; i < unlinked_elems.size(); i++) {
    FDASSERT(
        gst_element_link(unlinked_elems[i - 1], unlinked_elems[i]),
        "Failed to link elements.");
  }
}

// bool AddNvUriSrcBin(GstElement* pipeline, const NvUriSrcBinConfig& config, std::vector<std::string>& uri_list) {
//   GstElement* streammux = gst_element_factory_make("nvstreammux", NULL);
//   gst_bin_add(GST_BIN(pipeline), streammux);
  
//   for (size_t i = 0; i < uri_list.size(); i++) {
//     GstElement* source_bin = create_source_bin(i, const_cast<char*>(uri_list[i].c_str()), config.gpu_id);
//     if (!source_bin) {
//       g_printerr ("Failed to create source bin. Exiting.\n");
//       return false;
//     }
//     gst_bin_add(GST_BIN(pipeline), source_bin);
//     std::string pad_name = "sink_" + std::to_string(i);
//     GstPad* sinkpad = gst_element_get_request_pad(streammux, pad_name.c_str());
//     if (!sinkpad) {
//       g_printerr("Streammux request sink pad failed. Exiting.\n");
//       return -1;
//     }

//     GstPad* srcpad = gst_element_get_static_pad(source_bin, "src");
//     if (!srcpad) {
//       g_printerr("Failed to get src pad of source bin. Exiting.\n");
//       return -1;
//     }

//     if (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK) {
//       g_printerr("Failed to link source bin to stream muxer. Exiting.\n");
//       return -1;
//     }

//     gst_object_unref(srcpad);
//     gst_object_unref(sinkpad);
//   }
//   g_object_set(G_OBJECT(streammux), "gpu-id", config.gpu_id,
//                "batch-size", 4,
//                "width", 1920, "height", 1080,
//                "batched-push-timeout", 40000,
//                NULL);
//   // source_bins.push_back(source_bin);
//   unlinked_elems.push_back(streammux);

//   // GstElement* fakesink = gst_element_factory_make("fakesink", NULL);
//   // gst_bin_add(GST_BIN(pipeline), fakesink);
//   // unlinked_elems.push_back(fakesink);
//   return true;
// }

bool AddNvUriSrcBin(GstElement* pipeline, const NvUriSrcBinConfig& config) {
  GstElement* source_bin = create_source_bin(config.source_id, const_cast<char*>(config.uri.c_str()), config.gpu_id);
  if (!source_bin) {
    g_printerr("Failed to create source bin. Exiting.\n");
    return false;
  }
  gst_bin_add(GST_BIN(pipeline), source_bin);
  std::cout << "add: " << std::hex << source_bin << std::endl;
  source_bins.push_back(source_bin);
  std::cout << "add size: " << source_bins.size() << std::endl;
  return true;
}

bool AddNvStreamMux(GstElement* pipeline, const NvStreamMuxConfig& config) {
  GstElement* streammux = gst_element_factory_make("nvstreammux", NULL);
  gst_bin_add(GST_BIN(pipeline), streammux);
  std::cout << "mux size: " << source_bins.size() << std::endl;
  for (size_t i = 0; i < source_bins.size(); i++) {
    std::string pad_name = "sink_" + std::to_string(i);
    GstPad* sinkpad = gst_element_get_request_pad(streammux, pad_name.c_str());
    if (!sinkpad) {
      g_printerr("Streammux request sink pad failed. Exiting.\n");
      return -1;
    }
    std::cout << "mux: " << std::hex << source_bins[i] << std::endl;
    GstPad* srcpad = gst_element_get_static_pad(source_bins[i], "src");
    if (!srcpad) {
      g_printerr("Failed to get src pad of source bin. Exiting.\n");
      return -1;
    }

    if (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK) {
      g_printerr("Failed to link source bin to stream muxer. Exiting.\n");
      return -1;
    }

    gst_object_unref(srcpad);
    gst_object_unref(sinkpad);
  }
  g_object_set(G_OBJECT(streammux), "gpu-id", config.gpu_id,
               "batch-size", config.batch_size,
               "width", config.width, "height", config.height,
               "batched-push-timeout", config.batched_push_timeout,
               NULL);
  // gst_bin_add(GST_BIN(pipeline), streammux);
  unlinked_elems.push_back(streammux);
  return true;
}

bool AddNvInfer(GstElement* pipeline, const NvInferConfig& config) {
  GstElement* nvinfer = gst_element_factory_make("nvinfer", NULL);
  g_object_set(G_OBJECT(nvinfer), "gpu-id", config.gpu_id,
               "config-file-path", config.config_file_path.c_str(),
               NULL);
  gst_bin_add(GST_BIN(pipeline), nvinfer);
  unlinked_elems.push_back(nvinfer);
  return true;
}

bool AddNvTracker(GstElement* pipeline, const NvTrackerConfig& config) {
  GstElement* nvtracker = gst_element_factory_make("nvtracker", NULL);
  g_object_set(G_OBJECT(nvtracker), "gpu-id", config.gpu_id,
               "tracker-width", config.tracker_width,
               "tracker-height", config.tracker_height,
               "ll-lib-file", config.ll_lib_file.c_str(),
               "ll-config-file", config.ll_config_file.c_str(),
               "enable-batch-process", config.enable_batch_process, NULL);
  gst_bin_add(GST_BIN(pipeline), nvtracker);
  unlinked_elems.push_back(nvtracker);
  return true;
}

bool AddNvMultiStreamTiler(GstElement* pipeline, const NvMultiStreamTilerConfig& config) {
  GstElement* tiler = gst_element_factory_make("nvmultistreamtiler", NULL);
  g_object_set(G_OBJECT(tiler), "gpu-id", config.gpu_id,
               "rows", config.rows,
               "columns", config.columns, NULL);
  gst_bin_add(GST_BIN(pipeline), tiler);
  unlinked_elems.push_back(tiler);
  return true;
}

bool AddNvOsdBin(GstElement* pipeline, const NvOsdBinConfig& config) {
  GstElement* osd = gst_element_factory_make("nvosdbin", NULL);
  g_object_set(G_OBJECT(osd), "gpu-id", config.gpu_id, NULL);
  gst_bin_add(GST_BIN(pipeline), osd);
  unlinked_elems.push_back(osd);
  return true;
}

bool AddNvVideoEncFileSinkBin(GstElement* pipeline, const NvVideoEncFileSinkBin& config) {
  GstElement* sink = gst_element_factory_make("nvvideoencfilesinkbin", NULL);
  g_object_set(G_OBJECT(sink), "gpu-id", config.gpu_id,
               "bitrate", config.bitrate,
               "output-file", config.output_file.c_str(), NULL);
  gst_bin_add(GST_BIN(pipeline), sink);
  unlinked_elems.push_back(sink);
  return true;
}

bool BuildPipeline(const std::string& config_file, GstElement* pipeline) {
  // TODO: validate config
  YAML::Node cfg;
  try {
    cfg = YAML::LoadFile(config_file);
  } catch (YAML::BadFile& e) {
    FDERROR << "Failed to load yaml file " << config_file
            << ", maybe you should check this file." << std::endl;
    return false;
  }
  for (const auto& elem : cfg) {
    std::string elem_name = elem.first.as<std::string>();
    std::cout << elem_name << std::endl;

    if (elem_name == "app") {

    } else if (elem_name == "nvurisrcbin_list") {
      NvUriSrcBinConfig config;
      auto uri_list = elem.second["uri-list"].as<std::vector<std::string>>();
      config.gpu_id = elem.second["gpu-id"].as<int>();
      for (size_t i = 0; i < uri_list.size(); i++) {
        std::cout << uri_list[i] << std::endl;
        config.uri = uri_list[i];
        config.source_id = i;
        // AddNvUriSrcBin(pipeline, config, uri_list);
        AddNvUriSrcBin(pipeline, config);
      }
    } else if (elem_name == "nvstreammux") {
      NvStreamMuxConfig config;
      config.gpu_id = elem.second["gpu-id"].as<int>();
      config.batch_size = elem.second["batch-size"].as<int>();
      config.width = elem.second["width"].as<int>();
      config.height = elem.second["height"].as<int>();
      config.batched_push_timeout = elem.second["batched-push-timeout"].as<int>();
      AddNvStreamMux(pipeline, config);
    } else if (elem_name == "nvinfer") {
      NvInferConfig config;
      config.gpu_id = elem.second["gpu-id"].as<int>();
      config.config_file_path = elem.second["config-file-path"].as<std::string>();
      AddNvInfer(pipeline, config);
    } else if (elem_name == "nvtracker") {
      NvTrackerConfig config;
      config.gpu_id = elem.second["gpu-id"].as<int>();
      config.tracker_width = elem.second["tracker-width"].as<int>();
      config.tracker_height = elem.second["tracker-height"].as<int>();
      config.ll_lib_file = elem.second["ll-lib-file"].as<std::string>();
      config.ll_config_file = elem.second["ll-config-file"].as<std::string>();
      config.enable_batch_process = elem.second["enable-batch-process"].as<bool>();
      AddNvTracker(pipeline, config);
    } else if (elem_name == "nvmultistreamtiler") {
      NvMultiStreamTilerConfig config;
      config.gpu_id = elem.second["gpu-id"].as<int>();
      config.rows = elem.second["rows"].as<int>();
      config.columns = elem.second["columns"].as<int>();
      AddNvMultiStreamTiler(pipeline, config);
    } else if (elem_name == "nvosdbin") {
      NvOsdBinConfig config;
      config.gpu_id = elem.second["gpu-id"].as<int>();
      AddNvOsdBin(pipeline, config);
    } else if (elem_name == "nvvideoencfilesinkbin") {
      NvVideoEncFileSinkBin config;
      config.gpu_id = elem.second["gpu-id"].as<int>();
      config.bitrate = elem.second["bitrate"].as<int>();
      config.output_file = elem.second["output-file"].as<std::string>();
      AddNvVideoEncFileSinkBin(pipeline, config);
    } else {
      FDASSERT(false, "Unsupported element: %s.", elem_name.c_str());
    }
  }

  // GstBus* bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
  // guint bus_watch_id_ = gst_bus_add_watch(bus, bus_watch_callback, loop_);
  // gst_object_unref(bus);


  LinkElements();
  return true;
}

bool FDStreamer::Init(const std::string& config_file) {
  gst_init(NULL, NULL);
  loop_ = g_main_loop_new(NULL, FALSE);
  pipeline_ = gst_pipeline_new("dstest1-pipeline");

  GstBus* bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline_));
  guint bus_watch_id_ = gst_bus_add_watch(bus, bus_watch_callback, loop_);
  gst_object_unref(bus);


  BuildPipeline(config_file, pipeline_);

  return true;
}

bool FDStreamer::Run() {
  /* Set the pipeline to "playing" state */
  // g_print("Now playing: %s\n", argv[1]);
  gst_element_set_state(pipeline_, GST_STATE_PLAYING);

  /* Wait till pipeline encounters an error or EOS */
  g_print("Running...\n");
  g_main_loop_run(loop_);

  /* Out of the main loop, clean up nicely */
  g_print("Returned, stopping playback\n");
  gst_element_set_state(pipeline_, GST_STATE_NULL);
  g_print("Deleting pipeline\n");
  gst_object_unref(GST_OBJECT(pipeline_));
  // g_source_remove(bus_watch_id_);
  g_main_loop_unref(loop_);
  return true;
}

}  // namespace streamer
}  // namespace fastdeploy
