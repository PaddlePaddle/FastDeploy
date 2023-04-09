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

#include "fdinfer.h"

#include <gst/base/gstbasetransform.h>
#include <gst/gst.h>
#include <gst/video/video.h>

#include <iostream>

#include "fdmodel.h"
#include "gstreamer/meta/meta.h"

GST_DEBUG_CATEGORY_STATIC(gst_fdinfer_debug_category);
#define GST_CAT_DEFAULT gst_fdinfer_debug_category

/* prototypes */

static void gst_fdinfer_set_property(GObject* object, guint property_id,
                                     const GValue* value, GParamSpec* pspec);
static void gst_fdinfer_get_property(GObject* object, guint property_id,
                                     GValue* value, GParamSpec* pspec);
static gboolean gst_fdinfer_set_caps(GstBaseTransform* trans, GstCaps* incaps,
                                     GstCaps* outcaps);
static gboolean gst_fdinfer_start(GstBaseTransform* trans);
static gboolean gst_fdinfer_stop(GstBaseTransform* trans);
static GstFlowReturn gst_fdinfer_transform_ip(GstBaseTransform* trans,
                                              GstBuffer* buf);

enum { PROP_0, PROP_MODEL_DIR };

/* pad templates */
#define VIDEO_CAPS                     \
  GST_VIDEO_CAPS_MAKE("{ NV12, BGR }") \
  ";" GST_VIDEO_CAPS_MAKE_WITH_FEATURES("memory:NVMM", "{ NV12, BGR }")

static GstStaticPadTemplate gst_fdinfer_src_template = GST_STATIC_PAD_TEMPLATE(
    "src", GST_PAD_SRC, GST_PAD_ALWAYS, GST_STATIC_CAPS(VIDEO_CAPS));

static GstStaticPadTemplate gst_fdinfer_sink_template = GST_STATIC_PAD_TEMPLATE(
    "sink", GST_PAD_SINK, GST_PAD_ALWAYS, GST_STATIC_CAPS(VIDEO_CAPS));

G_DEFINE_TYPE_WITH_CODE(
    GstFdinfer, gst_fdinfer, GST_TYPE_BASE_TRANSFORM,
    GST_DEBUG_CATEGORY_INIT(gst_fdinfer_debug_category, "fdinfer", 0,
                            "debug category for fdinfer element"));

static void gst_fdinfer_class_init(GstFdinferClass* klass) {
  std::cout << "gst_fdinfer_class_init" << std::endl;
  GObjectClass* gobject_class = G_OBJECT_CLASS(klass);
  GstBaseTransformClass* base_transform_class = GST_BASE_TRANSFORM_CLASS(klass);

  gst_element_class_add_pad_template(
      GST_ELEMENT_CLASS(klass),
      gst_static_pad_template_get(&gst_fdinfer_src_template));
  gst_element_class_add_pad_template(
      GST_ELEMENT_CLASS(klass),
      gst_static_pad_template_get(&gst_fdinfer_sink_template));

  gst_element_class_set_static_metadata(
      GST_ELEMENT_CLASS(klass), "FIXME Long name", "Generic",
      "FIXME Description", "FIXME <fixme@example.com>");

  gobject_class->set_property = gst_fdinfer_set_property;
  gobject_class->get_property = gst_fdinfer_get_property;
  base_transform_class->set_caps = GST_DEBUG_FUNCPTR(gst_fdinfer_set_caps);
  base_transform_class->start = GST_DEBUG_FUNCPTR(gst_fdinfer_start);
  base_transform_class->stop = GST_DEBUG_FUNCPTR(gst_fdinfer_stop);
  base_transform_class->transform_ip =
      GST_DEBUG_FUNCPTR(gst_fdinfer_transform_ip);

  g_object_class_install_property(
      gobject_class, PROP_MODEL_DIR,
      g_param_spec_string(
          "model-dir", "Model Directory", "Path to the model directory", "",
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
                        GST_PARAM_MUTABLE_PLAYING)));
}

static void gst_fdinfer_init(GstFdinfer* fdinfer) {
  /* We will not be generating a new buffer. Just adding / updating
   * metadata. */
  gst_base_transform_set_in_place(GST_BASE_TRANSFORM(fdinfer), TRUE);
  /* We do not want to change the input caps. Set to passthrough. transform_ip
   * is still called. */
  gst_base_transform_set_passthrough(GST_BASE_TRANSFORM(fdinfer), TRUE);
}

void gst_fdinfer_set_property(GObject* object, guint property_id,
                              const GValue* value, GParamSpec* pspec) {
  GstFdinfer* fdinfer = GST_FDINFER(object);

  GST_DEBUG_OBJECT(fdinfer, "set_property");

  switch (property_id) {
    case PROP_MODEL_DIR:
      fdinfer->model_dir = g_value_dup_string(value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, pspec);
      break;
  }
}

void gst_fdinfer_get_property(GObject* object, guint property_id, GValue* value,
                              GParamSpec* pspec) {
  GstFdinfer* fdinfer = GST_FDINFER(object);

  GST_DEBUG_OBJECT(fdinfer, "get_property");

  switch (property_id) {
    case PROP_MODEL_DIR:
      g_value_set_string(value, fdinfer->model_dir);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, pspec);
      break;
  }
}

gboolean gst_fdinfer_set_caps(GstBaseTransform* trans, GstCaps* incaps,
                              GstCaps* outcaps) {
  GstFdinfer* fdinfer = GST_FDINFER(trans);

  GST_DEBUG_OBJECT(fdinfer, "set_caps");
  GST_WARNING_OBJECT(fdinfer, "set_caps");

  std::cout << "incaps: " << gst_caps_to_string(incaps) << std::endl;
  std::cout << "outcaps: " << gst_caps_to_string(outcaps) << std::endl;

  GstCapsFeatures* features = gst_caps_get_features(incaps, 0);
  std::cout << "in features: " << gst_caps_features_to_string(features)
            << std::endl;

  return TRUE;
}

static gboolean gst_fdinfer_start(GstBaseTransform* trans) {
  GstFdinfer* fdinfer = GST_FDINFER(trans);

  GST_DEBUG_OBJECT(fdinfer, "start");

  std::string model_dir(fdinfer->model_dir);
  auto model_file = model_dir + "model.pdmodel";
  auto params_file = model_dir + "model.pdiparams";
  auto config_file = model_dir + "infer_cfg.yml";

  auto option = fastdeploy::RuntimeOption();
  option.UseGpu();
  auto model = fastdeploy::streamer::CreateModel("PPYOLOE", option, model_file,
                                                 params_file, config_file);

  fdinfer->model = model;
  fdinfer->model_name = g_strdup("PPYOLOE");

  return TRUE;
}

static gboolean gst_fdinfer_stop(GstBaseTransform* trans) {
  GstFdinfer* fdinfer = GST_FDINFER(trans);

  GST_DEBUG_OBJECT(fdinfer, "stop");

  delete reinterpret_cast<fastdeploy::vision::detection::PPYOLOE*>(
      fdinfer->model);

  return TRUE;
}

static GstFlowReturn gst_fdinfer_transform_ip(GstBaseTransform* trans,
                                              GstBuffer* buf) {
  GstFdinfer* fdinfer = GST_FDINFER(trans);

  std::cout << "transform ip" << std::endl;

  GST_DEBUG_OBJECT(fdinfer, "transform_ip");

  fastdeploy::vision::DetectionResult res;
  fastdeploy::streamer::ModelPredict(fdinfer->model_name, fdinfer->model, buf,
                                     1920, 1080, res);
  fastdeploy::streamer::AddDetectionMeta(buf, res, 0.5);
  fastdeploy::streamer::PrintROIMeta(buf);

  return GST_FLOW_OK;
}

static gboolean plugin_init(GstPlugin* plugin) {
  return gst_element_register(plugin, "fdinfer", GST_RANK_PRIMARY,
                              GST_TYPE_FDINFER);
}

#define VERSION "0.0.1"
#define PACKAGE "fdinfer"
#define PACKAGE_NAME "PaddlePaddle FastDeploy Streamer FDInfer plugin"
#define GST_PACKAGE_ORIGIN "https://github.com/PaddlePaddle/FastDeploy"

// GSTreamer is under LGPL license, while FastDeploy is under Apache-2.0
// license, so please follow both when using this plugin.
GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, fdinfer,
                  "FIXME plugin description", plugin_init, VERSION, "LGPL",
                  PACKAGE_NAME, GST_PACKAGE_ORIGIN)
