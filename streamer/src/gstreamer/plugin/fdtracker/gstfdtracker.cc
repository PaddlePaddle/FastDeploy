// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "gstfdtracker.h"
#include <gst/base/gstbasetransform.h>
#include <gst/gst.h>
#include <gst/video/video.h>
#include <stdio.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include "gstnvdsmeta.h"

GST_DEBUG_CATEGORY_STATIC(gst_fdtracker_debug_category);
#define GST_CAT_DEFAULT gst_fdtracker_debug_category

/* prototypes */

static void gst_fdtracker_set_property(GObject *object, guint property_id,
                                       const GValue *value, GParamSpec *pspec);
static void gst_fdtracker_get_property(GObject *object, guint property_id,
                                       GValue *value, GParamSpec *pspec);
static gboolean gst_fdtracker_set_caps(GstBaseTransform *trans, GstCaps *incaps,
                                       GstCaps *outcaps);
static gboolean gst_fdtracker_start(GstBaseTransform *trans);
static gboolean gst_fdtracker_stop(GstBaseTransform *trans);
static GstFlowReturn gst_fdtracker_transform_ip(GstBaseTransform *trans,
                                                GstBuffer *buf);

enum { PROP_0 };

/* pad templates */
#define VIDEO_CAPS                      \
  GST_VIDEO_CAPS_MAKE("{ NV12, RGBA }") \
  ";" GST_VIDEO_CAPS_MAKE_WITH_FEATURES("memory:NVMM", "{ NV12, RGBA }")

static GstStaticPadTemplate gst_fdtracker_src_template =
    GST_STATIC_PAD_TEMPLATE("src", GST_PAD_SRC, GST_PAD_ALWAYS,
                            GST_STATIC_CAPS(GST_VIDEO_CAPS_MAKE_WITH_FEATURES(
                                "memory:NVMM", "{ NV12, RGBA }")));

static GstStaticPadTemplate gst_fdtracker_sink_template =
    GST_STATIC_PAD_TEMPLATE("sink", GST_PAD_SINK, GST_PAD_ALWAYS,
                            GST_STATIC_CAPS(GST_VIDEO_CAPS_MAKE_WITH_FEATURES(
                                "memory:NVMM", "{ NV12, RGBA }")));

/* class initialization */
G_DEFINE_TYPE_WITH_CODE(
    GstFdtracker, gst_fdtracker, GST_TYPE_BASE_TRANSFORM,
    GST_DEBUG_CATEGORY_INIT(gst_fdtracker_debug_category, "fdtracker", 0,
                            "debug category for fdtracker element"));

static void gst_fdtracker_class_init(GstFdtrackerClass *klass) {
  std::cout << "gst_fdtracker_class_init" << std::endl;
  GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
  GstBaseTransformClass *base_transform_class = GST_BASE_TRANSFORM_CLASS(klass);

  /* Setting up pads and setting metadata should be moved to
     base_class_init if you intend to subclass this class. */
  gst_element_class_add_pad_template(
      GST_ELEMENT_CLASS(klass),
      gst_static_pad_template_get(&gst_fdtracker_src_template));
  gst_element_class_add_pad_template(
      GST_ELEMENT_CLASS(klass),
      gst_static_pad_template_get(&gst_fdtracker_sink_template));

  gst_element_class_set_static_metadata(
      GST_ELEMENT_CLASS(klass), "FIXME Long name", "Generic",
      "FIXME Description", "FIXME <fixme@example.com>");

  gobject_class->set_property = gst_fdtracker_set_property;
  gobject_class->get_property = gst_fdtracker_get_property;
  base_transform_class->set_caps = GST_DEBUG_FUNCPTR(gst_fdtracker_set_caps);
  base_transform_class->start = GST_DEBUG_FUNCPTR(gst_fdtracker_start);
  base_transform_class->stop = GST_DEBUG_FUNCPTR(gst_fdtracker_stop);
  base_transform_class->transform_ip =
      GST_DEBUG_FUNCPTR(gst_fdtracker_transform_ip);
}

static void gst_fdtracker_init(GstFdtracker *fdtracker) {
  /* We will not be generating a new buffer. Just adding / updating
   * metadata. */
  gst_base_transform_set_in_place(GST_BASE_TRANSFORM(fdtracker), TRUE);
  /* We do not want to change the input caps. Set to passthrough. transform_ip
   * is still called. */
  gst_base_transform_set_passthrough(GST_BASE_TRANSFORM(fdtracker), TRUE);
}

void gst_fdtracker_set_property(GObject *object, guint property_id,
                                const GValue *value, GParamSpec *pspec) {
  GstFdtracker *fdtracker = GST_FDTRACKER(object);

  GST_DEBUG_OBJECT(fdtracker, "set_property");

  switch (property_id) {
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, pspec);
      break;
  }
}

void gst_fdtracker_get_property(GObject *object, guint property_id,
                                GValue *value, GParamSpec *pspec) {
  GstFdtracker *fdtracker = GST_FDTRACKER(object);

  GST_DEBUG_OBJECT(fdtracker, "get_property");

  switch (property_id) {
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, pspec);
      break;
  }
}

static gboolean gst_fdtracker_set_caps(GstBaseTransform *trans, GstCaps *incaps,
                                       GstCaps *outcaps) {
  GstFdtracker *fdtracker = GST_FDTRACKER(trans);

  GST_DEBUG_OBJECT(fdtracker, "set_caps");

  return TRUE;
}

/* states */
static gboolean gst_fdtracker_start(GstBaseTransform *trans) {
  GstFdtracker *fdtracker = GST_FDTRACKER(trans);

  GST_DEBUG_OBJECT(fdtracker, "start");

  fdtracker->tracker_per_class = new std::map<int, OcSortTracker *>;
  return TRUE;
}

static gboolean gst_fdtracker_stop(GstBaseTransform *trans) {
  GstFdtracker *fdtracker = GST_FDTRACKER(trans);

  GST_DEBUG_OBJECT(fdtracker, "stop");

  delete fdtracker->tracker_per_class;
  return TRUE;
}

static GstFlowReturn gst_fdtracker_transform_ip(GstBaseTransform *trans,
                                                GstBuffer *buf) {
  GstFdtracker *fdtracker = GST_FDTRACKER(trans);

  NvDsObjectMeta *obj_meta = NULL;
  NvDsMetaList *l_frame = NULL;
  NvDsMetaList *l_obj = NULL;
  NvBbox_Coords detector_bbox;

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

  std::cout << "This is a new frame!" << std::endl;
  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
       l_frame = l_frame->next) {
    std::map<int, std::vector<float>> bbox_per_class;
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);

    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
         l_obj = l_obj->next) {
      obj_meta = (NvDsObjectMeta *)(l_obj->data);
      if (bbox_per_class.find(obj_meta->class_id) == bbox_per_class.end()) {
        std::vector<float> vec;
        bbox_per_class[obj_meta->class_id] = vec;
      }
      detector_bbox = obj_meta->detector_bbox_info.org_bbox_coords;
      bbox_per_class[obj_meta->class_id].emplace_back(obj_meta->class_id);
      bbox_per_class[obj_meta->class_id].emplace_back(obj_meta->confidence);
      bbox_per_class[obj_meta->class_id].emplace_back(detector_bbox.left);
      bbox_per_class[obj_meta->class_id].emplace_back(detector_bbox.top);
      bbox_per_class[obj_meta->class_id].emplace_back(detector_bbox.left +
                                                      detector_bbox.width);
      bbox_per_class[obj_meta->class_id].emplace_back(detector_bbox.top +
                                                      detector_bbox.height);
    }

    nvds_clear_obj_meta_list(frame_meta, frame_meta->obj_meta_list);

    for (std::map<int, std::vector<float>>::iterator iter =
             bbox_per_class.begin();
         iter != bbox_per_class.end(); iter++) {
      std::map<int, OcSortTracker *> *octracker = fdtracker->tracker_per_class;
      if (octracker->find(iter->first) == octracker->end()) {
        octracker->insert(
            std::make_pair(iter->first, new OcSortTracker(iter->first)));
      }
      cv::Mat dets(iter->second.size() / 6, 6, CV_32FC1, cv::Scalar(0));
      memcpy(dets.data, iter->second.data(),
             iter->second.size() * sizeof(float));
      (*octracker)[iter->first]->update(dets, true, false);
      cv::Mat trackers = (*octracker)[iter->first]->get_trackers();
      for (int i = 0; i < trackers.rows; i++) {
        NvDsBatchMeta *batch_meta_pool = frame_meta->base_meta.batch_meta;
        NvDsObjectMeta *object_meta =
            nvds_acquire_obj_meta_from_pool(batch_meta_pool);
        NvOSD_RectParams &rect_params = object_meta->rect_params;
        NvOSD_TextParams &text_params = object_meta->text_params;
        detector_bbox = object_meta->detector_bbox_info.org_bbox_coords;
        detector_bbox.left = *(trackers.ptr<float>(i, 2));
        detector_bbox.top = *(trackers.ptr<float>(i, 3));
        detector_bbox.width =
            *(trackers.ptr<float>(i, 4)) - *(trackers.ptr<float>(i, 2));
        detector_bbox.height =
            *(trackers.ptr<float>(i, 5)) - *(trackers.ptr<float>(i, 3));
        rect_params.left = detector_bbox.left;
        rect_params.top = detector_bbox.top;
        rect_params.width = detector_bbox.width;
        rect_params.height = detector_bbox.height;
        /* Font to be used for label text. */
        static gchar font_name[] = "Serif";
        /* Semi-transparent yellow background. */
        rect_params.has_bg_color = 0;
        rect_params.bg_color = (NvOSD_ColorParams){1, 1, 0, 0.4};
        /* Red border of width 6. */
        rect_params.border_width = 3;
        rect_params.border_color = (NvOSD_ColorParams){1, 0, 0, 1};
        object_meta->class_id = *(trackers.ptr<float>(i, 0));
        object_meta->object_id = *(trackers.ptr<float>(i, 1));
        std::string text = std::to_string(object_meta->object_id);
        text_params.display_text = g_strdup(text.c_str());
        /* Display text above the left top corner of the object. */
        text_params.x_offset = rect_params.left;
        text_params.y_offset = rect_params.top - 10;
        /* Set black background for the text. */
        text_params.set_bg_clr = 1;
        text_params.text_bg_clr = (NvOSD_ColorParams){0, 0, 0, 1};
        /* Font face, size and color. */
        text_params.font_params.font_name = font_name;
        text_params.font_params.font_size = 11;
        text_params.font_params.font_color = (NvOSD_ColorParams){1, 1, 1, 1};
        nvds_add_obj_meta_to_frame(frame_meta, object_meta,
                                   object_meta->parent);
      }
    }
  }
  GST_DEBUG_OBJECT(fdtracker, "transform_ip");

  return GST_FLOW_OK;
}

static gboolean plugin_init(GstPlugin *plugin) {
  return gst_element_register(plugin, "fdtracker", GST_RANK_PRIMARY,
                              GST_TYPE_FDTRACKER);
}

#define VERSION "0.0.1"
#define PACKAGE "fdtracker"
#define PACKAGE_NAME "PaddlePaddle FastDeploy Streamer FDInfer plugin"
#define GST_PACKAGE_ORIGIN "https://github.com/PaddlePaddle/FastDeploy"

// GSTreamer is under LGPL license, while FastDeploy is under Apache-2.0
// license,
// so please follow both when using this plugin.
GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, fdtracker,
                  "FIXME plugin description", plugin_init, VERSION, "LGPL",
                  PACKAGE_NAME, GST_PACKAGE_ORIGIN)
