/* GStreamer
 * Copyright (C) 2023 FIXME <fixme@example.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin Street, Suite 500,
 * Boston, MA 02110-1335, USA.
 */
/**
 * SECTION:element-gstfdtracker
 *
 * The fdtracker element does FIXME stuff.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch-1.0 -v fakesrc ! fdtracker ! FIXME ! fakesink
 * ]|
 * FIXME Describe what the pipeline does.
 * </refsect2>
 */

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <gst/video/video.h>
#include "gstfdtracker.h"
#include "gstnvdsmeta.h"
#include <cmath>
#include <algorithm>
#include <stdio.h>
#include <iostream>
#include <string>
#include <fstream>

GST_DEBUG_CATEGORY_STATIC (gst_fdtracker_debug_category);
#define GST_CAT_DEFAULT gst_fdtracker_debug_category

/* prototypes */


static void gst_fdtracker_set_property (GObject * object,
    guint property_id, const GValue * value, GParamSpec * pspec);
static void gst_fdtracker_get_property (GObject * object,
    guint property_id, GValue * value, GParamSpec * pspec);
static void gst_fdtracker_dispose (GObject * object);
static void gst_fdtracker_finalize (GObject * object);

static GstCaps *gst_fdtracker_transform_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * filter);
static GstCaps *gst_fdtracker_fixate_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * othercaps);
static gboolean gst_fdtracker_accept_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps);
static gboolean gst_fdtracker_set_caps (GstBaseTransform * trans,
    GstCaps * incaps, GstCaps * outcaps);
static gboolean gst_fdtracker_query (GstBaseTransform * trans,
    GstPadDirection direction, GstQuery * query);
static gboolean gst_fdtracker_decide_allocation (GstBaseTransform * trans,
    GstQuery * query);
static gboolean gst_fdtracker_filter_meta (GstBaseTransform * trans,
    GstQuery * query, GType api, const GstStructure * params);
static gboolean gst_fdtracker_propose_allocation (GstBaseTransform * trans,
    GstQuery * decide_query, GstQuery * query);
static gboolean gst_fdtracker_transform_size (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, gsize size, GstCaps * othercaps,
    gsize * othersize);
static gboolean gst_fdtracker_get_unit_size (GstBaseTransform * trans,
    GstCaps * caps, gsize * size);
static gboolean gst_fdtracker_start (GstBaseTransform * trans);
static gboolean gst_fdtracker_stop (GstBaseTransform * trans);
static gboolean gst_fdtracker_sink_event (GstBaseTransform * trans,
    GstEvent * event);
static gboolean gst_fdtracker_src_event (GstBaseTransform * trans,
    GstEvent * event);
static GstFlowReturn gst_fdtracker_prepare_output_buffer (GstBaseTransform *
    trans, GstBuffer * input, GstBuffer ** outbuf);
static gboolean gst_fdtracker_copy_metadata (GstBaseTransform * trans,
    GstBuffer * input, GstBuffer * outbuf);
static gboolean gst_fdtracker_transform_meta (GstBaseTransform * trans,
    GstBuffer * outbuf, GstMeta * meta, GstBuffer * inbuf);
static void gst_fdtracker_before_transform (GstBaseTransform * trans,
    GstBuffer * buffer);
static GstFlowReturn gst_fdtracker_transform (GstBaseTransform * trans,
    GstBuffer * inbuf, GstBuffer * outbuf);
static GstFlowReturn gst_fdtracker_transform_ip (GstBaseTransform * trans,
    GstBuffer * buf);
static float IOU(const Bbox_cache& bbox_1, const Bbox_cache& bbox_2);

enum
{
  PROP_0
};

/* pad templates */
#define VIDEO_CAPS GST_VIDEO_CAPS_MAKE("{ NV12, RGBA }") ";" \
    GST_VIDEO_CAPS_MAKE_WITH_FEATURES("memory:NVMM", "{ NV12, RGBA }")

static GstStaticPadTemplate gst_fdtracker_src_template =
GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES("memory:NVMM", "{ NV12, RGBA }"))
    );

static GstStaticPadTemplate gst_fdtracker_sink_template =
GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES("memory:NVMM", "{ NV12, RGBA }"))
    );


/* class initialization */

G_DEFINE_TYPE_WITH_CODE (GstFdtracker, gst_fdtracker, GST_TYPE_BASE_TRANSFORM,
  GST_DEBUG_CATEGORY_INIT (gst_fdtracker_debug_category, "fdtracker", 0,
  "debug category for fdtracker element"));

static void
gst_fdtracker_class_init (GstFdtrackerClass * klass)
{

  std::cout << "gst_fdtracker_class_init" << std::endl;
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstBaseTransformClass *base_transform_class = GST_BASE_TRANSFORM_CLASS (klass);

  /* Setting up pads and setting metadata should be moved to
     base_class_init if you intend to subclass this class. */
  // gst_element_class_add_static_pad_template (GST_ELEMENT_CLASS(klass),
  //     &gst_fdtracker_src_template);
  // gst_element_class_add_static_pad_template (GST_ELEMENT_CLASS(klass),
  //     &gst_fdtracker_sink_template);

  gst_element_class_add_pad_template (GST_ELEMENT_CLASS(klass),
      gst_static_pad_template_get (&gst_fdtracker_src_template));
  gst_element_class_add_pad_template (GST_ELEMENT_CLASS(klass),
      gst_static_pad_template_get (&gst_fdtracker_sink_template));


  gst_element_class_set_static_metadata (GST_ELEMENT_CLASS(klass),
      "FIXME Long name", "Generic", "FIXME Description",
      "FIXME <fixme@example.com>");

  gobject_class->set_property = gst_fdtracker_set_property;
  gobject_class->get_property = gst_fdtracker_get_property;
  // gobject_class->dispose = gst_fdtracker_dispose;
  // gobject_class->finalize = gst_fdtracker_finalize;
  // base_transform_class->transform_caps = GST_DEBUG_FUNCPTR (gst_fdtracker_transform_caps);
  // base_transform_class->fixate_caps = GST_DEBUG_FUNCPTR (gst_fdtracker_fixate_caps);
  // base_transform_class->accept_caps = GST_DEBUG_FUNCPTR (gst_fdtracker_accept_caps);
  base_transform_class->set_caps = GST_DEBUG_FUNCPTR (gst_fdtracker_set_caps);
  // base_transform_class->query = GST_DEBUG_FUNCPTR (gst_fdtracker_query);
  // base_transform_class->decide_allocation = GST_DEBUG_FUNCPTR (gst_fdtracker_decide_allocation);
  // base_transform_class->filter_meta = GST_DEBUG_FUNCPTR (gst_fdtracker_filter_meta);
  // base_transform_class->propose_allocation = GST_DEBUG_FUNCPTR (gst_fdtracker_propose_allocation);
  // base_transform_class->transform_size = GST_DEBUG_FUNCPTR (gst_fdtracker_transform_size);
  // base_transform_class->get_unit_size = GST_DEBUG_FUNCPTR (gst_fdtracker_get_unit_size);
  base_transform_class->start = GST_DEBUG_FUNCPTR (gst_fdtracker_start);
  base_transform_class->stop = GST_DEBUG_FUNCPTR (gst_fdtracker_stop);
  // base_transform_class->sink_event = GST_DEBUG_FUNCPTR (gst_fdtracker_sink_event);
  // base_transform_class->src_event = GST_DEBUG_FUNCPTR (gst_fdtracker_src_event);
  // base_transform_class->prepare_output_buffer = GST_DEBUG_FUNCPTR (gst_fdtracker_prepare_output_buffer);
  // base_transform_class->copy_metadata = GST_DEBUG_FUNCPTR (gst_fdtracker_copy_metadata);
  // base_transform_class->transform_meta = GST_DEBUG_FUNCPTR (gst_fdtracker_transform_meta);
  // base_transform_class->before_transform = GST_DEBUG_FUNCPTR (gst_fdtracker_before_transform);
  // base_transform_class->transform = GST_DEBUG_FUNCPTR (gst_fdtracker_transform);
  base_transform_class->transform_ip = GST_DEBUG_FUNCPTR (gst_fdtracker_transform_ip);

}

static void
gst_fdtracker_init (GstFdtracker *fdtracker)
{
  /* We will not be generating a new buffer. Just adding / updating
   * metadata. */
  gst_base_transform_set_in_place (GST_BASE_TRANSFORM (fdtracker), TRUE);
  /* We do not want to change the input caps. Set to passthrough. transform_ip
   * is still called. */
  gst_base_transform_set_passthrough (GST_BASE_TRANSFORM (fdtracker), TRUE);
}

void
gst_fdtracker_set_property (GObject * object, guint property_id,
    const GValue * value, GParamSpec * pspec)
{
  GstFdtracker *fdtracker = GST_FDTRACKER (object);

  GST_DEBUG_OBJECT (fdtracker, "set_property");

  switch (property_id) {
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
      break;
  }
}

void
gst_fdtracker_get_property (GObject * object, guint property_id,
    GValue * value, GParamSpec * pspec)
{
  GstFdtracker *fdtracker = GST_FDTRACKER (object);

  GST_DEBUG_OBJECT (fdtracker, "get_property");

  switch (property_id) {
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
      break;
  }
}

void
gst_fdtracker_dispose (GObject * object)
{
  GstFdtracker *fdtracker = GST_FDTRACKER (object);

  GST_DEBUG_OBJECT (fdtracker, "dispose");

  /* clean up as possible.  may be called multiple times */

  G_OBJECT_CLASS (gst_fdtracker_parent_class)->dispose (object);
}

void
gst_fdtracker_finalize (GObject * object)
{
  GstFdtracker *fdtracker = GST_FDTRACKER (object);

  GST_DEBUG_OBJECT (fdtracker, "finalize");

  /* clean up object here */

  G_OBJECT_CLASS (gst_fdtracker_parent_class)->finalize (object);
}

static GstCaps *
gst_fdtracker_transform_caps (GstBaseTransform * trans, GstPadDirection direction,
    GstCaps * caps, GstCaps * filter)
{
  GstFdtracker *fdtracker = GST_FDTRACKER (trans);
  GstCaps *othercaps;

  GST_DEBUG_OBJECT (fdtracker, "transform_caps");

  othercaps = gst_caps_copy (caps);

  /* Copy other caps and modify as appropriate */
  /* This works for the simplest cases, where the transform modifies one
   * or more fields in the caps structure.  It does not work correctly
   * if passthrough caps are preferred. */
  if (direction == GST_PAD_SRC) {
    /* transform caps going upstream */
  } else {
    /* transform caps going downstream */
  }

  if (filter) {
    GstCaps *intersect;

    intersect = gst_caps_intersect (othercaps, filter);
    gst_caps_unref (othercaps);

    return intersect;
  } else {
    return othercaps;
  }
}

static GstCaps *
gst_fdtracker_fixate_caps (GstBaseTransform * trans, GstPadDirection direction,
    GstCaps * caps, GstCaps * othercaps)
{
  GstFdtracker *fdtracker = GST_FDTRACKER (trans);

  GST_DEBUG_OBJECT (fdtracker, "fixate_caps");

  return NULL;
}

static gboolean
gst_fdtracker_accept_caps (GstBaseTransform * trans, GstPadDirection direction,
    GstCaps * caps)
{
  GstFdtracker *fdtracker = GST_FDTRACKER (trans);

  GST_DEBUG_OBJECT (fdtracker, "accept_caps");

  return TRUE;
}

static gboolean
gst_fdtracker_set_caps (GstBaseTransform * trans, GstCaps * incaps,
    GstCaps * outcaps)
{
  GstFdtracker *fdtracker = GST_FDTRACKER (trans);

  GST_DEBUG_OBJECT (fdtracker, "set_caps");

  return TRUE;
}

static gboolean
gst_fdtracker_query (GstBaseTransform * trans, GstPadDirection direction,
    GstQuery * query)
{
  GstFdtracker *fdtracker = GST_FDTRACKER (trans);

  GST_DEBUG_OBJECT (fdtracker, "query");

  return TRUE;
}

/* decide allocation query for output buffers */
static gboolean
gst_fdtracker_decide_allocation (GstBaseTransform * trans, GstQuery * query)
{
  GstFdtracker *fdtracker = GST_FDTRACKER (trans);

  GST_DEBUG_OBJECT (fdtracker, "decide_allocation");

  return TRUE;
}

static gboolean
gst_fdtracker_filter_meta (GstBaseTransform * trans, GstQuery * query, GType api,
    const GstStructure * params)
{
  GstFdtracker *fdtracker = GST_FDTRACKER (trans);

  GST_DEBUG_OBJECT (fdtracker, "filter_meta");

  return TRUE;
}

/* propose allocation query parameters for input buffers */
static gboolean
gst_fdtracker_propose_allocation (GstBaseTransform * trans,
    GstQuery * decide_query, GstQuery * query)
{
  GstFdtracker *fdtracker = GST_FDTRACKER (trans);

  GST_DEBUG_OBJECT (fdtracker, "propose_allocation");

  return TRUE;
}

/* transform size */
static gboolean
gst_fdtracker_transform_size (GstBaseTransform * trans, GstPadDirection direction,
    GstCaps * caps, gsize size, GstCaps * othercaps, gsize * othersize)
{
  GstFdtracker *fdtracker = GST_FDTRACKER (trans);

  GST_DEBUG_OBJECT (fdtracker, "transform_size");

  return TRUE;
}

static gboolean
gst_fdtracker_get_unit_size (GstBaseTransform * trans, GstCaps * caps,
    gsize * size)
{
  GstFdtracker *fdtracker = GST_FDTRACKER (trans);

  GST_DEBUG_OBJECT (fdtracker, "get_unit_size");

  return TRUE;
}

/* states */
static gboolean
gst_fdtracker_start (GstBaseTransform * trans)
{
  GstFdtracker *fdtracker = GST_FDTRACKER (trans);

  GST_DEBUG_OBJECT (fdtracker, "start");

  fdtracker->previous_frame = new std::queue<std::vector<Bbox_cache>>;
  fdtracker->tracker_per_class = new std::map<int, OcSortTracker*>;
  return TRUE;
}

static gboolean
gst_fdtracker_stop (GstBaseTransform * trans)
{
  GstFdtracker *fdtracker = GST_FDTRACKER (trans);

  GST_DEBUG_OBJECT (fdtracker, "stop");

  delete fdtracker->previous_frame;
  delete fdtracker->tracker_per_class;
  return TRUE;
}

/* sink and src pad event handlers */
static gboolean
gst_fdtracker_sink_event (GstBaseTransform * trans, GstEvent * event)
{
  GstFdtracker *fdtracker = GST_FDTRACKER (trans);

  GST_DEBUG_OBJECT (fdtracker, "sink_event");

  return GST_BASE_TRANSFORM_CLASS (gst_fdtracker_parent_class)->sink_event (
      trans, event);
}

static gboolean
gst_fdtracker_src_event (GstBaseTransform * trans, GstEvent * event)
{
  GstFdtracker *fdtracker = GST_FDTRACKER (trans);

  GST_DEBUG_OBJECT (fdtracker, "src_event");

  return GST_BASE_TRANSFORM_CLASS (gst_fdtracker_parent_class)->src_event (
      trans, event);
}

static GstFlowReturn
gst_fdtracker_prepare_output_buffer (GstBaseTransform * trans, GstBuffer * input,
    GstBuffer ** outbuf)
{
  GstFdtracker *fdtracker = GST_FDTRACKER (trans);

  GST_DEBUG_OBJECT (fdtracker, "prepare_output_buffer");

  return GST_FLOW_OK;
}

/* metadata */
static gboolean
gst_fdtracker_copy_metadata (GstBaseTransform * trans, GstBuffer * input,
    GstBuffer * outbuf)
{
  GstFdtracker *fdtracker = GST_FDTRACKER (trans);

  GST_DEBUG_OBJECT (fdtracker, "copy_metadata");

  return TRUE;
}

static gboolean
gst_fdtracker_transform_meta (GstBaseTransform * trans, GstBuffer * outbuf,
    GstMeta * meta, GstBuffer * inbuf)
{
  GstFdtracker *fdtracker = GST_FDTRACKER (trans);

  GST_DEBUG_OBJECT (fdtracker, "transform_meta");

  return TRUE;
}

static void
gst_fdtracker_before_transform (GstBaseTransform * trans, GstBuffer * buffer)
{
  GstFdtracker *fdtracker = GST_FDTRACKER (trans);

  GST_DEBUG_OBJECT (fdtracker, "before_transform");

}

/* transform */
static GstFlowReturn
gst_fdtracker_transform (GstBaseTransform * trans, GstBuffer * inbuf,
    GstBuffer * outbuf)
{
  GstFdtracker *fdtracker = GST_FDTRACKER (trans);

  std::cout << "transform" << std::endl;

  GST_DEBUG_OBJECT (fdtracker, "transform");

  return GST_FLOW_OK;
}
OcSortTracker* octracker = new OcSortTracker(0);
static GstFlowReturn
gst_fdtracker_transform_ip (GstBaseTransform * trans, GstBuffer * buf)
{
  GstFdtracker *fdtracker = GST_FDTRACKER (trans);

  NvDsObjectMeta *obj_meta = NULL;
  NvDsMetaList * l_frame = NULL;
  NvDsMetaList * l_obj = NULL;
  NvBbox_Coords detector_bbox;
  NvBbox_Coords tracker_bbox;

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

  std::cout << "This is a new frame!" << std::endl;
  for(l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
      std::vector<float> rects;
      std::map<int, std::vector<float>> bbox_per_class;
      NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);

    for(l_obj = frame_meta->obj_meta_list; l_obj != NULL;
        l_obj = l_obj->next) {
      obj_meta = (NvDsObjectMeta *) (l_obj->data);
      if(bbox_per_class.find(obj_meta->class_id) == bbox_per_class.end()) {
        std::vector<float> vec;
        bbox_per_class[obj_meta->class_id] = vec;
      }
      detector_bbox = obj_meta->detector_bbox_info.org_bbox_coords;
      bbox_per_class[obj_meta->class_id].emplace_back(obj_meta->class_id);
      bbox_per_class[obj_meta->class_id].emplace_back(obj_meta->confidence);
      bbox_per_class[obj_meta->class_id].emplace_back(detector_bbox.left);
      bbox_per_class[obj_meta->class_id].emplace_back(detector_bbox.top);
      bbox_per_class[obj_meta->class_id].emplace_back(detector_bbox.left+detector_bbox.width);
      bbox_per_class[obj_meta->class_id].emplace_back(detector_bbox.top+detector_bbox.height);
    }	
      
    nvds_clear_obj_meta_list(frame_meta, frame_meta->obj_meta_list);   

    for(std::map<int, std::vector<float>>::iterator iter = bbox_per_class.begin(); iter != bbox_per_class.end(); iter++) {
      std::map<int, OcSortTracker*>* octracker = fdtracker->tracker_per_class;
      if(octracker->find(iter->first) == octracker->end()) {
        octracker->insert(std::make_pair(iter->first, new OcSortTracker(iter->first)));
      }
      cv::Mat dets(iter->second.size()/6, 6, CV_32FC1, cv::Scalar(0));
	    memcpy(dets.data, iter->second.data(), iter->second.size() * sizeof(float));
      (*octracker)[iter->first]->update(dets, true, false);
      cv::Mat trackers = (*octracker)[iter->first]->get_trackers();
      for(int i = 0; i < trackers.rows; i++) {
        NvDsBatchMeta *batch_meta_pool = frame_meta->base_meta.batch_meta;
        NvDsObjectMeta *object_meta = nvds_acquire_obj_meta_from_pool(batch_meta_pool);
        NvOSD_RectParams & rect_params = object_meta->rect_params;
        NvOSD_TextParams & text_params = object_meta->text_params;
        detector_bbox = object_meta->detector_bbox_info.org_bbox_coords;
        detector_bbox.left = *(trackers.ptr<float>(i, 2));
        detector_bbox.top = *(trackers.ptr<float>(i, 3));
        detector_bbox.width = *(trackers.ptr<float>(i, 4)) - *(trackers.ptr<float>(i, 2));
        detector_bbox.height = *(trackers.ptr<float>(i, 5)) - *(trackers.ptr<float>(i, 3));
        rect_params.left = detector_bbox.left;
        rect_params.top = detector_bbox.top;
        rect_params.width = detector_bbox.width;
        rect_params.height = detector_bbox.height;
        /* Font to be used for label text. */
        static gchar font_name[] = "Serif";
        /* Semi-transparent yellow background. */
        rect_params.has_bg_color = 0;
        rect_params.bg_color = (NvOSD_ColorParams) {
        1, 1, 0, 0.4};
        /* Red border of width 6. */
        rect_params.border_width = 3;
        rect_params.border_color = (NvOSD_ColorParams) {
        1, 0, 0, 1};
        object_meta->class_id = *(trackers.ptr<float>(i, 0));
        object_meta->object_id = *(trackers.ptr<float>(i, 1));
        std::string text = std::to_string(std::to_string(object_meta->object_id);
        text_params.display_text = g_strdup(text.c_str());
        /* Display text above the left top corner of the object. */
        text_params.x_offset = rect_params.left;
        text_params.y_offset = rect_params.top - 10;
        /* Set black background for the text. */
        text_params.set_bg_clr = 1;
        text_params.text_bg_clr = (NvOSD_ColorParams) {
        0, 0, 0, 1};
        /* Font face, size and color. */
        text_params.font_params.font_name = font_name;
        text_params.font_params.font_size = 11;
        text_params.font_params.font_color = (NvOSD_ColorParams) {
        1, 1, 1, 1};
        nvds_add_obj_meta_to_frame(frame_meta, object_meta, object_meta->parent);
      }
    }
  }
  GST_DEBUG_OBJECT (fdtracker, "transform_ip");

  return GST_FLOW_OK;
}

static gboolean
plugin_init (GstPlugin * plugin)
{
  return gst_element_register (plugin, "fdtracker", GST_RANK_PRIMARY,
      GST_TYPE_FDTRACKER);
}


#define VERSION "0.0.1"
#define PACKAGE "fdtracker"
#define PACKAGE_NAME "PaddlePaddle FastDeploy Streamer FDInfer plugin"
#define GST_PACKAGE_ORIGIN "https://github.com/PaddlePaddle/FastDeploy"

// GSTreamer is under LGPL license, while FastDeploy is under Apache-2.0 license,
// so please follow both when using this plugin.
GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    fdtracker,
    "FIXME plugin description",
    plugin_init, VERSION, "LGPL", PACKAGE_NAME, GST_PACKAGE_ORIGIN)

