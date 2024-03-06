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

#include "meta.h"

#include <gst/video/gstvideometa.h>

#include <iostream>

namespace fastdeploy {
namespace streamer {

void AddDetectionMeta(GstBuffer* buffer, vision::DetectionResult& result,
                      float thresh) {
  if (!gst_buffer_is_writable(buffer)) {
    throw std::runtime_error("Buffer is not writable.");
  }

  for (size_t i = 0; i < result.boxes.size(); ++i) {
    if (result.scores[i] < thresh) {
      continue;
    }
    float x1 = result.boxes[i][0];
    float y1 = result.boxes[i][1];
    float x2 = result.boxes[i][2];
    float y2 = result.boxes[i][3];
    GstVideoRegionOfInterestMeta* meta =
        gst_buffer_add_video_region_of_interest_meta(buffer, "detection", x1,
                                                     y1, x2 - x1, y2 - y1);
    meta->id = gst_util_seqnum_next();

    // Add detection tensor
    GstStructure* detection = gst_structure_new(
        "detection", "x1", G_TYPE_FLOAT, x1, "y1", G_TYPE_FLOAT, y1, "x2",
        G_TYPE_FLOAT, x2, "y2", G_TYPE_FLOAT, y2, NULL);
    gst_structure_set(detection, "score", G_TYPE_FLOAT, result.scores[i], NULL);
    gst_structure_set(detection, "label_id", G_TYPE_INT, result.label_ids[i],
                      NULL);
    gst_video_region_of_interest_meta_add_param(meta, detection);
  }
}

void AddTrackerMeta(GstBuffer* buffer, cv::Mat& result) {
  if (!gst_buffer_is_writable(buffer)) {
    throw std::runtime_error("Buffer is not writable.");
  }

  for (size_t i = 0; i < result.rows; ++i) {
    float x1 = *(result.ptr<float>(i, 2));
    float y1 = *(result.ptr<float>(i, 3));
    float x2 = *(result.ptr<float>(i, 4));
    float y2 = *(result.ptr<float>(i, 5));
    GstVideoRegionOfInterestMeta* meta =
        gst_buffer_add_video_region_of_interest_meta(buffer, "tracker", x1, y1,
                                                     x2 - x1, y2 - y1);
    meta->id = gst_util_seqnum_next();

    // Add detection tensor
    GstStructure* tracker = gst_structure_new(
        "tracker", "x1", G_TYPE_FLOAT, x1, "y1", G_TYPE_FLOAT, y1, "x2",
        G_TYPE_FLOAT, x2, "y2", G_TYPE_FLOAT, y2, NULL);
    gst_structure_set(tracker, "class_id", G_TYPE_INT,
                      *(result.ptr<float>(i, 4)), NULL);
    gst_structure_set(tracker, "object_id", G_TYPE_INT,
                      *(result.ptr<float>(i, 4)), NULL);
    gst_video_region_of_interest_meta_add_param(meta, tracker);
  }
}

void ReadDetectionMeta(GstBuffer* buffer, vision::DetectionResult& result) {
  GstMeta* meta = NULL;
  gpointer state = NULL;

  while ((meta = gst_buffer_iterate_meta_filtered(
              buffer, &state, GST_VIDEO_REGION_OF_INTEREST_META_API_TYPE))) {
    float x1, y1, x2, y2, score;
    int class_id;
    GstVideoRegionOfInterestMeta* m = (GstVideoRegionOfInterestMeta*)meta;
    GstStructure* detection =
        gst_video_region_of_interest_meta_get_param(m, "detection");
    gst_structure_get(detection, "x1", G_TYPE_FLOAT, &x1, NULL);
    gst_structure_get(detection, "y1", G_TYPE_FLOAT, &y1, NULL);
    gst_structure_get(detection, "x2", G_TYPE_FLOAT, &x2, NULL);
    gst_structure_get(detection, "y2", G_TYPE_FLOAT, &y2, NULL);
    gst_structure_get(detection, "class_id", G_TYPE_INT, &class_id, NULL);
    gst_structure_get(detection, "score", G_TYPE_FLOAT, &score, NULL);
    result.boxes.emplace_back(std::array<float, 4>{x1, y1, x2, y2});
    result.label_ids.emplace_back(class_id);
    result.scores.emplace_back(score);
  }
}

void PrintROIMeta(GstBuffer* buffer) {
  GstMeta* meta = NULL;
  gpointer state = NULL;

  std::cout << "ROI meta:" << std::endl;
  while ((meta = gst_buffer_iterate_meta_filtered(
              buffer, &state, GST_VIDEO_REGION_OF_INTEREST_META_API_TYPE))) {
    GstVideoRegionOfInterestMeta* m = (GstVideoRegionOfInterestMeta*)meta;
    for (GList* l = m->params; l; l = g_list_next(l)) {
      GstStructure* s = GST_STRUCTURE(l->data);
      std::cout << gst_structure_to_string(s) << std::endl;
    }
  }
}

}  // namespace streamer
}  // namespace fastdeploy
