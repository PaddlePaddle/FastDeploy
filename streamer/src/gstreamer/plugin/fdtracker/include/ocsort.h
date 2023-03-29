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

#pragma once

#include <map>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "gstreamer/plugin/fdtracker/include/kalmantracker.h"

class OcSortTracker {
 public:
  explicit OcSortTracker(int classid);
  virtual ~OcSortTracker(void) {}
  // static OcSortTracker *instance(void);
  virtual bool update(const cv::Mat &dets, bool use_byte, bool use_angle_cost);
  cv::Mat get_trackers(void);
  void associate_only_iou(cv::Mat detections, cv::Mat trackers,
                          float iou_threshold, std::map<int, int> &matches,
                          std::vector<int> &mismatch_row,
                          std::vector<int> &mismatch_col);
  void associate(cv::Mat detections, cv::Mat trackers, float iou_threshold,
                 cv::Mat velocities, cv::Mat previous_obs, float vdc_weight,
                 std::map<int, int> &matches, std::vector<int> &mismatch_row,
                 std::vector<int> &mismatch_col);
  std::vector<KalmanTracker*> trackers;
  std::vector<KalmanTracker*> unused_trackers;

 private:
  int timestamp;
  float lambda;
  float det_thresh = 0.1;
  int delta_t = 3;
  int inertia;
  float iou_threshold = 0.5;
  float score_thresh = 0.3;
  int max_age = 30;
  int min_hits = 0;
  int frame_count = 0;
  int classid = 0;
};
