// Copyright (c) 2023 niuzhibo. All Rights Reserved.

#pragma once

#include <map>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "include/kalmantracker.h"

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
