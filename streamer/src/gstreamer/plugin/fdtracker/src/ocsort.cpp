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

#include <limits.h>
#include <stdio.h>
#include <algorithm>
#include <map>
#include <utility>

#include "gstreamer/plugin/fdtracker/include/kalmantracker.h"
#include "gstreamer/plugin/fdtracker/include/lapjv.h"
#include "gstreamer/plugin/fdtracker/include/ocsort.h"

bool isnan_in_pred(cv::Mat &mat) {
  for (int i = 0; i < mat.rows; i++) {
    for (int j = 0; j < mat.cols; j++) {
      if (isnan(*(mat.ptr<float>(i, j)))) {
        return true;
      }
    }
  }

  return false;
}

cv::Vec4f k_previous_obs(KalmanTracker *trajectory, int cur_age, int delta) {
  if (trajectory->observations.size() == 0) {
    return cv::Vec4f{-1, -1, -1, -1};
  }
  for (int i = 0; i < delta; i++) {
    int idx = delta - i;
    if (trajectory->observations.find(idx) != trajectory->observations.end()) {
      return trajectory->observations[idx];
    }
  }
  return trajectory->observations.rbegin()->second;
}

OcSortTracker::OcSortTracker(int classid)
    : timestamp(0),
      max_age(30),
      lambda(0.98f),
      score_thresh(0.3f),
      iou_threshold(0.3f),
      delta_t(3),
      classid(classid) {
  std::cout << "Init Octracker: " << classid << std::endl;
}

bool OcSortTracker::update(const cv::Mat &dets, bool use_byte = true,
                           bool use_angle_cost = false) {
  ++timestamp;

  cv::Mat candidates_first, candidates_second;
  for (int i = 0; i < dets.rows; ++i) {
    float score = *dets.ptr<float>(i, 1);
    if (score > this->score_thresh) {
      candidates_first.push_back(dets.row(i));
    } else if (score > 0.1) {
      candidates_second.push_back(dets.row(i));
    }
  }

  cv::Mat trks;
  std::vector<int> to_del;
  for (int i = 0; i < this->trackers.size(); i++) {
    // cv::Mat pos_pred;
    cv::Mat pos_pred = xyah2ltrb(this->trackers[i]->predict());
    if (isnan_in_pred(pos_pred)) {
      to_del.push_back(i);
    } else {
      trks.push_back(pos_pred);
    }
  }

  for (int i = to_del.size() - 1; i >= 0; i--) {
    this->unused_trackers.push_back(this->trackers[i]);
    this->trackers.erase(this->trackers.begin() + i);
  }

  cv::Mat last_boxes;
  for (auto tracker : this->trackers) {
    last_boxes.push_back(cv::Mat(tracker->last_observation).t());
  }

  // First round of association
  std::map<int, int> matches;
  std::vector<int> mismatch_row, mismatch_col;
  std::vector<int> mismatch_row_temp, mismatch_col_temp;
  if (use_angle_cost) {
    cv::Mat velocities, k_observations;
    for (auto tracker : this->trackers) {
      if (!tracker->velocity[0] == -1) {
        velocities.push_back(tracker->velocity);
      } else {
        velocities.push_back(cv::Vec2f{0, 0});
      }

      k_observations.push_back(
          k_previous_obs(tracker, tracker->age, this->delta_t));
    }
    associate(candidates_first, trks, this->iou_threshold, velocities,
              k_observations, this->inertia, matches, mismatch_row,
              mismatch_col);
  } else {
    associate_only_iou(candidates_first, trks, this->iou_threshold, matches,
                       mismatch_row, mismatch_col);
  }

  for (auto item : matches) {
    this->trackers[item.second]->update(
        candidates_first.row(item.first).colRange(2, 6), use_angle_cost);
  }

  // Second round of associaton
  if (use_byte && candidates_second.rows > 0 && mismatch_col.size() > 0) {
    cv::Mat u_trks;
    for (auto idx : mismatch_col) {
      u_trks.push_back(trks.row(idx));
    }
    associate_only_iou(candidates_second, u_trks, this->iou_threshold, matches,
                       mismatch_row_temp, mismatch_col_temp);
    for (auto pair : matches) {
      this->trackers[mismatch_col[pair.second]]->update(
          candidates_second.row(pair.first).colRange(2, 6), use_angle_cost);
    }
    std::vector<int> mismatch_col_copy(mismatch_col);
    mismatch_col.clear();
    for (auto mistrk : mismatch_col_temp) {
      mismatch_col.push_back(mismatch_col_copy[mistrk]);
    }
  }

  if (mismatch_col.size() > 0 && mismatch_row.size() > 0) {
    cv::Mat left_dets, left_trks;
    for (auto det_idx : mismatch_row) {
      left_dets.push_back(candidates_first.row(det_idx));
    }
    for (auto trk_idx : mismatch_col) {
      left_trks.push_back(last_boxes.row(trk_idx));
    }
    associate_only_iou(left_dets, left_trks, this->iou_threshold, matches,
                       mismatch_row_temp, mismatch_col_temp);
    for (auto pair : matches) {
      this->trackers[mismatch_col[pair.second]]->update(
          left_dets.row(pair.first).colRange(2, 6), use_angle_cost);
    }

    std::vector<int> mismatch_row_copy(mismatch_row);
    mismatch_row.clear();
    for (auto mistrk : mismatch_row_temp) {
      mismatch_row.push_back(mismatch_row_copy[mistrk]);
    }
    std::vector<int> mismatch_col_copy(mismatch_col);
    mismatch_col.clear();
    for (auto mistrk : mismatch_col_temp) {
      mismatch_col.push_back(mismatch_col_copy[mistrk]);
    }
  }

  for (auto idx : mismatch_row) {
    cv::Mat rect = candidates_first.row(idx);
    float score = *rect.ptr<float>(0, 1);
    cv::Vec4f ltrb = mat2vec4f(rect.colRange(2, 6));
    printf(
        "find new obj with score: %.2f,ltrb xmin: %.2f, ymin: %.2f,, xmax: "
        "%.2f, ymax: %.2f,",
        score, ltrb[0], ltrb[1], ltrb[2], ltrb[3]);
    KalmanTracker *tracker = new KalmanTracker(ltrb, score);
    this->trackers.push_back(
        tracker);  // todo, is the tracker memory will be released out this {}
  }

  std::vector<std::vector<int>> tracker_res;
  int tracker_num = this->trackers.size();
  for (int idx = tracker_num - 1; idx >= 0; idx--) {
    if (this->trackers[idx]->time_since_update > this->max_age) {
      this->unused_trackers.push_back(this->trackers[idx]);
      this->trackers.erase(this->trackers.begin() + idx);
    }
  }
  return true;
}

cv::Mat OcSortTracker::get_trackers(void) {
  // return [class, id, xmin, ymin, xmax, ymax]
  cv::Mat tracker_res;
  int tracker_num = this->trackers.size();
  for (int idx = tracker_num - 1; idx >= 0; idx--) {
    cv::Mat rect(1, 6, CV_32FC1);
    if (this->trackers[idx]->last_observation[0] < 0) {
      rect(cv::Rect(2, 0, 4, 1)) = this->trackers[idx]->get_state();
    } else {
      *rect.ptr<float>(0, 2) = this->trackers[idx]->last_observation[0];
      *rect.ptr<float>(0, 3) = this->trackers[idx]->last_observation[1];
      *rect.ptr<float>(0, 4) = this->trackers[idx]->last_observation[2];
      *rect.ptr<float>(0, 5) = this->trackers[idx]->last_observation[3];
    }
    if (this->trackers[idx]->time_since_update < this->max_age &&
        (this->trackers[idx]->hit_streak >= this->min_hits)) {
      *rect.ptr<float>(0, 1) = this->trackers[idx]->id;
      *rect.ptr<float>(0, 0) = this->classid;
      tracker_res.push_back(rect);
    }
  }
  return tracker_res;
}

void speed_direction_batch(const cv::Mat &detections,
                           const cv::Mat &previous_obs, cv::Mat &directx,
                           cv::Mat &directy) {
  cv::Mat det_center =
      (detections.colRange(1, 3) + detections.colRange(3, 5)) / 2.0;
  cv::Mat pre_center =
      (previous_obs.colRange(1, 3) + previous_obs.colRange(3, 5)) / 2.0;
  for (int i = 0; i < previous_obs.rows; i++) {
    cv::Mat dist = det_center - pre_center.rowRange(i, i + 1);
    sqrt(dist.colRange(1, 2) * dist.colRange(1, 2) +
             dist.colRange(2, 3) * dist.colRange(2, 3),
         dist);
    directx(cv::Rect(0, i, detections.cols, 1)) =
        (det_center.colRange(1, 2) -
         pre_center.rowRange(i, i + 1).colRange(1, 2)) /
        dist;
    directy(cv::Rect(0, i, detections.cols, 1)) =
        (det_center.colRange(2, 3) -
         pre_center.rowRange(i, i + 1).colRange(2, 3)) /
        dist;
  }
}

void iou_batch(cv::Mat &bboxes1, cv::Mat &bboxes2, cv::Mat &iou_matrix) {
  for (int row = 0; row < bboxes1.rows; row++) {
    for (int col = 0; col < bboxes2.rows; col++) {
      cv::Mat box1 = bboxes1(cv::Rect(2, row, 4, 1));
      cv::Mat box2 = bboxes2(cv::Rect(0, col, 4, 1));
      float inner_xmin = std::max(box1.at<float>(0, 0), box2.at<float>(0, 0));
      float inner_ymin = std::max(box1.at<float>(0, 1), box2.at<float>(0, 1));
      float inner_xmax = std::min(box1.at<float>(0, 2), box2.at<float>(0, 2));
      float inner_ymax = std::min(box1.at<float>(0, 3), box2.at<float>(0, 3));
      float inner_area = (inner_xmax - inner_xmin) * (inner_ymax - inner_ymin);

      float outer_xmin = std::min(box1.at<float>(0, 0), box2.at<float>(0, 0));
      float outer_ymin = std::min(box1.at<float>(0, 1), box2.at<float>(0, 1));
      float outer_xmax = std::max(box1.at<float>(0, 2), box2.at<float>(0, 2));
      float outer_ymax = std::max(box1.at<float>(0, 3), box2.at<float>(0, 3));
      float outer_area = (outer_xmax - outer_xmin) * (outer_ymax - outer_ymin);

      float iou = inner_area / outer_area;
      iou_matrix.at<float>(row, col) = iou;
    }
  }
}

void OcSortTracker::associate_only_iou(cv::Mat detections, cv::Mat trackers,
                                       float iou_threshold,
                                       std::map<int, int> &matches,
                                       std::vector<int> &mismatch_row,
                                       std::vector<int> &mismatch_col) {
  matches.clear();
  mismatch_row.clear();
  mismatch_col.clear();
  if (detections.empty() || trackers.empty()) {
    for (int i = 0; i < detections.rows; ++i) mismatch_row.push_back(i);
    for (int i = 0; i < trackers.rows; ++i) mismatch_col.push_back(i);
    return;
  }

  cv::Mat iou_matrix(detections.rows, trackers.rows, CV_32FC1);
  iou_batch(detections, trackers, iou_matrix);

  cv::Mat x(iou_matrix.rows, 1, CV_32S, cv::Scalar(-1));
  cv::Mat y(iou_matrix.cols, 1, CV_32S, cv::Scalar(-1));
  float maxValue =
      *std::max_element(iou_matrix.begin<float>(), iou_matrix.end<float>());
  if (maxValue > iou_threshold) {
    lapjv_internal(-iou_matrix, true, 1000.f, reinterpret_cast<int *>(x.data),
                   reinterpret_cast<int *>(y.data));
  }

  for (int i = 0; i < x.rows; ++i) {
    int j = *x.ptr<int>(i);
    if (j >= 0 && iou_matrix.at<float>(i, j) >= iou_threshold)
      matches.insert({i, j});
    else
      mismatch_row.push_back(i);
  }

  for (int i = 0; i < y.rows; ++i) {
    int j = *y.ptr<int>(i);
    if (j < 0 || iou_matrix.at<float>(j, i) < iou_threshold)
      mismatch_col.push_back(i);
  }
  return;
}

void OcSortTracker::associate(cv::Mat detections, cv::Mat trackers,
                              float iou_threshold, cv::Mat velocities,
                              cv::Mat previous_obs, float vdc_weight,
                              std::map<int, int> &matches,
                              std::vector<int> &mismatch_row,
                              std::vector<int> &mismatch_col) {
  if (trackers.empty()) {
    return;
  }
  cv::Mat directx(detections.rows, previous_obs.rows, CV_32SC1),
      directy(detections.rows, previous_obs.rows, CV_32SC1);
  speed_direction_batch(detections, previous_obs, directx, directy);
}
