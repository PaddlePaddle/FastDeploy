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
//
// Part of the following code in this file refs to
// https://github.com/CnybTseng/JDE/blob/master/platforms/common/trajectory.h
//
// Copyright (c) 2020 CnybTseng
// Licensed under The MIT License

#pragma once

#include <vector>
#include <map>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#define mat2vec4f(m)             \
  cv::Vec4f(*m.ptr<float>(0, 0), \
            *m.ptr<float>(0, 1), \
            *m.ptr<float>(0, 2), \
            *m.ptr<float>(0, 3))

typedef enum { New = 0, Tracked = 1, Lost = 2, Removed = 3 } TrajectoryState;

inline void printmat(cv::Mat &mat) {
  std::string str;
  for (int i=0; i < mat.rows; i++) {
    for (int j=0; j < mat.cols; j++) {
      str += std::to_string(*(mat.ptr<float>(i, j)))+" ";
    }
  }
  printf("\nmat rows: %d, cols: %d, str: %s", mat.rows, mat.cols, str.c_str());
}

class TKalmanFilter : public cv::KalmanFilter {
 public:
  TKalmanFilter(void);
  virtual ~TKalmanFilter(void) {}
  virtual void init(const cv::Mat &measurement);
  virtual const cv::Mat &predict();
  virtual const cv::Mat &correct(const cv::Mat &measurement);
  virtual void project(cv::Mat *mean, cv::Mat *covariance) const;

 private:
  float std_weight_position;
  float std_weight_velocity;
};

inline TKalmanFilter::TKalmanFilter(void) : cv::KalmanFilter(8, 4) {
  cv::KalmanFilter::transitionMatrix = cv::Mat::eye(8, 8, CV_32F);
  for (int i = 0; i < 4; ++i)
    cv::KalmanFilter::transitionMatrix.at<float>(i, i + 4) = 1;
  cv::KalmanFilter::measurementMatrix = cv::Mat::eye(4, 8, CV_32F);
  std_weight_position = 1 / 20.f;
  std_weight_velocity = 1 / 160.f;
}

class KalmanTracker : public TKalmanFilter {
 public:
  KalmanTracker();
  KalmanTracker(const cv::Vec4f &ltrb, float score);
  KalmanTracker(const KalmanTracker &other);
  KalmanTracker &operator=(const KalmanTracker &rhs);
  virtual ~KalmanTracker(void) {}

  void alloc_id(void);
  virtual const cv::Mat &predict(void);
  virtual void update(cv::Vec4f dets, bool angle_cost);
  virtual void activate(int timestamp);
  virtual void reactivate(KalmanTracker *traj, int timestamp,
                          bool newid = false);
  virtual void mark_lost(void);
  virtual void mark_removed(void);
  virtual int get_length(void);
  const cv::Mat get_state(void);

  TrajectoryState state;
  cv::Vec4f ltrb;
  cv::Vec4f last_observation{-1, -1, -1, -1};
  cv::Vec2f velocity;
  std::map<int, cv::Vec4f> observations;
  int id;
  bool is_activated = false;
  int time_since_update = 0;
  int timestamp;
  int starttime;
  float score;
  int age = 0;
  int delta_t = 3;
  int hits = 0;
  int hit_streak = 0;

 private:
  static int count;
  cv::Vec4f xyah;
  float eta;
  int length = 0;
};

inline cv::Vec4f ltrb2xyah(const cv::Vec4f &ltrb) {
  cv::Vec4f xyah;
  xyah[0] = (ltrb[0] + ltrb[2]) * 0.5f;
  xyah[1] = (ltrb[1] + ltrb[3]) * 0.5f;
  xyah[3] = ltrb[3] - ltrb[1];
  xyah[2] = (ltrb[2] - ltrb[0]) / xyah[3];
  return xyah;
}

inline cv::Mat xyah2ltrb(const cv::Mat &xyah_in) {
  cv::Vec4f ltrb;
  cv::Vec4f xyah = mat2vec4f(xyah_in);
  ltrb[0] = xyah[0] - (xyah[3]*xyah[2])*0.5f;
  ltrb[1] = xyah[1] - (xyah[3]/xyah[2])*0.5f;
  ltrb[2] = xyah[0] + (xyah[3]*xyah[2])*0.5f;
  ltrb[3] = xyah[1] + (xyah[3]/xyah[2])*0.5f;
  return cv::Mat(ltrb).t();
}

inline KalmanTracker::KalmanTracker()
    : state(New),
      ltrb(cv::Vec4f()),
      id(0),
      is_activated(false),
      timestamp(0),
      starttime(0),
      time_since_update(0),
      score(0),
      eta(0.9),
      length(0) {
        xyah = ltrb2xyah(ltrb);
        alloc_id();
        TKalmanFilter::init(cv::Mat(xyah));
      }

inline KalmanTracker::KalmanTracker(const cv::Vec4f &ltrb_,
                              float score_)
    : state(New),
      ltrb(ltrb_),
      last_observation(ltrb_),
      id(0),
      age(0),
      is_activated(false),
      timestamp(0),
      starttime(0),
      time_since_update(0),
      score(score_),
      eta(0.9),
      length(1) {
  this->observations[this->age] = ltrb_;
  xyah = ltrb2xyah(ltrb);
  alloc_id();
  TKalmanFilter::init(cv::Mat(xyah));
}

inline KalmanTracker::KalmanTracker(const KalmanTracker &other)
    : state(other.state),
      ltrb(other.ltrb),
      id(other.id),
      is_activated(other.is_activated),
      timestamp(other.timestamp),
      starttime(other.starttime),
      xyah(other.xyah),
      score(other.score),
      eta(other.eta),
      length(other.length) {
  // copy state in KalmanFilter

  other.statePre.copyTo(cv::KalmanFilter::statePre);
  other.statePost.copyTo(cv::KalmanFilter::statePost);
  other.errorCovPre.copyTo(cv::KalmanFilter::errorCovPre);
  other.errorCovPost.copyTo(cv::KalmanFilter::errorCovPost);
}

inline KalmanTracker &KalmanTracker::operator=(const KalmanTracker &rhs) {
  this->state = rhs.state;
  this->ltrb = rhs.ltrb;
  this->id = rhs.id;
  this->is_activated = rhs.is_activated;
  this->timestamp = rhs.timestamp;
  this->starttime = rhs.starttime;
  this->xyah = rhs.xyah;
  this->score = rhs.score;
  this->eta = rhs.eta;
  this->length = rhs.length;

  // copy state in KalmanFilter

  rhs.statePre.copyTo(cv::KalmanFilter::statePre);
  rhs.statePost.copyTo(cv::KalmanFilter::statePost);
  rhs.errorCovPre.copyTo(cv::KalmanFilter::errorCovPre);
  rhs.errorCovPost.copyTo(cv::KalmanFilter::errorCovPost);

  return *this;
}

inline void KalmanTracker::alloc_id() {
  this->id = count;
  ++count;
}

inline void KalmanTracker::mark_lost(void) { state = Lost; }

inline void KalmanTracker::mark_removed(void) { state = Removed; }

inline int KalmanTracker::get_length(void) { return length; }
