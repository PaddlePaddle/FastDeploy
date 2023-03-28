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

#include <stdio.h>
#include <vector>

#include "gstreamer/plugin/fdtracker/include/trajectory.h"

std::vector<std::vector<int>> Trajectory::entrance_count(
    OcSortTracker* octracker) {
  std::vector<int> breakin, breakout;
  std::vector<std::vector<int>> res;
  for (auto tracker : octracker->trackers) {
    if (tracker->observations.size() < 2) {
      continue;
    }
    if (this->inout_type == horizontal) {
      auto precenter = tracker->observations[tracker->age - 1][3];
      auto curcenter = tracker->observations[tracker->age][3];
      if (precenter <= this->region[1] && curcenter > this->region[1]) {
        breakin.emplace_back(tracker->id);
      } else if (precenter >= this->region[1] && curcenter < this->region[1]) {
        breakout.emplace_back(tracker->id);
      }
    } else {
      auto precenter = (tracker->observations[tracker->age - 1][0] +
                        tracker->observations[tracker->age - 1][2]) /
                       2.0;
      auto curcenter = (tracker->observations[tracker->age][0] +
                        tracker->observations[tracker->age][2]) /
                       2.0;
      if (precenter <= this->region[0] && curcenter > this->region[0]) {
        breakin.emplace_back(tracker->id);
      } else if (precenter >= this->region[0] && curcenter < this->region[0]) {
        breakout.emplace_back(tracker->id);
      }
    }
  }
  res.emplace_back(breakin);
  res.emplace_back(breakout);
  countin += breakin.size();
  countout += breakout.size();
  return res;
}

bool checkinarea(cv::Point point, std::vector<std::vector<float>> area,
                 int num_pts) {
  if (area.size() == 0) {
    return false;
  }
  cv::Point points[1][num_pts];
  int maxw = 0;
  int maxh = 0;
  for (int i = 0; i < num_pts; i++) {
    if (area[i][0] > maxw) {
      maxw = area[i][0];
    }
    if (area[i][1] > maxh) {
      maxh = area[i][1];
    }
    points[0][i] = cv::Point(area[i][0], area[i][1]);
  }
  maxw = std::max(maxw, point.x);
  maxh = std::max(maxh, point.y);

  cv::Mat img = cv::Mat::zeros(maxh + 1, maxw + 1, CV_32FC1);
  // img.setTo(0);

  const cv::Point* ppt[] = {points[0]};
  int npt[] = {num_pts};
  cv::fillPoly(img, ppt, npt, 1, cv::Scalar(1.0));
  if (img.at<float>(point) > 0.5) {
    return true;
  } else {
    return false;
  }
}

std::vector<int> Trajectory::breakin_count(OcSortTracker* octracker) {
  std::vector<int> res;
  for (auto tracker : octracker->trackers) {
    float locx =
        (tracker->last_observation[0] + tracker->last_observation[1]) / 2.0;
    float locy = tracker->last_observation[3];
    cv::Point2f loc(locx, locy);
    if (checkinarea(loc, this->breakarea, this->num_pts)) {
      res.push_back(tracker->id);
    }
  }
  return res;
}

bool Trajectory::set_region(region_type inout_type, std::vector<int> region) {
  if (region.size() != 2) {
    printf("illegal region set, the region should be a vector of size=2(x,y)");
    return false;
  }
  this->inout_type = inout_type;
  if (inout_type == horizontal) {
    this->region = region;
  } else {
    this->region = region;
  }
  return true;
}

bool Trajectory::set_area(std::vector<int> area) {
  if (area.size() < 6) {
    printf(
        "illegal area set, the area should include at least 3 points, so the "
        "vector should have a size>=6");
    return false;
  }
  if (area.size() % 2 != 0) {
    printf(
        "illegal area set, the area vector should have even nunmbers, while "
        "got number of %d",
        static_cast<int>(area.size()));
    return false;
  }
  this->breakarea.clear();

  this->num_pts = area.size() / 2;
  for (int i = 0; i < this->num_pts; i++) {
    std::vector<float> temp;
    temp.push_back(area[i * 2]);
    temp.push_back(area[i * 2 + 1]);
    this->breakarea.push_back(temp);
  }

  return true;
}

void Trajectory::clearset(void) {
  this->countin = 0;
  this->countout = 0;
  this->breakin.clear();
  this->region.clear();
  this->breakarea.clear();
  this->inout_type = horizontal;
}
