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

#include <vector>
#include "gstreamer/plugin/fdtracker/include/ocsort.h"

typedef enum { horizontal = 0, vertical = 1} region_type;

class Trajectory {
 public:
  Trajectory(void) {}
  ~Trajectory(void) {}

  std::vector<std::vector<int>> entrance_count(OcSortTracker* octracker);
  std::vector<int> breakin_count(OcSortTracker* octracker);
  bool set_region(region_type inout_type, std::vector<int> region);
  bool set_area(std::vector<int> area);
  void clearset(void);

  std::vector<int> get_region() {return this->region;}
  std::vector<std::vector<float>> get_breakarea() {return  this->breakarea;}
  std::vector<int> get_count() {return std::vector<int>{this->countin,
                                                        this->countout};}

 private:
  int countin = 0;
  int countout = 0;
  int num_pts = 0;
  std::vector<int> breakin;
  std::vector<int> breakout;
  std::vector<int> region;
  std::vector<std::vector<float>> breakarea = {{0.f, 0.f}, {0.f, 0.f},
                                               {0.f, 0.f}};
  region_type inout_type = horizontal;
};
