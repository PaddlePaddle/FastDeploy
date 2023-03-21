//Copyright (c) 2023 niuzhibo. All Rights Reserved.

#pragma once

#include <vector>
#include "ocsort.h"

typedef enum { horizontal = 0, vertical = 1} region_type;

class Trajectory {
 public:
  Trajectory(void){};
  ~Trajectory(void){};

  std::vector<std::vector<int>> entrance_count(OcSortTracker* octracker);
  std::vector<int> breakin_count(OcSortTracker* octracker);
  bool set_region(region_type inout_type, std::vector<int> region);
  bool set_area(std::vector<int> area);
  void clearset(void);

  std::vector<int> get_region() {return this->region;};
  std::vector<std::vector<float>> get_breakarea() {return  this->breakarea;};
  std::vector<int> get_count() {return std::vector<int>{this->countin, this->countout};};
  

 private:
  int countin=0;
  int countout=0;
  int num_pts=0;
  std::vector<int> breakin;
  std::vector<int> breakout;
  std::vector<int> region;
  std::vector<std::vector<float>> breakarea = {{0.f,0.f}, {0.f,0.f}, {0.f,0.f}};
  region_type inout_type = horizontal;
};

