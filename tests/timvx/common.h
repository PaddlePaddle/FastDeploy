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
#pragma once
#include <fstream>
#include "fastdeploy/vision.h"

std::vector<std::string> stringSplit(const std::string& str, char delim) {
    std::stringstream ss(str);
    std::string item;
    std::vector<std::string> elems;
    while (std::getline(ss, item, delim)) {
        if (!item.empty()) {
            elems.push_back(item);
        }
    }
    return elems;
}


bool CompareDetResult(const fastdeploy::vision::DetectionResult& res,
                      const std::string& det_result_file) {
  std::ifstream res_str(det_result_file);
  if (!res_str.is_open()) {
    std::cout<< "Could not open detect result file : "
              << det_result_file <<"\n"<< std::endl;
    return false;
  }
  int obj_num = 0;
  while (!res_str.eof()) {
    std::string line;
    std::getline(res_str, line);
    if (line.find("DetectionResult") == line.npos
        && line.find(",") != line.npos ) {
      auto strs = stringSplit(line, ',');
      if (strs.size() != 6) {
        std::cout<< "Failed to parse result file : "
                  << det_result_file <<"\n"<< std::endl;
        return false;
      }
      std::vector<float> vals;
      for (auto str : strs) {
        vals.push_back(atof(str.c_str()));
      }
      if (abs(res.scores[obj_num] - vals[4]) > 0.3) {
        std::cout<< "Score error, the result is: "
                  << res.scores[obj_num] << " but the expected is: "
                  << vals[4] << std::endl;
        return false;
      }
      if (abs(res.label_ids[obj_num] - vals[5]) > 0) {
        std::cout<< "label error, the result is: "
                  << res.label_ids[obj_num] << " but the expected is: "
                  << vals[5] <<std::endl;
        return false;
      }
      std::array<float, 4> boxes = res.boxes[obj_num++];
      for (auto i = 0; i < 4; i++) {
        if (abs(boxes[i] - vals[i]) > 5) {
           std::cout<< "position error, the result is: "
           << boxes[i] << " but the expected is: " << vals[i] <<std::endl;
           return false;
        }
      }
    }
  }
  return true;
}


bool CompareClsResult(const fastdeploy::vision::ClassifyResult& res,
                      const std::string& cls_result_file) {
  std::ifstream res_str(cls_result_file);
  if (!res_str.is_open()) {
    std::cout<< "Could not open detect result file : "
              << cls_result_file << "\n" << std::endl;
    return false;
  }
  int obj_num = 0;
  while (!res_str.eof()) {
    std::string line;
    std::getline(res_str, line);
    if (line.find("label_ids") != line.npos
        && line.find(":") != line.npos) {
      auto strs = stringSplit(line, ':');
      if (strs.size() != 2) {
        std::cout<< "Failed to parse result file : "
                  << cls_result_file <<"\n"<< std::endl;
        return false;
      }
      int32_t label = static_cast<int32_t>(atof(strs[1].c_str()));
      if (res.label_ids[obj_num] != label) {
        std::cout<< "label error, the result is: "
                << res.label_ids[obj_num] << " but the expected is: "
                << label<< "\n" << std::endl;
        return false;
      }
    } else if (line.find("scores") != line.npos
               && line.find(":") != line.npos) {
      auto strs = stringSplit(line, ':');
      if (strs.size() != 2) {
        std::cout<< "Failed to parse result file : "
                  << cls_result_file << "\n" << std::endl;
        return false;
      }
      float score = atof(strs[1].c_str());
      if (abs(res.scores[obj_num] - score) > 1e-1) {
        std::cout << "score error, the result is: "
                  << res.scores[obj_num] << " but the expected is: "
                  << score << "\n" << std::endl;
        return false;
      } else {
        obj_num++;
      }
    } else if (line.size()) {
      std::cout << "Unknown File. \n" << std::endl;
      return false;
    }
  }
  return true;
}

bool WriteSegResult(const fastdeploy::vision::SegmentationResult& res,
                    const std::string& seg_result_file) {
  std::ofstream res_str(seg_result_file);
  if (!res_str.is_open()) {
    std::cerr<< "Could not open segmentation result file : "
            << seg_result_file <<" to write.\n"<< std::endl;
      return false;
  }
  std::string out;
  out = "";
  // save shape
  for (auto shape : res.shape) {
    out += std::to_string(shape) + ",";
  }
  out += "\n";
  // save label
  for (auto label : res.label_map) {
    out +=  std::to_string(label) + ",";
  }
  out += "\n";
  // save score
  if (res.contain_score_map) {
     for (auto score : res.score_map) {
      out +=  std::to_string(score) + ",";
    }
  }
  res_str << out;
  return true;
}

bool CompareSegResult(const fastdeploy::vision::SegmentationResult& res,
                      const std::string& seg_result_file) {
  std::ifstream res_str(seg_result_file);
  if (!res_str.is_open()) {
    std::cout<< "Could not open detect result file : "
              << seg_result_file <<"\n"<< std::endl;
    return false;
  }
  std::string line;
  std::getline(res_str, line);
  if (line.find(",") == line.npos) {
    std::cout << "Unexpected File." << std::endl;
    return false;
  }
  // check shape diff
  auto shape_strs = stringSplit(line, ',');
  std::vector<int64_t> shape;
  for (auto str : shape_strs) {
    shape.push_back(static_cast<int64_t>(atof(str.c_str())));
  }
  if (shape.size() != res.shape.size()) {
    std::cout << "Output shape and expected shape size mismatch, shape size: "
              << res.shape.size() << " expected shape size: "
              << shape.size() << std::endl;
    return false;
  }
  for (auto i = 0; i < res.shape.size(); i++) {
    if (res.shape[i] != shape[i]) {
      std::cout << "Output Shape and expected shape mismatch, shape: "
                << res.shape[i] << " expected: " << shape[i] << std::endl;
      return false;
    }
  }
  std::cout << "Shape check passed!" << std::endl;

  std::getline(res_str, line);
  if (line.find(",") == line.npos) {
    std::cout << "Unexpected File." << std::endl;
    return false;
  }
  // check label
  auto label_strs = stringSplit(line, ',');
  std::vector<uint8_t> labels;
  for (auto str : label_strs) {
    labels.push_back(static_cast<uint8_t>(atof(str.c_str())));
  }
  if (labels.size() != res.label_map.size()) {
    std::cout << "Output labels and expected shape size mismatch." << std::endl;
    return false;
  }
  for (auto i = 0; i < res.label_map.size(); i++) {
    if (res.label_map[i] != labels[i]) {
      std::cout << "Output labels and expected labels mismatch." << std::endl;
      return false;
    }
  }
  std::cout << "Label check passed!" << std::endl;

  // check score_map
  if (res.contain_score_map) {
    auto scores_strs = stringSplit(line, ',');
    std::vector<float> scores;
    for (auto str : scores_strs) {
      scores.push_back(static_cast<float>(atof(str.c_str())));
    }
    if (scores.size() != res.score_map.size()) {
      std::cout << "Output scores and expected score_map size mismatch."
              <<std::endl;
      return false;
    }
    for (auto i = 0; i < res.score_map.size(); i++) {
      if (abs(res.score_map[i] - scores[i]) > 3e-1) {
        std::cout << "Output scores and expected scores mismatch."
                 << std::endl;
        return false;
      }
    }
  }
  return true;
}
