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

#include <algorithm>

#include "fastdeploy/vision/visualize/visualize.h"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"

namespace fastdeploy {
namespace vision {

using matrix = std::vector<std::vector<float>>;

matrix multiple(const matrix a, const matrix b) {
  const int m = a.size();  // a rows
  if (m == 0) {
    matrix c;
    return c;
  }
  // assert(a[0].size() == b.size(), "A[m,n] * B[p,q], n must equal to p.");
  const int n = a[0].size();  // a cols
  const int p = b[0].size();  // b cols
  matrix c(m, std::vector<float>(p, 0));
  for (auto i = 0; i < m; i++) {
    for (auto j = 0; j < p; j++) {
      for (auto k = 0; k < n; k++) c[i][j] += a[i][k] * b[k][j];
    }
  }
  return c;
}

cv::Mat VisDetection3D(const cv::Mat& im, const Detection3DResult& result,
                       float score_threshold, int line_size, float font_size) {
  if (result.scores.empty()) {
    return im;
  }
  matrix k_data = {{721.53771973, 0.0, 609.55932617},
                   {0.0, 721.53771973, 172.85400391},
                   {0, 0, 1}};
  std::vector<double> rvec = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  std::vector<double> tvec = {0, 0, 0};

  matrix connect_line_id = {{1, 0}, {2, 7}, {3, 6}, {4, 5}, {1, 2}, {2, 3},
                            {3, 4}, {4, 1}, {0, 7}, {7, 6}, {6, 5}, {5, 0}};

  int max_label_id =
      *std::max_element(result.label_ids.begin(), result.label_ids.end());
  std::vector<int> color_map = GenerateColorMap(max_label_id);
  int h = im.rows;
  int w = im.cols;
  cv::Mat vis_im;
  cv::resize(im, vis_im, cv::Size(1280, 384), 0, 0, 0);
  // auto vis_im = im.clone();
  for (size_t i = 0; i < result.scores.size(); ++i) {
    if (result.scores[i] < 0.5) {
      continue;
    }
    float h = result.boxes[i][4];
    float w = result.boxes[i][5];
    float l = result.boxes[i][6];

    float x = result.center[i][0];
    float y = result.center[i][1];
    float z = result.center[i][2];
    std::vector<float> x_corners = {0, l, l, l, l, 0, 0, 0};
    std::vector<float> y_corners = {0, 0, h, h, 0, 0, h, h};
    std::vector<float> z_corners = {0, 0, 0, w, w, w, w, 0};

    // x_corners += -np.float32(l) / 2
    // y_corners += -np.float32(h)
    // z_corners += -np.float32(w) / 2
    for (auto j = 0; j < x_corners.size(); j++) {
      x_corners[j] = x_corners[j] - l / 2;
      y_corners[j] = y_corners[j] - h;
      z_corners[j] = z_corners[j] - w / 2;
    }

    // corners_3d = np.array([x_corners, y_corners, z_corners])
    matrix corners_3d = {x_corners, y_corners, z_corners};

    // rot_mat = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry),
    // 0, np.cos(ry)]])
    float ry = result.yaw_angle[i];
    matrix rot_mat = {
        {cosf(ry), 0, sinf(ry)}, {0, 1, 0}, {sinf(ry), 0, cosf(ry)}};

    // corners_3d = np.matmul(rot_mat, corners_3d)
    matrix rot_corners_3d = multiple(rot_mat, corners_3d);

    // corners_3d += np.array([x, y, z]).reshape([3, 1])
    for (auto j = 0; j < rot_corners_3d[0].size(); j++) {
      rot_corners_3d[0][j] += x;
      rot_corners_3d[1][j] += y;
      rot_corners_3d[2][j] += z;
    }
    // corners_2d = np.matmul(K, corners_3d)
    auto corners_2d = multiple(k_data, rot_corners_3d);

    // corners_2d = corners_2d[:2] / corners_2d[2]
    for (auto j = 0; j < corners_2d[0].size(); j++) {
      corners_2d[0][j] /= corners_2d[2][j];
      corners_2d[1][j] /= corners_2d[2][j];
    }
    // box2d =
    // np.array([min(corners_2d[0]),min(corners_2d[1]),max(corners_2d[0]),max(corners_2d[1])])
    std::vector<float> box2d = {
        *std::min_element(corners_2d[0].begin(), corners_2d[0].end()),
        *std::min_element(corners_2d[1].begin(), corners_2d[1].end()),
        *std::max_element(corners_2d[0].begin(), corners_2d[0].end()),
        *std::max_element(corners_2d[1].begin(), corners_2d[1].end())};

    if (box2d[0] == 0 && box2d[1] == 0 && box2d[2] == 0 && box2d[3] == 0) {
      continue;
    }
    // cv2.projectPoints(box3d.T, rvec, tvec, K, 0)
    std::vector<cv::Point3f> points3d;
    for (auto j = 0; j < rot_corners_3d[0].size(); j++) {
      points3d.push_back(cv::Point3f(rot_corners_3d[0][j], rot_corners_3d[1][j],
                                     rot_corners_3d[2][j]));
    }
    cv::Mat rVec(3, 3, cv::DataType<double>::type, rvec.data());
    cv::Mat tVec(3, 1, cv::DataType<double>::type, tvec.data());
    std::vector<float> vec_k;
    for (auto&& v : k_data) {
      vec_k.insert(vec_k.end(), v.begin(), v.end());
    }
    cv::Mat intrinsicMat(3, 3, cv::DataType<float>::type, vec_k.data());
    cv::Mat distCoeffs(5, 1, cv::DataType<double>::type);
    std::vector<cv::Point2f> projectedPoints;
    cv::projectPoints(points3d, rVec, tVec, intrinsicMat, distCoeffs,
                      projectedPoints);

    int c0 = color_map[3 * result.label_ids[i] + 0];
    int c1 = color_map[3 * result.label_ids[i] + 1];
    int c2 = color_map[3 * result.label_ids[i] + 2];
    cv::Scalar color = cv::Scalar(c0, c1, c2);
    for (auto id = 0; id < connect_line_id.size(); id++) {
      int p1 = connect_line_id[id][0];
      int p2 = connect_line_id[id][1];
      cv::line(vis_im, projectedPoints[p1], projectedPoints[p2], color, 1);
    }
    int font = cv::FONT_HERSHEY_SIMPLEX;
    std::string score = std::to_string(result.scores[i]);
    if (score.size() > 4) {
      score = score.substr(0, 4);
    }
    std::string text = std::to_string(result.label_ids[i]) + ", " + score;
    cv::Point2f original;
    original.x = box2d[0];
    original.y = box2d[1];
    cv::putText(vis_im, text, original, font, font_size,
                cv::Scalar(255, 255, 255), 1);
  }
  return vis_im;
}

}  // namespace vision
}  // namespace fastdeploy
