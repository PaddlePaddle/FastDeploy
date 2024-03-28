// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "ocr_postprocess_op.h"
#include <map>
#include "clipper.h"

namespace fastdeploy {
namespace vision {
namespace ocr {

void PostProcessor::GetContourArea(const std::vector<std::vector<float>> &box,
                                   float unclip_ratio, float &distance) {
  int pts_num = box.size();
  float area = 0.0f;
  float dist = 0.0f;
  for (int i = 0; i < pts_num; i++) {
    area += box[i][0] * box[(i + 1) % pts_num][1] -
            box[i][1] * box[(i + 1) % pts_num][0];
    dist += sqrtf((box[i][0] - box[(i + 1) % pts_num][0]) *
                      (box[i][0] - box[(i + 1) % pts_num][0]) +
                  (box[i][1] - box[(i + 1) % pts_num][1]) *
                      (box[i][1] - box[(i + 1) % pts_num][1]));
  }
  area = fabs(float(area / 2.0));

  distance = area * unclip_ratio / dist;
}

std::vector<std::vector<cv::Point>> PostProcessor::UnClip(std::vector<std::vector<float>> box,
                                      const float &unclip_ratio) {
  float distance = 1.0;

  GetContourArea(box, unclip_ratio, distance);

  ClipperLib::ClipperOffset offset;
  ClipperLib::Path p;

  for (int i = 0; i < box.size(); i++) {
    p << ClipperLib::IntPoint(int(box[i][0]), int(box[i][1]));
  }
  offset.AddPath(p, ClipperLib::jtRound, ClipperLib::etClosedPolygon);

  ClipperLib::Paths soln;
  offset.Execute(soln, distance);
  std::vector<std::vector<cv::Point>> paths;

  for (int j = 0; j < soln.size(); j++) {
    std::vector<cv::Point> path;
    for (int i = 0; i < soln[j].size(); i++) {
      path.emplace_back(soln[j][i].X, soln[j][i].Y);
    }
    paths.push_back(path);
  }

  return paths;
}

float **PostProcessor::Mat2Vec(cv::Mat mat) {
  auto **array = new float *[mat.rows];
  for (int i = 0; i < mat.rows; ++i) array[i] = new float[mat.cols];
  for (int i = 0; i < mat.rows; ++i) {
    for (int j = 0; j < mat.cols; ++j) {
      array[i][j] = mat.at<float>(i, j);
    }
  }

  return array;
}

std::vector<std::vector<int>> PostProcessor::OrderPointsClockwise(
    std::vector<std::vector<int>> pts) {
  std::vector<std::vector<int>> box = pts;
  std::sort(box.begin(), box.end(), XsortInt);

  std::vector<std::vector<int>> leftmost = {box[0], box[1]};
  std::vector<std::vector<int>> rightmost = {box[2], box[3]};

  if (leftmost[0][1] > leftmost[1][1]) std::swap(leftmost[0], leftmost[1]);

  if (rightmost[0][1] > rightmost[1][1]) std::swap(rightmost[0], rightmost[1]);

  std::vector<std::vector<int>> rect = {leftmost[0], rightmost[0], rightmost[1],
                                        leftmost[1]};
  return rect;
}

std::vector<std::vector<float>> PostProcessor::Mat2Vector(cv::Mat mat) {
  std::vector<std::vector<float>> img_vec;
  std::vector<float> tmp;

  for (int i = 0; i < mat.rows; ++i) {
    tmp.clear();
    for (int j = 0; j < mat.cols; ++j) {
      tmp.push_back(mat.at<float>(i, j));
    }
    img_vec.push_back(tmp);
  }
  return img_vec;
}

bool PostProcessor::XsortFp32(std::vector<float> a, std::vector<float> b) {
  if (a[0] != b[0]) return a[0] < b[0];
  return false;
}

bool PostProcessor::XsortInt(std::vector<int> a, std::vector<int> b) {
  if (a[0] != b[0]) return a[0] < b[0];
  return false;
}

std::vector<std::vector<float>> PostProcessor::GetMiniBoxes(std::vector<cv::Point> contour,
                                                            float &ssid) {
  cv::RotatedRect box = cv::minAreaRect(contour);
  ssid = std::max(box.size.width, box.size.height);

  cv::Mat points;
  cv::boxPoints(box, points);

  auto array = Mat2Vector(points);
  std::sort(array.begin(), array.end(), XsortFp32);

  std::vector<float> idx1 = array[0], idx2 = array[1], idx3 = array[2],
                     idx4 = array[3];
  if (array[3][1] <= array[2][1]) {
    idx2 = array[3];
    idx3 = array[2];
  } else {
    idx2 = array[2];
    idx3 = array[3];
  }
  if (array[1][1] <= array[0][1]) {
    idx1 = array[1];
    idx4 = array[0];
  } else {
    idx1 = array[0];
    idx4 = array[1];
  }

  array[0] = idx1;
  array[1] = idx2;
  array[2] = idx3;
  array[3] = idx4;

  return array;
}

float PostProcessor::PolygonScoreAcc(std::vector<cv::Point> contour,
                                     cv::Mat pred) {
  int width = pred.cols;
  int height = pred.rows;
  std::vector<float> box_x;
  std::vector<float> box_y;
  for (int i = 0; i < contour.size(); ++i) {
    box_x.push_back(contour[i].x);
    box_y.push_back(contour[i].y);
  }

  int xmin =
      clamp(int(std::floor(*(std::min_element(box_x.begin(), box_x.end())))), 0,
            width - 1);
  int xmax =
      clamp(int(std::ceil(*(std::max_element(box_x.begin(), box_x.end())))), 0,
            width - 1);
  int ymin =
      clamp(int(std::floor(*(std::min_element(box_y.begin(), box_y.end())))), 0,
            height - 1);
  int ymax =
      clamp(int(std::ceil(*(std::max_element(box_y.begin(), box_y.end())))), 0,
            height - 1);

  cv::Mat mask;
  mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1);

  cv::Point *rook_point = new cv::Point[contour.size()];

  for (int i = 0; i < contour.size(); ++i) {
    rook_point[i] = cv::Point(int(box_x[i]) - xmin, int(box_y[i]) - ymin);
  }
  const cv::Point *ppt[1] = {rook_point};
  int npt[] = {int(contour.size())};

  cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(1));

  cv::Mat croppedImg;
  pred(cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1))
      .copyTo(croppedImg);
  float score = cv::mean(croppedImg, mask)[0];

  delete[] rook_point;
  return score;
}

float PostProcessor::BoxScoreFast(std::vector<std::vector<float>> box_array,
                                  cv::Mat pred) {
  auto array = box_array;
  int width = pred.cols;
  int height = pred.rows;

  float box_x[4] = {array[0][0], array[1][0], array[2][0], array[3][0]};
  float box_y[4] = {array[0][1], array[1][1], array[2][1], array[3][1]};

  int xmin = clamp(int(std::floor(*(std::min_element(box_x, box_x + 4)))), 0,
                   width - 1);
  int xmax = clamp(int(std::ceil(*(std::max_element(box_x, box_x + 4)))), 0,
                   width - 1);
  int ymin = clamp(int(std::floor(*(std::min_element(box_y, box_y + 4)))), 0,
                   height - 1);
  int ymax = clamp(int(std::ceil(*(std::max_element(box_y, box_y + 4)))), 0,
                   height - 1);

  cv::Mat mask;
  mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1);

  cv::Point root_point[4];
  root_point[0] = cv::Point(int(array[0][0]) - xmin, int(array[0][1]) - ymin);
  root_point[1] = cv::Point(int(array[1][0]) - xmin, int(array[1][1]) - ymin);
  root_point[2] = cv::Point(int(array[2][0]) - xmin, int(array[2][1]) - ymin);
  root_point[3] = cv::Point(int(array[3][0]) - xmin, int(array[3][1]) - ymin);
  const cv::Point *ppt[1] = {root_point};
  int npt[] = {4};
  cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(1));

  cv::Mat croppedImg;
  pred(cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1))
      .copyTo(croppedImg);

  auto score = cv::mean(croppedImg, mask)[0];
  return score;
}

std::vector<std::vector<std::array<int, 2>>> PostProcessor::PloygonsFromBitmap(
    const cv::Mat pred, const cv::Mat bitmap, const float &box_thresh,
    const float &det_db_unclip_ratio, const std::string &det_db_score_mode) {
  const int min_size = 3;
  const int max_candidates = 1000;

  int width = bitmap.cols;
  int height = bitmap.rows;

  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;

  cv::findContours(bitmap, contours, hierarchy, cv::RETR_LIST,
                   cv::CHAIN_APPROX_SIMPLE);

  int num_contours =
      contours.size() >= max_candidates ? max_candidates : contours.size();

  std::vector<std::vector<std::array<int, 2>>> boxes;

  for (int _i = 0; _i < num_contours; _i++) {
    std::vector<cv::Point> contour = contours[_i];
    double epsilon = 0.002 * cv::arcLength(contour, true);
    std::vector<cv::Point> approx;
    cv::approxPolyDP(contour, approx, epsilon, true);
    if (approx.size() < 4) continue;

    float score = PolygonScoreAcc(contours[_i], pred);
    if (score < box_thresh) continue;

    std::vector<std::vector<float>> box;
    for(int i=0; i < approx.size(); i++) {
      box.push_back({float(approx[i].x), float(approx[i].y)});
    }

    // start for unclip
    std::vector<std::vector<cv::Point>> paths = UnClip(box, det_db_unclip_ratio);
    if (paths.size() > 1) continue;

    std::vector<cv::Point> path = paths[0];

    // end for unclip

    float ssid;
    GetMiniBoxes(path, ssid);

    if (ssid < min_size + 2) continue;

    int dest_width = pred.cols;
    int dest_height = pred.rows;
    std::vector<std::array<int, 2>> intcliparray;

    for (int num_pt = 0; num_pt < path.size(); num_pt++) {
      std::array<int, 2> a{
          int(clampf(
              roundf(path[num_pt].x / float(width) * float(dest_width)),
              0, float(dest_width))),
          int(clampf(
              roundf(path[num_pt].y / float(height) * float(dest_height)),
              0, float(dest_height)))};
      intcliparray.push_back(a);
    }

    boxes.push_back(intcliparray);

  }  // end for
  return boxes;
}

std::vector<std::vector<std::array<int, 2>>> PostProcessor::BoxesFromBitmap(
    const cv::Mat pred, const cv::Mat bitmap, const float &box_thresh,
    const float &det_db_unclip_ratio, const std::string &det_db_score_mode) {
  const int min_size = 3;
  const int max_candidates = 1000;

  int width = bitmap.cols;
  int height = bitmap.rows;

  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;

  cv::findContours(bitmap, contours, hierarchy, cv::RETR_LIST,
                   cv::CHAIN_APPROX_SIMPLE);

  int num_contours =
      contours.size() >= max_candidates ? max_candidates : contours.size();

  std::vector<std::vector<std::array<int, 2>>> boxes;

  for (int _i = 0; _i < num_contours; _i++) {
    if (contours[_i].size() <= 2) {
      continue;
    }
    float ssid;
    auto array = GetMiniBoxes(contours[_i], ssid);

    auto box_for_unclip = array;
    // end get_mini_box

    if (ssid < min_size) {
      continue;
    }

    float score;
    if (det_db_score_mode == "slow") /* compute using polygon*/
      score = PolygonScoreAcc(contours[_i], pred);
    else
      score = BoxScoreFast(array, pred);

    if (score < box_thresh) continue;

    // start for unclip
    std::vector<std::vector<cv::Point>> paths = UnClip(box_for_unclip, det_db_unclip_ratio);
    // end for unclip

    auto cliparray = GetMiniBoxes(paths[0], ssid);

    if (ssid < min_size + 2) continue;

    int dest_width = pred.cols;
    int dest_height = pred.rows;
    std::vector<std::array<int, 2>> intcliparray;

    for (int num_pt = 0; num_pt < 4; num_pt++) {
      std::array<int, 2> a{
          int(clampf(
              roundf(cliparray[num_pt][0] / float(width) * float(dest_width)),
              0, float(dest_width))),
          int(clampf(
              roundf(cliparray[num_pt][1] / float(height) * float(dest_height)),
              0, float(dest_height)))};
      intcliparray.push_back(a);
    }
    boxes.push_back(intcliparray);

  }  // end for
  return boxes;
}

std::vector<std::vector<std::array<int, 2>>> PostProcessor::FilterTagDetRes(
    std::vector<std::vector<std::array<int, 2>>> boxes,
    const std::array<int,4>& det_img_info) {
  int oriimg_w = det_img_info[0];
  int oriimg_h = det_img_info[1];
  float ratio_w = float(det_img_info[2])/float(oriimg_w);
  float ratio_h = float(det_img_info[3])/float(oriimg_h);

  for (int n = 0; n < boxes.size(); n++) {
    // boxes[n] = OrderPointsClockwise(boxes[n]);

    for (int m = 0; m < boxes[n].size(); m++) {
      boxes[n][m][0] /= ratio_w;
      boxes[n][m][1] /= ratio_h;

      boxes[n][m][0] = int(_min(_max(boxes[n][m][0], 0), oriimg_w - 1));
      boxes[n][m][1] = int(_min(_max(boxes[n][m][1], 0), oriimg_h - 1));
    }
  }

  return boxes;
}

}  // namespace ocr
}  // namespace vision
}  // namespace fastdeploy