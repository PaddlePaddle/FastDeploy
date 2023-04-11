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

#include "fastdeploy/vision/detection/ppdet/multiclass_nms_rotated.h"
#include "fastdeploy/vision/detection/ppdet/multiclass_nms.h"

#include <algorithm>

#include "fastdeploy/core/fd_tensor.h"
#include "fastdeploy/utils/utils.h"
#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>


namespace fastdeploy {
namespace vision {
namespace detection {

template <typename T> struct RotatedBox { T x_ctr, y_ctr, w, h, a; };

template <typename T> struct Point {
  T x, y;
  Point(const T &px = 0, const T &py = 0) : x(px), y(py) {}
  Point operator+(const Point &p) const {
    return Point(x + p.x, y + p.y);
  }
  Point &operator+=(const Point &p) {
    x += p.x;
    y += p.y;
    return *this;
  }
  Point operator-(const Point &p) const {
    return Point(x - p.x, y - p.y);
  }
  Point operator*(const T coeff) const {
    return Point(x * coeff, y * coeff);
  }
};

template <typename T>
T dot_2d(const Point<T> &A, const Point<T> &B) {
  return A.x * B.x + A.y * B.y;
}

template <typename T>
T cross_2d(const Point<T> &A, const Point<T> &B) {
  return A.x * B.y - B.x * A.y;
}

template <typename T>
void get_rotated_vertices(const RotatedBox<T> &box,
                                             Point<T> (&pts)[4]) {
  // M_PI / 180. == 0.01745329251
  // double theta = box.a * 0.01745329251;
  // MODIFIED
  double theta = box.a;
  T cosTheta2 = (T)cos(theta) * 0.5f;
  T sinTheta2 = (T)sin(theta) * 0.5f;

  // y: top --> down; x: left --> right
  pts[0].x = box.x_ctr - sinTheta2 * box.h - cosTheta2 * box.w;
  pts[0].y = box.y_ctr + cosTheta2 * box.h - sinTheta2 * box.w;
  pts[1].x = box.x_ctr + sinTheta2 * box.h - cosTheta2 * box.w;
  pts[1].y = box.y_ctr - cosTheta2 * box.h - sinTheta2 * box.w;
  pts[2].x = 2 * box.x_ctr - pts[0].x;
  pts[2].y = 2 * box.y_ctr - pts[0].y;
  pts[3].x = 2 * box.x_ctr - pts[1].x;
  pts[3].y = 2 * box.y_ctr - pts[1].y;
}

template <typename T>
int get_intersection_points(const Point<T> (&pts1)[4],
                                               const Point<T> (&pts2)[4],
                                               Point<T> (&intersections)[24]) {
  // Line vector
  // A line from p1 to p2 is: p1 + (p2-p1)*t, t=[0,1]
  Point<T> vec1[4], vec2[4];
  for (int i = 0; i < 4; i++) {
    vec1[i] = pts1[(i + 1) % 4] - pts1[i];
    vec2[i] = pts2[(i + 1) % 4] - pts2[i];
  }

  // Line test - test all line combos for intersection
  int num = 0; // number of intersections
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      // Solve for 2x2 Ax=b
      T det = cross_2d<T>(vec2[j], vec1[i]);

      // This takes care of parallel lines
      if (fabs(det) <= 1e-14) {
        continue;
      }

      auto vec12 = pts2[j] - pts1[i];

      T t1 = cross_2d<T>(vec2[j], vec12) / det;
      T t2 = cross_2d<T>(vec1[i], vec12) / det;

      if (t1 >= 0.0f && t1 <= 1.0f && t2 >= 0.0f && t2 <= 1.0f) {
        intersections[num++] = pts1[i] + vec1[i] * t1;
      }
    }
  }

  // Check for vertices of rect1 inside rect2
  {
    const auto &AB = vec2[0];
    const auto &DA = vec2[3];
    auto ABdotAB = dot_2d<T>(AB, AB);
    auto ADdotAD = dot_2d<T>(DA, DA);
    for (int i = 0; i < 4; i++) {
      // assume ABCD is the rectangle, and P is the point to be judged
      // P is inside ABCD iff. P's projection on AB lies within AB
      // and P's projection on AD lies within AD

      auto AP = pts1[i] - pts2[0];

      auto APdotAB = dot_2d<T>(AP, AB);
      auto APdotAD = -dot_2d<T>(AP, DA);

      if ((APdotAB >= 0) && (APdotAD >= 0) && (APdotAB <= ABdotAB) &&
          (APdotAD <= ADdotAD)) {
        intersections[num++] = pts1[i];
      }
    }
  }

  // Reverse the check - check for vertices of rect2 inside rect1
  {
    const auto &AB = vec1[0];
    const auto &DA = vec1[3];
    auto ABdotAB = dot_2d<T>(AB, AB);
    auto ADdotAD = dot_2d<T>(DA, DA);
    for (int i = 0; i < 4; i++) {
      auto AP = pts2[i] - pts1[0];

      auto APdotAB = dot_2d<T>(AP, AB);
      auto APdotAD = -dot_2d<T>(AP, DA);

      if ((APdotAB >= 0) && (APdotAD >= 0) && (APdotAB <= ABdotAB) &&
          (APdotAD <= ADdotAD)) {
        intersections[num++] = pts2[i];
      }
    }
  }

  return num;
}

template <typename T>
int convex_hull_graham(const Point<T> (&p)[24],
                                          const int &num_in, Point<T> (&q)[24],
                                          bool shift_to_zero = false) {
  assert(num_in >= 2);

  // Step 1:
  // Find point with minimum y
  // if more than 1 points have the same minimum y,
  // pick the one with the minimum x.
  int t = 0;
  for (int i = 1; i < num_in; i++) {
    if (p[i].y < p[t].y || (p[i].y == p[t].y && p[i].x < p[t].x)) {
      t = i;
    }
  }
  auto &start = p[t]; // starting point

  // Step 2:
  // Subtract starting point from every points (for sorting in the next step)
  for (int i = 0; i < num_in; i++) {
    q[i] = p[i] - start;
  }

  // Swap the starting point to position 0
  auto tmp = q[0];
  q[0] = q[t];
  q[t] = tmp;

  // Step 3:
  // Sort point 1 ~ num_in according to their relative cross-product values
  // (essentially sorting according to angles)
  // If the angles are the same, sort according to their distance to origin
  T dist[24];
  for (int i = 0; i < num_in; i++) {
    dist[i] = dot_2d<T>(q[i], q[i]);
  }

  // CPU version
  std::sort(q + 1, q + num_in,
            [](const Point<T> &A, const Point<T> &B) -> bool {
              T temp = cross_2d<T>(A, B);
              if (fabs(temp) < 1e-6) {
                return dot_2d<T>(A, A) < dot_2d<T>(B, B);
              } else {
                return temp > 0;
              }
            });

  // Step 4:
  // Make sure there are at least 2 points (that don't overlap with each other)
  // in the stack
  int k; // index of the non-overlapped second point
  for (k = 1; k < num_in; k++) {
    if (dist[k] > 1e-8) {
      break;
    }
  }
  if (k == num_in) {
    // We reach the end, which means the convex hull is just one point
    q[0] = p[t];
    return 1;
  }
  q[1] = q[k];
  int m = 2; // 2 points in the stack
  // Step 5:
  // Finally we can start the scanning process.
  // When a non-convex relationship between the 3 points is found
  // (either concave shape or duplicated points),
  // we pop the previous point from the stack
  // until the 3-point relationship is convex again, or
  // until the stack only contains two points
  for (int i = k + 1; i < num_in; i++) {
    while (m > 1 && cross_2d<T>(q[i] - q[m - 2], q[m - 1] - q[m - 2]) >= 0) {
      m--;
    }
    q[m++] = q[i];
  }

  // Step 6 (Optional):
  // In general sense we need the original coordinates, so we
  // need to shift the points back (reverting Step 2)
  // But if we're only interested in getting the area/perimeter of the shape
  // We can simply return.
  if (!shift_to_zero) {
    for (int i = 0; i < m; i++) {
      q[i] += start;
    }
  }

  return m;
}

template <typename T>
T polygon_area(const Point<T> (&q)[24], const int &m) {
  if (m <= 2) {
    return 0;
  }

  T area = 0;
  for (int i = 1; i < m - 1; i++) {
    area += fabs(cross_2d<T>(q[i] - q[0], q[i + 1] - q[0]));
  }

  return area / 2.0;
}

template <typename T>
T rboxes_intersection(const RotatedBox<T> &box1,
                                         const RotatedBox<T> &box2) {
  // There are up to 4 x 4 + 4 + 4 = 24 intersections (including dups) returned
  // from rotated_rect_intersection_pts
  Point<T> intersectPts[24], orderedPts[24];

  Point<T> pts1[4];
  Point<T> pts2[4];
  get_rotated_vertices<T>(box1, pts1);
  get_rotated_vertices<T>(box2, pts2);

  int num = get_intersection_points<T>(pts1, pts2, intersectPts);

  if (num <= 2) {
    return 0.0;
  }

  // Convex Hull to order the intersection points in clockwise order and find
  // the contour area.
  int num_convex = convex_hull_graham<T>(intersectPts, num, orderedPts, true);
  return polygon_area<T>(orderedPts, num_convex);
}

template <typename T>
void Poly2Rbox(T const *const poly_raw, RotatedBox<T>& box){
  std::vector<cv::Point2f> contour_poly {
    cv::Point2f(poly_raw[0], poly_raw[1]),
    cv::Point2f(poly_raw[2], poly_raw[3]),
    cv::Point2f(poly_raw[4], poly_raw[5]),
    cv::Point2f(poly_raw[6], poly_raw[7]),
  };
  cv::RotatedRect rotate_rect = cv::minAreaRect(contour_poly);
  box.x_ctr = rotate_rect.center.x;
  box.y_ctr = rotate_rect.center.y;
  box.w = rotate_rect.size.width;
  box.h = rotate_rect.size.height;
  box.a = rotate_rect.angle;
}

template <typename T>
T rbox_iou_single(T const *const poly1_raw, T const *const poly2_raw) {
  // shift center to the middle point to achieve higher precision in result
  RotatedBox<T> box1, box2;
  Poly2Rbox<T>(poly1_raw, box1);
  Poly2Rbox<T>(poly2_raw, box2);
//  printf("poly1_raw: %f,%f,%f,%f,%f,%f,%f,%f\n",
//    poly1_raw[0],poly1_raw[1],poly1_raw[2],poly1_raw[3],poly1_raw[4],poly1_raw[5],poly1_raw[6],poly1_raw[7]);
//  printf("poly2_raw: %f,%f,%f,%f,%f,%f,%f,%f\n",
//    poly2_raw[0],poly2_raw[1],poly2_raw[2],poly2_raw[3],poly2_raw[4],poly2_raw[5],poly2_raw[6],poly2_raw[7]);
//
//    printf("before box1: %f,%f,%f,%f,%f\n",
//        box1.x_ctr,box1.y_ctr, box1.w, box1.h, box1.a);
//    printf("before box2: %f,%f,%f,%f,%f\n",
//        box2.x_ctr,box2.y_ctr, box2.w, box2.h, box2.a);

  auto center_shift_x = (box1.x_ctr + box2.x_ctr) / 2.0;
  auto center_shift_y = (box1.y_ctr + box2.y_ctr) / 2.0;
  box1.x_ctr -= center_shift_x;
  box1.y_ctr -= center_shift_y;
  box2.x_ctr -= center_shift_x;
  box2.y_ctr -= center_shift_y;

//      printf("after box1: %f,%f,%f,%f,%f\n",
//        box1.x_ctr,box1.y_ctr, box1.w, box1.h, box1.a);
//    printf("after box2: %f,%f,%f,%f,%f\n",
//        box2.x_ctr,box2.y_ctr, box2.w, box2.h, box2.a);

  if (box1.w < 1e-2 || box1.h < 1e-2 || box2.w < 1e-2 || box2.h < 1e-2) {
    return 0.f;
  }
  const T area1 = box1.w * box1.h;
  const T area2 = box2.w * box2.h;

  const T intersection = rboxes_intersection<T>(box1, box2);
  const T iou = intersection / (area1 + area2 - intersection);
  return iou;
}

template <typename T>
bool SortScorePairDescendRotated(const std::pair<float, T>& pair1,
                          const std::pair<float, T>& pair2) {
  return pair1.first > pair2.first;
}

void GetMaxScoreIndexRotated(const float* scores, const int& score_size,
                      const float& threshold, const int& top_k,
                      std::vector<std::pair<float, int>>* sorted_indices) {
                      printf("score size: %d score thrd: %f, top_k: %d\n", score_size, threshold, top_k);
  for (size_t i = 0; i < score_size; ++i) {
    if (scores[i] > threshold) {
      sorted_indices->push_back(std::make_pair(scores[i], i));
    }
  }
  // Sort the score pair according to the scores in descending order
  std::stable_sort(sorted_indices->begin(), sorted_indices->end(),
                   SortScorePairDescendRotated<int>);
  // Keep top_k scores if needed.
  if (top_k > -1 && top_k < static_cast<int>(sorted_indices->size())) {
    sorted_indices->resize(top_k);
  }
}

void PaddleMultiClassNMSRotated::FastNMSRotated(const float* boxes, const float* scores,
                            const int& num_boxes,
                            std::vector<int>* keep_indices) {
  std::vector<std::pair<float, int>> sorted_indices;
  GetMaxScoreIndexRotated(scores, num_boxes, score_threshold, nms_top_k,
                   &sorted_indices);
   printf("nms thrd: %f, sort dim: %d\n", nms_threshold, int(sorted_indices.size()));
  float adaptive_threshold = nms_threshold;
  while (sorted_indices.size() != 0) {
    const int idx = sorted_indices.front().second;
    bool keep = true;
    for (size_t k = 0; k < keep_indices->size(); ++k) {
      if (!keep) {
        break;
      }
      const int kept_idx = (*keep_indices)[k];
      float overlap =
          rbox_iou_single<float>(boxes + idx * 8, boxes + kept_idx * 8);

//      printf("overlap check: %f vs %f， keep num: %d\n", overlap, adaptive_threshold, int(keep_indices->size()));
      keep = overlap <= adaptive_threshold;

//      if (!keep)
//        printf("overlap check: %f vs %f， keep idx: %d\n", overlap, adaptive_threshold, kept_idx);
    }
    if (keep) {
      keep_indices->push_back(idx);
    }
    sorted_indices.erase(sorted_indices.begin());
    if (keep && nms_eta<1.0 & adaptive_threshold> 0.5) {
      adaptive_threshold *= nms_eta;
    }
  }
}

int PaddleMultiClassNMSRotated::NMSRotatedForEachSample(
    const float* boxes, const float* scores, int num_boxes, int num_classes,
    std::map<int, std::vector<int>>* keep_indices) {

  for (int i = 0; i < num_classes; ++i) {
    if (i == background_label) {
      continue;
    }
    const float* score_for_class_i = scores + i * num_boxes;
    printf("--->cls id: %d, numbox: %d", i, num_boxes);
    FastNMSRotated(boxes, score_for_class_i, num_boxes, &((*keep_indices)[i]));
  }
  int num_det = 0;
  for (auto iter = keep_indices->begin(); iter != keep_indices->end(); ++iter) {
    num_det += iter->second.size();
    printf("keep id: %d, \n", iter->first);
    for (auto &kid: iter->second)
        printf("%d ", kid);
    printf("\n");
  }


  printf("num det: %d, keep_top_k: %d， num_classes: %d\n", num_det, keep_top_k, num_classes);

  if (keep_top_k > -1 && num_det > keep_top_k) {
    std::vector<std::pair<float, std::pair<int, int>>> score_index_pairs;
    for (const auto& it : *keep_indices) {
      int label = it.first;
      const float* current_score = scores + label * num_boxes;
      auto& label_indices = it.second;
      for (size_t j = 0; j < label_indices.size(); ++j) {
        int idx = label_indices[j];
        score_index_pairs.push_back(
            std::make_pair(current_score[idx], std::make_pair(label, idx)));
      }
    }

    std::stable_sort(score_index_pairs.begin(), score_index_pairs.end(),
                     SortScorePairDescendRotated<std::pair<int, int>>);
    score_index_pairs.resize(keep_top_k);

    std::map<int, std::vector<int>> new_indices;
    for (size_t j = 0; j < score_index_pairs.size(); ++j) {
      int label = score_index_pairs[j].second.first;
      int idx = score_index_pairs[j].second.second;
      new_indices[label].push_back(idx);
    }
    new_indices.swap(*keep_indices);
    num_det = keep_top_k;
  }
  return num_det;
}

void PaddleMultiClassNMSRotated::Compute(const float* boxes_data,
                                  const float* scores_data,
                                  const std::vector<int64_t>& boxes_dim,
                                  const std::vector<int64_t>& scores_dim) {

  int score_size = scores_dim.size();

  int64_t batch_size = scores_dim[0];
  int64_t box_dim = boxes_dim[2];
  int64_t out_dim = box_dim + 2;
  printf("---> %d %d %d\n", boxes_dim[0], boxes_dim[1], boxes_dim[2]);

  int num_nmsed_out = 0;
  FDASSERT(score_size == 3,
           "Require rank of input scores be 3, but now it's %d.", score_size);
  FDASSERT(boxes_dim[2] == 8,
           "Require the 3-dimension of input boxes be 8, but now it's %lld.",
           box_dim);
  out_num_rois_data.resize(batch_size);

  std::vector<std::map<int, std::vector<int>>> all_indices;
  for (size_t i = 0; i < batch_size; ++i) {
    std::map<int, std::vector<int>> indices;  // indices kept for each class
    const float* current_boxes_ptr =
        boxes_data + i * boxes_dim[1] * boxes_dim[2];
    const float* current_scores_ptr =
        scores_data + i * scores_dim[1] * scores_dim[2];
    int num = NMSRotatedForEachSample(current_boxes_ptr, current_scores_ptr,
                               boxes_dim[1], scores_dim[1], &indices);
                               printf("num: %d\n", num);
    num_nmsed_out += num;
    out_num_rois_data[i] = num;
    all_indices.emplace_back(indices);
  }
  std::vector<int64_t> out_box_dims = {num_nmsed_out, 10};
  std::vector<int64_t> out_index_dims = {num_nmsed_out, 1};
  if (num_nmsed_out == 0) {
    for (size_t i = 0; i < batch_size; ++i) {
      out_num_rois_data[i] = 0;
    }
    return;
  }
  out_box_data.resize(num_nmsed_out * 10);
  out_index_data.resize(num_nmsed_out);

  int count = 0;
  for (size_t i = 0; i < batch_size; ++i) {
    const float* current_boxes_ptr =
        boxes_data + i * boxes_dim[1] * boxes_dim[2];
    const float* current_scores_ptr =
        scores_data + i * scores_dim[1] * scores_dim[2];
    for (const auto& it : all_indices[i]) {
      int label = it.first;
      const auto& indices = it.second;
      const float* current_scores_class_ptr =
          current_scores_ptr + label * scores_dim[2];
      for (size_t j = 0; j < indices.size(); ++j) {
        int start = count * 10;
        out_box_data[start] = label;
        out_box_data[start + 1] = current_scores_class_ptr[indices[j]];
        for (int k = 0; k <  8; k++) {
            out_box_data[start + 2 + k] = current_boxes_ptr[indices[j] * 8 + k];
        }
        out_index_data[count] = i * boxes_dim[1] + indices[j];
        count += 1;
      }
    }
  }
}
}  // namespace detection
}  // namespace vision
}  // namespace fastdeploy
