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

/*
3D IoU Calculation and Rotated NMS(modified from 2D NMS written by others)
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
*/

#include <stdio.h>
#define THREADS_PER_BLOCK 16
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

const int THREADS_PER_BLOCK_NMS = sizeof(int64_t) * 8;
const float EPS = 1e-8;
struct Point {
  float x, y;
  __device__ Point() {}
  __device__ Point(double _x, double _y) { x = _x, y = _y; }

  __device__ void set(float _x, float _y) {
    x = _x;
    y = _y;
  }

  __device__ Point operator+(const Point &b) const {
    return Point(x + b.x, y + b.y);
  }

  __device__ Point operator-(const Point &b) const {
    return Point(x - b.x, y - b.y);
  }
};

__device__ inline float cross(const Point &a, const Point &b) {
  return a.x * b.y - a.y * b.x;
}

__device__ inline float cross(const Point &p1, const Point &p2,
                              const Point &p0) {
  return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
}

__device__ int check_rect_cross(const Point &p1, const Point &p2,
                                const Point &q1, const Point &q2) {
  int ret = min(p1.x, p2.x) <= max(q1.x, q2.x) &&
            min(q1.x, q2.x) <= max(p1.x, p2.x) &&
            min(p1.y, p2.y) <= max(q1.y, q2.y) &&
            min(q1.y, q2.y) <= max(p1.y, p2.y);
  return ret;
}

__device__ inline int check_in_box2d(const float *box, const Point &p) {
  // params: (7) [x, y, z, dx, dy, dz, heading]
  const float MARGIN = 1e-2;

  float center_x = box[0], center_y = box[1];
  // rotate the point in the opposite direction of box
  float angle_cos = cos(-box[6]), angle_sin = sin(-box[6]);
  float rot_x = (p.x - center_x) * angle_cos + (p.y - center_y) * (-angle_sin);
  float rot_y = (p.x - center_x) * angle_sin + (p.y - center_y) * angle_cos;

  return (fabs(rot_x) < box[3] / 2 + MARGIN &&
          fabs(rot_y) < box[4] / 2 + MARGIN);
}

__device__ inline int intersection(const Point &p1, const Point &p0,
                                   const Point &q1, const Point &q0,
                                   Point *ans) {
  // fast exclusion
  if (check_rect_cross(p0, p1, q0, q1) == 0) return 0;

  // check cross standing
  float s1 = cross(q0, p1, p0);
  float s2 = cross(p1, q1, p0);
  float s3 = cross(p0, q1, q0);
  float s4 = cross(q1, p1, q0);

  if (!(s1 * s2 > 0 && s3 * s4 > 0)) return 0;

  // calculate intersection of two lines
  float s5 = cross(q1, p1, p0);
  if (fabs(s5 - s1) > EPS) {
    ans->x = (s5 * q0.x - s1 * q1.x) / (s5 - s1);
    ans->y = (s5 * q0.y - s1 * q1.y) / (s5 - s1);

  } else {
    float a0 = p0.y - p1.y, b0 = p1.x - p0.x, c0 = p0.x * p1.y - p1.x * p0.y;
    float a1 = q0.y - q1.y, b1 = q1.x - q0.x, c1 = q0.x * q1.y - q1.x * q0.y;
    float D = a0 * b1 - a1 * b0;

    ans->x = (b0 * c1 - b1 * c0) / D;
    ans->y = (a1 * c0 - a0 * c1) / D;
  }

  return 1;
}

__device__ inline void rotate_around_center(const Point &center,
                                            const float angle_cos,
                                            const float angle_sin, Point *p) {
  float new_x = (p->x - center.x) * angle_cos +
                (p->y - center.y) * (-angle_sin) + center.x;
  float new_y =
      (p->x - center.x) * angle_sin + (p->y - center.y) * angle_cos + center.y;
  p->set(new_x, new_y);
}

__device__ inline int point_cmp(const Point &a, const Point &b,
                                const Point &center) {
  return atan2(a.y - center.y, a.x - center.x) >
         atan2(b.y - center.y, b.x - center.x);
}

__device__ inline float box_overlap(const float *box_a, const float *box_b) {
  // params box_a: [x, y, z, dx, dy, dz, heading]
  // params box_b: [x, y, z, dx, dy, dz, heading]

  float a_angle = box_a[6], b_angle = box_b[6];
  float a_dx_half = box_a[3] / 2, b_dx_half = box_b[3] / 2,
        a_dy_half = box_a[4] / 2, b_dy_half = box_b[4] / 2;
  float a_x1 = box_a[0] - a_dx_half, a_y1 = box_a[1] - a_dy_half;
  float a_x2 = box_a[0] + a_dx_half, a_y2 = box_a[1] + a_dy_half;
  float b_x1 = box_b[0] - b_dx_half, b_y1 = box_b[1] - b_dy_half;
  float b_x2 = box_b[0] + b_dx_half, b_y2 = box_b[1] + b_dy_half;

  Point center_a(box_a[0], box_a[1]);
  Point center_b(box_b[0], box_b[1]);

  Point box_a_corners[5];
  box_a_corners[0].set(a_x1, a_y1);
  box_a_corners[1].set(a_x2, a_y1);
  box_a_corners[2].set(a_x2, a_y2);
  box_a_corners[3].set(a_x1, a_y2);

  Point box_b_corners[5];
  box_b_corners[0].set(b_x1, b_y1);
  box_b_corners[1].set(b_x2, b_y1);
  box_b_corners[2].set(b_x2, b_y2);
  box_b_corners[3].set(b_x1, b_y2);

  // get oriented corners
  float a_angle_cos = cos(a_angle), a_angle_sin = sin(a_angle);
  float b_angle_cos = cos(b_angle), b_angle_sin = sin(b_angle);

  for (int k = 0; k < 4; k++) {
    rotate_around_center(center_a, a_angle_cos, a_angle_sin, box_a_corners + k);
    rotate_around_center(center_b, b_angle_cos, b_angle_sin, box_b_corners + k);
  }

  box_a_corners[4] = box_a_corners[0];
  box_b_corners[4] = box_b_corners[0];

  // get intersection of lines
  Point cross_points[16];
  Point poly_center;
  int cnt = 0, flag = 0;

  poly_center.set(0, 0);
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      flag = intersection(box_a_corners[i + 1], box_a_corners[i],
                          box_b_corners[j + 1], box_b_corners[j],
                          cross_points + cnt);
      if (flag) {
        poly_center = poly_center + cross_points[cnt];
        cnt++;
      }
    }
  }

  // check corners
  for (int k = 0; k < 4; k++) {
    if (check_in_box2d(box_a, box_b_corners[k])) {
      poly_center = poly_center + box_b_corners[k];
      cross_points[cnt] = box_b_corners[k];
      cnt++;
    }
    if (check_in_box2d(box_b, box_a_corners[k])) {
      poly_center = poly_center + box_a_corners[k];
      cross_points[cnt] = box_a_corners[k];
      cnt++;
    }
  }

  poly_center.x /= cnt;
  poly_center.y /= cnt;

  // sort the points of polygon
  Point temp;
  for (int j = 0; j < cnt - 1; j++) {
    for (int i = 0; i < cnt - j - 1; i++) {
      if (point_cmp(cross_points[i], cross_points[i + 1], poly_center)) {
        temp = cross_points[i];
        cross_points[i] = cross_points[i + 1];
        cross_points[i + 1] = temp;
      }
    }
  }

  // get the overlap areas
  float area = 0;
  for (int k = 0; k < cnt - 1; k++) {
    area += cross(cross_points[k] - cross_points[0],
                  cross_points[k + 1] - cross_points[0]);
  }

  return fabs(area) / 2.0;
}

__device__ inline float iou_bev(const float *box_a, const float *box_b) {
  // params box_a: [x, y, z, dx, dy, dz, heading]
  // params box_b: [x, y, z, dx, dy, dz, heading]
  float sa = box_a[3] * box_a[4];
  float sb = box_b[3] * box_b[4];
  float s_overlap = box_overlap(box_a, box_b);
  return s_overlap / fmaxf(sa + sb - s_overlap, EPS);
}

__global__ void nms_kernel(const int num_bboxes, const int num_bboxes_for_nms,
                           const float nms_overlap_thresh,
                           const int decode_bboxes_dims, const float *bboxes,
                           const int *index, const int64_t *sorted_index,
                           int64_t *mask) {
  // params: boxes (N, 7) [x, y, z, dx, dy, dz, heading]
  // params: mask (N, N/THREADS_PER_BLOCK_NMS)

  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  // if (row_start > col_start) return;

  const int row_size =
      fminf(num_bboxes_for_nms - row_start * THREADS_PER_BLOCK_NMS,
            THREADS_PER_BLOCK_NMS);
  const int col_size =
      fminf(num_bboxes_for_nms - col_start * THREADS_PER_BLOCK_NMS,
            THREADS_PER_BLOCK_NMS);

  __shared__ float block_boxes[THREADS_PER_BLOCK_NMS * 7];

  if (threadIdx.x < col_size) {
    int box_idx =
        index[sorted_index[THREADS_PER_BLOCK_NMS * col_start + threadIdx.x]];
    block_boxes[threadIdx.x * 7 + 0] = bboxes[box_idx * decode_bboxes_dims];
    block_boxes[threadIdx.x * 7 + 1] = bboxes[box_idx * decode_bboxes_dims + 1];
    block_boxes[threadIdx.x * 7 + 2] = bboxes[box_idx * decode_bboxes_dims + 2];
    block_boxes[threadIdx.x * 7 + 3] = bboxes[box_idx * decode_bboxes_dims + 4];
    block_boxes[threadIdx.x * 7 + 4] = bboxes[box_idx * decode_bboxes_dims + 3];
    block_boxes[threadIdx.x * 7 + 5] = bboxes[box_idx * decode_bboxes_dims + 5];
    block_boxes[threadIdx.x * 7 + 6] =
        -bboxes[box_idx * decode_bboxes_dims + decode_bboxes_dims - 1] -
        3.141592653589793 / 2;
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = THREADS_PER_BLOCK_NMS * row_start + threadIdx.x;
    const int act_box_idx = index[sorted_index[cur_box_idx]];
    float cur_box[7];
    cur_box[0] = bboxes[act_box_idx * decode_bboxes_dims];
    cur_box[1] = bboxes[act_box_idx * decode_bboxes_dims + 1];
    cur_box[2] = bboxes[act_box_idx * decode_bboxes_dims + 2];
    cur_box[3] = bboxes[act_box_idx * decode_bboxes_dims + 4];
    cur_box[4] = bboxes[act_box_idx * decode_bboxes_dims + 3];
    cur_box[5] = bboxes[act_box_idx * decode_bboxes_dims + 5];
    cur_box[6] =
        -bboxes[act_box_idx * decode_bboxes_dims + decode_bboxes_dims - 1] -
        3.141592653589793 / 2;

    int i = 0;
    int64_t t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (iou_bev(cur_box, block_boxes + i * 7) > nms_overlap_thresh) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = DIVUP(num_bboxes_for_nms, THREADS_PER_BLOCK_NMS);
    mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

void NmsLauncher(const cudaStream_t &stream, const float *bboxes,
                 const int *index, const int64_t *sorted_index,
                 const int num_bboxes, const int num_bboxes_for_nms,
                 const float nms_overlap_thresh, const int decode_bboxes_dims,
                 int64_t *mask) {
  dim3 blocks(DIVUP(num_bboxes_for_nms, THREADS_PER_BLOCK_NMS),
              DIVUP(num_bboxes_for_nms, THREADS_PER_BLOCK_NMS));
  dim3 threads(THREADS_PER_BLOCK_NMS);
  nms_kernel<<<blocks, threads, 0, stream>>>(
      num_bboxes, num_bboxes_for_nms, nms_overlap_thresh, decode_bboxes_dims,
      bboxes, index, sorted_index, mask);
}
