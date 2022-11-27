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

#ifdef ENABLE_VISION_VISUALIZE

#include "fastdeploy/vision/visualize/visualize.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace fastdeploy {
namespace vision {

#ifdef __ARM_NEON  
static inline void QuantizeBlendingWeight8(
  float weight, uint8_t* old_multi_factor, uint8_t* new_multi_factor) {
  // Quantize the weight to boost blending performance.
  // if 0.0 < w <= 1/8, w ~ 1/8=1/(2^3) shift right 3 mul 1, 7
  // if 1/8 < w <= 2/8, w ~ 2/8=1/(2^3) shift right 3 mul 2, 6
  // if 2/8 < w <= 3/8, w ~ 3/8=1/(2^3) shift right 3 mul 3, 5
  // if 3/8 < w <= 4/8, w ~ 4/8=1/(2^3) shift right 3 mul 4, 4
  // Shift factor is always 3, but the mul factor is different.
  // Moving 7 bits to the right tends to result in a zero value,
  // So, We choose to shift 3 bits to get an approximation.
  uint8_t weight_quantize = static_cast<uint8_t>(weight * 8.0f);
  *new_multi_factor = weight_quantize;
  *old_multi_factor = (8 - weight_quantize);
}

static cv::Mat FastVisSegmentationNEON(
  const cv::Mat& im, const SegmentationResult& result,
  float weight, bool quantize_weight = true) {
  int64_t height = result.shape[0];
  int64_t width = result.shape[1];
  auto vis_img = cv::Mat(height, width, CV_8UC3);
  
  int32_t size = static_cast<int32_t>(height * width);
  uint8_t *vis_ptr = static_cast<uint8_t*>(vis_img.data);
  const uint8_t *label_ptr = static_cast<const uint8_t*>(result.label_map.data());
  const uint8_t *im_ptr = static_cast<const uint8_t*>(im.data);

  if (!quantize_weight) {
    // int32_t i = 0;
    #pragma omp parallel for num_threads(2)
    for (int i = 0; i < size - 15; i += 16) {
      uint8x16_t labelx16 = vld1q_u8(label_ptr + i); // 16 bytes
      // e.g 0b00000001 << 7 -> 0b10000000 128;
      uint8x16x3_t vbgrx16x3;
      vbgrx16x3.val[0] = vshlq_n_u8(labelx16, 7); 
      vbgrx16x3.val[1] = vshlq_n_u8(labelx16, 4); 
      vbgrx16x3.val[2] = vshlq_n_u8(labelx16, 3); 
      vst3q_u8(vis_ptr + i * 3, vbgrx16x3);
    }
    for (int i = size - 15; i < size; i++) {
      uint8_t label = label_ptr[i];
      vis_ptr[i * 3 + 0] = (label << 7); 
      vis_ptr[i * 3 + 1] = (label << 4); 
      vis_ptr[i * 3 + 2] = (label << 3); 
    }
    // Blend colors use opencv
    cv::addWeighted(im, 1.0 - weight, vis_img, weight, 0, vis_img);
    return vis_img;
  }
  
  // Quantize the weight to boost blending performance.
  // After that, we can directly use shift instructions
  // to blend the colors from input im and mask. Please 
  // check QuantizeBlendingWeight8 for more details.
  uint8_t old_multi_factor, new_multi_factor;
  QuantizeBlendingWeight8(weight, &old_multi_factor,
                          &new_multi_factor);     
  if (new_multi_factor == 0) {
    return im; // Only keep origin image.
  }                                            
  
  if (new_multi_factor == 8) {
    // Only keep mask, no need to blending with origin image.
    #pragma omp parallel for num_threads(2)
    for (int i = 0; i < size - 15; i += 16) {
      uint8x16_t labelx16 = vld1q_u8(label_ptr + i); // 16 bytes
      // e.g 0b00000001 << 7 -> 0b10000000 128;
      uint8x16_t mbx16 = vshlq_n_u8(labelx16, 7); 
      uint8x16_t mgx16 = vshlq_n_u8(labelx16, 4); 
      uint8x16_t mrx16 = vshlq_n_u8(labelx16, 3); 
      uint8x16x3_t vbgr16x3;
      vbgr16x3.val[0] = mbx16;
      vbgr16x3.val[1] = mgx16;
      vbgr16x3.val[2] = mrx16;
      vst3q_u8(vis_ptr + i * 3, vbgr16x3);
    }
    for (int i = size - 15; i < size; i++) {
      uint8_t label = label_ptr[i];
      vis_ptr[i * 3 + 0] = (label << 7); 
      vis_ptr[i * 3 + 1] = (label << 4); 
      vis_ptr[i * 3 + 2] = (label << 3); 
    }  
    return vis_img;
  }
  
  // int32_t i = 0;
  uint8x16_t old_mulx16 = vdupq_n_u8(old_multi_factor);
  uint8x16_t new_mulx16 = vdupq_n_u8(new_multi_factor);
  // Blend the two colors together with quantize 'weight'.
  #pragma omp parallel for num_threads(2)
  for (int i = 0; i < size - 15; i += 16) {
    uint8x16x3_t bgrx16x3 = vld3q_u8(im_ptr + i * 3);  // 48 bytes
    uint8x16_t labelx16 = vld1q_u8(label_ptr + i); // 16 bytes
    uint8x16_t ibx16 = bgrx16x3.val[0];
    uint8x16_t igx16 = bgrx16x3.val[1];
    uint8x16_t irx16 = bgrx16x3.val[2];
    // e.g 0b00000001 << 7 -> 0b10000000 128;
    uint8x16_t mbx16 = vshlq_n_u8(labelx16, 7); 
    uint8x16_t mgx16 = vshlq_n_u8(labelx16, 4); 
    uint8x16_t mrx16 = vshlq_n_u8(labelx16, 3); 
    // TODO: keep the pixels of input im if mask = 0
    uint8x16_t ibx16_mshr, igx16_mshr, irx16_mshr;
    uint8x16_t mbx16_mshr, mgx16_mshr, mrx16_mshr;
    // Moving 7 bits to the right tends to result in zero,
    // So, We choose to shift 3 bits to get an approximation 
    ibx16_mshr = vmulq_u8(vshrq_n_u8(ibx16, 3), old_mulx16);
    igx16_mshr = vmulq_u8(vshrq_n_u8(igx16, 3), old_mulx16);   
    irx16_mshr = vmulq_u8(vshrq_n_u8(irx16, 3), old_mulx16);
    mbx16_mshr = vmulq_u8(vshrq_n_u8(mbx16, 3), new_mulx16);
    mgx16_mshr = vmulq_u8(vshrq_n_u8(mgx16, 3), new_mulx16);
    mrx16_mshr = vmulq_u8(vshrq_n_u8(mrx16, 3), new_mulx16);  
    uint8x16x3_t vbgr16x3;
    vbgr16x3.val[0] = vaddq_u8(ibx16_mshr, mbx16_mshr);
    vbgr16x3.val[1] = vaddq_u8(igx16_mshr, mgx16_mshr);
    vbgr16x3.val[2] = vaddq_u8(irx16_mshr, mrx16_mshr);
    // Store the blended pixels to vis img
    vst3q_u8(vis_ptr + i * 3, vbgr16x3);
  }
  for (int i = size - 15; i < size; i++) {
    uint8_t label = label_ptr[i];
    vis_ptr[i * 3 + 0] = (im_ptr[i * 3 + 0] >> 3) * old_multi_factor 
      + ((label << 7) >> 3) * new_multi_factor; 
    vis_ptr[i * 3 + 1] = (im_ptr[i * 3 + 1] >> 3) * old_multi_factor 
      + ((label << 4) >> 3) * new_multi_factor; 
    vis_ptr[i * 3 + 2] = (im_ptr[i * 3 + 2] >> 3) * old_multi_factor 
      + ((label << 3) >> 3) * new_multi_factor;   
  }  
  return vis_img;
}
#endif

static cv::Mat VisSegmentationCommonCpu(
  const cv::Mat& im, const SegmentationResult& result,
  float weight) {
  // Use the native c++ version without any optimization.
  auto color_map = GenerateColorMap(1000);
  int64_t height = result.shape[0];
  int64_t width = result.shape[1];
  auto vis_img = cv::Mat(height, width, CV_8UC3);

  int64_t index = 0;
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      int category_id = result.label_map[index++];
      vis_img.at<cv::Vec3b>(i, j)[0] = color_map[3 * category_id + 0];
      vis_img.at<cv::Vec3b>(i, j)[1] = color_map[3 * category_id + 1];
      vis_img.at<cv::Vec3b>(i, j)[2] = color_map[3 * category_id + 2];
    }
  }
  cv::addWeighted(im, 1.0 - weight, vis_img, weight, 0, vis_img);
  return vis_img;
}

cv::Mat VisSegmentation(const cv::Mat& im, const SegmentationResult& result,
                        float weight) {
  // TODO: Support SSE/AVX on x86_64 platforms                        
#ifdef __ARM_NEON 
  return FastVisSegmentationNEON(im, result, weight, true);
#else  
  return VisSegmentationCommonCpu(im, result, weight);
#endif  
}

cv::Mat Visualize::VisSegmentation(const cv::Mat& im,
                                   const SegmentationResult& result) {
  FDWARNING << "DEPRECATED: fastdeploy::vision::Visualize::VisSegmentation is "
               "deprecated, please use fastdeploy::vision:VisSegmentation "
               "function instead."
            << std::endl;     
#ifdef __ARM_NEON 
  return FastVisSegmentationNEON(im, result, 0.5f, true);
#else  
  return VisSegmentationCommonCpu(im, result, 0.5f);
#endif  
}

}  // namespace vision
}  // namespace fastdeploy
#endif
