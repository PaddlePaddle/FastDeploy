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

#include "fastdeploy/fastdeploy_model.h"
#include "fastdeploy/vision/common/processors/transform.h"
#include "fastdeploy/vision/common/result.h"

namespace fastdeploy {

namespace vision {

namespace meituan {

class FASTDEPLOY_DECL YOLOv6 : public FastDeployModel {
 public:
  // 当model_format为ONNX时，无需指定params_file
  // 当model_format为Paddle时，则需同时指定model_file & params_file
  YOLOv6(const std::string& model_file, const std::string& params_file = "",
         const RuntimeOption& custom_option = RuntimeOption(),
         const Frontend& model_format = Frontend::ONNX);

  // 定义模型的名称
  virtual std::string ModelName() const { return "meituan/YOLOv6"; }

  // 初始化函数，包括初始化后端，以及其它模型推理需要涉及的操作
  virtual bool Initialize();

  // 输入图像预处理操作
  // Mat为FastDeploy定义的数据结构
  // FDTensor为预处理后的Tensor数据，传给后端进行推理
  // im_info为预处理过程保存的数据，在后处理中需要用到
  virtual bool Preprocess(Mat* mat, FDTensor* outputs,
                          std::map<std::string, std::array<float, 2>>* im_info);

  // 后端推理结果后处理，输出给用户
  // infer_result 为后端推理后的输出Tensor
  // result 为模型预测的结果
  // im_info 为预处理记录的信息，后处理用于还原box
  // conf_threshold 后处理时过滤box的置信度阈值
  // nms_iou_threshold 后处理时NMS设定的iou阈值
  virtual bool Postprocess(
      FDTensor& infer_result, DetectionResult* result,
      const std::map<std::string, std::array<float, 2>>& im_info,
      float conf_threshold, float nms_iou_threshold);

  // 模型预测接口，即用户调用的接口
  // im 为用户的输入数据，目前对于CV均定义为cv::Mat
  // result 为模型预测的输出结构体
  // conf_threshold 为后处理的参数
  // nms_iou_threshold 为后处理的参数
  virtual bool Predict(cv::Mat* im, DetectionResult* result,
                       float conf_threshold = 0.25,
                       float nms_iou_threshold = 0.5);

  // 用户可以通过该接口 查看输入的模型是否为动态维度
  virtual bool IsDynamicShape() const { return is_dynamic_shape_; }                     

  // 以下为模型在预测时的一些参数，基本是前后处理所需
  // 用户在创建模型后，可根据模型的要求，以及自己的需求
  // 对参数进行修改
  // tuple of (width, height)
  std::vector<int> size;
  // padding value, size should be same with Channels
  std::vector<float> padding_value;
  // only pad to the minimum rectange which height and width is times of stride
  bool is_mini_pad;
  // while is_mini_pad = false and is_no_pad = true, will resize the image to
  // the set size
  bool is_no_pad;
  // if is_scale_up is false, the input image only can be zoom out, the maximum
  // resize scale cannot exceed 1.0
  bool is_scale_up;
  // padding stride, for is_mini_pad
  int stride;
  // for offseting the boxes by classes when using NMS, default 4096 in meituan/YOLOv6
  float max_wh;

 protected:
  // whether to inference with dynamic shape (e.g ONNX export with dynamic shape or not.)
  // meituan/YOLOv6 official 'export_onnx.py' script will export static ONNX by default.
  // while is_dynamic_shape if 'false', is_mini_pad will force 'false'. This value will
  // auto check by fastdeploy after the internal Runtime already initialized. 
  bool is_dynamic_shape_;
};
}  // namespace meituan
}  // namespace vision
}  // namespace fastdeploy