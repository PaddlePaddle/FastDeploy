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

namespace linzaer {

class FASTDEPLOY_DECL UltraFace : public FastDeployModel {
 public:
  // 当model_format为ONNX时，无需指定params_file
  // 当model_format为Paddle时，则需同时指定model_file & params_file
  UltraFace(const std::string& model_file, const std::string& params_file = "",
            const RuntimeOption& custom_option = RuntimeOption(),
            const Frontend& model_format = Frontend::ONNX);

  // 定义模型的名称
  std::string ModelName() const {
    return "Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB";
  }

  // 模型预测接口，即用户调用的接口
  // im 为用户的输入数据，目前对于CV均定义为cv::Mat
  // result 为模型预测的输出结构体
  // conf_threshold 为后处理的参数
  // nms_iou_threshold 为后处理的参数
  virtual bool Predict(cv::Mat* im, FaceDetectionResult* result,
                       float conf_threshold = 0.7f,
                       float nms_iou_threshold = 0.3f);

  // 以下为模型在预测时的一些参数，基本是前后处理所需
  // 用户在创建模型后，可根据模型的要求，以及自己的需求
  // 对参数进行修改
  // tuple of (width, height), default (320, 240)
  std::vector<int> size;

 private:
  // 初始化函数，包括初始化后端，以及其它模型推理需要涉及的操作
  bool Initialize();

  // 输入图像预处理操作
  // Mat为FastDeploy定义的数据结构
  // FDTensor为预处理后的Tensor数据，传给后端进行推理
  // im_info为预处理过程保存的数据，在后处理中需要用到
  bool Preprocess(Mat* mat, FDTensor* outputs,
                  std::map<std::string, std::array<float, 2>>* im_info);

  // 后端推理结果后处理，输出给用户
  // infer_result 为后端推理后的输出Tensor
  // result 为模型预测的结果
  // im_info 为预处理记录的信息，后处理用于还原box
  // conf_threshold 后处理时过滤box的置信度阈值
  // nms_iou_threshold 后处理时NMS设定的iou阈值
  bool Postprocess(std::vector<FDTensor>& infer_result,
                   FaceDetectionResult* result,
                   const std::map<std::string, std::array<float, 2>>& im_info,
                   float conf_threshold, float nms_iou_threshold);

  // 查看输入是否为动态维度的 不建议直接使用 不同模型的逻辑可能不一致
  bool IsDynamicInput() const { return is_dynamic_input_; }

  bool is_dynamic_input_;
};

}  // namespace linzaer
}  // namespace vision
}  // namespace fastdeploy
