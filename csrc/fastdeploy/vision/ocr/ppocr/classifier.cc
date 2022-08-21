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

#include "fastdeploy/vision/ocr/ppocr/classifier.h"
#include "fastdeploy/utils/perf.h"
#include "fastdeploy/vision/ocr/ppocr/utils/ocr_utils.h"

namespace fastdeploy {
namespace vision {
namespace ppocr {

//构造
Classifier::Classifier(const std::string& model_file,
                       const std::string& params_file,
                       const RuntimeOption& custom_option,
                       const Frontend& model_format) {
  if (model_format == Frontend::ONNX) {
    valid_cpu_backends = {Backend::ORT};  // 指定可用的CPU后端
    valid_gpu_backends = {Backend::ORT, Backend::TRT};  // 指定可用的GPU后端
  } else {
    // Cls模型暂不支持ORT后端推理
    valid_cpu_backends = {Backend::PDINFER};
    valid_gpu_backends = {Backend::PDINFER, Backend::TRT};
  }
  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;

  initialized = Initialize();
}

// Init
bool Classifier::Initialize() {
  // pre&post process parameters
  cls_thresh = 0.9;
  cls_batch_num = 1;
  mean = {0.485f, 0.456f, 0.406f};
  scale = {0.5f, 0.5f, 0.5f};
  is_scale = true;

  if (!InitRuntime()) {
    FDERROR << "Failed to initialize fastdeploy backend." << std::endl;
    return false;
  }

  return true;
}

//预处理
//预处理
bool Classifier::Preprocess(const std::vector<cv::Mat>& img_list,
                            FDTensor* output,
                            const std::vector<int>& cls_image_shape,
                            const int& cur_index) {
  int img_num = img_list.size();
  int end_img_no = std::min(img_num, cur_index + cls_batch_num);
  int batch_num = end_img_no - cur_index;

  // 1. cls resizes,用到cv::copyMakeBorders
  // 2. normalize
  // 3. batch_permute
  for (int ino = cur_index; ino < end_img_no; ino++) {
    cv::Mat resize_img;
    cls_resize_img(img_list[ino], resize_img, cls_image_shape);

    // cv::Mat to Mat
    Mat img(resize_img);
    Normalize::Run(&img, mean, scale, true);

    HWC2CHW::Run(&img);
    Cast::Run(&img, "float");

    img.ShareWithTensor(output);
    output->shape.insert(output->shape.begin(), 1);
  }

  return true;
}

//后处理
bool Classifier::Postprocess(FDTensor& infer_result,
                             std::vector<int>& cls_labels,
                             std::vector<float>& cls_scores,
                             const int& cur_index) {
  // infer_result : n, c, h , w
  std::vector<int64_t> output_shape = infer_result.shape;
  FDASSERT(output_shape[0] == 1, "Only support batch =1 now.");

  float* out_data = static_cast<float*>(infer_result.Data());

  for (int batch_idx = 0; batch_idx < output_shape[0]; batch_idx++) {
    int label = std::distance(
        &out_data[batch_idx * output_shape[1]],
        std::max_element(&out_data[batch_idx * output_shape[1]],
                         &out_data[(batch_idx + 1) * output_shape[1]]));
    float score =
        float(*std::max_element(&out_data[batch_idx * output_shape[1]],
                                &out_data[(batch_idx + 1) * output_shape[1]]));

    cls_labels[cur_index + batch_idx] = label;
    cls_scores[cur_index + batch_idx] = score;
  }

  return true;
}

//预测
bool Classifier::Predict(const std::vector<cv::Mat>& img_list,
                         std::vector<int>& cls_labels,
                         std::vector<float>& cls_socres) {
#ifdef FASTDEPLOY_DEBUG
  TIMERECORD_START(0)
#endif

  int img_num = img_list.size();
  std::vector<int> cls_image_shape = {3, 48, 192};

  // cls_batch_num, 默认为1
  for (int ino = 0; ino < img_num; ino += cls_batch_num) {
    // PPOCR套件支持推理一个batch. 目前FD暂支持推理一张图
    std::vector<FDTensor> input_tensors(1);

    if (!Preprocess(img_list, &input_tensors[0], cls_image_shape, ino)) {
      FDERROR << "Failed to preprocess input image." << std::endl;
      return false;
    }

#ifdef FASTDEPLOY_DEBUG
    TIMERECORD_END(0, "Preprocess")
    TIMERECORD_START(1)
#endif

    input_tensors[0].name = InputInfoOfRuntime(0).name;
    std::vector<FDTensor> output_tensors;
    if (!Infer(input_tensors, &output_tensors)) {
      FDERROR << "Failed to inference." << std::endl;
      return false;
    }
#ifdef FASTDEPLOY_DEBUG
    TIMERECORD_END(1, "Inference")
    TIMERECORD_START(2)
#endif

    if (!Postprocess(output_tensors[0], cls_labels, cls_socres, ino)) {
      FDERROR << "Failed to post process." << std::endl;
      return false;
    }

#ifdef FASTDEPLOY_DEBUG
    TIMERECORD_END(2, "Postprocess")
#endif
  }

  return true;
}

}  // namesapce ppocr
}  // namespace vision
}  // namespace fastdeploy