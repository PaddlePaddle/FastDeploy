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

#include "fastdeploy/vision/ppocr/dbdetector.h"
#include "fastdeploy/utils/perf.h"
#include "fastdeploy/vision/ppocr/utils/ocr_utils.h"

namespace fastdeploy {
namespace vision {
namespace ppocr {

//构造
DBDetector::DBDetector(const std::string& model_file,
                       const std::string& params_file,
                       const RuntimeOption& custom_option,
                       const Frontend& model_format) {
  if (model_format == Frontend::ONNX) {
    valid_cpu_backends = {Backend::ORT};  // 指定可用的CPU后端
    valid_gpu_backends = {Backend::ORT, Backend::TRT};  // 指定可用的GPU后端
  } else {
    valid_cpu_backends = {Backend::PDINFER, Backend::ORT};
    valid_gpu_backends = {Backend::PDINFER, Backend::ORT, Backend::TRT};
  }

  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;
  runtime_option.UsePaddleBackend();

  initialized = Initialize();
}

// Init
bool DBDetector::Initialize() {
  // pre&post process parameters
  max_side_len = 960;

  det_db_thresh = 0.3;
  det_db_box_thresh = 0.6;    // 0.5
  det_db_unclip_ratio = 1.5;  // 2.0
  det_db_score_mode = "slow";
  use_dilation = false;

  mean = {0.485f, 0.456f, 0.406f};
  scale = {0.229f, 0.224f, 0.225f};  // scale即std
  is_scale = true;

  if (!InitRuntime()) {
    FDERROR << "Failed to initialize fastdeploy backend." << std::endl;
    return false;
  }

  return true;
}

void OcrDetectorResizeImage(Mat* img, int max_size_len, float& ratio_h,
                            float& ratio_w) {
  int w = img->Width();
  int h = img->Height();

  float ratio = 1.f;
  int max_wh = w >= h ? w : h;
  if (max_wh > max_size_len) {
    if (h > w) {
      ratio = float(max_size_len) / float(h);
    } else {
      ratio = float(max_size_len) / float(w);
    }
  }

  int resize_h = int(float(h) * ratio);
  int resize_w = int(float(w) * ratio);

  resize_h = std::max(int(std::round(float(resize_h) / 32) * 32), 32);
  resize_w = std::max(int(std::round(float(resize_w) / 32) * 32), 32);

  Resize::Run(img, resize_w, resize_h);

  ratio_h = float(resize_h) / float(h);
  ratio_w = float(resize_w) / float(w);
}

//预处理
bool DBDetector::Preprocess(
    Mat* mat, FDTensor* output,
    std::map<std::string, std::array<float, 2>>* im_info) {
  // Resize
  OcrDetectorResizeImage(mat, max_side_len, ratio_h, ratio_w);
  // Normalize
  Normalize::Run(mat, mean, scale, true);

  (*im_info)["output_shape"] = {static_cast<float>(mat->Height()),
                                static_cast<float>(mat->Width())};
  //-CHW
  HWC2CHW::Run(mat);
  Cast::Run(mat, "float");

  mat->ShareWithTensor(output);
  output->shape.insert(output->shape.begin(), 1);
  return true;
}

//后处理
bool DBDetector::Postprocess(
    FDTensor& infer_result, std::vector<std::vector<std::vector<int>>>* boxes,
    const std::map<std::string, std::array<float, 2>>& im_info) {
  // infer_result : n, c, h , w
  std::vector<int64_t> output_shape = infer_result.shape;
  FDASSERT(output_shape[0] == 1, "Only support batch =1 now.");
  int n2 = output_shape[2];
  int n3 = output_shape[3];
  int n = n2 * n3;

  float* out_data = static_cast<float*>(infer_result.Data());
  // prepare bitmap
  std::vector<float> pred(n, 0.0);
  std::vector<unsigned char> cbuf(n, ' ');

  for (int i = 0; i < n; i++) {
    pred[i] = float(out_data[i]);
    cbuf[i] = (unsigned char)((out_data[i]) * 255);
  }
  cv::Mat cbuf_map(n2, n3, CV_8UC1, (unsigned char*)cbuf.data());
  cv::Mat pred_map(n2, n3, CV_32F, (float*)pred.data());

  const double threshold = det_db_thresh * 255;
  const double maxvalue = 255;
  cv::Mat bit_map;
  cv::threshold(cbuf_map, bit_map, threshold, maxvalue, cv::THRESH_BINARY);
  if (use_dilation) {
    cv::Mat dila_ele =
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::dilate(bit_map, bit_map, dila_ele);
  }

  post_processor_.BoxesFromBitmap(pred_map, boxes, bit_map, det_db_box_thresh,
                                  det_db_unclip_ratio, det_db_score_mode);

  post_processor_.FilterTagDetRes(boxes, ratio_h, ratio_w, im_info);

  return true;
}

//预测
bool DBDetector::Predict(
    cv::Mat* img, std::vector<std::vector<std::vector<int>>>& boxes_result) {
#ifdef FASTDEPLOY_DEBUG
  TIMERECORD_START(0)
#endif
  Mat mat(*img);
  // output boxes
  // std::vector<std::vector<std::vector<int>>> boxes_result;

  std::vector<FDTensor> input_tensors(1);

  std::map<std::string, std::array<float, 2>> im_info;

  // Record the shape of image and the shape of preprocessed image
  im_info["input_shape"] = {static_cast<float>(mat.Height()),
                            static_cast<float>(mat.Width())};
  im_info["output_shape"] = {static_cast<float>(mat.Height()),
                             static_cast<float>(mat.Width())};

  if (!Preprocess(&mat, &input_tensors[0], &im_info)) {
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

  if (!Postprocess(output_tensors[0], &boxes_result, im_info)) {
    FDERROR << "Failed to post process." << std::endl;
    return false;
  }

#ifdef FASTDEPLOY_DEBUG
  TIMERECORD_END(2, "Postprocess")
#endif
  return true;
}

}  // namesapce ppocr
}  // namespace vision
}  // namespace fastdeploy