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
#include "fastdeploy/vision/utils/utils.h"

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
  initialized = Initialize();
}

// Init
bool DBDetector::Initialize() {
  // pre&post process parameters
  max_side_len = 960;

  det_db_thresh = 0.3;
  det_db_box_thresh = 0.5;
  det_db_unclip_ratio = 2.0;
  det_db_score_mode = "slow";
  use_dilation = false;

  mean = {0.485f, 0.456f, 0.406f};
  scale = {1 / 0.229f, 1 / 0.224f, 1 / 0.225f};
  is_scale = true;

  if (!InitRuntime()) {
    FDERROR << "Failed to initialize fastdeploy backend." << std::endl;
    return false;
  }

  is_dynamic_input_ = false;

  return true;
}

//预处理
bool DBDetector::Preprocess(
    cv::Mat& mat, cv::Mat& resize_img, FDTensor* output,
    std::map<std::string, std::array<float, 2>>* im_info) {
  // OCR的preprocess需要的CV算子，目前FastDeploy应该都支持. 暂时先用opencv
  // resize
  utils::OcrDetectorResizeImage(mat, resize_img, max_side_len, ratio_h,
                                ratio_w);
  // normalize
  utils::OcrNormalize(&resize_img, mean, scale, is_scale);
  // permute
  // std::vector<float> input(1 * 3 * resize_img.rows * resize_img.cols, 0.0f);
  // utils::OcrPermute(&resize_img, input.data()));
  Mat resize_mat(resize_img);
  (*im_info)["output_shape"] = {static_cast<float>(resize_mat.Height()),
                                static_cast<float>(resize_mat.Width())};

  HWC2CHW::Run(&resize_mat);
  resize_mat.ShareWithTensor(output);
  output->shape.insert(output->shape.begin(), 1);

  return true;
}

//后处理
bool DBDetector::Postprocess(
    FDTensor& infer_result, std::vector<std::vector<std::vector<int>>>& boxes,
    const std::map<std::string, std::array<float, 2>>& im_info,
    cv::Mat srcimg) {
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

  boxes =
      post_processor_.BoxesFromBitmap(pred_map, bit_map, det_db_box_thresh,
                                      det_db_unclip_ratio, det_db_score_mode);

  boxes = post_processor_.FilterTagDetRes(boxes, ratio_h, ratio_w, srcimg);
  std::cout << "Finish the PostProcess!" << std::endl;
  return true;
}

//预测
bool DBDetector::Predict(cv::Mat& img,
                         std::vector<OCRPredictResult>& ocr_results) {
#ifdef FASTDEPLOY_DEBUG
  TIMERECORD_START(0)
#endif

  cv::Mat srcimg;
  cv::Mat resize_img;
  img.copyTo(srcimg);
  // output boxes
  std::vector<std::vector<std::vector<int>>> boxes_result;

  std::vector<FDTensor> input_tensors(1);

  std::map<std::string, std::array<float, 2>> im_info;

  // Record the shape of image and the shape of preprocessed image
  im_info["input_shape"] = {static_cast<float>(srcimg.rows),
                            static_cast<float>(srcimg.cols)};
  im_info["output_shape"] = {static_cast<float>(srcimg.rows),
                             static_cast<float>(srcimg.cols)};

  if (!Preprocess(srcimg, resize_img, &input_tensors[0], &im_info)) {
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

  if (!Postprocess(output_tensors[0], boxes_result, im_info, srcimg)) {
    FDERROR << "Failed to post process." << std::endl;
    return false;
  }

  for (int i = 0; i < boxes_result.size(); i++) {
    OCRPredictResult res;
    res.boxes = boxes_result
        [i];  // boxes_result里装着一张输入图片中，识别出的所有的小box
    ocr_results.push_back(res);
  }

#ifdef FASTDEPLOY_DEBUG
  TIMERECORD_END(2, "Postprocess")
#endif
  return true;
}

}  // namesapce ppocr
}  // namespace vision
}  // namespace fastdeploy