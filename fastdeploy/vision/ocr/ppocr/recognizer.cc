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

#include "fastdeploy/vision/ocr/ppocr/recognizer.h"
#include "fastdeploy/utils/perf.h"
#include "fastdeploy/vision/ocr/ppocr/utils/ocr_utils.h"

namespace fastdeploy {
namespace vision {
namespace ocr {

std::vector<std::string> ReadDict(const std::string& path) {
  std::ifstream in(path);
  std::string line;
  std::vector<std::string> m_vec;
  if (in) {
    while (getline(in, line)) {
      m_vec.push_back(line);
    }
  } else {
    std::cout << "no such label file: " << path << ", exit the program..."
              << std::endl;
    exit(1);
  }
  return m_vec;
}

Recognizer::Recognizer() {}

Recognizer::Recognizer(const std::string& model_file,
                       const std::string& params_file,
                       const std::string& label_path,
                       const RuntimeOption& custom_option,
                       const ModelFormat& model_format) {
  if (model_format == ModelFormat::ONNX) {
    valid_cpu_backends = {Backend::ORT,
                          Backend::OPENVINO};  
    valid_gpu_backends = {Backend::ORT, Backend::TRT};  
  } else {
    valid_cpu_backends = {Backend::PDINFER, Backend::ORT, Backend::OPENVINO};
    valid_gpu_backends = {Backend::PDINFER, Backend::ORT, Backend::TRT};
  }

  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;
  runtime_option.DeletePaddleBackendPass("matmul_transpose_reshape_fuse_pass");
  runtime_option.DeletePaddleBackendPass(
      "matmul_transpose_reshape_mkldnn_fuse_pass");

  initialized = Initialize();

  // init label_lsit
  label_list = ReadDict(label_path);
  label_list.insert(label_list.begin(), "#");  // blank char for ctc
  label_list.push_back(" ");
}

// Init
bool Recognizer::Initialize() {
  // pre&post process parameters
  rec_batch_num = 1;
  rec_img_h = 48;
  rec_img_w = 320;
  rec_image_shape = {3, rec_img_h, rec_img_w};

  mean = {0.5f, 0.5f, 0.5f};
  scale = {0.5f, 0.5f, 0.5f};
  is_scale = true;

  if (!InitRuntime()) {
    FDERROR << "Failed to initialize fastdeploy backend." << std::endl;
    return false;
  }

  return true;
}

void OcrRecognizerResizeImage(Mat* mat, const float& wh_ratio,
                              const std::vector<int>& rec_image_shape) {
  int imgC, imgH, imgW;
  imgC = rec_image_shape[0];
  imgH = rec_image_shape[1];
  imgW = rec_image_shape[2];

  imgW = int(imgH * wh_ratio);

  float ratio = float(mat->Width()) / float(mat->Height());
  int resize_w;
  if (ceilf(imgH * ratio) > imgW)
    resize_w = imgW;
  else
    resize_w = int(ceilf(imgH * ratio));

  Resize::Run(mat, resize_w, imgH);

  std::vector<float> value = {127, 127, 127};
  Pad::Run(mat, 0, 0, 0, int(imgW - mat->Width()), value);
}

bool Recognizer::Preprocess(Mat* mat, FDTensor* output,
                            const std::vector<int>& rec_image_shape) {
  int imgH = rec_image_shape[1];
  int imgW = rec_image_shape[2];
  float wh_ratio = imgW * 1.0 / imgH;

  float ori_wh_ratio = mat->Width() * 1.0 / mat->Height();
  wh_ratio = std::max(wh_ratio, ori_wh_ratio);

  OcrRecognizerResizeImage(mat, wh_ratio, rec_image_shape);

  Normalize::Run(mat, mean, scale, true);

  HWC2CHW::Run(mat);
  Cast::Run(mat, "float");

  mat->ShareWithTensor(output);
  output->shape.insert(output->shape.begin(), 1);

  return true;
}

bool Recognizer::Postprocess(FDTensor& infer_result,
                             std::tuple<std::string, float>* rec_result) {
  std::vector<int64_t> output_shape = infer_result.shape;
  FDASSERT(output_shape[0] == 1, "Only support batch =1 now.");

  float* out_data = static_cast<float*>(infer_result.Data());

  std::string str_res;
  int argmax_idx;
  int last_index = 0;
  float score = 0.f;
  int count = 0;
  float max_value = 0.0f;

  for (int n = 0; n < output_shape[1]; n++) {
    argmax_idx = int(
        std::distance(&out_data[n * output_shape[2]],
                      std::max_element(&out_data[n * output_shape[2]],
                                       &out_data[(n + 1) * output_shape[2]])));

    max_value = float(*std::max_element(&out_data[n * output_shape[2]],
                                        &out_data[(n + 1) * output_shape[2]]));

    if (argmax_idx > 0 && (!(n > 0 && argmax_idx == last_index))) {
      score += max_value;
      count += 1;
      str_res += label_list[argmax_idx];
    }
    last_index = argmax_idx;
  }

  score /= count;

  std::get<0>(*rec_result) = str_res;
  std::get<1>(*rec_result) = score;

  return true;
}

bool Recognizer::Predict(cv::Mat* img,
                         std::tuple<std::string, float>* rec_result) {
  Mat mat(*img);

  std::vector<FDTensor> input_tensors(1);

  if (!Preprocess(&mat, &input_tensors[0], rec_image_shape)) {
    FDERROR << "Failed to preprocess input image." << std::endl;
    return false;
  }

  input_tensors[0].name = InputInfoOfRuntime(0).name;
  std::vector<FDTensor> output_tensors;

  if (!Infer(input_tensors, &output_tensors)) {
    FDERROR << "Failed to inference." << std::endl;
    return false;
  }

  if (!Postprocess(output_tensors[0], rec_result)) {
    FDERROR << "Failed to post process." << std::endl;
    return false;
  }

  return true;
}

}  // namesapce ocr
}  // namespace vision
}  // namespace fastdeploy
