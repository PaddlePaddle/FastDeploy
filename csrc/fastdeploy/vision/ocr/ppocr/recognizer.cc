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
namespace ppocr {

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

std::vector<int> argsort(const std::vector<float>& array) {
  const int array_len(array.size());
  std::vector<int> array_index(array_len, 0);
  for (int i = 0; i < array_len; ++i) array_index[i] = i;

  std::sort(
      array_index.begin(), array_index.end(),
      [&array](int pos1, int pos2) { return (array[pos1] < array[pos2]); });

  return array_index;
}

Recognizer::Recognizer(const std::string& label_path,
                       const std::string& model_file,
                       const std::string& params_file,
                       const RuntimeOption& custom_option,
                       const Frontend& model_format) {
  if (model_format == Frontend::ONNX) {
    valid_cpu_backends = {Backend::ORT};  // 指定可用的CPU后端
    valid_gpu_backends = {Backend::ORT, Backend::TRT};  // 指定可用的GPU后端
  } else {
    // NOTE:此模型暂不支持paddle-inference-Gpu推理
    valid_cpu_backends = {Backend::ORT, Backend::PDINFER};
    valid_gpu_backends = {Backend::ORT, Backend::TRT};
  }

  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;
  // Recognizer在使用CPU推理，并把PaddleInference作为推理后端时,需要删除以下2个pass//
  runtime_option.EnablePaddleDeletePass("matmul_transpose_reshape_fuse_pass");
  runtime_option.EnablePaddleDeletePass(
      "matmul_transpose_reshape_mkldnn_fuse_pass");
  runtime_option.EnablePaddleLogInfo();

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
  scale = {0.5f, 0.5f, 0.5f};  // scale即std
  is_scale = true;

  if (!InitRuntime()) {
    FDERROR << "Failed to initialize fastdeploy backend." << std::endl;
    return false;
  }

  return true;
}

//预处理
bool Recognizer::Preprocess(const std::vector<cv::Mat>& img_list,
                            FDTensor* output,
                            const std::vector<int>& rec_image_shape,
                            const int& cur_index,
                            const std::vector<int>& indices) {
  int img_num = img_list.size();
  int end_img_no = std::min(img_num, cur_index + rec_batch_num);
  int batch_num = end_img_no - cur_index;

  int imgH = rec_image_shape[1];
  int imgW = rec_image_shape[2];
  float max_wh_ratio = imgW * 1.0 / imgH;

  for (int ino = cur_index; ino < end_img_no; ino++) {
    //这个batch中的max_wh_ratio
    int h = img_list[indices[ino]].rows;
    int w = img_list[indices[ino]].cols;
    float wh_ratio = w * 1.0 / h;
    max_wh_ratio = std::max(max_wh_ratio, wh_ratio);
  }

  int batch_width = imgW;

  for (int ino = cur_index; ino < end_img_no; ino++) {
    cv::Mat resize_img;

    crnn_resize_img(img_list[indices[ino]], resize_img, max_wh_ratio,
                    rec_image_shape);
    // cv::Mat to Mat
    Mat img(resize_img);

    Normalize::Run(&img, mean, scale, true);

    batch_width =
        std::max(resize_img.cols, batch_width);  //支持batch推理时要用，暂时没用

    HWC2CHW::Run(&img);
    Cast::Run(&img, "float");

    img.ShareWithTensor(output);
    output->shape.insert(output->shape.begin(), 1);
  }

  return true;
}

//后处理
bool Recognizer::Postprocess(FDTensor& infer_result, const int& cur_index,
                             const std::vector<int>& indices,
                             std::vector<std::string>& rec_texts,
                             std::vector<float>& rec_text_scores) {
  // infer_result : n, c, h , w
  std::vector<int64_t> output_shape = infer_result.shape;
  FDASSERT(output_shape[0] == 1, "Only support batch =1 now.");

  float* out_data = static_cast<float*>(infer_result.Data());

  for (int m = 0; m < output_shape[0]; m++) {
    std::string str_res;
    int argmax_idx;
    int last_index = 0;
    float score = 0.f;
    int count = 0;
    float max_value = 0.0f;

    for (int n = 0; n < output_shape[1]; n++) {
      argmax_idx = int(std::distance(
          &out_data[(m * output_shape[1] + n) * output_shape[2]],
          std::max_element(
              &out_data[(m * output_shape[1] + n) * output_shape[2]],
              &out_data[(m * output_shape[1] + n + 1) * output_shape[2]])));

      max_value = float(*std::max_element(
          &out_data[(m * output_shape[1] + n) * output_shape[2]],
          &out_data[(m * output_shape[1] + n + 1) * output_shape[2]]));

      if (argmax_idx > 0 && (!(n > 0 && argmax_idx == last_index))) {
        score += max_value;
        count += 1;
        str_res += label_list[argmax_idx];
      }
      last_index = argmax_idx;
    }

    score /= count;

    if (std::isnan(score)) {
      continue;
    }

    rec_texts[indices[cur_index + m]] = str_res;
    rec_text_scores[indices[cur_index + m]] = score;
  }

  return true;
}

//预测
bool Recognizer::Predict(const std::vector<cv::Mat>& img_list,
                         std::vector<std::string>& rec_texts,
                         std::vector<float>& rec_text_scores) {
#ifdef FASTDEPLOY_DEBUG
  TIMERECORD_START(0)
#endif

  int img_num = img_list.size();
  std::vector<float> width_list;
  for (int i = 0; i < img_num; i++) {
    width_list.push_back(float(img_list[i].cols) / img_list[i].rows);
  }
  std::vector<int> indices = argsort(width_list);

  // rec_batch_num, 默认为1
  for (int ino = 0; ino < img_num; ino += rec_batch_num) {
    // PPOCR套件支持推理一个batch. 目前FD暂支持推理一张图
    std::vector<FDTensor> input_tensors(1);

    if (!Preprocess(img_list, &input_tensors[0], rec_image_shape, ino,
                    indices)) {
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
    if (!Postprocess(output_tensors[0], ino, indices, rec_texts,
                     rec_text_scores)) {
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