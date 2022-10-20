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

#include "fastdeploy/vision/detection/ppdet/picodet.h"
#include "yaml-cpp/yaml.h"

inline float fast_exp(float x) {
  union {
    uint32_t i;
    float f;
  } v{};
  v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
  return v.f;
}

inline float sigmoid(float x) { return 1.0f / (1.0f + fast_exp(-x)); }

template <typename Tp>
int activation_function_softmax(const Tp* src, Tp* dst, int length) {
  const Tp alpha = *std::max_element(src, src + length);
  Tp denominator{0};

  for (int i = 0; i < length; ++i) {
    dst[i] = fast_exp(src[i] - alpha);
    denominator += dst[i];
  }

  for (int i = 0; i < length; ++i) {
    dst[i] /= denominator;
  }

  return 0;
}
namespace fastdeploy {
namespace vision {
namespace detection {

PicoDet::PicoDet(const std::string& model_file, const std::string& params_file,
                 const std::string& config_file,
                 const RuntimeOption& custom_option,
                 const ModelFormat& model_format) {
  config_file_ = config_file;
  valid_cpu_backends = {Backend::PDINFER, Backend::ORT, Backend::LITE};
  valid_gpu_backends = {Backend::ORT, Backend::PDINFER, Backend::TRT};
  if (model_format == ModelFormat::RKNN or model_format == ModelFormat::ONNX) {
    has_nms_ = false;
  }
  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;

  // NMS parameters come from picodet_s_nms
  background_label = -1;
  keep_top_k = 100;
  nms_eta = 1;
  nms_threshold = 0.5;
  nms_top_k = 1000;
  normalized = true;
  score_threshold = 0.3;
  if (has_nms_) {
    CheckIfContainDecodeAndNMS();
  }
  initialized = Initialize();
}

bool PicoDet::Preprocess(Mat* mat, std::vector<FDTensor>* outputs) {
  int origin_w = mat->Width();
  int origin_h = mat->Height();
  for (size_t i = 0; i < processors_.size(); ++i) {
    if (!(*(processors_[i].get()))(mat)) {
      FDERROR << "Failed to process image data in " << processors_[i]->Name()
              << "." << std::endl;
      return false;
    }
  }

  Cast::Run(mat, "float");

  if (this->has_nms_) {
    outputs->resize(2);
    (*outputs)[0].name = InputInfoOfRuntime(0).name;
    mat->ShareWithTensor(&((*outputs)[0]));
    // reshape to [1, c, h, w]
    (*outputs)[0].shape.insert((*outputs)[0].shape.begin(), 1);
    (*outputs)[1].Allocate({1, 2}, FDDataType::FP32,
                           InputInfoOfRuntime(1).name);
    float* ptr = static_cast<float*>((*outputs)[1].MutableData());
    ptr[0] = mat->Height() * 1.0 / origin_h;
    ptr[1] = mat->Width() * 1.0 / origin_w;
  } else {
    ptr.resize(2);
    ptr[0] = mat->Height() * 1.0 / origin_h;
    ptr[1] = mat->Width() * 1.0 / origin_w;

    outputs->resize(1);
    (*outputs)[0].name = InputInfoOfRuntime(0).name;
    mat->ShareWithTensor(&((*outputs)[0]));
    // reshape to [1, c, h, w]
    (*outputs)[0].shape.insert((*outputs)[0].shape.begin(), 1);
  }
  return true;
}

bool PicoDet::CheckIfContainDecodeAndNMS() {
  YAML::Node cfg;
  try {
    cfg = YAML::LoadFile(config_file_);
  } catch (YAML::BadFile& e) {
    FDERROR << "Failed to load yaml file " << config_file_
            << ", maybe you should check this file." << std::endl;
    return false;
  }

  if (cfg["NMS"])
    return true;

  // if cfg don't have the NMS parameter, print ERROR and return false.
  FDERROR << "The arch in config file is PicoDet, which means this model "
             "doesn contain box decode and nms, please export model with "
             "decode and nms."
          << std::endl;
  return false;
}
bool PicoDet::Postprocess(std::vector<FDTensor>& infer_result,
                          DetectionResult* result) {
  FDASSERT(infer_result[1].shape[0] == 1,
           "Only support batch = 1 in FastDeploy now.");

  if (!has_nms_) {
    // get output
    std::vector<std::vector<BoxInfo>> box_infos;
    box_infos.resize(this->num_class_);
    int num_outs = (int)(infer_result.size() / 2);
    for (size_t out_idx = 0; out_idx < num_outs; ++out_idx) {
      const auto* cls_pred =
          static_cast<const float*>(infer_result[out_idx].Data());
      const auto* dis_pred =
          static_cast<const float*>(infer_result[out_idx + num_outs].Data());
      this->decode_infer(cls_pred, dis_pred, strides[out_idx], score_threshold,
                         box_infos);
    }
    std::vector<BoxInfo> dets;
    for (auto& box_info : box_infos) {
      fastdeploy::vision::detection::PicoDet::picodet_nms(box_info,
                                                          nms_threshold);

      for (auto& box : box_info) {
        dets.push_back(box);
      }
    }
    result->Reserve(static_cast<int>(dets.size()));
    for (size_t i = 0; i < dets.size(); ++i) {
      result->label_ids.push_back(dets[i].label);
      result->scores.push_back(dets[i].score);
      result->boxes.emplace_back(std::array<float, 4>{
          static_cast<float>(dets[i].x1 / this->ptr[1]),
          static_cast<float>(dets[i].y1 / this->ptr[0]),
          static_cast<float>(dets[i].x2 / this->ptr[1]),
          static_cast<float>(dets[i].y2 / this->ptr[0])});
    }
    return true;
  } else {
    int box_num = 0;
    if (infer_result[1].dtype == FDDataType::INT32) {
      box_num = *(static_cast<int32_t*>(infer_result[1].Data()));
    } else if (infer_result[1].dtype == FDDataType::INT64) {
      box_num = *(static_cast<int64_t*>(infer_result[1].Data()));
    } else {
      FDASSERT(
          false,
          "The output box_num of PPYOLOE model should be type of int32/int64.");
    }
    result->Reserve(box_num);
    float* box_data = static_cast<float*>(infer_result[0].Data());
    for (size_t i = 0; i < box_num; ++i) {
      result->label_ids.push_back(box_data[i * 6]);
      result->scores.push_back(box_data[i * 6 + 1]);
      result->boxes.emplace_back(
          std::array<float, 4>{box_data[i * 6 + 2], box_data[i * 6 + 3],
                               box_data[i * 6 + 4], box_data[i * 6 + 5]});
    }
  }
  return true;
}
void PicoDet::decode_infer(const float*& cls_pred, const float*& dis_pred,
                           int stride, float threshold,
                           std::vector<std::vector<BoxInfo>>& results) {
  int feature_h = ceil((float)input_size_ / stride);
  int feature_w = ceil((float)input_size_ / stride);
  for (int idx = 0; idx < feature_h * feature_w; idx++) {
    int row = idx / feature_w;
    int col = idx % feature_w;
    float score = 0;
    int cur_label = 0;

    for (int label = 0; label < num_class_; label++) {
      if (cls_pred[idx * num_class_ + label] > score) {
        score = cls_pred[idx * num_class_ + label];
        cur_label = label;
      }
    }
    if (score > threshold) {
      const float* bbox_pred = dis_pred + idx * (reg_max_ + 1) * 4;
      results[cur_label].push_back(
          this->disPred2Bbox(bbox_pred, cur_label, score, col, row, stride));
    }
  }
}
BoxInfo PicoDet::disPred2Bbox(const float*& dfl_det, int label, float score,
                              int x, int y, int stride) {
  float ct_x = (x + 0.5) * stride;
  float ct_y = (y + 0.5) * stride;
  std::vector<float> dis_pred;
  dis_pred.resize(4);
  for (int i = 0; i < 4; i++) {
    float dis = 0;
    float* dis_after_sm = new float[reg_max_ + 1];
    activation_function_softmax(dfl_det + i * (reg_max_ + 1), dis_after_sm,
                                reg_max_ + 1);
    for (int j = 0; j < reg_max_ + 1; j++) {
      dis += j * dis_after_sm[j];
    }
    dis *= stride;
    dis_pred[i] = dis;
    delete[] dis_after_sm;
  }
  float xmin = (std::max)(ct_x - dis_pred[0], .0f);
  float ymin = (std::max)(ct_y - dis_pred[1], .0f);
  float xmax = (std::min)(ct_x + dis_pred[2], (float)this->input_size_);
  float ymax = (std::min)(ct_y + dis_pred[3], (float)this->input_size_);
  return BoxInfo{xmin, ymin, xmax, ymax, score, label};
}

void PicoDet::picodet_nms(std::vector<BoxInfo>& input_boxes, float NMS_THRESH) {
  std::sort(input_boxes.begin(), input_boxes.end(),
            [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
  std::vector<float> vArea(input_boxes.size());
  for (int i = 0; i < int(input_boxes.size()); ++i) {
    vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1) *
               (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
  }
  for (int i = 0; i < int(input_boxes.size()); ++i) {
    for (int j = i + 1; j < int(input_boxes.size());) {
      float xx1 = (std::max)(input_boxes[i].x1, input_boxes[j].x1);
      float yy1 = (std::max)(input_boxes[i].y1, input_boxes[j].y1);
      float xx2 = (std::min)(input_boxes[i].x2, input_boxes[j].x2);
      float yy2 = (std::min)(input_boxes[i].y2, input_boxes[j].y2);
      float w = (std::max)(float(0), xx2 - xx1 + 1);
      float h = (std::max)(float(0), yy2 - yy1 + 1);
      float inter = w * h;
      float ovr = inter / (vArea[i] + vArea[j] - inter);
      if (ovr >= NMS_THRESH) {
        input_boxes.erase(input_boxes.begin() + j);
        vArea.erase(vArea.begin() + j);
      } else {
        j++;
      }
    }
  }
}
} // namespace detection
} // namespace vision
} // namespace fastdeploy
