//
// Created by aichao on 2022/10/10.
//

#pragma once
#include "fastdeploy/vision/common/processors/transform.h"
#include "fastdeploy/fastdeploy_model.h"
#include "fastdeploy/vision/tracking/pptracking/utils.h"
#include "fastdeploy/vision/tracking/pptracking/postprocess.h"

namespace fastdeploy {
namespace vision {
namespace tracking {

class FASTDEPLOY_DECL PPTracking: public FastDeployModel {

  public:
    PPTracking(const std::string& model_file, const std::string& params_file = "",
               const RuntimeOption& custom_option = RuntimeOption(),
               const ModelFormat& model_format = ModelFormat::PADDLE);

  std::string ModelName() const { return "pptracking/jde"; }

  virtual bool Predict(cv::Mat* img, MOTResult* result);
  static cv::Mat Visualize(const cv::Mat& img,
                      const MOTResult& results,
                      float fps,
                      int frame_id);

 private:


  std::vector<float> mean_;
  std::vector<float> scale_;
  bool is_scale_ = true;
  std::vector<int> target_size_;
  float conf_thresh_;
  float threshold_;
  float min_box_area_;

  bool Initialize();


  void LetterBoxResize(Mat* im);

  bool Preprocess(Mat* img, std::vector<FDTensor>* outputs);

  bool Postprocess(std::vector<FDTensor>& infer_result, MOTResult *result);


};

} // namespace tracking
} // namespace vision
} // namespace fastdeploy

