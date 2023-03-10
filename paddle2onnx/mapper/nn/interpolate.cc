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

#include "paddle2onnx/mapper/nn/interpolate.h"

namespace paddle2onnx {
REGISTER_MAPPER(bilinear_interp, InterpolateMapper)
REGISTER_MAPPER(bilinear_interp_v2, InterpolateMapper)
REGISTER_MAPPER(nearest_interp_v2, InterpolateMapper)
REGISTER_MAPPER(bicubic_interp_v2, InterpolateMapper)
REGISTER_MAPPER(linear_interp_v2, InterpolateMapper)
REGISTER_MAPPER(trilinear_interp_v2, InterpolateMapper)

int32_t InterpolateMapper::GetMinOpset(bool verbose) {
  if (data_layout_ == "NHWC") {
    Error() << "Data format of NHWC is not supported." << std::endl;
    return -1;
  }
  auto x_info = GetInput("X");
  if (x_info[0].Rank() > 5 && x_info[0].Rank() < 3) {
    Error() << "Only support 3D/4D/5D tensor, but now its dimension is "
            << x_info[0].Rank() << std::endl;
    return -1;
  }
  Logger(verbose, 11) << RequireOpset(11) << std::endl;
  return 11;
}

std::string InterpolateMapper::ComputeOutSize() {
  bool has_out_size = HasInput("OutSize");
  bool has_size_tensor = HasInput("SizeTensor");
  if (has_out_size) {
    auto out_size_info = GetInput("OutSize");
    return helper_->AutoCast(out_size_info[0].name, out_size_info[0].dtype,
                             P2ODataType::INT64);
  } else {
    auto size_tensor_info = GetInput("SizeTensor");
    return helper_->ConcatIndices(size_tensor_info);
  }
}

std::string InterpolateMapper::ComputeScale() {
  auto scale_info = GetInput("Scale");
  auto scale = helper_->AutoCast(scale_info[0].name, scale_info[0].dtype,
                                 P2ODataType::FP32);
  auto padding = helper_->Constant(ONNX_NAMESPACE::TensorProto::FLOAT,
                                   std::vector<float>(2, 1.0));
  scale = helper_->Concat({padding, scale}, 0);
  return scale;
}

void InterpolateMapper::Opset11() {
  auto x_info = GetInput("X");
  auto out_info = GetOutput("Out");
  std::string coordinate_transformation_mode = "half_pixel";
  auto resize_type = resize_mapper_[method_];
  if (align_corners_) {
    coordinate_transformation_mode = "align_corners";
  } else if (resize_type == "nearest") {
    coordinate_transformation_mode = "asymmetric";
  } else if (align_mode_ == 1 && resize_type != "cubic") {
    coordinate_transformation_mode = "asymmetric";
  }
  std::string scale = "";
  std::string size = "";
  bool has_out_size = HasInput("OutSize");
  bool has_size_tensor = HasInput("SizeTensor");
  bool has_scale_tensor = HasInput("Scale");
  if (has_out_size || has_size_tensor) {
    size = ComputeOutSize();
  } else if (has_scale_tensor) {
    scale = ComputeScale();
  } else {
    // get size or scale from attribute
    if (out_d_ > 0 || out_w_ > 0 || out_h_ > 0) {
      std::vector<int64_t> out_size;
      if (x_info[0].Rank() == 5) {
        out_size.push_back(out_d_);
        out_size.push_back(out_h_);
      }
      if (x_info[0].Rank() == 4) {
        out_size.push_back(out_h_);
      }
      out_size.push_back(out_w_);
      size = helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64, out_size);
    } else {
      std::vector<float> scale_;
      GetAttr("scale", &scale_);
      float padding = 1.0;
      scale_.insert(scale_.begin(), padding);
      scale_.insert(scale_.begin(), padding);
      scale = helper_->Constant(ONNX_NAMESPACE::TensorProto::FLOAT, scale_);
    }
  }
  std::string roi = helper_->Constant(ONNX_NAMESPACE::TensorProto::FLOAT, std::vector<float>());
  if (scale == "") {
    // has to generate a empty tensor for resize
    scale = helper_->Constant(ONNX_NAMESPACE::TensorProto::FLOAT,
                              std::vector<float>());
  }
  if (size != "") {
    auto ipt_shape = helper_->MakeNode("Shape", {x_info[0].name})->output(0);
    auto nc = helper_->Slice(ipt_shape, {0}, {0}, {2});
    size = helper_->Concat({nc, size}, 0);
  }
  std::shared_ptr<ONNX_NAMESPACE::NodeProto> node;
  if (size != "") {
    node = helper_->MakeNode("Resize", {x_info[0].name, roi, scale, size},
                             {out_info[0].name});
  } else {
    node = helper_->MakeNode("Resize", {x_info[0].name, roi, scale},
                             {out_info[0].name});
  }
  Assert(resize_mapper_.find(OpType()) != resize_mapper_.end(),
         "Cannot find " + OpType() + " in resize_mapper.");
  AddAttribute(node, "mode", resize_mapper_[OpType()]);
  AddAttribute(node, "coordinate_transformation_mode",
               coordinate_transformation_mode);
  if (resize_mapper_[OpType()] == "nearest" &&
      coordinate_transformation_mode == "asymmetric") {
    AddAttribute(node, "nearest_mode", "floor");
  }
}

}  // namespace paddle2onnx
