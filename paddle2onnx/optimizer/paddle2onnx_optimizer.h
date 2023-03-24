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
#include <onnx/onnx_pb.h>
#include <map>
#include <string>
#include <vector>

namespace ONNX_NAMESPACE {
namespace optimization {

struct OptimizerOption {
  std::vector<std::string> passes;
  OptimizerOption() {
    passes.push_back("eliminate_identity");
    passes.push_back("eliminate_deadend");
    passes.push_back("fuse_constant_reshape");
    passes.push_back("fuse_constant_unsqueeze");
    passes.push_back("fuse_paddle_conv_bias");
    passes.push_back("fuse_consecutive_transposes");
    passes.push_back("eliminate_non_transpose");
    passes.push_back("replace_mul_to_identity");
    passes.push_back("replace_add_to_identity");
    passes.push_back("fuse_matmul_add_bias_into_gemm");
    passes.push_back("eliminate_identity");
    passes.push_back("eliminate_deadend");
  }
};

ONNX_NAMESPACE::ModelProto OptimizeOnnxModel(
    const ONNX_NAMESPACE::ModelProto& model);

bool OptimizePaddle2ONNX(const std::string& model_path,
                         const std::string& optimized_model_path,
                         const OptimizerOption& option = OptimizerOption());

bool OptimizePaddle2ONNX(
    const std::string& model_path, const std::string& optimized_model_path,
    const std::map<std::string, std::vector<int>>& shape_infos,
    const OptimizerOption& option = OptimizerOption());

bool Paddle2ONNXFP32ToFP16(const std::string& model_path,
                           const std::string& optimized_model_path);

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
