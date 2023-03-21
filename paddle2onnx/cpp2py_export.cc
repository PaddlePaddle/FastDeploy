//   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "paddle2onnx/converter.h"
#include "paddle2onnx/mapper/exporter.h"
#include "paddle2onnx/optimizer/paddle2onnx_optimizer.h"

namespace paddle2onnx {

typedef std::map<std::string, std::string> CustomOpInfo;
PYBIND11_MODULE(paddle2onnx_cpp2py_export, m) {
  m.doc() = "Paddle2ONNX: export PaddlePaddle to ONNX";
  m.def("export", [](const std::string& model_filename,
                     const std::string& params_filename, int opset_version = 9,
                     bool auto_upgrade_opset = true, bool verbose = true,
                     bool enable_onnx_checker = true,
                     bool enable_experimental_op = true,
                     bool enable_optimize = true,
                     const CustomOpInfo& info = CustomOpInfo(),
                     const std::string& deploy_backend = "onnxruntime",
                     const std::string& calibration_file = "",
                     const std::string& external_file = "",
                     const bool& export_fp16_model = false) {
    P2OLogger(verbose) << "Start to parse PaddlePaddle model..." << std::endl;
    P2OLogger(verbose) << "Model file path: " << model_filename << std::endl;
    P2OLogger(verbose) << "Paramters file path: " << params_filename
                       << std::endl;
    if (info.size() == 0) {
      char* out = nullptr;
      int size = 0;
      char* calibration_cache = nullptr;
      int cache_size = 0;
      bool save_external;
      if (!Export(model_filename.c_str(), params_filename.c_str(), &out, &size,
                  opset_version, auto_upgrade_opset, verbose,
                  enable_onnx_checker, enable_experimental_op, enable_optimize,
                  nullptr, 0, deploy_backend.c_str(), &calibration_cache,
                  &cache_size, external_file.c_str(), &save_external,
                  export_fp16_model)) {
        P2OLogger(verbose) << "Paddle model convert failed." << std::endl;
        return pybind11::bytes("");
      }
      if (cache_size) {
        std::string calibration_cache_str(calibration_cache,
                                          calibration_cache + cache_size);
        std::ofstream cache_file;
        cache_file.open(calibration_file, std::ios::out);
        cache_file << calibration_cache_str;
        delete calibration_cache;
        calibration_cache = nullptr;
        P2OLogger(verbose) << "TensorRT calibration cache path: "
                           << calibration_file << std::endl;
      }
      std::string onnx_proto(out, out + size);
      delete out;
      out = nullptr;
      return pybind11::bytes(onnx_proto);
    }

    std::vector<CustomOp> ops;
    ops.resize(info.size());
    int index = 0;
    for (auto& item : info) {
      strcpy(ops[index].op_name, item.first.c_str());
      strcpy(ops[index].export_op_name, item.second.c_str());
      index += 1;
    }
    char* out = nullptr;
    int size = 0;
    char* calibration_cache = nullptr;
    int cache_size = 0;
    bool save_external;
    if (!Export(model_filename.c_str(), params_filename.c_str(), &out, &size,
                opset_version, auto_upgrade_opset, verbose, enable_onnx_checker,
                enable_experimental_op, enable_optimize, ops.data(),
                info.size(), deploy_backend.c_str(), &calibration_cache,
                &cache_size, external_file.c_str(), &save_external,
                export_fp16_model)) {
      P2OLogger(verbose) << "Paddle model convert failed." << std::endl;
      return pybind11::bytes("");
    }
    if (cache_size) {
      std::string calibration_cache_str(calibration_cache,
                                        calibration_cache + cache_size);
      std::ofstream cache_file;
      cache_file.open(calibration_file, std::ios::out);
      cache_file << calibration_cache_str;
      delete calibration_cache;
      calibration_cache = nullptr;
      P2OLogger(verbose) << "TensorRT calibration cache path: "
                         << calibration_file << std::endl;
    }
    std::string onnx_proto(out, out + size);
    delete out;
    out = nullptr;
    return pybind11::bytes(onnx_proto);
  });
  m.def(
      "optimize",
      [](const std::string& model_path, const std::string& optimized_model_path,
         const std::map<std::string, std::vector<int>>& shape_infos) {
        ONNX_NAMESPACE::optimization::OptimizePaddle2ONNX(
            model_path, optimized_model_path, shape_infos);
      });
  m.def("convert_to_fp16", [](const std::string& fp32_model_path,
                              const std::string& fp16_model_path) {
    paddle2onnx::optimization::Paddle2ONNXFP32ToFP16(fp32_model_path,
                                                     fp16_model_path);
  });
}
}  // namespace paddle2onnx
