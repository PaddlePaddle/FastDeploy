/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <onnxoptimizer/optimize.h>

#include <onnx/checker.h>
#include <onnx/onnx_pb.h>

#include <fstream>

int main(int argc, char **argv) {
  ONNX_NAMESPACE::ModelProto model;
  std::ifstream ifs(argv[1]);
  bool success = model.ParseFromIstream(&ifs);
  if (!success) {
    std::cout << "load failed" << std::endl;
    return -1;
  }
  onnx::checker::check_model(model);
  const auto new_model = onnx::optimization::Optimize(
      model, onnx::optimization::GetFuseAndEliminationPass());
  onnx::checker::check_model(new_model);
  std::ofstream ofs(argv[2]);
  success = new_model.SerializePartialToOstream(&ofs);
  if (!success) {
    std::cout << "save failed" << std::endl;
    return -1;
  }
  return 0;
}
