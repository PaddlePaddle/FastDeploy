/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#include "onnxoptimizer/pass_registry.h"

namespace ONNX_NAMESPACE {
namespace optimization {

const std::vector<std::string> GlobalPassRegistry::GetAvailablePasses() {
  std::vector<std::string> names;
  for (const auto& pass : this->passes) {
    names.push_back(pass.first);
  }
  return names;
}

const std::vector<std::string> GlobalPassRegistry::GetFuseAndEliminationPass() {
  std::vector<std::string> names;
  for (const auto& pass : this->passes) {
    const auto pass_type = pass.second->getPassType();
    if (pass_type == PassType::Fuse || pass_type == PassType::Nop) {
      names.push_back(pass.first);
    }
  }
  return names;
}

} // namespace optimization
} // namespace ONNX_NAMESPACE
