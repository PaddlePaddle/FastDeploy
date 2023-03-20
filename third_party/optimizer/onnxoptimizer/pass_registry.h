/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/common/ir.h"
#include "onnx/common/ir_pb_converter.h"
#include "onnx/common/stl_backports.h"
#include "onnx/proto_utils.h"

#include "onnxoptimizer/passes/eliminate_deadend.h"
#include "onnxoptimizer/passes/eliminate_duplicate_initializer.h"
#include "onnxoptimizer/passes/eliminate_identity.h"
#include "onnxoptimizer/passes/eliminate_if_with_const_cond.h"
#include "onnxoptimizer/passes/eliminate_nop_cast.h"
#include "onnxoptimizer/passes/eliminate_nop_dropout.h"
#include "onnxoptimizer/passes/eliminate_nop_flatten.h"
#include "onnxoptimizer/passes/eliminate_nop_monotone_argmax.h"
#include "onnxoptimizer/passes/eliminate_nop_pad.h"
#include "onnxoptimizer/passes/eliminate_nop_transpose.h"
#include "onnxoptimizer/passes/eliminate_unused_initializer.h"
#include "onnxoptimizer/passes/extract_constant_to_initializer.h"
#include "onnxoptimizer/passes/fuse_add_bias_into_conv.h"
#include "onnxoptimizer/passes/fuse_bn_into_conv.h"
#include "onnxoptimizer/passes/fuse_consecutive_concats.h"
#include "onnxoptimizer/passes/fuse_consecutive_log_softmax.h"
#include "onnxoptimizer/passes/fuse_consecutive_reduce_unsqueeze.h"
#include "onnxoptimizer/passes/fuse_consecutive_squeezes.h"
#include "onnxoptimizer/passes/fuse_consecutive_transposes.h"
#include "onnxoptimizer/passes/fuse_matmul_add_bias_into_gemm.h"
#include "onnxoptimizer/passes/fuse_pad_into_conv.h"
#include "onnxoptimizer/passes/fuse_transpose_into_gemm.h"
#include "onnxoptimizer/passes/lift_lexical_references.h"
#include "onnxoptimizer/passes/nop.h"
#include "onnxoptimizer/passes/split.h"

#include <unordered_set>
#include <vector>

namespace ONNX_NAMESPACE {
namespace optimization {

// Registry containing all passes available in ONNX.
struct GlobalPassRegistry {
  std::map<std::string, std::shared_ptr<Pass>> passes;

  GlobalPassRegistry() {
    // Register the optimization passes to the optimizer.
    registerPass<NopEmptyPass>();
    registerPass<EliminateDeadEnd>();
    registerPass<EliminateDuplicateInitializer>();
    registerPass<EliminateNopCast>();
    registerPass<EliminateNopDropout>();
    registerPass<EliminateNopFlatten>();
    registerPass<EliminateIdentity>();
    registerPass<EliminateIfWithConstCond>();
    registerPass<EliminateNopMonotoneArgmax>();
    registerPass<EliminateNopPad>();
    registerPass<EliminateNopTranspose>();
    registerPass<EliminateUnusedInitializer>();
    registerPass<ExtractConstantToInitializer>();
    registerPass<FuseAddBiasIntoConv>();
    registerPass<FuseBNIntoConv>();
    registerPass<FuseConsecutiveConcats>();
    registerPass<FuseConsecutiveLogSoftmax>();
    registerPass<FuseConsecutiveReduceUnsqueeze>();
    registerPass<FuseConsecutiveSqueezes>();
    registerPass<FuseConsecutiveTransposes>();
    registerPass<FuseMatMulAddBiasIntoGemm>();
    registerPass<FusePadIntoConv>();
    registerPass<FuseTransposeIntoGemm>();
    registerPass<LiftLexicalReferences>();
    registerPass<SplitInit>();
    registerPass<SplitPredict>();
  }

  ~GlobalPassRegistry() {
    this->passes.clear();
  }

  std::shared_ptr<Pass> find(std::string pass_name) {
    auto it = this->passes.find(pass_name);
    ONNX_ASSERTM(it != this->passes.end(), "pass %s is unknown.",
                 pass_name.c_str());
    return it->second;
  }
  const std::vector<std::string> GetAvailablePasses();

  const std::vector<std::string> GetFuseAndEliminationPass();

  template <typename T>
  void registerPass() {
    static_assert(std::is_base_of<Pass, T>::value, "T must inherit from Pass");
    std::shared_ptr<Pass> pass(new T());
    passes[pass->getPassName()] = pass;
  }
};
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
