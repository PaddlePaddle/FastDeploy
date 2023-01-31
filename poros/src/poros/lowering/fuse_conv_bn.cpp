// Copyright (c) 2022 Baidu, Inc.  All Rights Reserved.
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

/**
* @file fuse_conv_bn.cpp
* @author Lin Xiao Chun (linxiaochun@baidu.com)
* @date 2022-03-31 16:11:19
* @brief
**/

#include "poros/lowering/fuse_conv_bn.h"

#include "poros/util/poros_util.h"

namespace baidu {
namespace mirana {
namespace poros {

struct TORCH_API ConvBNParameters {
    at::Tensor conv_w;
    at::Tensor conv_b;
    at::Tensor bn_rm;
    at::Tensor bn_rv;
    double bn_eps = 0.0;
    at::Tensor bn_w;
    at::Tensor bn_b;
};

// calculate weights and bias
std::tuple<at::Tensor, at::Tensor> CalcFusedConvWeightAndBias(
        const ConvBNParameters &p) {
    at::Tensor bn_var_rsqrt = at::rsqrt(p.bn_rv + p.bn_eps);
    const int64_t ndim = p.conv_w.dim();
    at::DimVector sizes(ndim, 1);
    sizes.at(0) = -1;
    at::Tensor new_w = p.conv_w * (p.bn_w * bn_var_rsqrt).reshape(sizes);
    at::Tensor new_b = (p.conv_b - p.bn_rm) * bn_var_rsqrt * p.bn_w + p.bn_b;
    return std::make_tuple(new_w, new_b);
}

bool FuseConvBatchNorm::fuse(std::shared_ptr<torch::jit::Graph> graph) {
    graph_ = graph;
    return try_to_fuse_conv_batchnorm(graph_->block());
}

FuseConvBatchNorm::FuseConvBatchNorm() = default;

bool FuseConvBatchNorm::try_to_fuse_conv_batchnorm(torch::jit::Block *block) {
    bool graph_changed = false;
    auto it = block->nodes().begin();
    while (it != block->nodes().end()) {
        auto node = *it;
        ++it;  //++it first, node may be destroyed later。
        for (auto sub_block: node->blocks()) {
            if (try_to_fuse_conv_batchnorm(sub_block)) {
                graph_changed = true;
            }
        }
        //只处理 aten::conv + batch_norm的场景
        if (node->kind() != torch::jit::aten::batch_norm) {
            continue;
        }

        auto all_users = node->inputs()[0]->uses();
        if (all_users.size() != 1 || ((node->inputs()[0])->node()->kind() != torch::jit::aten::conv1d &&
                (node->inputs()[0])->node()->kind() != torch::jit::aten::conv2d &&
                (node->inputs()[0])->node()->kind() != torch::jit::aten::conv3d && 
                (node->inputs()[0])->node()->kind() != torch::jit::aten::_convolution)) {
            continue;
        }

        auto bn = node;
        auto conv = (node->inputs()[0])->node();

        // More parameters need to be checked when node is aten::_convolution.
        if (conv->schema().operator_name() == torch::jit::parseSchema("aten::_convolution(Tensor input, Tensor weight, "
        "Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, "
        "bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32) -> Tensor ").operator_name()) {
            bool transposed = toIValue(conv->input(6)).value().toBool();
            // deconvolution is not supported.
            if (transposed) {
                LOG(INFO) << "It is found that the transposed of aten::_convolution is true, which is not support to fuse conv+bn currently.";
                continue;
            }
            // output_padding is not supported.
            std::vector<int64_t> output_padding = toIValue(conv->input(7)).value().toIntVector();
            for (int64_t o : output_padding) {
                if (o != 0) {
                    LOG(INFO) << "It is found that the output_padding of aten::_convolution is not equal to zero, "
                    "which is not support to fuse conv+bn currently.";
                    continue;
                }
            }
            // other parameters like benchmark, deterministic, cudnn_enabled and allow_tf do not need to be checked for now.
        }

        ConvBNParameters params;
        // conv weights and bias
        if (!(conv->inputs()[1])->type()->isSubtypeOf(c10::TensorType::get()) || //conv_weight
            // !(conv->inputs()[2])->type()->isSubtypeOf(c10::TensorType::get()) || //conv_bias (maybe is None)
            !(bn->inputs()[1])->type()->isSubtypeOf(c10::TensorType::get()) ||   //bn_weight
            !(bn->inputs()[2])->type()->isSubtypeOf(c10::TensorType::get()) ||   //bn_bias
            !(bn->inputs()[3])->type()->isSubtypeOf(c10::TensorType::get()) ||   //bn_mean
            !(bn->inputs()[4])->type()->isSubtypeOf(c10::TensorType::get()) ||   //bn_var
            !(bn->inputs()[7])->type()->isSubtypeOf(c10::FloatType::get())) {    //bn_esp(default=1e-5)
            continue;
        }

        // record the fusing ops for debug
        record_transform(conv, bn)->to(conv);

        params.conv_w = toIValue(conv->inputs()[1]).value().toTensor();
        if (toIValue(conv->inputs()[2]).value().isNone()) {
            params.conv_b = torch::zeros({params.conv_w.size(0)}, {params.conv_w.device()}).to(params.conv_w.type());
        } else {
            params.conv_b = toIValue(conv->inputs()[2]).value().toTensor();
        }
        params.bn_w = toIValue(bn->inputs()[1]).value().toTensor();
        params.bn_b = toIValue(bn->inputs()[2]).value().toTensor();
        params.bn_rm = toIValue(bn->inputs()[3]).value().toTensor();
        params.bn_rv = toIValue(bn->inputs()[4]).value().toTensor();
        params.bn_eps = toIValue(bn->inputs()[7]).value().toDouble();

        // calc new weights and bias
        auto w_b = CalcFusedConvWeightAndBias(params);

        at::Tensor weights = std::get<0>(w_b);
        at::Tensor bias = std::get<1>(w_b);

        torch::jit::WithInsertPoint guard(graph_->block()->nodes().front());
        auto conv_w = graph_->insertConstant(weights);
        auto conv_b = graph_->insertConstant(bias);
        conv_w->node()->moveBefore(conv);
        conv_b->node()->moveBefore(conv);

        conv->replaceInput(1, conv_w);
        conv->replaceInput(2, conv_b);

        bn->output()->replaceAllUsesWith(conv->output());
        bn->removeAllInputs();
        bn->destroy();

        graph_changed = true;
    }
    return graph_changed;
}

REGISTER_OP_FUSER(FuseConvBatchNorm)

}
}
}// namespace