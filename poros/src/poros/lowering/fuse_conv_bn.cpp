/*******************************************************************************

 Copyright (c) 2022 Baidu, Inc.  All Rights Reserved.

 *******************************************************************************

 @file fuse_conv_bn.cpp
 @author Lin Xiao Chun (linxiaochun@baidu.com)
 @date 2022-03-31 16:11:19
 @brief
 */

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
                (node->inputs()[0])->node()->kind() != torch::jit::aten::conv3d)) {
            continue;
        }

        auto bn = node;
        auto conv = (node->inputs()[0])->node();

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