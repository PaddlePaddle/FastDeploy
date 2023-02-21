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
* @file lowering_pass.h
* @author Lin Xiao Chun (linxiaochun@baidu.com)
* @date 2022-03-31 16:11:18
* @brief
**/
#pragma once

#include <vector>

#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/ir/ir.h>

namespace baidu {
namespace mirana {
namespace poros {

/**
* @brief 删除graph中，纯粹的prim::RaiseException分支。
*       （否则graph会被分割成过多的block）
**/
void eliminate_exception_pass(std::shared_ptr<torch::jit::Graph> graph);

/**
* @brief 替换graph中，prim::ListConstruct 类型的节点后面，紧跟 prim::ListUnpack 类型的情况。
*        prim::ListConstruct 用来将多个元素构建成list
*        prim::ListUnpack 用来将一个list打散成多个元素。
*        当这两个节点处理同一个list，且节点间没有其他可能改变该list的情况时，将这两个节点抵消。
**/
void eliminate_some_list(std::shared_ptr<torch::jit::Graph> graph);

/**
* @brief 替换graph中，prim::DictConstruct 类型的节点后面，跟的全部是 aten::__getitem__ 类型的情况（且dict的key是常量）。
*        prim::DictConstruct 用来将多个元素构建成dict
*        aten::__getitem__  用来从list 或者 dict 中获取元素。
*        当DictConstruct生成的dict，只被aten::__getitem__ 调用，且没有其他可能改变该dict的情况时，将这两类op抵消。
**/
void eliminate_some_dict(std::shared_ptr<torch::jit::Graph> graph);

/**
* @brief 删除graph中，未被使用的aten::copy_节点。
*
**/
void eliminate_useless_copy(std::shared_ptr<torch::jit::Graph> graph);


/**
 * @brief 尝试用maxpool 代替 maxpool_with_indeces.
 * 以 maxpoll2d 为例：
 * maxpoll2d_with_indices 的schema为：aten::max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)
 * 而 maxpoll 的schema为：aten::max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor
 * 这两个op，输入参数完全一致，输出上，max_pool2d_with_indices有两个输出，第一个输出与max_pool2d的输出完全一致，第二个输出为indeces信息。
 * 当 max_pool2d_with_indices 的第二个输出indices，后续没有其他op使用该value的时候，
 * 我们直接用max_pool2d 替代 max_pool2d_with_indices。
 **/
void eliminate_maxpool_with_indices(std::shared_ptr<torch::jit::Graph> graph);

/**
* @brief 将符合条件的loop进行循环展开，避免过多的block，影响子图分割的逻辑。
*        本function很大程度上参考了jit原生的UnrollLoop的实现，
*        考虑到原生的UnrollLoop支持的bodysize和loopcount不符合poros的预期，且原生实现不提供修改参数的接口。
*        故重新实现该function, 调整loop展开的条件和部分细节。
**/
void unrolling_loop(std::shared_ptr<torch::jit::Graph> graph);

/**
* @brief 替换graph中，aten::std 的算子为 aten::var + aten::sqrt。
*        依据：标准差(aten::std) = 方差(aten::var) 的算术平方根(aten::sqrt)
**/
void unpack_std(std::shared_ptr<torch::jit::Graph>& graph);

/**
* @brief 替换graph中，aten::var 的算子为 aten::mul + aten::mean 等。
*        参考pytorch-1.9.0 中该算子的实现: https://github.com/pytorch/pytorch/blob/v1.9.0/aten/src/ATen/native/ReduceOps.cpp#L1380
**/
void unpack_var(std::shared_ptr<torch::jit::Graph>& graph);

/**
* @brief 尝试将aten::percentFormat 的结果变成常量。
*        背景： aten::percentFormat 的功能主要是用于字符串的组装，
*              且常常配合 prim::If 这个op进行字符串的比较，实现分支选择。
*        当precentFormat 的输入都是常量的时候，尝试直接计算出这个算子的结果，替换成常量
*        进一步配合prim::If 的条件判断是否为常量，最终配合达到删除不必要的分支的目的。
**/
void freeze_percentformat(std::shared_ptr<torch::jit::Graph> graph);

/**
* @brief 尝试固定aten::size的结果。需要配合后续aten::size的输出的使用进行判断。
*        注意： 本function必须在预热数据处理完整个graph之后再使用，且依赖于预热数据覆盖的全面程度。
**/
void freeze_aten_size(std::shared_ptr<torch::jit::Graph> graph);

/**
* @brief 尝试固定aten::len的结果。需要配合后续aten::len的输出的使用进行判断。
*        注意： 本function必须在预热数据处理完整个graph之后再使用，且依赖于预热数据覆盖的全面程度。
**/
void freeze_aten_len(std::shared_ptr<torch::jit::Graph> graph);

/**
* @brief 尝试固定aten::dim的结果。需要配合后续aten::dim的输出的使用进行判断。
*        注意： 本function必须在预热数据处理完整个graph之后再使用，且依赖于预热数据覆盖的全面程度。
**/
void freeze_aten_dim(std::shared_ptr<torch::jit::Graph> graph);

/**
* @brief 当遇到使用ListConstruct对1个constant进行append时，可以讲output直接替换为1个constant
*  例如：float 替换为 (float, float,..)
**/
void freeze_list_construct(std::shared_ptr<torch::jit::Graph> graph);

/**
* @brief 针对graph的简单类型输入【bool or int 类型】，尝试进行剪枝。
*        当多轮预热数据的简单类型输入保持不变，则认为该输入可以用常量进行替代。
*        注意： 本function必须在预热数据处理完整个graph之后再使用，且依赖于预热数据覆盖的全面程度。
**/
void input_param_propagate(std::shared_ptr<torch::jit::Graph> graph,
                        std::vector<std::vector<c10::IValue>>& stack_vec);

/**
* @brief 移除graph中，跟踪bool类型的prim::profile节点。
*        这些prim::profile在数据预热阶段(IvalueAnalysis)添加进graph，数据预热完成后，需要相应移除这些节点。
**/
void remove_simple_type_profile_nodes(std::shared_ptr<torch::jit::Graph> graph);

/**
* @brief 使用log(softmax())替代log_softmax()
**/
void replace_log_softmax(std::shared_ptr<torch::jit::Graph> graph);

/**
* @brief 使用log(sigmoid())替代log_sigmoid()
**/
void replace_log_sigmoid(std::shared_ptr<torch::jit::Graph> graph);

/**
* @brief 将包含list类型引用语义的op输入与输出串联起来。
**/
void link_mutable_list(std::shared_ptr<torch::jit::Graph> graph);

/**
* @brief 直接删除图中与infer无关的节点。
**/
void eliminate_simple_useless_nodes(std::shared_ptr<torch::jit::Graph> graph);

/**
 * @brief 删除子图内部或输入相关的无用节点。（当前支持aten::to.device，aten::contiguous，aten::dropout和aten::detach）
 *        注意：1、删除子图内部节点（is_input == false）必须在拷贝的子图上，否则fallback会出错。 
 *             2、删除（替换）子图输入节点（is_input == true）必须在子图转engine成功后。
 *
 * @param [in] subgraph : 要删无用节点的子图
 * @param [in] subgraph_node : subgraph对应的子图节点，类型必须是prim::CudaFusionGroup
 * @param [in] is_input : true表示要删除的是子图输入的节点，false表示删除子图内部节点
 * 
 * @return bool
 * @retval true => 删除节点成功  false => 如果删完无用节点后的子图node数量为0，返回false unmerge
**/
bool eliminate_subgraph_useless_nodes(std::shared_ptr<torch::jit::Graph> subgraph, 
                            torch::jit::Node& subgraph_node, 
                            const bool is_input);

/**
* @brief 检查并替换有问题的constant
**/
void replace_illegal_constant(std::shared_ptr<torch::jit::Graph> graph);

/**
* @brief 替换aten::pad。该op可视做多种pad的集合，只是用mode来设置pad方式（包括：constant、reflect、replicate还有circular）。
*        mode == constant时，可替换为aten::constant_pad_nd，已实现。
*        todo:
*        mode == refect时，替换为aten::reflection_pad
*        mode == replicate时，替换为aten::replication_pad
**/
void replace_pad(std::shared_ptr<torch::jit::Graph> graph);
}  // namespace poros 
}  // namespace mirana
}  // namespace baidu