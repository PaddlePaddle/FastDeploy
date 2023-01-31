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
* @file: fuse_copy.h
* @author: tianjinjin@baidu.com
* @data: Mon Aug 22 11:33:45 CST 2022
* @brief: 
**/ 

#pragma once

#include "poros/lowering/op_fuse_pass.h"

namespace baidu {
namespace mirana {
namespace poros {

/***
 * torchscript 中有个重要的概念，是view，
 * 当slice这个op出现时（包括连续出现时），不会真正的执行内存的copy，而是通过view去尽可能的复用buffer
 * 直到出现copy_ 才会进行真正的buffer的拷贝。
 * 比如下面的graph，执行到最后，真正发生了变化的是 %out.4。
 * typical graph for copy_

   %none : NoneType = prim::Constant()
   %false : bool = prim::Constant[value=0]()
   %0 : int = prim::Constant[value=0]()
   %1 : int = prim::Constant[value=1]()
   %2 : int = prim::Constant[value=2]()
   %11 : int = prim::Constant[value=-1]()
   %out.4 : Float(*, 16, 64, 16, 16) = aten::zeros_like(%x, %none, %none, %none, %none, %none)
   %303 : Float(*, 16, 64, 16, 16) = aten::slice(%x, %0, %none, %none, %1) 
   %304 : Float(*, 15, 64, 16, 16) = aten::slice(%303, %1, %1, %none, %1) 
   %305 : Float(*, 15, 8, 16, 16) = aten::slice(%304, %2, %none, %y, %1) 

   %306 : Float(*, 16, 64, 16, 16) = aten::slice(%out.4, %0, %none, %none, %1) 
   %307 : Float(*, 15, 64, 16, 16) = aten::slice(%306, %1, %none, %11, %1) 
   %308 : Float(*, 15, 8, 16, 16) = aten::slice(%307, %2, %none, %y, %1) 
   %309 : Tensor = aten::copy_(%308, %305, %false) 

   %310 : Float(*, 15, 64, 16, 16) = aten::slice(%303, %1, %none, %11, %1)
   %311 : int = aten::mul(%2, %y) 
   %312 : Float(*, 15, 8, 16, 16) = aten::slice(%310, %2, %y, %311, %1) 
   %313 : Float(*, 16, 64, 16, 16) = aten::slice(%out.4, %0, %none, %none, %1) 
   %314 : Float(*, 15, 64, 16, 16) = aten::slice(%313, %1, %1, %none, %1) 
   %315 : Float(*, 15, 8, 16, 16) = aten::slice(%314, %2, %y, %311, %1) 
   %316 : Tensor = aten::copy_(%315, %312, %false) 

   %317 : Float(*, 16, 64, 16, 16) = aten::slice(%303, %1, %none, %none, %1) 
   %318 : Float(*, 16, 48, 16, 16) = aten::slice(%317, %2, %311, %none, %1) 
   %319 : Float(*, 16, 64, 16, 16) = aten::slice(%out.4, %0, %none, %none, %1) 
   %320 : Float(*, 16, 64, 16, 16) = aten::slice(%319, %1, %none, %none, %1) 
   %321 : Float(*, 16, 48, 16, 16) = aten::slice(%320, %2, %311, %none, %1) 
   %322 : Tensor = aten::copy_(%321, %318, %false)
   
   %323 : int[] = prim::ListConstruct(%nt.3, %c.3, %h.3, %w.3)
   %final : Float(*, 64, 16, 16) = aten::view(%out.4, %323)
 * 
 * the implementation of index_put:
 * aten/src/ATen/native/cuda/indexing.cu
 * https://github.com/pytorch/pytorch/blob/v1.9.0-rc1/aten/src/ATen/native/cuda/Indexing.cu#L209
 * ***/

struct ConvertedIndex {
    ConvertedIndex(torch::jit::Value* index, c10::Symbol orig_node_kind)
        : index(index), orig_node_kind(orig_node_kind) {}
      
    torch::jit::Value* index = nullptr;
    c10::Symbol orig_node_kind;
};

/**
 * 目前可以处理的场景包括:
 * 1. 纯slice的场景:  out[:, :-1, :3] = x[:, 1:, :3]
 * 2. slice + 单个select的场景(等号右侧为单值): out[2:3:1, :, :, 0, :] = 1
 * 3. slice + 多个select的场景(等号右侧为单值): out[2:3:1, 3, :, 0, :] = 1
 * 4. slice + 单个select的场景(等号右侧为tensor): boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
 * 5. sclie + 多个select的场景(等号右侧为tensor): boxes[:, 0, :, 1] = torch.clamp(boxes[:, 0, :, 1], min=0)
 * **/
class FuseCopy : public IFuser {
public:
    FuseCopy();

    bool fuse(std::shared_ptr<torch::jit::Graph> graph);

private:
    bool try_to_fuse_copy(torch::jit::Block *block);

    bool prepare_copy(torch::jit::Node* node);
    bool prepare_index_put(torch::jit::Node* index_put_node);

    torch::jit::Value* create_size_of_dim(torch::jit::Value* input,
                                        int64_t dim,
                                        torch::jit::Node* insertBefore);
    torch::jit::Value* convert_select_to_index(torch::jit::Value* index,
                                            torch::jit::Node* insertBefore);
    torch::jit::Value* convert_slice_to_index(torch::jit::Node* slice,
                                            torch::jit::Value* size,
                                            torch::jit::Node* insertBefore);

    std::vector<torch::jit::Node*> fetch_slice_and_select_pattern(const torch::jit::Node* node);

    std::unordered_map<int64_t, ConvertedIndex> merge_slice_and_select_to_indices(
                                            torch::jit::Graph* graph,
                                            torch::jit::Node* index_put_node,
                                            const std::vector<torch::jit::Node*>& slice_and_select_nodes,
                                            torch::jit::Value* orig_data);

    std::vector<torch::jit::Value*> reshape_to_advanced_indexing_format(
                                            torch::jit::Graph* graph,
                                            torch::jit::Node* index_put_node,
                                            std::unordered_map<int64_t, ConvertedIndex>& dim_index_map);

    void adjust_value(torch::jit::Graph* graph,
                    torch::jit::Node* index_put_node,
                    const std::vector<torch::jit::Node*>& slice_and_select_nodes,
                    torch::jit::Value* orig_data);

    std::shared_ptr<torch::jit::Graph> graph_;
};

}
}
}
