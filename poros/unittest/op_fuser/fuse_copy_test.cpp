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
* @file fuse_copy_test.cpp
* @author tianjinjin@baidu.com
* @date Mon Aug 22 10:47:14 CST 2022
* @brief
**/

#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "poros/lowering/fuse_copy.h"
#include "poros/lowering/op_fuse_pass.h"
#include "poros/util/graph_test_helper.h"

static void fuse_test_helper(const std::string &graph_IR,
                             std::shared_ptr<baidu::mirana::poros::IFuser> fuser,
                             std::vector<int64_t> input_shape,
                             bool with_single_value
) {
    std::vector<at::IValue> input_data;
    input_data.push_back(at::randn(input_shape, {at::kCPU}));
    std::vector<baidu::mirana::poros::graphtester::InputTypeEnum> input_data_type_mask = {
            baidu::mirana::poros::graphtester::InputTensor,
    };

    if (with_single_value) {
        input_data.push_back(at::randn({1}, {at::kCPU}));
        input_data_type_mask.push_back(baidu::mirana::poros::graphtester::ConstantTensor);
    }

    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    // poros_option.debug = true;
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> fused_output;
    ASSERT_TRUE(baidu::mirana::poros::graphtester::run_graph_and_fused_graph(graph_IR, poros_option, fuser,
                                                                             input_data, input_data_type_mask,
                                                                             graph_output, fused_output));

    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, fused_output.size());
    ASSERT_TRUE(baidu::mirana::poros::graphtester::almost_equal(graph_output[0], fused_output[0], 1e-6));
}

/**
 * this IR is generated from python code below:
def shift(x, n_segment, fold_div=3, inplace=False):  
    nt, c, h, w = x.size()
    n_batch = nt // n_segment
    x = x.view(n_batch, n_segment, c, h, w)
    
    fold = c // fold_div
    
    out = torch.zeros_like(x)
    out[:, :-1, :fold] = x[:, 1:, :fold]
    return out.view(nt, c, h, w)
 * **/
static std::string gen_simple_slice_graph() {
    std::string graph = R"IR(
      graph(%x : Tensor):
        %none : NoneType = prim::Constant()
        %0 : int = prim::Constant[value=0]()
        %1 : int = prim::Constant[value=1]()
        %2 : int = prim::Constant[value=2]()
        %3 : int = prim::Constant[value=-1]()
        %8 : int = prim::Constant[value=8]()
        %16 : int = prim::Constant[value=16]()
        %false : bool = prim::Constant[value=0]()
        %fold : int = prim::Constant[value=21]()

        %292 : int[] = aten::size(%x)
        %nt.3 : int, %c.3 : int, %h.3 : int, %w.3 : int = prim::ListUnpack(%292)
        %n_batch.3 : int = aten::floordiv(%nt.3, %16)
        %298 : int[] = prim::ListConstruct(%n_batch.3, %16, %c.3, %h.3, %w.3)
        %x1.7 : Tensor = aten::view(%x, %298)

        %out : Tensor = aten::zeros_like(%x1.7, %none, %none, %none, %none, %none) # temporal_shift.py:12:18
        
        %302 : Tensor = aten::slice(%x1.7, %0, %none, %none, %1) # temporal_shift.py:13:33
        %303 : Tensor = aten::slice(%302, %1, %1, %none, %1) # temporal_shift.py:13:33
        %304 : Tensor = aten::slice(%303, %2, %none, %fold, %1) # temporal_shift.py:13:33
   
        %305 : Tensor = aten::slice(%out, %0, %none, %none, %1) # temporal_shift.py:13:12
        %306 : Tensor = aten::slice(%305, %1, %none, %3, %1) # temporal_shift.py:13:12
        %307 : Tensor = aten::slice(%306, %2, %none, %fold, %1) # temporal_shift.py:13:12
        %308 : Tensor = aten::copy_(%307, %304, %false) # temporal_shift.py:13:12

        %322 : int[] = prim::ListConstruct(%nt.3, %c.3, %h.3, %w.3)
        %x0.7 : Tensor = aten::view(%out, %322)
        return (%x0.7))IR";
    return graph;
}

/**
 * this IR is generated from python code below:
def shift(x, n_segment, fold_div=3, inplace=False):  
    nt, c, h, w = x.size()
    n_batch = nt // n_segment
    x = x.view(n_batch, n_segment, c, h, w)
    
    fold = c // fold_div
    
    out = torch.zeros_like(x)
    out[:, :-1, :fold] = x[:, 1:, :fold]  
    out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]
    out[:, :, 2 * fold:] = x[:, :, 2 * fold:]
    return out.view(nt, c, h, w)
 * **/
static std::string gen_complex_slice_graph() {
    std::string graph = R"IR(
      graph(%x : Tensor):
        %none : NoneType = prim::Constant()
        %0 : int = prim::Constant[value=0]()
        %1 : int = prim::Constant[value=1]()
        %2 : int = prim::Constant[value=2]()
        %3 : int = prim::Constant[value=-1]()
        %8 : int = prim::Constant[value=8]()
        %16 : int = prim::Constant[value=16]()
        %false : bool = prim::Constant[value=0]()
        %fold : int = prim::Constant[value=21]()

        %292 : int[] = aten::size(%x)
        %nt.3 : int, %c.3 : int, %h.3 : int, %w.3 : int = prim::ListUnpack(%292)
        %n_batch.3 : int = aten::floordiv(%nt.3, %16)
        %298 : int[] = prim::ListConstruct(%n_batch.3, %16, %c.3, %h.3, %w.3)
        %x1.7 : Tensor = aten::view(%x, %298)

        %out : Tensor = aten::zeros_like(%x1.7, %none, %none, %none, %none, %none) # temporal_shift.py:12:18
        
        %302 : Tensor = aten::slice(%x1.7, %0, %none, %none, %1) # temporal_shift.py:13:33
        %303 : Tensor = aten::slice(%302, %1, %1, %none, %1) # temporal_shift.py:13:33
        %304 : Tensor = aten::slice(%303, %2, %none, %fold, %1) # temporal_shift.py:13:33
   
        %305 : Tensor = aten::slice(%out, %0, %none, %none, %1) # temporal_shift.py:13:12
        %306 : Tensor = aten::slice(%305, %1, %none, %3, %1) # temporal_shift.py:13:12
        %307 : Tensor = aten::slice(%306, %2, %none, %fold, %1) # temporal_shift.py:13:12
        %308 : Tensor = aten::copy_(%307, %304, %false) # temporal_shift.py:13:12

        %309 : Tensor = aten::slice(%302, %1, %none, %3, %1) # temporal_shift.py:14:41
        %310 : int = aten::mul(%2, %fold) # temporal_shift.py:14:57
        %311 : Tensor = aten::slice(%309, %2, %fold, %310, %1) # temporal_shift.py:14:41
        
        %312 : Tensor = aten::slice(%out, %0, %none, %none, %1) # temporal_shift.py:14:12
        %313 : Tensor = aten::slice(%312, %1, %1, %none, %1) # temporal_shift.py:14:12
        %314 : Tensor = aten::slice(%313, %2, %fold, %310, %1) # temporal_shift.py:14:12
        %315 : Tensor = aten::copy_(%314, %311, %false) # temporal_shift.py:14:12
        
        %316 : Tensor = aten::slice(%302, %1, %none, %none, %1) # temporal_shift.py:15:35
        %317 : Tensor = aten::slice(%316, %2, %310, %none, %1) # temporal_shift.py:15:35
        
        %318 : Tensor = aten::slice(%out, %0, %none, %none, %1) # temporal_shift.py:15:12
        %319 : Tensor = aten::slice(%318, %1, %none, %none, %1) # temporal_shift.py:15:12
        %320 : Tensor = aten::slice(%319, %2, %310, %none, %1) # temporal_shift.py:15:12
        %321 : Tensor = aten::copy_(%320, %317, %false) # temporal_shift.py:15:12

        %322 : int[] = prim::ListConstruct(%nt.3, %c.3, %h.3, %w.3)
        %x0.7 : Tensor = aten::view(%out, %322)
        return (%x0.7))IR";
    return graph;
}

/**
 * this IR is generated from python code below:
 * class SliceTest(torch.nn.Module):
    def __init__(self):
        super(SliceTest, self).__init__()

    def forward(self, x):
        size = x.size()
        #resize = size[:-1]
        attention_mask = torch.zeros(size)
        attention_mask[2:3:1, 2, :, 0, :] = 1
        out = attention_mask * 3
        return out
 * **/
static std::string gen_select_graph_with_single_value() {
    std::string graph = R"IR(
      graph(%x.1 : Tensor, %value : Tensor):
        %33 : bool = prim::Constant[value=0]()
        %5 : NoneType = prim::Constant()
        %10 : int = prim::Constant[value=1]() # ../../test.py:11:44
        %12 : int = prim::Constant[value=2]() # ../../test.py:11:30
        %13 : int = prim::Constant[value=0]() # ../../test.py:11:36
        %15 : int = prim::Constant[value=3]() # ../../test.py:11:25
        %size.1 : int[] = aten::size(%x.1) # ../../test.py:8:15
        %attention_mask.1 : Tensor = aten::zeros(%size.1, %5, %5, %5, %5) # ../../test.py:10:25
        %16 : Tensor = aten::slice(%attention_mask.1, %13, %12, %15, %10) # ../../test.py:11:8
        %18 : Tensor = aten::select(%16, %10, %12) # ../../test.py:11:8
        %23 : Tensor = aten::slice(%18, %10, %5, %5, %10) # ../../test.py:11:8
        %25 : Tensor = aten::select(%23, %12, %13) # ../../test.py:11:8
        %30 : Tensor = aten::slice(%25, %12, %5, %5, %10) # ../../test.py:11:8
        %36 : Tensor = aten::copy_(%30, %value, %33) # ../../test.py:11:8
        %out.1 : Tensor = aten::mul(%attention_mask.1, %15) # ../../test.py:12:14
        return (%out.1))IR";
    return graph;
}

/**
 * this IR is generated from python code below:
 * class SliceTest(torch.nn.Module):
    def __init__(self):
        super(SliceTest, self).__init__()

    def forward(self, x):
        size = x.size()
        #resize = size[:-1]
        attention_mask = torch.zeros(size)
        attention_mask[0, 2, 1:4:1, 0, :] = 1
        out = attention_mask * 3
        return out
 * **/
static std::string gen_select_graph_with_single_value2() {
    std::string graph = R"IR(
      graph(%x.1 : Tensor, %value : Tensor):
        %30 : bool = prim::Constant[value=0]()
        %5 : NoneType = prim::Constant()
        %10 : int = prim::Constant[value=1]() # ../../test.py:11:44
        %12 : int = prim::Constant[value=0]() # ../../test.py:11:23
        %13 : int = prim::Constant[value=2]() # ../../test.py:11:26
        %19 : int = prim::Constant[value=4]() # ../../test.py:11:31
        %35 : int = prim::Constant[value=3]() # ../../test.py:12:31
        %size.1 : int[] = aten::size(%x.1) # ../../test.py:8:15
        %attention_mask.1 : Tensor = aten::zeros(%size.1, %5, %5, %5, %5) # ../../test.py:10:25
        %15 : Tensor = aten::select(%attention_mask.1, %12, %12) # ../../test.py:11:8
        %17 : Tensor = aten::select(%15, %12, %13) # ../../test.py:11:8
        %20 : Tensor = aten::slice(%17, %12, %10, %19, %10) # ../../test.py:11:8
        %22 : Tensor = aten::select(%20, %10, %12) # ../../test.py:11:8
        %27 : Tensor = aten::slice(%22, %10, %5, %5, %10) # ../../test.py:11:8
        %33 : Tensor = aten::copy_(%27, %value, %30) # ../../test.py:11:8
        %out.1 : Tensor = aten::mul(%attention_mask.1, %35) # ../../test.py:12:14
        return (%out.1))IR";
    return graph;
}

/**
 * this IR is generated from python code below:
class ClipBoxes(torch.nn.Module):
    def __init__(self):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes):
        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        return boxes * 2
 * **/
static std::string gen_select_with_tensor_value() {
    std::string graph = R"IR(
        graph(%boxes.1 : Tensor):
            %31 : bool = prim::Constant[value=0]()
            %6 : NoneType = prim::Constant()
            %5 : int = prim::Constant[value=1]() # test.py:54:37
            %3 : int = prim::Constant[value=0]() # test.py:54:49
            %34 : int = prim::Constant[value=2]() # test.py:60:23
            %8 : Tensor = aten::slice(%boxes.1, %3, %6, %6, %5) # test.py:54:37
            %13 : Tensor = aten::slice(%8, %5, %6, %6, %5) # test.py:54:37
            %15 : Tensor = aten::select(%13, %34, %3) # test.py:54:37
            %17 : Tensor = aten::clamp(%15, %3, %6) # test.py:54:25
            %23 : Tensor = aten::slice(%boxes.1, %3, %6, %6, %5) # test.py:54:8
            %28 : Tensor = aten::slice(%23, %5, %6, %6, %5) # test.py:54:8
            %30 : Tensor = aten::select(%28, %34, %3) # test.py:54:8
            %32 : Tensor = aten::copy_(%30, %17, %31) # test.py:54:8
            %35 : Tensor = aten::mul(%boxes.1, %34) # test.py:60:15
            return (%35))IR";
    return graph;
}

/**
 * this IR is generated from python code below:
class ClipBoxes(torch.nn.Module):
    def __init__(self):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes):
        boxes[:, 0, :, :] = torch.clamp(boxes[:, 0, :, :], min=0)
        return boxes * 2
 * **/
static std::string gen_select_with_tensor_value2() {
    std::string graph = R"IR(
        graph(%boxes.1 : Tensor):
            %41 : bool = prim::Constant[value=0]()
            %6 : NoneType = prim::Constant()
            %5 : int = prim::Constant[value=1]() # test.py:46:40
            %3 : int = prim::Constant[value=0]() # test.py:46:49
            %44 : int = prim::Constant[value=2]() # test.py:47:23
            %8 : Tensor = aten::slice(%boxes.1, %3, %6, %6, %5) # test.py:46:40
            %10 : Tensor = aten::select(%8, %5, %3) # test.py:46:40
            %15 : Tensor = aten::slice(%10, %5, %6, %6, %5) # test.py:46:40
            %20 : Tensor = aten::slice(%15, %44, %6, %6, %5) # test.py:46:40
            %22 : Tensor = aten::clamp(%20, %3, %6) # test.py:46:28
            %28 : Tensor = aten::slice(%boxes.1, %3, %6, %6, %5) # test.py:46:8
            %30 : Tensor = aten::select(%28, %5, %3) # test.py:46:8
            %35 : Tensor = aten::slice(%30, %5, %6, %6, %5) # test.py:46:8
            %40 : Tensor = aten::slice(%35, %44, %6, %6, %5) # test.py:46:8
            %42 : Tensor = aten::copy_(%40, %22, %41) # test.py:46:8
            %45 : Tensor = aten::mul(%boxes.1, %44) # test.py:47:15
            return (%45))IR";
    return graph;
}

/**
 * this IR is generated from python code below:
class ClipBoxes(torch.nn.Module):
    def __init__(self):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes):
        boxes[:, 0, :, 1] = torch.clamp(boxes[:, 0, :, 1], min=0)
        return boxes * 2
 * **/
static std::string gen_select_with_tensor_value3() {
    std::string graph = R"IR(
        graph(%boxes.1 : Tensor):
            %36 : bool = prim::Constant[value=0]()
            %7 : NoneType = prim::Constant()
            %3 : int = prim::Constant[value=0]() # test.py:50:49
            %4 : int = prim::Constant[value=1]() # test.py:50:55
            %39 : int = prim::Constant[value=2]() # test.py:51:23
            %9 : Tensor = aten::slice(%boxes.1, %3, %7, %7, %4) # test.py:50:40
            %11 : Tensor = aten::select(%9, %4, %3) # test.py:50:40
            %16 : Tensor = aten::slice(%11, %4, %7, %7, %4) # test.py:50:40
            %18 : Tensor = aten::select(%16, %39, %4) # test.py:50:40
            %20 : Tensor = aten::clamp(%18, %3, %7) # test.py:50:28
            %26 : Tensor = aten::slice(%boxes.1, %3, %7, %7, %4) # test.py:50:8
            %28 : Tensor = aten::select(%26, %4, %3) # test.py:50:8
            %33 : Tensor = aten::slice(%28, %4, %7, %7, %4) # test.py:50:8
            %35 : Tensor = aten::select(%33, %39, %4) # test.py:50:8
            %37 : Tensor = aten::copy_(%35, %20, %36) # test.py:50:8
            %40 : Tensor = aten::mul(%boxes.1, %39) # test.py:51:15
            return (%40))IR";
    return graph;
}

TEST(Fusers, ATenFuseCopySliceTest) {
    auto fuser = std::make_shared<baidu::mirana::poros::FuseCopy>();
    //situation1: out[:, :-1, :fold] = x[:, 1:, :fold] 
    const auto slice_graph_simple = gen_simple_slice_graph();
    fuse_test_helper(slice_graph_simple, fuser, {16, 64, 16, 16}, false);
    //situation2:  multi-copy 
    const auto slice_graph = gen_complex_slice_graph();
    fuse_test_helper(slice_graph, fuser, {16, 64, 16, 16}, false);
}

TEST(Fusers, ATenFuseCopySelectWithSingleValueTest) {
    auto fuser = std::make_shared<baidu::mirana::poros::FuseCopy>();
    //situation1: attention_mask[2:3:1, 2, :, 0, :] = 1
    const auto select_graph_IR = gen_select_graph_with_single_value();
    fuse_test_helper(select_graph_IR, fuser, {4, 4, 5, 4, 3}, true);

    //situation2: attention_mask[0, 2, 1:4:1, 0, :] = 1
    const auto select_graph_IR2 = gen_select_graph_with_single_value2();
    fuse_test_helper(select_graph_IR2, fuser, {4, 4, 5, 4, 3}, true);
}

TEST(Fusers, ATenFuseCopySelectWithTensorValueTest) {
    auto fuser = std::make_shared<baidu::mirana::poros::FuseCopy>();
    //situation1:  boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
    const auto  select_graph_IR= gen_select_with_tensor_value();
    fuse_test_helper(select_graph_IR, fuser, {1, 20, 4}, false);

    //situation2: boxes[:, 0, :, :] = torch.clamp(boxes[:, 0, :, :], min=0)
    const auto select_graph_IR2 = gen_select_with_tensor_value2();
    fuse_test_helper(select_graph_IR2, fuser, {1, 20, 4, 5}, false);

    //situation3: boxes[:, 0, :, 1] = torch.clamp(boxes[:, 0, :, 1], min=0)
    const auto select_graph_IR3 = gen_select_with_tensor_value3();
    fuse_test_helper(select_graph_IR3, fuser, {1, 20, 4, 5}, false);
}