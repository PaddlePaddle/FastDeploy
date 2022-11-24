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
* @file fuse_hard_swish_test.cpp
* @author Lin Xiao Chun (linxiaochun@baidu.com)
* @date 2022-04-07 15:31:03
* @brief
**/

#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "poros/lowering/fuse_hard_swish.h"
#include "poros/lowering/op_fuse_pass.h"
#include "poros/util/graph_test_helper.h"

static void fuse_test_helper(const std::string &graph_IR,
                             std::shared_ptr<baidu::mirana::poros::IFuser> fuser,
                             std::vector<int64_t> input_shape
) {
    std::vector<at::IValue> input_data;
    input_data.push_back(at::randn(input_shape, {at::kCPU}));

    const std::vector<baidu::mirana::poros::graphtester::InputTypeEnum> input_data_type_mask = {
            baidu::mirana::poros::graphtester::InputTensor,
    };

    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
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

static std::string gen_hardswish_graph() {

    std::string hardsiwsh = R"IR(
    graph(%x):
        %out: Tensor = aten::hardswish(%x)
        return (%out))IR";
    return hardsiwsh;
}

TEST(Fusers, ATenFuseHardSwish_Test) {
    const auto graph_IR = gen_hardswish_graph();
    auto fuser = std::make_shared<baidu::mirana::poros::FuseHardSwish>();

    fuse_test_helper(graph_IR, fuser, {2, 3, 4, 5});
    fuse_test_helper(graph_IR, fuser, {3, 4, 5});
    fuse_test_helper(graph_IR, fuser, {4, 5});
    fuse_test_helper(graph_IR, fuser, {5});
}

