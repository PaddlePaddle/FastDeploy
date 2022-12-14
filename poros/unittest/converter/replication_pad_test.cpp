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
* @file replication_pad_test.cpp
* @author tianshaoqing@baidu.com
* @date Wed Sep 27 11:24:21 CST 2021
* @brief 
**/
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "poros/converter/gpu/replication_pad.h"
#include "poros/util/test_util.h"

static void replicationpad_test_helper(const std::string& graph_IR,
                                    std::vector<int64_t> shape) {
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn(shape, {at::kCUDA}));
    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    baidu::mirana::poros::ReplicationPadConverter replicationpadconverter;
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, &replicationpadconverter, 
                input_data, graph_output, poros_output));
    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());
    // ASSERT_TRUE(baidu::mirana::poros::testutil::almost_equal(graph_output[0], poros_output[0], 2e-6));
    ASSERT_TRUE(graph_output[0].equal(poros_output[0]));
}

static std::string gen_replicationpad_graph(const std::string& op, 
                                            const std::string& padding) {
    return R"IR(
      graph(%0 : Tensor):
        %1 : int[] = prim::Constant[value=[)IR" + padding + R"IR(]]()
        %2 : Tensor = aten::)IR" + op + R"IR((%0, %1)
        return (%2))IR";
}

TEST(Converters, ATenReplicationPad1DConvertsCorrectly) {
    // aten::replication_pad1d(Tensor self, int[2] padding) -> Tensor
    const auto graph_IR = gen_replicationpad_graph("replication_pad1d", "2, 3");
    replicationpad_test_helper(graph_IR, {1, 3, 4});
}

TEST(Converters, ATenReplicationPad1DRightZeroConvertsCorrectly) {
    // aten::replication_pad1d(Tensor self, int[2] padding) -> Tensor
    const auto graph_IR = gen_replicationpad_graph("replication_pad1d", "2, 0");
    replicationpad_test_helper(graph_IR, {1, 3, 4});
}

TEST(Converters, ATenReplicationPad1DLeftZeroConvertsCorrectly) {
    // aten::replication_pad1d(Tensor self, int[2] padding) -> Tensor
    const auto graph_IR = gen_replicationpad_graph("replication_pad1d", "0, 3");
    replicationpad_test_helper(graph_IR, {1, 3, 4});
}

TEST(Converters, ATenReplicationPad2DConvertsCorrectly) {
    // aten::replication_pad2d(Tensor self, int[4] padding) -> Tensor
    const auto graph_IR = gen_replicationpad_graph("replication_pad2d", "2, 3, 2, 3");
    replicationpad_test_helper(graph_IR, {1, 3, 4, 5});
}

TEST(Converters, ATenReplicationPad2DBottomZeroConvertsCorrectly) {
    // aten::replication_pad2d(Tensor self, int[4] padding) -> Tensor
    const auto graph_IR = gen_replicationpad_graph("replication_pad2d", "2, 0, 2, 0");
    replicationpad_test_helper(graph_IR, {1, 3, 4, 5});
}

TEST(Converters, ATenReplicationPad2DTopZeroConvertsCorrectly) {
    // aten::replication_pad2d(Tensor self, int[4] padding) -> Tensor
    const auto graph_IR = gen_replicationpad_graph("replication_pad2d", "0, 3, 0, 3");
    replicationpad_test_helper(graph_IR, {1, 3, 4, 5});
}

TEST(Converters, ATenReplicationPad3DConvertsCorrectly) {
    // aten::replication_pad3d(Tensor self, int[6] padding) -> Tensor
    const auto graph_IR = gen_replicationpad_graph("replication_pad3d", "2, 3, 2, 3, 1, 4");
    replicationpad_test_helper(graph_IR, {1, 3, 4, 5, 3});
}

TEST(Converters, ATenReplicationPad3DRightBottomZeroConvertsCorrectly) {
    // aten::replication_pad3d(Tensor self, int[6] padding) -> Tensor
    const auto graph_IR = gen_replicationpad_graph("replication_pad3d", "2, 0, 2, 0, 1, 0");
    replicationpad_test_helper(graph_IR, {1, 3, 4, 5, 3});
}

TEST(Converters, ATenReplicationPad3DLeftTopZeroConvertsCorrectly) {
    // aten::replication_pad3d(Tensor self, int[6] padding) -> Tensor
    const auto graph_IR = gen_replicationpad_graph("replication_pad3d", "2, 0, 2, 0, 1, 0");
    replicationpad_test_helper(graph_IR, {1, 3, 4, 5, 3});
}