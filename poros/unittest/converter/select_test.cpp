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
* @file select_test.cpp
* @author tianshaoqing@baidu.com
* @date Wed Sep 27 11:24:21 CST 2021
* @brief 
**/
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "poros/converter/gpu/select.h"
#include "poros/util/test_util.h"

static void select_test_helper(const std::string& graph_IR, 
                                baidu::mirana::poros::IConverter* converter,
                                std::vector<int64_t> shape,
                                bool is_dynamic = false,
                                std::vector<std::vector<at::Tensor>>* prewarm_data = nullptr) {
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn(shape, {at::kCUDA}));

    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    poros_option.is_dynamic = is_dynamic;
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, converter, 
                input_data, graph_output, poros_output, prewarm_data));

    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());
    ASSERT_TRUE(graph_output[0].equal(poros_output[0]));
}

static void split_test_helper(const std::string& graph_IR, 
                                baidu::mirana::poros::IConverter* converter,
                                std::vector<int64_t> shape,
                                const int64_t& output_size) {
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn(shape, {at::kCUDA}));
    
    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, converter, 
                input_data, graph_output, poros_output));

    ASSERT_EQ(output_size, graph_output.size());
    ASSERT_EQ(output_size, poros_output.size());

    for (int64_t i = 0; i < output_size; i++) {
        ASSERT_TRUE(graph_output[i].equal(poros_output[i]));
    }
}

static void embedding_test_helper(const std::string& graph_IR, 
                                baidu::mirana::poros::IConverter* converter) {

    std::vector<at::Tensor> input_data;
    auto options_pyt = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kInt64);
    auto weight = at::randn({10, 4}, {at::kCUDA});
    auto input = at::tensor({2, 3, 4}, options_pyt);
    input_data.push_back(weight);
    input_data.push_back(input);
                            
    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, converter, 
                input_data, graph_output, poros_output));
    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());

    ASSERT_TRUE(baidu::mirana::poros::testutil::almost_equal(graph_output[0], poros_output[0], 2e-6));
}

static std::string gen_select_graph(const std::string& dim, const std::string& index) {
    return R"IR(
        graph(%0 : Tensor):
          %1 : int = prim::Constant[value=)IR" + dim + R"IR(]()
          %2 : int = prim::Constant[value=)IR" + index + R"IR(]()
          %3 : Tensor = aten::select(%0, %1, %2)
          return (%3))IR";
}

static std::string gen_slice_graph(const std::string& dim, 
                                const std::string& start,
                                const std::string& end,
                                const std::string& step) {
    std::string start_ir, end_ir;
    if (start.empty()) {
        start_ir = "%2 : None = prim::Constant()";
    } else {
        start_ir = "%2 : int = prim::Constant[value=" + start + "]()";
    }
    if (end.empty()) {
        end_ir = "%3 : None = prim::Constant()";
    } else {
        end_ir = "%3 : int = prim::Constant[value=" + end + "]()";
    }
    return R"IR(
        graph(%0 : Tensor):
          %1 : int = prim::Constant[value=)IR" + dim + R"IR(]()
          )IR" + start_ir + R"IR(
          )IR" + end_ir + R"IR(
          %4 : int = prim::Constant[value=)IR" + step + R"IR(]()
          %5 : Tensor = aten::slice(%0, %1, %2, %3, %4)
          return (%5))IR";
}

static std::string gen_narrow_graph(const std::string& dim, 
                                    const std::string& start,
                                    const std::string& length,
                                    bool singleinput) {
    if (singleinput) {
        return R"IR(
            graph(%0 : Tensor):
              %1 : int = prim::Constant[value=)IR" + dim + R"IR(]()
              %2 : int = prim::Constant[value=)IR" + start + R"IR(]()
              %3 : int = prim::Constant[value=)IR" + length + R"IR(]()
              %4 : Tensor = aten::narrow(%0, %1, %2, %3)
              return (%4))IR";
    } else {
        return R"IR(
            graph(%0 : Tensor, %1 : Tensor):
              %2 : int = prim::Constant[value=)IR" + dim + R"IR(]()
              %3 : int = prim::Constant[value=)IR" + length + R"IR(]()
              %4 : Tensor = aten::narrow(%0, %2, %1, %3)
              return (%4))IR";
    }
}

static std::string gen_indexput_graph(const std::string& fold) {
    return  R"IR(
    graph(%x : Tensor):
        %none : NoneType = prim::Constant()
        %0 : int = prim::Constant[value=0]()
        %1 : int = prim::Constant[value=1]()
        %2 : int = prim::Constant[value=2]()
        %4 : int = prim::Constant[value=4]()
        %negtive : int = prim::Constant[value=-1]()
        %fold : int = prim::Constant[value=)IR" + fold + R"IR(]()
        %false : bool = prim::Constant[value=0]()

        %out : Tensor = aten::zeros_like(%x, %none, %none, %none, %none, %none)
        %302 : Tensor = aten::slice(%x, %0, %none, %none, %1)
        %303 : Tensor = aten::slice(%302, %1, %1, %none, %1)
        %304 : Tensor = aten::slice(%303, %2, %none, %fold, %1)

        %2726 : int = aten::size(%out, %0)
        %2731 : Tensor = aten::arange(%2726, %4, %none, %none, %none)
        %2733 : Tensor = aten::slice(%2731, %0, %none, %none, %1)

        %2735 : int = aten::size(%out, %1)
        %2740 : Tensor = aten::arange(%2735, %4, %none, %none, %none)
        %2742 : Tensor = aten::slice(%2740, %0, %none, %negtive, %1)

        %2744 : int = aten::size(%out, %2)
        %2749 : Tensor = aten::arange(%2744, %4, %none, %none, %none)
        %2751 : Tensor = aten::slice(%2749, %0, %none, %fold, %1)

        %2752 : int[] = prim::Constant[value=[-1, 1, 1]]()
        %2753 : Tensor = aten::view(%2733, %2752)
        %2754 : int[] = prim::Constant[value=[-1, 1]]()
        %2755 : Tensor = aten::view(%2742, %2754)
        %2756 : Tensor?[] = prim::ListConstruct(%2753, %2755, %2751)
        %2757 : Tensor = aten::index_put(%out, %2756, %304, %false)
        return (%2757))IR";
}

static std::string gen_indexput_with_singular_value_graph() {
    return R"IR(
      graph(%x : Tensor):
        %false : bool = prim::Constant[value=0]()
        %none  : NoneType = prim::Constant()
        %neg1 : int = prim::Constant[value=-1]()
        %0 : int = prim::Constant[value=0]()
        %1 : int = prim::Constant[value=1]()
        %4 : int = prim::Constant[value=4]()
        %device : Device = prim::Constant[value="cuda:0"]()

        %size : int[] = aten::size(%x)
        %input_shape : int[] = aten::slice(%size, %none, %neg1, %1)
        %attention_mask : Tensor = aten::zeros(%input_shape, %none, %none, %device, %none)
        %92 : int = aten::size(%attention_mask, %1)
        %90 : Tensor = aten::arange(%92, %4, %none, %none, %none)
        %86 : Tensor = aten::slice(%90, %0, %none, %none, %1)
        %2326 : int = prim::dtype(%86)
        %101 : Tensor = aten::tensor(%0, %2326, %device, %false)

        %index : Tensor?[] = prim::ListConstruct(%101, %86)

        %28 : int = prim::dtype(%attention_mask)
        %value : Tensor = aten::tensor(%1, %28, %device, %false)
        %tmp : Tensor = aten::index_put(%attention_mask, %index, %value, %false)
        %out : Tensor = aten::mul(%tmp, %4)
        return (%out))IR";
}

TEST(Converters, ATenSelectIntConvertsCorrectly) {
    // aten::select.int(Tensor(a) self, int dim, int index) -> Tensor(a)
    const auto graph_IR = gen_select_graph("0", "0");
    baidu::mirana::poros::SelectConverter selectconverter;
    select_test_helper(graph_IR, &selectconverter, {4, 4, 4});
}

TEST(Converters, ATenSelectIntDimIsOneConvertsCorrectly) {
    // aten::select.int(Tensor(a) self, int dim, int index) -> Tensor(a)
    const auto graph_IR = gen_select_graph("1", "0");
    baidu::mirana::poros::SelectConverter selectconverter;
    select_test_helper(graph_IR, &selectconverter, {4, 4, 4});
}

TEST(Converters, ATenSelectIntDimNegativeConvertsCorrectly) {
    // aten::select.int(Tensor(a) self, int dim, int index) -> Tensor(a)
    const auto graph_IR = gen_select_graph("-2", "0");
    baidu::mirana::poros::SelectConverter selectconverter;
    select_test_helper(graph_IR, &selectconverter, {4, 4, 4});
}

TEST(Converters, ATenSelectIntNegIndexConvertsCorrectly) {
    // aten::select.int(Tensor(a) self, int dim, int index) -> Tensor(a)
    const auto graph_IR = gen_select_graph("0", "-1");
    baidu::mirana::poros::SelectConverter selectconverter;
    select_test_helper(graph_IR, &selectconverter, {4, 4, 4});
}

TEST(Converters, ATenSelectSelfDynaimcConverterCorrectly) {
    // aten::select.int(Tensor(a) self, int dim, int index) -> Tensor(a)
    const auto graph_IR = gen_select_graph("3", "0");
    baidu::mirana::poros::SelectConverter selectconverter;

    std::vector<at::Tensor> input_data;
    auto options_pyt = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kInt);
    input_data.push_back(at::randint(0, 100, {3, 4, 5, 6}, options_pyt)); // indices
    
    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randint(0, 3, {3, 4, 5, 6}, options_pyt)); // indices
    prewarm_data[1].push_back(at::randint(0, 3, {2, 3, 4, 5}, options_pyt)); // indices
    prewarm_data[2].push_back(at::randint(0, 3, {2, 3, 4, 5}, options_pyt)); // indices

    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    poros_option.is_dynamic = true;
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, &selectconverter, 
                input_data, graph_output, poros_output, &prewarm_data));

    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());
    ASSERT_TRUE(graph_output[0].equal(poros_output[0]));
}

TEST(Converters, ATenSliceConvertsCorrectly) {
    // aten::slice.Tensor(Tensor(a) self, int dim=0, int? start=None, int? end=None, int step=1) -> Tensor(a)
    const auto graph_IR = gen_slice_graph("2", "0", "2", "1");
    baidu::mirana::poros::SliceConverter sliceconverter;
    select_test_helper(graph_IR, &sliceconverter, {3, 4, 5, 6});
}

TEST(Converters, ATenSliceDimNegConvertsCorrectly) {
    // aten::slice.Tensor(Tensor(a) self, int dim=0, int? start=None, int? end=None, int step=1) -> Tensor(a)
    const auto graph_IR = gen_slice_graph("-2", "0", "2", "1");
    
    baidu::mirana::poros::SliceConverter sliceconverter;
    select_test_helper(graph_IR, &sliceconverter, {3, 4, 5, 6});
}

TEST(Converters, ATenSliceStartNoneConvertsCorrectly) {
    // aten::slice.Tensor(Tensor(a) self, int dim=0, int? start=None, int? end=None, int step=1) -> Tensor(a)
    const auto graph_IR = gen_slice_graph("2", "", "3", "1");
    baidu::mirana::poros::SliceConverter sliceconverter;
    select_test_helper(graph_IR, &sliceconverter, {3, 4, 5, 6});
}

TEST(Converters, ATenSliceStartNegConvertsCorrectly) {
    // aten::slice.Tensor(Tensor(a) self, int dim=0, int? start=None, int? end=None, int step=1) -> Tensor(a)
    const auto graph_IR = gen_slice_graph("2", "-2", "3", "1");
    baidu::mirana::poros::SliceConverter sliceconverter;
    select_test_helper(graph_IR, &sliceconverter, {3, 4, 5, 6});
}

TEST(Converters, ATenSliceEndNoneConvertsCorrectly) {
    // aten::slice.Tensor(Tensor(a) self, int dim=0, int? start=None, int? end=None, int step=1) -> Tensor(a)
    const auto graph_IR = gen_slice_graph("2", "1", "", "1");
    baidu::mirana::poros::SliceConverter sliceconverter;
    select_test_helper(graph_IR, &sliceconverter, {3, 4, 5, 6});
}

TEST(Converters, ATenSliceEndNegConvertsCorrectly) {
    // aten::slice.Tensor(Tensor(a) self, int dim=0, int? start=None, int? end=None, int step=1) -> Tensor(a)
    const auto graph_IR = gen_slice_graph("2", "0", "-2", "2");
    baidu::mirana::poros::SliceConverter sliceconverter;
    select_test_helper(graph_IR, &sliceconverter, {3, 4, 5, 6});
}

TEST(Converters, ATenSliceStartEndNegConvertsCorrectly) {
    // aten::slice.Tensor(Tensor(a) self, int dim=0, int? start=None, int? end=None, int step=1) -> Tensor(a)
    const auto graph_IR = gen_slice_graph("2", "-3", "-1", "2");
    baidu::mirana::poros::SliceConverter sliceconverter;
    select_test_helper(graph_IR, &sliceconverter, {3, 4, 5, 6});
}

TEST(Converters, ATenSliceStartEndNoneConvertsCorrectly) {
    // aten::slice.Tensor(Tensor(a) self, int dim=0, int? start=None, int? end=None, int step=1) -> Tensor(a)
    const auto graph_IR = gen_slice_graph("2", "", "", "2");
    baidu::mirana::poros::SliceConverter sliceconverter;
    select_test_helper(graph_IR, &sliceconverter, {3, 4, 5, 6});
}

TEST(Converters, ATenSliceStepConvertsCorrectly) {
    // aten::slice.Tensor(Tensor(a) self, int dim=0, int? start=None, int? end=None, int step=1) -> Tensor(a)
    const auto graph_IR = gen_slice_graph("2", "0", "3", "2");
    baidu::mirana::poros::SliceConverter sliceconverter;
    select_test_helper(graph_IR, &sliceconverter, {3, 4, 5, 6});
}

TEST(Converters, ATenSliceResnetTestConvertsCorrectly) {
    // aten::slice.Tensor(Tensor(a) self, int dim=0, int? start=None, int? end=None, int step=1) -> Tensor(a)
    const auto graph_IR = gen_slice_graph("1", "0", "8", "1");
    baidu::mirana::poros::SliceConverter sliceconverter;
    select_test_helper(graph_IR, &sliceconverter, {1, 8, 256, 56, 56});
}

TEST(Converters, ATenSliceDimDynamicTestConvertsCorrectly) {
    // aten::slice.Tensor(Tensor(a) self, int dim=0, int? start=None, int? end=None, int step=1) -> Tensor(a)
    const auto graph_IR = gen_slice_graph("0", "2", "8", "1");
    baidu::mirana::poros::SliceConverter sliceconverter;

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randn({20, 16, 32}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({5, 16, 32}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({10, 16, 32}, {at::kCUDA}));

    select_test_helper(graph_IR, &sliceconverter, {10, 16, 32}, true, &prewarm_data);
}

TEST(Converters, ATenSliceDimDynamicStartEndBothNegTestConvertsCorrectly) {
    // aten::slice.Tensor(Tensor(a) self, int dim=0, int? start=None, int? end=None, int step=1) -> Tensor(a)
    const auto graph_IR = gen_slice_graph("0", "-5", "-1", "1");
    baidu::mirana::poros::SliceConverter sliceconverter;

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randn({20, 16, 32}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({5, 16, 32}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({10, 16, 32}, {at::kCUDA}));

    select_test_helper(graph_IR, &sliceconverter, {10, 16, 32}, true, &prewarm_data);
}

TEST(Converters, ATenSliceDimDynamicStartEndBothNoneTestConvertsCorrectly) {
    // aten::slice.Tensor(Tensor(a) self, int dim=0, int? start=None, int? end=None, int step=1) -> Tensor(a)
    const auto graph_IR = gen_slice_graph("0", "", "", "1");
    baidu::mirana::poros::SliceConverter sliceconverter;

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randn({20, 16, 32}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({5, 16, 32}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({10, 16, 32}, {at::kCUDA}));

    select_test_helper(graph_IR, &sliceconverter, {10, 16, 32}, true, &prewarm_data);
}

TEST(Converters, ATenSliceDimDynamicTestStepConvertsCorrectly) {
    // aten::slice.Tensor(Tensor(a) self, int dim=0, int? start=None, int? end=None, int step=1) -> Tensor(a)
    const auto graph_IR = gen_slice_graph("0", "-8", "-1", "2");
    baidu::mirana::poros::SliceConverter sliceconverter;

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randn({20, 16, 32}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({5, 16, 32}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({10, 16, 32}, {at::kCUDA}));

    select_test_helper(graph_IR, &sliceconverter, {10, 16, 32}, true, &prewarm_data);
}

TEST(Converters, ATenSliceDimNotDynamicTestConvertsCorrectly) {
    // aten::slice.Tensor(Tensor(a) self, int dim=0, int? start=None, int? end=None, int step=1) -> Tensor(a)
    const auto graph_IR = gen_slice_graph("1", "10", "16", "1");
    baidu::mirana::poros::SliceConverter sliceconverter;

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randn({20, 16, 32}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({5, 16, 32}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({10, 16, 32}, {at::kCUDA}));

    select_test_helper(graph_IR, &sliceconverter, {10, 16, 32}, true, &prewarm_data);
}

TEST(Converters, ATenSliceDimNotDynamicStartEndBothNegTestConvertsCorrectly) {
    // aten::slice.Tensor(Tensor(a) self, int dim=0, int? start=None, int? end=None, int step=1) -> Tensor(a)
    const auto graph_IR = gen_slice_graph("1", "-10", "-5", "1");
    baidu::mirana::poros::SliceConverter sliceconverter;

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randn({20, 16, 32}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({5, 16, 32}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({10, 16, 32}, {at::kCUDA}));

    select_test_helper(graph_IR, &sliceconverter, {10, 16, 32}, true, &prewarm_data);
}

TEST(Converters, ATenSliceDimNotDynamicStartEndBothNoneTestConvertsCorrectly) {
    // aten::slice.Tensor(Tensor(a) self, int dim=0, int? start=None, int? end=None, int step=1) -> Tensor(a)
    const auto graph_IR = gen_slice_graph("1", "", "", "1");
    baidu::mirana::poros::SliceConverter sliceconverter;

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randn({20, 16, 32}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({5, 16, 32}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({10, 16, 32}, {at::kCUDA}));

    select_test_helper(graph_IR, &sliceconverter, {10, 16, 32}, true, &prewarm_data);
}

TEST(Converters, ATenSliceDimNotDynamicTestStepConvertsCorrectly) {
    // aten::slice.Tensor(Tensor(a) self, int dim=0, int? start=None, int? end=None, int step=1) -> Tensor(a)
    const auto graph_IR = gen_slice_graph("1", "-10", "-5", "2");
    baidu::mirana::poros::SliceConverter sliceconverter;

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randn({20, 16, 32}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({5, 16, 32}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({10, 16, 32}, {at::kCUDA}));

    select_test_helper(graph_IR, &sliceconverter, {10, 16, 32}, true, &prewarm_data);
}

TEST(Converters, ATenSliceTStartEndBothNoneDynamicConvertsCorrectly) {
    // aten::slice.t(t[] l, int? start=None, int? end=None, int step=1) -> (t[])
    const auto graph_IR = R"IR(
        graph(%0 : Tensor):
          %1 : int[] = aten::size(%0)
          %2 : None = prim::Constant()
          %3 : Device = prim::Constant[value="cuda"]()
          %4 : int = prim::Constant[value=6]()
          %5 : int = prim::Constant[value=1]()
          %6 : int[] = aten::slice(%1, %2, %2, %5)
          %7 : Tensor = aten::ones(%6, %4, %2, %3, %2)
          return (%7))IR";
    baidu::mirana::poros::SliceConverter sliceconverter;

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randn({5, 6, 7, 8}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({4, 5, 6, 7}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({4, 5, 6, 7}, {at::kCUDA}));

    select_test_helper(graph_IR, &sliceconverter, {4, 5, 6, 7}, true, &prewarm_data);
}

TEST(Converters, ATenSliceTStartEndDynamicConvertsCorrectly) {
    // aten::slice.t(t[] l, int? start=None, int? end=None, int step=1) -> (t[])
    const auto graph_IR = R"IR(
        graph(%0 : Tensor):
          %start : int = prim::Constant[value=1]()
          %end : int = prim::Constant[value=3]()
          %1 : int = prim::Constant[value=1]()
          %2 : None = prim::Constant()
          %3 : int[] = aten::size(%0)
          %4 : int[] = aten::slice(%3, %start, %end, %1)
          %5 : Device = prim::Constant[value="cuda"]()
          %6 : int = prim::Constant[value=6]()
          %7 : Tensor = aten::ones(%4, %6, %2, %5, %2)
          return (%7))IR";
    baidu::mirana::poros::SliceConverter sliceconverter;

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randn({5, 6, 7, 8}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({4, 5, 6, 7}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({4, 5, 6, 7}, {at::kCUDA}));

    select_test_helper(graph_IR, &sliceconverter, {4, 5, 6, 7}, true, &prewarm_data);
}

TEST(Converters, ATenSliceTStartEndNegDynamicConvertsCorrectly) {
    // aten::slice.t(t[] l, int? start=None, int? end=None, int step=1) -> (t[])
    const auto graph_IR = R"IR(
        graph(%0 : Tensor):
          %start : int = prim::Constant[value=-3]()
          %end : int = prim::Constant[value=-1]()
          %1 : int = prim::Constant[value=1]()
          %2 : None = prim::Constant()
          %3 : int[] = aten::size(%0)
          %4 : int[] = aten::slice(%3, %start, %end, %1)
          %5 : Device = prim::Constant[value="cuda"]()
          %6 : int = prim::Constant[value=6]()
          %7 : Tensor = aten::ones(%4, %6, %2, %5, %2)
          return (%7))IR";
    baidu::mirana::poros::SliceConverter sliceconverter;

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randn({5, 6, 7, 8}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({4, 5, 6, 7}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({4, 5, 6, 7}, {at::kCUDA}));

    select_test_helper(graph_IR, &sliceconverter, {4, 5, 6, 7}, true, &prewarm_data);
}

TEST(Converters, ATenSliceTStartEndStepDynamicConvertsCorrectly) {
    // aten::slice.t(t[] l, int? start=None, int? end=None, int step=1) -> (t[])
    const auto graph_IR = R"IR(
        graph(%0 : Tensor):
          %start : int = prim::Constant[value=0]()
          %end : int = prim::Constant[value=3]()
          %step : int = prim::Constant[value=2]()
          %1 : int = prim::Constant[value=1]()
          %2 : None = prim::Constant()
          %3 : int[] = aten::size(%0)
          %4 : int[] = aten::slice(%3, %start, %end, %step)
          %5 : Device = prim::Constant[value="cuda"]()
          %6 : int = prim::Constant[value=6]()
          %7 : Tensor = aten::ones(%4, %6, %2, %5, %2)
          return (%7))IR";
    baidu::mirana::poros::SliceConverter sliceconverter;

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randn({5, 6, 7, 8}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({4, 5, 6, 7}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({4, 5, 6, 7}, {at::kCUDA}));

    select_test_helper(graph_IR, &sliceconverter, {4, 5, 6, 7}, true, &prewarm_data);
}

TEST(Converters, ATenSliceFromSizeStartDynamicConvertsCorrectly) {
    // aten::slice.Tensor(Tensor(a) self, int dim=0, int? start=None, int? end=None, int step=1) -> Tensor(a)
    const auto graph_IR = R"IR(
        graph(%0 : Tensor):
          %1 : int = prim::Constant[value=1]()
          %2 : int = aten::size(%0, %1)
          %3 : int = prim::Constant[value=3]()
          %4 : int = aten::floordiv(%2, %3)
          %end : None = prim::Constant()
          %step : int = prim::Constant[value=1]()
          %5 : Tensor = aten::slice(%0, %1, %4, %end, %step)
          return (%5))IR";
    baidu::mirana::poros::SliceConverter sliceconverter;

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randn({5, 10, 7, 8}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({4, 5, 6, 7}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({4, 5, 6, 7}, {at::kCUDA}));

    select_test_helper(graph_IR, &sliceconverter, {4, 5, 6, 7}, true, &prewarm_data);
}

TEST(Converters, ATenSliceFromSizeEndDynamicConvertsCorrectly) {
    // aten::slice.Tensor(Tensor(a) self, int dim=0, int? start=None, int? end=None, int step=1) -> Tensor(a)
    const auto graph_IR = R"IR(
        graph(%0 : Tensor):
          %1 : int = prim::Constant[value=1]()
          %2 : int = aten::size(%0, %1)
          %3 : int = prim::Constant[value=3]()
          %4 : int = aten::floordiv(%2, %3)
          %start : None = prim::Constant()
          %step : int = prim::Constant[value=1]()
          %5 : Tensor = aten::slice(%0, %1, %start, %4, %step)
          return (%5))IR";
    baidu::mirana::poros::SliceConverter sliceconverter;

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randn({5, 10, 7, 8}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({4, 5, 6, 7}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({4, 5, 6, 7}, {at::kCUDA}));

    select_test_helper(graph_IR, &sliceconverter, {4, 5, 6, 7}, true, &prewarm_data);
}

TEST(Converters, ATenSliceFromSizeStartEndDynamicConvertsCorrectly) {
    // aten::slice.Tensor(Tensor(a) self, int dim=0, int? start=None, int? end=None, int step=1) -> Tensor(a)
    const auto graph_IR = R"IR(
        graph(%0 : Tensor):
          %1 : int = prim::Constant[value=1]()
          %2 : int = aten::size(%0, %1)
          %3 : int = prim::Constant[value=2]()
          %4 : int = prim::Constant[value=5]()
          %5 : int = aten::floordiv(%2, %3)
          %6 : int = aten::floordiv(%2, %4)
          %step : int = prim::Constant[value=1]()
          %5 : Tensor = aten::slice(%0, %1, %6, %5, %step)
          return (%5))IR";
    baidu::mirana::poros::SliceConverter sliceconverter;

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randn({5, 10, 7, 8}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randn({4, 5, 6, 7}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randn({4, 5, 6, 7}, {at::kCUDA}));

    select_test_helper(graph_IR, &sliceconverter, {4, 5, 6, 7}, true, &prewarm_data);
}

TEST(Converters, ATenNarrowScalarConvertsCorrectly) {
    // aten::narrow(Tensor(a) self, int dim, int start, int length) -> Tensor(a)
    const auto graph_IR = gen_narrow_graph("2", "0", "2", true);
    baidu::mirana::poros::NarrowConverter narrowconverter;
    select_test_helper(graph_IR, &narrowconverter, {4, 4, 4, 4});
}

TEST(Converters, ATenNarrowScalarNegtiveStartConvertsCorrectly) {
    // aten::narrow(Tensor(a) self, int dim, int start, int length) -> Tensor(a)
    const auto graph_IR = gen_narrow_graph("2", "-3", "2", true);
    baidu::mirana::poros::NarrowConverter narrowconverter;
    select_test_helper(graph_IR, &narrowconverter, {4, 4, 4, 4});
}

TEST(Converters, ATenNarrowScalarNegtiveDimConvertsCorrectly) {
    // aten::narrow(Tensor(a) self, int dim, int start, int length) -> Tensor(a)
    const auto graph_IR = gen_narrow_graph("-2", "0", "2", true);
    baidu::mirana::poros::NarrowConverter narrowconverter;
    select_test_helper(graph_IR, &narrowconverter, {4, 4, 4, 4});
}

TEST(Converters, ATenNarrowScalarNegtiveDimStartConvertsCorrectly) {
    // aten::narrow(Tensor(a) self, int dim, int start, int length) -> Tensor(a)
    const auto graph_IR = gen_narrow_graph("-2", "-3", "2", true);
    baidu::mirana::poros::NarrowConverter narrowconverter;
    select_test_helper(graph_IR, &narrowconverter, {4, 4, 4, 4});
}

TEST(Converters, ATenSplitFixedTensorsConvertsCorrectly) {
    // aten::split.Tensor(Tensor(a) self, int split_size, int dim=0) -> Tensor(a)[]
    const auto graph_IR = R"IR(
        graph(%1 : Tensor):
          %2 : int = prim::Constant[value=3]()
          %3 : int = prim::Constant[value=0]()
          %4 : Tensor[] = aten::split(%1, %2, %3)
          %5 : Tensor, %6 : Tensor = prim::ListUnpack(%4)
          return (%5, %6))IR";
    baidu::mirana::poros::SplitConverter splitconverter;
    split_test_helper(graph_IR, &splitconverter, {6, 4, 3, 1}, 2);
}

TEST(Converters, ATenSplitUnfixedTensorsConvertsCorrectly) {
    // aten::split.Tensor(Tensor(a) self, int split_size, int dim=0) -> Tensor(a)[]
    const auto graph_IR = R"IR(
        graph(%1 : Tensor):
          %2 : int = prim::Constant[value=2]()
          %3 : int = prim::Constant[value=1]()
          %4 : Tensor[] = aten::split(%1, %2, %3)
          %5 : Tensor, %6 : Tensor, %7 : Tensor = prim::ListUnpack(%4)
          return (%5, %6, %7))IR";
    baidu::mirana::poros::SplitConverter splitconverter;
    split_test_helper(graph_IR, &splitconverter, {4, 5, 3, 1}, 3);
}

TEST(Converters, ATenSplitWithSizeDoubleTensorsConvertsCorrectly) {
    // aten::split_with_sizes(Tensor(a) self, int[] split_sizes, int dim=0) -> Tensor(a)[]
    const auto graph_IR = R"IR(
        graph(%1 : Tensor):
          %2 : int[] = prim::Constant[value=[4, 2]]()
          %3 : int = prim::Constant[value=0]()
          %4 : Tensor[] = aten::split_with_sizes(%1, %2, %3)
          %5 : Tensor, %6 : Tensor = prim::ListUnpack(%4)
          return (%5, %6))IR";
    baidu::mirana::poros::SplitConverter splitconverter;
    split_test_helper(graph_IR, &splitconverter, {6, 4, 3, 1}, 2);
}

TEST(Converters, ATenSplitWithSizeTrippleTensorsConvertsCorrectly) {
    // aten::split_with_sizes(Tensor(a) self, int[] split_sizes, int dim=0) -> Tensor(a)[]
    const auto graph_IR = R"IR(
        graph(%1 : Tensor):
          %2 : int[] = prim::Constant[value=[5, 1, 2]]()
          %3 : int = prim::Constant[value=1]()
          %4 : Tensor[] = aten::split_with_sizes(%1, %2, %3)
          %5 : Tensor, %6 : Tensor, %7 : Tensor = prim::ListUnpack(%4)
          return (%5, %6, %7))IR";
    baidu::mirana::poros::SplitConverter splitconverter;
    split_test_helper(graph_IR, &splitconverter, {2, 8, 3, 1}, 3);
}

TEST(Converters, ATenUnbindTensorsConvertsCorrectly) {
    // aten::unbind.int(Tensor(a) self, int dim=0) -> Tensor(a)[]
    const auto graph_IR = R"IR(
        graph(%1 : Tensor):
          %2 : int = prim::Constant[value=1]()
          %3 : Tensor[] = aten::unbind(%1, %2)
          %4 : Tensor, %5 : Tensor, %6 : Tensor = prim::ListUnpack(%3)
          return (%4, %5, %6))IR";
    baidu::mirana::poros::SplitConverter splitconverter;
    split_test_helper(graph_IR, &splitconverter, {2, 3, 4}, 3);
}

TEST(Converters, EmbeddingConverterCorrectly) {
    const auto graph_IR = R"IR(
        graph(%weight.1 : Tensor,
          %input.1 : Tensor):
          %7 : bool = prim::Constant[value=0]()
          %6 : int = prim::Constant[value=-1]()
          %9 : Tensor = aten::embedding(%weight.1, %input.1, %6, %7, %7)
          return (%9))IR";

    baidu::mirana::poros::EmbeddingConverter embeddingconverter;
    embedding_test_helper(graph_IR, &embeddingconverter);
}

static void gather_test_helper(const std::string& graph_IR, 
                                baidu::mirana::poros::IConverter* converter,
                                const std::vector<at::Tensor>& input_data,
                                bool is_dynamic = false,
                                std::vector<std::vector<at::Tensor>>* prewarm_data = nullptr) {
    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    poros_option.is_dynamic = is_dynamic;
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, converter, 
                input_data, graph_output, poros_output, prewarm_data));

    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());
    ASSERT_TRUE(graph_output[0].equal(poros_output[0]));
}

TEST(Converters, ATenGatherConverterCorrectly) {
    const auto graph_IR = R"IR(
        graph(%0 : Tensor, %1 : Tensor):
          %2 : bool = prim::Constant[value=0]()
          %3 : int = prim::Constant[value=1]()
          %4 : Tensor = aten::gather(%0, %3, %1, %2)
          return (%4))IR";

    std::vector<at::Tensor> input_data;
    auto input = at::randn({3, 4, 5, 6}, {at::kCUDA});
    auto options_pyt = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kInt64);
    auto index = at::randint(0, 2, {3, 4, 5, 6}, options_pyt);
    
    input_data.push_back(input);
    input_data.push_back(index);

    baidu::mirana::poros::GatherConverter gatherconverter;
    gather_test_helper(graph_IR, &gatherconverter, input_data, false);
}

TEST(Converters, ATenGatherNegtiveDimConverterCorrectly) {
    const auto graph_IR = R"IR(
        graph(%0 : Tensor, %1 : Tensor):
          %2 : bool = prim::Constant[value=0]()
          %3 : int = prim::Constant[value=-1]()
          %4 : Tensor = aten::gather(%0, %3, %1, %2)
          return (%4))IR";

    std::vector<at::Tensor> input_data;
    auto input = at::randn({3, 4, 5, 6}, {at::kCUDA});
    auto options_pyt = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kInt64);
    auto index = at::randint(0, 2, {3, 4, 5, 6}, options_pyt);
    
    input_data.push_back(input);
    input_data.push_back(index);

    baidu::mirana::poros::GatherConverter gatherconverter;
    gather_test_helper(graph_IR, &gatherconverter, input_data, false);
}

TEST(Converters,  ATenGatherDynamicConverterCorrectly) {
    const auto graph_IR = R"IR(
        graph(%0 : Tensor, %1 : Tensor):
          %2 : bool = prim::Constant[value=0]()
          %3 : int = prim::Constant[value=-1]()
          %4 : Tensor = aten::gather(%0, %3, %1, %2)
          return (%4))IR";
    
    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};

    auto options_pyt = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kInt64);
    // max
    prewarm_data[0].push_back(at::randn({4, 5, 6, 7}, {at::kCUDA}));
    prewarm_data[0].push_back(at::randint(0, 3, {4, 5, 6, 7}, options_pyt));
    // min
    prewarm_data[1].push_back(at::randn({3, 4, 5, 6}, {at::kCUDA}));
    prewarm_data[1].push_back(at::randint(0, 2, {3, 4, 5, 6}, options_pyt));
    // opt
    prewarm_data[2].push_back(at::randn({3, 4, 5, 6}, {at::kCUDA}));
    prewarm_data[2].push_back(at::randint(0, 2, {3, 4, 5, 6}, options_pyt));

    std::vector<at::Tensor> input_data;
    auto input = at::randn({3, 4, 5, 6}, {at::kCUDA});
    auto index = at::randint(0, 2, {3, 4, 5, 6}, options_pyt);
    
    input_data.push_back(input);
    input_data.push_back(index);

    baidu::mirana::poros::GatherConverter gatherconverter;
    gather_test_helper(graph_IR, &gatherconverter, input_data, true, &prewarm_data);
}

TEST(Converters,  ATenMaskedfillScalarValueConverterCorrectly) {
// aten::masked_fill.Scalar(Tensor self, Tensor mask, Scalar value) -> Tensor
    const auto graph_IR = R"IR(
      graph(%0 : Tensor, %1 : Tensor):
        %2 : int = prim::Constant[value=-1]()
        %3 : Tensor = aten::masked_fill(%0, %1, %2)
        return (%3))IR";

    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({2, 2}, {at::kCUDA}));
    auto options_pyt = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kBool);
    input_data.push_back(torch::tensor({false, true, false, true}, options_pyt).reshape({2, 2}));

    baidu::mirana::poros::MaskedFillConverter maskedfillconverter;

    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    poros_option.is_dynamic = false;
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, &maskedfillconverter, 
                input_data, graph_output, poros_output));

    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());
    ASSERT_TRUE(graph_output[0].equal(poros_output[0]));        
}

TEST(Converters,  ATenMaskedfillScalarValueDynamicConverterCorrectly) {
// aten::masked_fill.Scalar(Tensor self, Tensor mask, Scalar value) -> Tensor
    const auto graph_IR = R"IR(
      graph(%0 : Tensor, %1 : Tensor):
        %2 : int = prim::Constant[value=-1]()
        %3 : Tensor = aten::masked_fill(%0, %1, %2)
        return (%3))IR";

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};

    auto options_pyt = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kBool);
    // max
    prewarm_data[0].push_back(at::randn({4, 2}, {at::kCUDA}));
    prewarm_data[0].push_back(torch::tensor({false, true, false, true, false, true, false, true}, options_pyt).reshape({4, 2}));
    // min
    prewarm_data[1].push_back(at::randn({1, 2}, {at::kCUDA}));
    prewarm_data[1].push_back(torch::tensor({false, true}, options_pyt).reshape({1, 2}));
    // opt
    prewarm_data[2].push_back(at::randn({2, 2}, {at::kCUDA}));
    prewarm_data[2].push_back(torch::tensor({false, true, false, true}, options_pyt).reshape({2, 2}));

    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({2, 2}, {at::kCUDA}));
    input_data.push_back(torch::tensor({false, true, false, true}, options_pyt).reshape({2, 2}));

    baidu::mirana::poros::MaskedFillConverter maskedfillconverter;

    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    poros_option.is_dynamic = true;
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, &maskedfillconverter,
                input_data, graph_output, poros_output, &prewarm_data));

    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());
    ASSERT_TRUE(graph_output[0].equal(poros_output[0]));
}

TEST(Converters,  ATenMaskedfillScalarValueDynamicMoreConverterCorrectly) {
// aten::masked_fill.Scalar(Tensor self, Tensor mask, Scalar value) -> Tensor
    const auto graph_IR = R"IR(
      graph(%0 : Tensor, %1 : Tensor):
        %2 : int = prim::Constant[value=-1]()
        %3 : Tensor = aten::masked_fill(%0, %1, %2)
        return (%3))IR";

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};

    auto options_pyt = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kBool);
    // max
    prewarm_data[0].push_back(at::randn({4, 2}, {at::kCUDA}));
    prewarm_data[0].push_back(torch::tensor({false, true}, options_pyt).reshape({2}));
    // min
    prewarm_data[1].push_back(at::randn({1, 2}, {at::kCUDA}));
    prewarm_data[1].push_back(torch::tensor({false, true}, options_pyt).reshape({2}));
    // opt
    prewarm_data[2].push_back(at::randn({2, 2}, {at::kCUDA}));
    prewarm_data[2].push_back(torch::tensor({true, true}, options_pyt).reshape({2}));

    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({2, 2}, {at::kCUDA}));
    input_data.push_back(torch::tensor({false, true}, options_pyt).reshape({2}));

    baidu::mirana::poros::MaskedFillConverter maskedfillconverter;

    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    poros_option.is_dynamic = true;
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, &maskedfillconverter,
                input_data, graph_output, poros_output, &prewarm_data));

    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());
    ASSERT_TRUE(graph_output[0].equal(poros_output[0]));
}

TEST(Converters,  ATenMaskedfillTensorValueConverterCorrectly) {
// aten::masked_fill.Tensor(Tensor self, Tensor mask, Tensor value) -> Tensor
    const auto graph_IR = R"IR(
      graph(%0 : Tensor, %1 : Tensor):
        %false : bool = prim::Constant[value=0]()
        %2 : int = prim::Constant[value=2]()
        %device : Device = prim::Constant[value="cuda:0"]()
        %type : int = prim::dtype(%0)
        %value : Tensor = aten::tensor(%2, %type, %device, %false)
        %4 : Tensor = aten::masked_fill(%0, %1, %value)
        return (%4))IR";

    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn({2, 2}, {at::kCUDA}));
    auto options_pyt = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kBool);
    input_data.push_back(torch::tensor({false, true, false, true}, options_pyt).reshape({2, 2}));

    baidu::mirana::poros::MaskedFillConverter maskedfillconverter;

    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    poros_option.is_dynamic = false;
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, &maskedfillconverter,
                input_data, graph_output, poros_output));

    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());
    ASSERT_TRUE(graph_output[0].equal(poros_output[0]));
}


TEST(Converters, ATenIndexOneDimConverterCorrectly) {
// aten::index.Tensor(Tensor self, Tensor?[] indices) -> Tensor 
    const auto graph_IR = R"IR(
      graph(%0 : Tensor, %1 : Tensor):
        %2 : Tensor?[] = prim::ListConstruct(%0)
        %3 : Tensor = aten::index(%1, %2)
        return (%3))IR";

    std::vector<at::Tensor> input_data;
    auto options_pyt = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kLong);
    input_data.push_back(at::randint(0, 3, {2, 2}, options_pyt)); // indices
    input_data.push_back(at::randint(0, 10, {3, 4, 5}, options_pyt)); // self

    baidu::mirana::poros::IndexConverter indexconverter;

    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    poros_option.is_dynamic = false;
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, &indexconverter, 
                input_data, graph_output, poros_output));

    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());
    ASSERT_TRUE(graph_output[0].equal(poros_output[0]));
}

TEST(Converters, ATenIndexOneDimDynamicConverterCorrectly) {
// aten::index.Tensor(Tensor self, Tensor?[] indices) -> Tensor 
    const auto graph_IR = R"IR(
      graph(%0 : Tensor, %1 : Tensor):
        %2 : Tensor?[] = prim::ListConstruct(%0)
        %3 : Tensor = aten::index(%1, %2)
        return (%3))IR";

    std::vector<at::Tensor> input_data;
    auto options_pyt = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kLong);
    input_data.push_back(at::randint(0, 3, {2, 2}, options_pyt)); // indices
    input_data.push_back(at::randint(0, 10, {3, 4, 5}, options_pyt)); // self

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::randint(0, 4, {3, 4}, options_pyt)); // indices
    prewarm_data[0].push_back(at::randint(0, 10, {4, 5, 6}, options_pyt)); // self
    prewarm_data[1].push_back(at::randint(0, 3, {2, 2}, options_pyt)); // indices
    prewarm_data[1].push_back(at::randint(0, 10, {3, 4, 5}, options_pyt)); // self
    prewarm_data[2].push_back(at::randint(0, 3, {2, 2}, options_pyt)); // indices
    prewarm_data[2].push_back(at::randint(0, 10, {3, 4, 5}, options_pyt)); // self

    baidu::mirana::poros::IndexConverter indexconverter;

    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    poros_option.is_dynamic = true;
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, &indexconverter, 
                input_data, graph_output, poros_output, &prewarm_data));

    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());
    ASSERT_TRUE(graph_output[0].equal(poros_output[0]));
}

TEST(Converters, ATenIndexPutConverterCorrectly) {
//aten::index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor"
    const auto graph_IR = R"IR(
      graph(%0 : Tensor, %1 : Tensor, %2 : Tensor, %3 : Tensor):
        %false : bool = prim::Constant[value=0]()
        %none  : NoneType = prim::Constant()
        %zeros : Tensor = aten::zeros_like(%0, %none, %none, %none, %none, %none)
        %index : Tensor?[] = prim::ListConstruct(%1, %2)
        %out : Tensor = aten::index_put(%zeros, %index, %3, %false)
        return (%out))IR";

    std::vector<at::Tensor> input_data;
    auto options_pyt_long = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kLong);
    auto options_pyt_float = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kFloat);
    input_data.push_back(at::zeros({2, 5}, options_pyt_float));
    input_data.push_back(torch::tensor({0, 0, 1, 1}, options_pyt_long));
    input_data.push_back(torch::tensor({0, 2, 1, 3}, options_pyt_long));
    input_data.push_back(torch::tensor({1, 2, 3, 4}, options_pyt_float));

    baidu::mirana::poros::IndexPutConverter indexputconverter;

    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    poros_option.is_dynamic = false;
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, &indexputconverter, 
                input_data, graph_output, poros_output));

    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());
    ASSERT_TRUE(graph_output[0].equal(poros_output[0]));
}

//2022.10.19 踩坑记录：不要对indexput 这个singular的IR 进行非dynamic的单测，
//因为这个graph在static的情况下，会在数据预热阶段, 直接全图计算出结果，生成一个constant结果给tensorrt，
//直接导致单测无法通过。
TEST(Converters, ATenIndexPutConverterSingularValueDynamicCorrectly) {
//aten::index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor"
    const auto graph_IR = gen_indexput_with_singular_value_graph();
    std::vector<at::Tensor> input_data;
    auto options_pyt_float = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kFloat);
    input_data.push_back(at::zeros({1, 16, 64}, options_pyt_float));

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::zeros({1, 30, 64}, options_pyt_float));
    prewarm_data[1].push_back(at::zeros({1, 8, 64}, options_pyt_float));
    prewarm_data[2].push_back(at::zeros({1, 20, 64}, options_pyt_float));

    baidu::mirana::poros::IndexPutConverter indexputconverter;

    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    poros_option.is_dynamic = true;
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, &indexputconverter,
                input_data, graph_output, poros_output, &prewarm_data));

    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());
    ASSERT_TRUE(graph_output[0].equal(poros_output[0]));
} 

TEST(Converters, ATenIndexPutConverterDynamicCorrectly) {
//aten::index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor"
    const auto graph_IR = R"IR(
      graph(%0 : Tensor, %1 : Tensor, %2 : Tensor, %3 : Tensor):
        %false : bool = prim::Constant[value=0]()
        %none  : NoneType = prim::Constant()
        %zeros : Tensor = aten::zeros_like(%0, %none, %none, %none, %none, %none)
        %index : Tensor?[] = prim::ListConstruct(%1, %2)
        %out : Tensor = aten::index_put(%zeros, %index, %3, %false)
        return (%out))IR";

    std::vector<at::Tensor> input_data;
    auto options_pyt_long = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kLong);
    auto options_pyt_float = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kFloat);
    input_data.push_back(at::zeros({2, 5}, options_pyt_float));
    input_data.push_back(torch::tensor({0, 0, 1, 1}, options_pyt_long));
    input_data.push_back(torch::tensor({0, 2, 1, 3}, options_pyt_long));
    input_data.push_back(torch::tensor({1, 2, 3, 4}, options_pyt_float));

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::zeros({8, 5}, options_pyt_float)); // indices
    prewarm_data[0].push_back(torch::tensor({2, 3, 3, 4}, options_pyt_long));
    prewarm_data[0].push_back(torch::tensor({0, 2, 1, 3}, options_pyt_long));
    prewarm_data[0].push_back(torch::tensor({8, 8, 8, 8}, options_pyt_float));

    prewarm_data[1].push_back(at::zeros({2, 5}, options_pyt_float)); // indices
    prewarm_data[1].push_back(torch::tensor({0, 0, 1, 1}, options_pyt_long));
    prewarm_data[1].push_back(torch::tensor({0, 2, 1, 3}, options_pyt_long));
    prewarm_data[1].push_back(torch::tensor({3, 3, 3, 3}, options_pyt_float));

    prewarm_data[2].push_back(at::zeros({4, 5}, options_pyt_float)); // indices
    prewarm_data[2].push_back(torch::tensor({0, 0, 1, 1}, options_pyt_long));
    prewarm_data[2].push_back(torch::tensor({0, 2, 1, 3}, options_pyt_long));
    prewarm_data[2].push_back(torch::tensor({1, 2, 3, 4}, options_pyt_float));

    baidu::mirana::poros::IndexPutConverter indexputconverter;

    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    poros_option.is_dynamic = true;
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, &indexputconverter, 
                input_data, graph_output, poros_output, &prewarm_data));

    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());
    ASSERT_TRUE(graph_output[0].equal(poros_output[0]));
}

TEST(Converters, ATenIndexPutConverterDynamicFromCopyCorrectly) {
//aten::index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor"
    const auto graph_IR = gen_indexput_graph("21");

    std::vector<at::Tensor> input_data;
    auto options_pyt_float = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kFloat);
    input_data.push_back(at::ones({8, 16, 64, 16, 16}, options_pyt_float));
   
    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    prewarm_data[0].push_back(at::ones({64, 16, 64, 16, 16}, options_pyt_float));
    prewarm_data[1].push_back(at::ones({8, 16, 64, 16, 16}, options_pyt_float));
    prewarm_data[2].push_back(at::ones({32, 16, 64, 16, 16}, options_pyt_float));

    baidu::mirana::poros::IndexPutConverter indexputconverter;

    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    poros_option.is_dynamic = true;
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, &indexputconverter, 
                input_data, graph_output, poros_output, &prewarm_data));

    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());
    ASSERT_TRUE(graph_output[0].equal(poros_output[0]));
}

TEST(Converters, ATenIndexPutConverterStaticFromCopyCorrectly) {
//aten::index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor"
    const auto graph_IR = gen_indexput_graph("21");

    std::vector<at::Tensor> input_data;
    auto options_pyt_float = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kFloat);
    input_data.push_back(at::ones({1, 16, 64, 56, 56}, options_pyt_float));

    baidu::mirana::poros::IndexPutConverter indexputconverter;
    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    poros_option.is_dynamic = false;

    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, &indexputconverter,
                input_data, graph_output, poros_output));

    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());
    ASSERT_TRUE(graph_output[0].equal(poros_output[0]));
}

TEST(Converters, ATenScatterConverterCorrectly) {
// aten::scatter.value(Tensor self, int dim, Tensor index, Scalar value) -> (Tensor)
    const auto graph_IR = R"IR(
      graph(%0 : Tensor, %1 : Tensor):
        %2 : int = prim::Constant[value=1]()
        %3 : float = prim::Constant[value=2.5]()
        %4 : Tensor = aten::scatter(%0, %2, %1, %3)
        return (%4))IR";

    std::vector<at::Tensor> input_data;
    auto options_pyt_long = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kLong);
    auto options_pyt_float = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kFloat);
    input_data.push_back(at::zeros({2, 4}, options_pyt_float));
    input_data.push_back(torch::tensor({{0, 1, 2, 0}, {1, 2, 0, 3}}, options_pyt_long));

    baidu::mirana::poros::ScatterConverter scatterconverter;

    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    poros_option.is_dynamic = false;
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, &scatterconverter, 
                input_data, graph_output, poros_output));

    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());
    ASSERT_TRUE(graph_output[0].equal(poros_output[0]));
}

TEST(Converters, ATenScatterSelfValueDiffTypeConverterCorrectly) {
// aten::scatter.value(Tensor self, int dim, Tensor index, Scalar value) -> (Tensor)
    const auto graph_IR = R"IR(
      graph(%0 : Tensor, %1 : Tensor):
        %2 : int = prim::Constant[value=1]()
        %3 : float = prim::Constant[value=2.5]()
        %4 : Tensor = aten::scatter(%0, %2, %1, %3)
        return (%4))IR";

    std::vector<at::Tensor> input_data;
    auto options_pyt_long = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kLong);
    auto options_pyt_int = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kInt);
    input_data.push_back(at::zeros({2, 4}, options_pyt_int));
    input_data.push_back(torch::tensor({{0, 1, 2, 3}, {1, 2, 0, 3}}, options_pyt_long));

    baidu::mirana::poros::ScatterConverter scatterconverter;

    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    poros_option.is_dynamic = false;
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, &scatterconverter, 
                input_data, graph_output, poros_output));

    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());
    ASSERT_TRUE(graph_output[0].equal(poros_output[0]));
}

TEST(Converters, ATenScatterSelfIndexDiffShapeConverterCorrectly) {
// aten::scatter.value(Tensor self, int dim, Tensor index, Scalar value) -> (Tensor)
// Index tensor 和 self tensor 的shape可以不一致但是rank（dim数）必须一致
    const auto graph_IR = R"IR(
      graph(%0 : Tensor, %1 : Tensor):
        %2 : int = prim::Constant[value=0]()
        %3 : float = prim::Constant[value=2.5]()
        %4 : Tensor = aten::scatter(%0, %2, %1, %3)
        return (%4))IR";

    std::vector<at::Tensor> input_data;
    auto options_pyt_long = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kLong);
    auto options_pyt_float = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kFloat);
    input_data.push_back(at::zeros({2, 4}, options_pyt_float));
    input_data.push_back(torch::tensor({{0}}, options_pyt_long));

    baidu::mirana::poros::ScatterConverter scatterconverter;

    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    poros_option.is_dynamic = false;
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, &scatterconverter, 
                input_data, graph_output, poros_output));

    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());
    ASSERT_TRUE(graph_output[0].equal(poros_output[0]));
}

TEST(Converters, ATenScatterDynamicConverterCorrectly) {
// aten::scatter.value(Tensor self, int dim, Tensor index, Scalar value) -> (Tensor)
    const auto graph_IR = R"IR(
      graph(%0 : Tensor, %1 : Tensor):
        %2 : int = prim::Constant[value=1]()
        %3 : float = prim::Constant[value=2.5]()
        %4 : Tensor = aten::scatter(%0, %2, %1, %3)
        return (%4))IR";

    std::vector<at::Tensor> input_data;
    auto options_pyt_long = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kLong);
    auto options_pyt_float = torch::TensorOptions().device(torch::kCUDA, 0).dtype(torch::kFloat);
    input_data.push_back(at::zeros({3, 4}, options_pyt_float));
    input_data.push_back(torch::tensor({{0, 1}, {1, 2}}, options_pyt_long));

    std::vector<std::vector<at::Tensor>> prewarm_data = {{}, {}, {}};
    // max
    prewarm_data[0].push_back(at::randint(0, 4, {3, 4}, options_pyt_float));
    prewarm_data[0].push_back(at::randint(0, 2, {3, 4}, options_pyt_long));
    // min
    prewarm_data[1].push_back(at::randint(0, 4, {2, 3}, options_pyt_float));
    prewarm_data[1].push_back(at::randint(0, 2, {1, 1}, options_pyt_long));
    // opt
    prewarm_data[2].push_back(at::randint(0, 4, {3, 4}, options_pyt_float));
    prewarm_data[2].push_back(at::randint(0, 2, {1, 1}, options_pyt_long));

    baidu::mirana::poros::ScatterConverter scatterconverter;

    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    poros_option.is_dynamic = true;
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, &scatterconverter, 
                input_data, graph_output, poros_output, &prewarm_data));

    ASSERT_EQ(1, graph_output.size());
    ASSERT_EQ(1, poros_output.size());
    ASSERT_TRUE(graph_output[0].equal(poros_output[0]));
}

static void chunk_test_helper(const std::string& graph_IR, 
                                std::vector<int64_t> shape,
                                const int& output_num) {
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn(shape, {at::kCUDA}));

    baidu::mirana::poros::PorosOptions poros_option; // default device GPU
    baidu::mirana::poros::ChunkConverter chunkconverter;
    // 运行原图与engine获取结果
    std::vector<at::Tensor> graph_output;
    std::vector<at::Tensor> poros_output;
    ASSERT_TRUE(baidu::mirana::poros::testutil::run_graph_and_poros(graph_IR, poros_option, &chunkconverter, 
                input_data, graph_output, poros_output));

    ASSERT_EQ(output_num, graph_output.size());
    ASSERT_EQ(graph_output.size(), poros_output.size());
    for (size_t i = 0; i < graph_output.size(); i++) {
        ASSERT_TRUE(graph_output[i].equal(poros_output[i]));
    }
}

TEST(Converters, PrimConstantChunkTwoOutputsConverterCorrectly) {
    // prim::chunk
    const auto graph_IR = R"IR(
      graph(%0 : Tensor):
        %1 : Tensor, %2 : Tensor = prim::ConstantChunk[chunks=2, dim=-1](%0)
        return (%1, %2))IR";
    chunk_test_helper(graph_IR, {5, 6, 8}, 2);
}

TEST(Converters, PrimConstantChunkThreeOutputsConverterCorrectly) {
    // prim::chunk
    const auto graph_IR = R"IR(
      graph(%0 : Tensor):
        %1 : Tensor, %2 : Tensor, %3 : Tensor = prim::ConstantChunk[chunks=3, dim=0](%0)
        return (%1, %2, %3))IR";
    chunk_test_helper(graph_IR, {11, 6, 8}, 3);
}

TEST(Converters, PrimConstantChunkFourOutputsConverterCorrectly) {
    // prim::chunk
    const auto graph_IR = R"IR(
      graph(%0 : Tensor):
        %1 : Tensor, %2 : Tensor, %3 : Tensor, %4 : Tensor = prim::ConstantChunk[chunks=4, dim=1](%0)
        return (%1, %2, %3, %4))IR";
    chunk_test_helper(graph_IR, {7, 13, 8}, 4);
}