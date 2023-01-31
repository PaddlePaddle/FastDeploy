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
* @file interpolate_test.cpp
* @author tianshaoqing@baidu.com
* @date Wed Sep 27 11:24:21 CST 2021
* @brief 
**/
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "poros/converter/gpu/interpolate.h"
#include "poros/util/test_util.h"

static void interpolate_test_helper(const std::string& graph_IR,
                            baidu::mirana::poros::IConverter* converter,
                            std::vector<int64_t> shape){
    std::vector<at::Tensor> input_data;
    input_data.push_back(at::randn(shape, {at::kCUDA}));
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

static std::string gen_upsample_nearest_nd_graph(bool vec_scales,
                                                const std::string& op,
                                                const std::string& output_size,
                                                const std::string& scales) {
    std::string output_ir("");
    std::string scales_ir("");
    std::string op_ir("");
    if (!vec_scales) {
        output_ir = "int[] = prim::Constant[value=[" + output_size + "]]()";
        if (scales.empty()) {
            scales_ir = "None = prim::Constant()";
        } else {
            scales_ir = "float = prim::Constant[value=" + scales + "]()";
        }
        if (op == "upsample_nearest1d") {
            op_ir = op + "(%0, %1, %2)";
        } else if (op == "upsample_nearest2d") {
            op_ir = op + "(%0, %1, %2, %2)";
        } else if (op == "upsample_nearest3d") {
            op_ir = op + "(%0, %1, %2, %2, %2)";
        } else {
            return "";
        }
    } else {
        if (output_size.empty()) {
            output_ir = "None = prim::Constant()";
        } else {
            output_ir = "int[] = prim::Constant[value=[" + output_size + "]]()";
        }
        if (scales.empty()) {
            scales_ir = "None = prim::Constant()";
        } else {
            scales_ir = "float[] = prim::Constant[value=[" + scales + "]]()";
        }
        op_ir = op + "(%0, %1, %2)";
    }
    return R"IR(
        graph(%0 : Tensor):
          %1 : )IR" + output_ir + R"IR(
          %2 : )IR" + scales_ir + R"IR(
          %3 : Tensor = aten::)IR" + op_ir + R"IR(
          return (%3))IR";
}

static std::string gen_upsample_linear_graph(bool vec_scales,
                                            const std::string& op,
                                            const std::string& output_size,
                                            const std::string& align_corners,
                                            const std::string& scales) {
    std::string output_ir("");
    std::string scales_ir("");
    std::string op_ir("");
    if (!vec_scales) {
        output_ir = "int[] = prim::Constant[value=[" + output_size + "]]()";
        if (scales.empty()) {
            scales_ir = "None = prim::Constant()";
        } else {
            scales_ir = "float = prim::Constant[value=" + scales + "]()";
        }
        if (op == "upsample_linear1d") {
            op_ir = op + "(%0, %1, %2, %3)";
        } else if (op == "upsample_bilinear2d") {
            op_ir = op + "(%0, %1, %2, %3, %3)";
        } else if (op == "upsample_trilinear3d") {
            op_ir = op + "(%0, %1, %2, %3, %3, %3)";
        } else {
            return "";
        }
    } else {
        if (output_size.empty()) {
            output_ir = "None = prim::Constant()";
        } else {
            output_ir = "int[] = prim::Constant[value=[" + output_size + "]]()";
        }
        if (scales.empty()) {
            scales_ir = "None = prim::Constant()";
        } else {
            scales_ir = "float[] = prim::Constant[value=[" + scales + "]]()";
        }
        op_ir = op + "(%0, %1, %2, %3)";
    }

    return R"IR(
        graph(%0 : Tensor):
          %1 : )IR" + output_ir + R"IR(
          %2 : bool = prim::Constant[value=)IR" + align_corners + R"IR(]()
          %3 : )IR" + scales_ir + R"IR(
          %4 : Tensor = aten::)IR" + op_ir + R"IR(
          return (%4))IR";
}

TEST(Converters, ATenUpsampleNearest1d) {
    // aten::upsample_nearest1d(Tensor self, int[1] output_size, float? scales=None) -> Tensor
    const auto graph_IR = gen_upsample_nearest_nd_graph(false, "upsample_nearest1d", "10", "");
    baidu::mirana::poros::UnsampleNearest1DConverter unsamplenearest1dconverter;
    interpolate_test_helper(graph_IR, &unsamplenearest1dconverter, {10, 2, 2});
}

TEST(Converters, ATenUpsampleNearest1dScalar) {
    // aten::upsample_nearest1d(Tensor self, int[1] output_size, float? scales=None) -> Tensor
    const auto graph_IR = gen_upsample_nearest_nd_graph(false, "upsample_nearest1d", "8", "4.0");
    baidu::mirana::poros::UnsampleNearest1DConverter unsamplenearest1dconverter;
    interpolate_test_helper(graph_IR, &unsamplenearest1dconverter, {10, 2, 2});
}

TEST(Converters, ATenUpsampleNearest1dVecScalar) {
    // aten::upsample_nearest1d.vec(Tensor input, int[]? output_size, float[]? scale_factors) -> Tensor
    const auto graph_IR = gen_upsample_nearest_nd_graph(true, "upsample_nearest1d", "", "4.0");          
    baidu::mirana::poros::UnsampleNearest1DConverter unsamplenearest1dconverter;
    interpolate_test_helper(graph_IR, &unsamplenearest1dconverter, {10, 2, 2});
}

TEST(Converters, ATenUpsampleNearest2d) {
    // aten::upsample_nearest2d(Tensor self, int[2] output_size, float? scales_h=None, float? scales_w=None) -> Tensor
    const auto graph_IR = gen_upsample_nearest_nd_graph(false, "upsample_nearest2d", "10, 8", "");
    baidu::mirana::poros::UnsampleNearest2DConverter unsamplenearest2dconverter;
    interpolate_test_helper(graph_IR, &unsamplenearest2dconverter, {10, 2, 2, 2});
}

TEST(Converters, ATenUpsampleNearest2dScalar) {
    // aten::upsample_nearest2d(Tensor self, int[2] output_size, float? scales_h=None, float? scales_w=None) -> Tensor
    const auto graph_IR = gen_upsample_nearest_nd_graph(false, "upsample_nearest2d", "8, 8", "4.0");
    baidu::mirana::poros::UnsampleNearest2DConverter unsamplenearest2dconverter;
    interpolate_test_helper(graph_IR, &unsamplenearest2dconverter, {10, 2, 2, 2});
}

TEST(Converters, ATenUpsampleNearest2dVecScalar) {
    // aten::upsample_nearest2d.vec(Tensor input, int[]? output_size, float[]? scale_factors) -> Tensor
    const auto graph_IR = gen_upsample_nearest_nd_graph(true, "upsample_nearest2d", "", "5.0, 4.0");
    baidu::mirana::poros::UnsampleNearest2DConverter unsamplenearest2dconverter;
    interpolate_test_helper(graph_IR, &unsamplenearest2dconverter, {10, 2, 2, 2});
}

TEST(Converters, ATenUpsampleNearest3d) {
    // aten::upsample_nearest3d(Tensor self, int[3] output_size, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor
    const auto graph_IR = gen_upsample_nearest_nd_graph(false, "upsample_nearest3d", "10, 8, 6", "");
    baidu::mirana::poros::UnsampleNearest3DConverter unsamplenearest3dconverter;
    interpolate_test_helper(graph_IR, &unsamplenearest3dconverter, {10, 2, 2, 2, 2});
}

TEST(Converters, ATenUpsampleNearest3dScalar) {
    // aten::upsample_nearest3d(Tensor self, int[3] output_size, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor
    const auto graph_IR = gen_upsample_nearest_nd_graph(false, "upsample_nearest3d", "8, 8, 8", "4.0");
    baidu::mirana::poros::UnsampleNearest3DConverter unsamplenearest3dconverter;
    interpolate_test_helper(graph_IR, &unsamplenearest3dconverter, {10, 2, 2, 2, 2});
}

TEST(Converters, ATenUpsampleNearest3dVecScalar) {
    // aten::upsample_nearest3d.vec(Tensor input, int[]? output_size, float[]? scale_factors) -> Tensor
    const auto graph_IR = gen_upsample_nearest_nd_graph(true, "upsample_nearest3d", "", "5.0, 4.0, 3.0");
    baidu::mirana::poros::UnsampleNearest3DConverter unsamplenearest3dconverter;
    interpolate_test_helper(graph_IR, &unsamplenearest3dconverter, {10, 2, 2, 2, 2});
}

// start almost equal
TEST(Converters, ATenUpsampleLinear1dWithAlignCorners) {
    // aten::upsample_linear1d(Tensor self, int[1] output_size, bool align_corners, float? scales=None) -> Tensor
    const auto graph_IR = gen_upsample_linear_graph(false, "upsample_linear1d", "10", "1", "");
    baidu::mirana::poros::UnsampleLinear1DConverter unsamplelinear1dconverter;
    interpolate_test_helper(graph_IR, &unsamplelinear1dconverter, {10, 2, 2});
}

TEST(Converters, ATenUpsampleLinear1dWithoutAlignCorners) {
    // aten::upsample_linear1d(Tensor self, int[1] output_size, bool align_corners, float? scales=None) -> Tensor
    const auto graph_IR = gen_upsample_linear_graph(false, "upsample_linear1d", "10", "0", "5.0");
    baidu::mirana::poros::UnsampleLinear1DConverter unsamplelinear1dconverter;
    interpolate_test_helper(graph_IR, &unsamplelinear1dconverter, {10, 2, 2});
}

TEST(Converters, ATenUpsampleLinear1dScalesWithoutAlignCorners) {
    // aten::upsample_linear1d(Tensor self, int[1] output_size, bool align_corners, float? scales=None) -> Tensor
    const auto graph_IR = gen_upsample_linear_graph(false, "upsample_linear1d", "8", "0", "4.0");
    baidu::mirana::poros::UnsampleLinear1DConverter unsamplelinear1dconverter;
    interpolate_test_helper(graph_IR, &unsamplelinear1dconverter, {10, 2, 2});
}
TEST(Converters, ATenUpsampleLinear1dVecScaleFactorsWithoutAlignCorners) {
    // aten::upsample_linear1d.vec(Tensor input, int[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor
    const auto graph_IR = gen_upsample_linear_graph(true, "upsample_linear1d", "", "0", "4.0");
    baidu::mirana::poros::UnsampleLinear1DConverter unsamplelinear1dconverter;
    interpolate_test_helper(graph_IR, &unsamplelinear1dconverter, {10, 2, 2});
}

TEST(Converters, ATenUpsampleBilinear2dWithAlignCorners) {
    // aten::upsample_bilinear2d(Tensor self, int[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor
    const auto graph_IR = gen_upsample_linear_graph(false, "upsample_bilinear2d", "10, 8", "1", "");
    baidu::mirana::poros::UnsampleBilinear2DConverter unsamplebilinear2dconverter;
    interpolate_test_helper(graph_IR, &unsamplebilinear2dconverter, {10, 2, 2, 2});
}

TEST(Converters, ATenUpsampleBilinear2dWithoutAlignCorners) {
    // aten::upsample_bilinear2d(Tensor self, int[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor
    const auto graph_IR = gen_upsample_linear_graph(false, "upsample_bilinear2d", "10, 8", "0", "");
    baidu::mirana::poros::UnsampleBilinear2DConverter unsamplebilinear2dconverter;
    interpolate_test_helper(graph_IR, &unsamplebilinear2dconverter, {10, 2, 2, 2});
}

TEST(Converters, ATenUpsampleBilinear2dScalesWithoutAlignCorners) {
    // aten::upsample_bilinear2d(Tensor self, int[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor
    const auto graph_IR = gen_upsample_linear_graph(false, "upsample_bilinear2d", "10, 10", "0", "5.0");
    baidu::mirana::poros::UnsampleBilinear2DConverter unsamplebilinear2dconverter;
    interpolate_test_helper(graph_IR, &unsamplebilinear2dconverter, {10, 2, 2, 2});
}

TEST(Converters, ATenUpsampleBilinear2dVecScaleFactorsWithoutAlignCorners) {
    // aten::upsample_bilinear2d.vec(Tensor input, int[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor
    const auto graph_IR = gen_upsample_linear_graph(true, "upsample_bilinear2d", "", "0", "5.0, 4.0");
    baidu::mirana::poros::UnsampleBilinear2DConverter unsamplebilinear2dconverter;
    interpolate_test_helper(graph_IR, &unsamplebilinear2dconverter, {10, 2, 2, 2});
}

TEST(Converters, ATenUpsampleTrilinear3dWithAlignCorners) {
    // aten::upsample_trilinear3d(Tensor self, int[3] output_size, bool align_corners, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor
    const auto graph_IR = gen_upsample_linear_graph(false, "upsample_trilinear3d", "10, 8, 6", "1", "");
    baidu::mirana::poros::UnsampleTrilinear3DConverter unsampletrilinear3dconverter;
    interpolate_test_helper(graph_IR, &unsampletrilinear3dconverter, {10, 2, 2, 2, 2});
}

TEST(Converters, ATenUpsampleTrilinear3dWithoutAlignCorners) {
    // aten::upsample_trilinear3d(Tensor self, int[3] output_size, bool align_corners, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor
    const auto graph_IR = gen_upsample_linear_graph(false, "upsample_trilinear3d", "10, 8, 6", "0", "");
    baidu::mirana::poros::UnsampleTrilinear3DConverter unsampletrilinear3dconverter;
    interpolate_test_helper(graph_IR, &unsampletrilinear3dconverter, {10, 2, 2, 2, 2});
}

TEST(Converters, ATenUpsampleTrilinear3dVecScaleFactorsWithoutAlignCorners) {
    // aten::upsample_trilinear3d.vec(Tensor input, int[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor
    const auto graph_IR = gen_upsample_linear_graph(true, "upsample_trilinear3d", "", "0", "5.0, 4.0, 3.0");
    baidu::mirana::poros::UnsampleTrilinear3DConverter unsampletrilinear3dconverter;
    interpolate_test_helper(graph_IR, &unsampletrilinear3dconverter, {10, 2, 2, 2, 2});
}