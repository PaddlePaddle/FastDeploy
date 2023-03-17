# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import

import numpy as np
from paddle2onnx.legacy.constant import dtypes
from paddle2onnx.legacy.op_mapper import OpMapper as op_mapper
from paddle2onnx.legacy.op_mapper import mapper_helper


@op_mapper('box_coder')
class BoxCoder():
    """
    we use the decode the prior box to target box,
    we just use the decode mode to transform this op.
    """
    support_opset_verison_range = (7, 12)

    @classmethod
    def opset_7(cls, graph, node, **kw):
        input_names = node.input_names

        t_size = node.input_shape('TargetBox', 0)
        p_size = node.input_shape('PriorBox', 0)

        # get the outout_name
        result_name = node.output('OutputBox', 0)
        # n is size of batch, m is boxes num of targe_boxes
        n = t_size[0]
        m = t_size[0]

        axis = int(node.attr('axis'))

        #norm
        norm = bool(node.attr('box_normalized'))

        name_slice_x1 = node.output('OutputBox')[0] + "@x1"
        name_slice_y1 = node.output('OutputBox')[0] + "@y1"
        name_slice_x2 = node.output('OutputBox')[0] + "@x2"
        name_slice_y2 = node.output('OutputBox')[0] + "@y2"

        #make onnx tensor to save the intermeidate reslut
        name_slice_indices = [
            [node.output('OutputBox')[0] + "@slice_" + str(i)]
            for i in range(1, 3)
        ]
        node_slice_indices = [None for i in range(1, 3)]

        # create the range(0, 4) const data to slice
        for i in range(1, 3):
            tmp_node = graph.make_node(
                'Constant',
                inputs=[],
                outputs=name_slice_indices[i - 1],
                dtype=dtypes.ONNX.FLOAT,
                dims=(),
                value=[i])
        # make node split data
        name_box_split = [
            name_slice_x1, name_slice_y1, name_slice_x2, name_slice_y2
        ]
        split_shape = list(p_size)
        split_shape[-1] = 1

        node_split_prior_node = graph.make_node(
            'Split',
            inputs=node.input('PriorBox'),
            outputs=name_box_split,
            axis=1)

        # make node get centor node for decode
        final_outputs_vars = []
        if not norm:
            name_centor_w_tmp = [node.output('OutputBox')[0] + "@centor_w_tmp"]
            name_centor_h_tmp = [node.output('OutputBox')[0] + "@centor_h_tmp"]
            node_centor_w_tmp = None
            node_centor_h_tmp = None
            name_centor_tmp_list = [name_centor_w_tmp, name_centor_h_tmp]
            node_centor_tmp_list = [node_centor_w_tmp, node_centor_h_tmp]

            count = 2
            for (name, op_node) in zip(name_centor_tmp_list,
                                       node_centor_tmp_list):
                tmp_node = graph.make_node('Add',
                       inputs=[node.output('OutputBox')[0] + "@slice_" + str(1)]\
                           + [name_box_split[count]],
                       outputs=name)
                count = count + 1
        if not norm:
            inputs_sub = [[name_centor_w_tmp[0], name_box_split[0]],
                          [name_centor_h_tmp[0], name_box_split[1]]]
        else:
            inputs_sub = [[name_box_split[2], name_box_split[0]],
                          [name_box_split[3], name_box_split[1]]]
        outputs_sub = [result_name + "@pb_w", result_name + "@pb_h"]
        for i in range(0, 2):
            tmp_node = graph.make_node(
                'Sub', inputs=inputs_sub[i], outputs=[outputs_sub[i]])
        # according to prior_box height and weight to get centor x, y
        name_half_value = [result_name + "@half_value"]
        node_half_value = graph.make_node(
            'Constant',
            inputs=[],
            outputs=name_half_value,
            dtype=dtypes.ONNX.FLOAT,
            dims=(),
            value=[0.5])
        outputs_half_wh = [[result_name + "@pb_w_half"],
                           [result_name + "@pb_h_half"]]
        inputs_half_wh = [[result_name + "@pb_w", name_half_value[0]],
                          [result_name + "@pb_h", name_half_value[0]]]

        for i in range(0, 2):
            tmp_node = graph.make_node(
                'Mul', inputs=inputs_half_wh[i], outputs=outputs_half_wh[i])

        inputs_centor_xy = [[outputs_half_wh[0][0], name_slice_x1],
                            [outputs_half_wh[1][0], name_slice_y1]]

        outputs_centor_xy = [[result_name + "@pb_x"], [result_name + "@pb_y"]]

        # final calc the centor x ,y
        for i in range(0, 2):
            tmp_node = graph.make_node(
                'Add', inputs=inputs_centor_xy[i], outputs=outputs_centor_xy[i])
        # reshape the data
        shape = (1, split_shape[0]) if axis == 0 else (split_shape[0], 1)

        # need to reshape the data
        inputs_transpose_pb = [
            [result_name + "@pb_w"],
            [result_name + "@pb_h"],
            [result_name + "@pb_x"],
            [result_name + "@pb_y"],
        ]
        outputs_transpose_pb = [
            [result_name + "@pb_w_transpose"],
            [result_name + "@pb_h_transpose"],
            [result_name + "@pb_x_transpose"],
            [result_name + "@pb_y_transpose"],
        ]
        if axis == 0:
            name_reshape_pb = [result_name + "@pb_transpose"]
            # reshape the data
            for i in range(0, 4):
                tmp_node = graph.make_node(
                    'Transpose',
                    inputs=inputs_transpose_pb[i],
                    outputs=outputs_transpose_pb[i])
        # decoder the box according to the target_box and variacne
        name_variance_raw = [result_name + "@variance_raw"]
        name_variance_unsqueeze = [result_name + "@variance_unsqueeze"]
        shape = []
        # make node to extend the data
        var_split_axis = 0
        var_split_inputs_name = []
        if 'PriorBoxVar' in input_names and len(node.input('PriorBoxVar')) > 0:
            if axis == 1:
                raise Exception(
                    "The op box_coder has variable do not support aixs broadcast"
                )
            axes = []
            var_split_inputs_name = [result_name + "@variance_split"]
            tmp_node = graph.make_node(
                'Transpose',
                inputs=node.input('PriorBoxVar'),
                outputs=var_split_inputs_name)
            var_split_axis = 0
        else:
            variances = [1.0, 1.0, 1.0, 1.0]
            if 'variance' in node.attrs and len(node.attr('variance')) > 0:
                variances = [float(var) for var in node.attr('variance')]
            node_variance_create = graph.make_node(
                'Constant',
                inputs=[],
                outputs=name_variance_raw,
                dtype=dtypes.ONNX.FLOAT,
                dims=[len(variances)],
                value=variances)
            var_split_axis = 0
            var_split_inputs_name = name_variance_raw

        # decode the result
        outputs_split_variance = [
            result_name + "@variance_split" + str(i) for i in range(0, 4)
        ]
        outputs_split_targebox = [
            result_name + "@targebox_split" + str(i) for i in range(0, 4)
        ]
        node_split_var = graph.make_node(
            'Split',
            inputs=var_split_inputs_name,
            outputs=outputs_split_variance,
            axis=var_split_axis)
        node_split_target = graph.make_node(
            'Split',
            inputs=node.input('TargetBox'),
            outputs=outputs_split_targebox,
            axis=2)

        outputs_squeeze_targebox = [
            result_name + "@targebox_squeeze" + str(i) for i in range(0, 4)
        ]
        for (input_name, output_name) in zip(outputs_split_targebox,
                                             outputs_squeeze_targebox):
            tmp_node = mapper_helper.squeeze_helper(graph, input_name, [2],
                                                    [output_name])

        output_shape_step1 = list(t_size)[:-1]

        inputs_tb_step1 = [
            [outputs_squeeze_targebox[0], outputs_split_variance[0]],
            [outputs_squeeze_targebox[1], outputs_split_variance[1]],
            [outputs_squeeze_targebox[2], outputs_split_variance[2]],
            [outputs_squeeze_targebox[3], outputs_split_variance[3]]
        ]
        outputs_tb_step1 = [[result_name + "@decode_x_step1"],
                            [result_name + "@decode_y_step1"],
                            [result_name + "@decode_w_step1"],
                            [result_name + "@decode_h_step1"]]

        for input_step1, output_step_1 in zip(inputs_tb_step1,
                                              outputs_tb_step1):
            tmp_node = graph.make_node(
                'Mul', inputs=input_step1, outputs=output_step_1)
        if axis == 0:
            inputs_tbxy_step2 = [[
                outputs_tb_step1[0][0], outputs_transpose_pb[0][0]
            ], [outputs_tb_step1[1][0], outputs_transpose_pb[1][0]]]
        else:
            inputs_tbxy_step2 = [[
                outputs_tb_step1[0][0], inputs_transpose_pb[0][0]
            ], [outputs_tb_step1[1][0], inputs_transpose_pb[1][0]]]

        outputs_tbxy_step2 = [[result_name + "@decode_x_step2"],
                              [result_name + "@decode_y_step2"]]

        for input_step2, output_step_2 in zip(inputs_tbxy_step2,
                                              outputs_tbxy_step2):
            tmp_node = graph.make_node(
                'Mul', inputs=input_step2, outputs=output_step_2)
        if axis == 0:
            inputs_tbxy_step3 = [[
                outputs_tbxy_step2[0][0], outputs_transpose_pb[2][0]
            ], [outputs_tbxy_step2[1][0], outputs_transpose_pb[3][0]]]
        else:
            inputs_tbxy_step3 = [[
                outputs_tbxy_step2[0][0], inputs_transpose_pb[2][0]
            ], [outputs_tbxy_step2[1][0], inputs_transpose_pb[3][0]]]

        outputs_tbxy_step3 = [[result_name + "@decode_x_step3"],
                              [result_name + "@decode_y_step3"]]

        for input_step3, output_step_3 in zip(inputs_tbxy_step3,
                                              outputs_tbxy_step3):
            tmp_node = graph.make_node(
                'Add', inputs=input_step3, outputs=output_step_3)

        # deal with width & height
        inputs_tbwh_step2 = [outputs_tb_step1[2], outputs_tb_step1[3]]
        outputs_tbwh_step2 = [[result_name + "@decode_w_step2"],
                              [result_name + "@decode_h_step2"]]

        for input_name, output_name in zip(inputs_tbwh_step2,
                                           outputs_tbwh_step2):
            tmp_node = graph.make_node(
                'Exp', inputs=input_name, outputs=output_name)

        if axis == 0:
            inputs_tbwh_step3 = [[
                outputs_tbwh_step2[0][0], outputs_transpose_pb[0][0]
            ], [outputs_tbwh_step2[1][0], outputs_transpose_pb[1][0]]]
        else:
            inputs_tbwh_step3 = [[
                outputs_tbwh_step2[0][0], inputs_transpose_pb[0][0]
            ], [outputs_tbwh_step2[1][0], inputs_transpose_pb[1][0]]]

        outputs_tbwh_step3 = [[result_name + "@decode_w_step3"],
                              [result_name + "@decode_h_step3"]]

        for input_name, output_name in zip(inputs_tbwh_step3,
                                           outputs_tbwh_step3):
            tmp_node = graph.make_node(
                'Mul', inputs=input_name, outputs=output_name)

        # final step to calc the result, and concat the result to output
        # return the output box, [(x1, y1), (x2, y2)]

        inputs_half_tbwh_step4 = [[
            outputs_tbwh_step3[0][0], result_name + "@slice_2"
        ], [outputs_tbwh_step3[1][0], result_name + "@slice_2"]]

        outputs_half_tbwh_step4 = [[result_name + "@decode_half_w_step4"],
                                   [result_name + "@decode_half_h_step4"]]
        for inputs_name, outputs_name in zip(inputs_half_tbwh_step4,
                                             outputs_half_tbwh_step4):
            tmp_node = graph.make_node(
                'Div', inputs=inputs_name, outputs=outputs_name)
        inputs_output_point1 = [[
            outputs_tbxy_step3[0][0], outputs_half_tbwh_step4[0][0]
        ], [outputs_tbxy_step3[1][0], outputs_half_tbwh_step4[1][0]]]

        outputs_output_point1 = [[result_name + "@ouput_x1"],
                                 [result_name + "@output_y1"]]
        for input_name, output_name in zip(inputs_output_point1,
                                           outputs_output_point1):
            tmp_node = graph.make_node(
                'Sub', inputs=input_name, outputs=output_name)

        inputs_output_point2 = [[
            outputs_tbxy_step3[0][0], outputs_half_tbwh_step4[0][0]
        ], [outputs_tbxy_step3[1][0], outputs_half_tbwh_step4[1][0]]]

        outputs_output_point2 = [[result_name + "@ouput_x2"],
                                 [result_name + "@output_y2"]]

        for input_name, output_name in zip(inputs_output_point2,
                                           outputs_output_point2):
            tmp_node = graph.make_node(
                'Add', inputs=input_name, outputs=output_name)
        if not norm:
            inputs_unnorm_point2 = [[
                outputs_output_point2[0][0], result_name + "@slice_1"
            ], [outputs_output_point2[1][0], result_name + "@slice_1"]]
            outputs_unnorm_point2 = [[result_name + "@ouput_unnorm_x2"],
                                     [result_name + "@ouput_unnorm_y2"]]

            for input_name, output_name in zip(inputs_unnorm_point2,
                                               outputs_unnorm_point2):
                tmp_node = graph.make_node(
                    'Sub', inputs=input_name, outputs=output_name)
            outputs_output_point2 = outputs_unnorm_point2

        outputs_output_point1.extend(outputs_output_point2)
        ouputs_points_unsqueeze = [[result_name + "@points_unsqueeze_x1"],
                                   [result_name + "points_unsqueeze_y1"],
                                   [result_name + "points_unsqueeze_x2"],
                                   [result_name + "points_unsqueeze_y2"]]

        for input_name, output_name in zip(outputs_output_point1,
                                           ouputs_points_unsqueeze):
            tmp_node = mapper_helper.unsqueeze_helper(
                graph, input_name, [len(output_shape_step1)], output_name)
        outputs_points_unsqueeze_list = [
            output[0] for output in ouputs_points_unsqueeze
        ]
        node_point_final = graph.make_node(
            'Concat',
            inputs=outputs_points_unsqueeze_list,
            outputs=node.output('OutputBox'),
            axis=len(output_shape_step1))
