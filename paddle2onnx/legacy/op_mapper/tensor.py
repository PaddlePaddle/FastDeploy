#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
import copy
import six
import paddle


@op_mapper('set_value')
class SetValue():
    support_opset_version_range = (11, 15)

    @classmethod
    def opset_11(cls, graph, node, **kw):
        axes = node.attr('axes')
        steps, is_steps_tensor = mapper_helper.get_node_attr_value(
            graph,
            node,
            'steps',
            'StepsTensor',
            'StepsTensorList',
            return_list=True,
            dtype=dtypes.ONNX.INT64)

        starts, is_starts_tensor = mapper_helper.get_node_attr_value(
            graph,
            node,
            'starts',
            'StartsTensor',
            'StartsTensorList',
            return_list=True,
            dtype=dtypes.ONNX.INT64)

        ends, is_ends_tensor = mapper_helper.get_node_attr_value(
            graph,
            node,
            'ends',
            'EndsTensor',
            'EndsTensorList',
            return_list=True,
            dtype=dtypes.ONNX.INT64)

        contain_step_bigger_than_1 = False
        for i in steps:
            contain_step_bigger_than_1 = i > 1
            if not isinstance(i, int) or contain_step_bigger_than_1:
                contain_step_bigger_than_1 = True
                break
        condition = is_steps_tensor or is_starts_tensor or is_ends_tensor or contain_step_bigger_than_1
        assert not condition, "Currently not supported convert now"

        input_x_shape = node.input_shape('Input', 0)
        onnx_paddings = [0] * len(input_x_shape) * 2
        value_shape = list(copy.copy(node.input_shape('Input', 0)))
        for i in range(len(axes)):
            axis = axes[i]
            if starts[i] < 0:
                starts[i] = starts[i] + input_x_shape[i]
            if ends[i] < 0:
                ends[i] = ends[i] + input_x_shape[i]
            onnx_paddings[axis] = starts[i]
            value_shape[axis] = value_shape[axis] - onnx_paddings[axis]
            onnx_paddings[axis + len(input_x_shape)] = input_x_shape[
                axis] - ends[i]
            if onnx_paddings[axis + len(input_x_shape)] < 0:
                onnx_paddings[axis + len(input_x_shape)] = 0
            value_shape[axis] = value_shape[axis] - onnx_paddings[axis + len(
                input_x_shape)]
        dtype_paddle = node.input_dtype('Input', 0)
        dtype = dtypes.DTYPE_PADDLE_ONNX_MAP[dtype_paddle]
        value_tensor = None
        shape = node.attr('shape')
        if len(shape) > 0:
            dtypes_list = [
                'fp32_values', 'fp64_values', 'int32_values', 'int64_values',
                'bool_values'
            ]
            for i in range(len(dtypes_list)):
                value = node.attr(dtypes_list[i])
                if value is not None:
                    break
            if len(value) == 1:
                total_nums = 1
                for i in value_shape:
                    total_nums *= i
                value = value * total_nums
                value_tensor = mapper_helper.constant_helper(
                    graph, dtype_paddle, value, shape=value_shape)
            else:
                value_tensor = mapper_helper.constant_helper(
                    graph, dtype_paddle, value, shape=shape)
        else:
            value_tensor = node.input('ValueTensor', 0)
        MAX_FLOAT32 = 3.402823466E+38
        max_node = graph.make_node(
            'Constant', attrs={'dtype': dtype,
                               'value': [MAX_FLOAT32]})
        pads_node = graph.make_node(
            'Constant',
            attrs={'dtype': dtypes.ONNX.INT64,
                   'value': onnx_paddings})
        value_pad_node = graph.make_node(
            'Pad', inputs=[value_tensor, pads_node, max_node])

        condition_dtype = graph.make_node(
            "Equal", inputs=[value_pad_node, max_node])
        condition_node = graph.make_node(
            'Cast', inputs=[condition_dtype], to=dtypes.ONNX.BOOL)
        graph.make_node(
            "Where",
            inputs=[condition_node, node.input('Input', 0), value_pad_node],
            outputs=node.output('Out'))


@op_mapper('one_hot_v2')
class OneHotV2():
    support_opset_version_range = (9, )

    @classmethod
    def opset_9(cls, graph, node, **kw):
        allow_out_of_range = node.attr('allow_out_of_range')
        assert not allow_out_of_range, "allow_out_of_range can not be true in one_hot_v2."
        in_dtype_paddle = node.input_dtype('X', 0)
        in_dtype = dtypes.DTYPE_PADDLE_ONNX_MAP[in_dtype_paddle]
        out_dtype = node.output_dtype('Out', 0)
        out_dtype = dtypes.DTYPE_PADDLE_ONNX_MAP[out_dtype]
        inputs = node.input('X', 0)
        if in_dtype_paddle == paddle.int32:
            inputs = graph.make_node(
                'Cast', inputs=[inputs], to=dtypes.ONNX.INT64)
            in_dtype = dtypes.ONNX.INT64
        value_node = graph.make_node('Constant', dtype=out_dtype, value=[0, 1])
        depth = node.attr('depth')
        if node.input('depth_tensor', 0) is not None:
            depth_node = node.input('depth_tensor', 0)
        else:
            depth_node = graph.make_node(
                'Constant', dtype=in_dtype, value=[depth])
        reshaped_input_node = graph.make_node(
            'OneHot',
            inputs=[inputs, depth_node, value_node],
            outputs=node.output('Out'))


@op_mapper('concat')
class Concat():
    support_opset_version_range = (4, 15)

    @classmethod
    def opset_4(cls, graph, node, **kw):
        inputs = node.input('X')

        input_dtypes = [node.input_dtype('X', i) for i in range(len(inputs))]
        inputs = mapper_helper.dtype_alignment(graph, inputs, input_dtypes)
        node_axis = node.input('AxisTensor')
        if node_axis is not None and len(node_axis) > 0:
            axis_node = node.input('AxisTensor')[0]
            try:
                axis = mapper_helper.get_value_from_parameters(graph,
                                                               axis_node)[0]
            except Exception as e:
                raise Exception(
                    "Currently does not support the axis parameter as input tensor"
                    + str(e))
        else:
            axis = node.attr('axis')
        if axis < 0:
            axis = axis + len(node.input_shape('X', 0))

        node = graph.make_node(
            'Concat', inputs=inputs, outputs=node.output('Out'), axis=axis)


@op_mapper('assign')
class Assign():
    support_opset_version_range = (1, 15)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        inputs = node.input('X')
        graph.make_node('Identity', inputs=inputs, outputs=node.output('Out'))


@op_mapper('lod_reset')
class LodReset():
    support_opset_version_range = (1, )

    @classmethod
    def opset_1(cls, graph, node, **kw):
        graph.make_node(
            'Identity', inputs=node.input('X'), outputs=node.output('Out'))


@op_mapper('eye')
class Eye():
    support_opset_version_range = (9, )

    @classmethod
    def opset_9(cls, graph, node, **kw):
        num_rows = node.attr('num_rows')
        num_columns = node.attr('num_columns')
        dtype = node.output_dtype('Out', 0)
        value = [0] * num_rows * num_columns
        value_tensor = mapper_helper.constant_helper(
            graph, dtype, value, shape=[num_rows, num_columns])
        graph.make_node(
            'EyeLike', inputs=[value_tensor], outputs=node.output('Out'))


@op_mapper('stack')
class Stack():
    support_opset_version_range = (4, 15)

    @classmethod
    def opset_4(cls, graph, node, **kw):
        inputs = node.input('X')
        input_dtypes = [node.input_dtype('X', i) for i in range(len(inputs))]
        inputs = mapper_helper.dtype_alignment(graph, inputs, input_dtypes)
        axis = node.attr('axis')

        unsqueezed_inputs = list()
        for ipt in inputs:
            unsqueezed_ipt = mapper_helper.unsqueeze_helper(graph, ipt, [axis])
            unsqueezed_inputs.append(unsqueezed_ipt)
        graph.make_node(
            'Concat',
            inputs=unsqueezed_inputs,
            outputs=node.output('Y'),
            axis=axis)


@op_mapper('unstack')
class Unstack():
    support_opset_version_range = (2, 15)

    @classmethod
    def opset_2(cls, graph, node, **kw):
        axis = node.attr('axis')
        ndim = node.block.vars[node.input('X')[0]].ndim
        axis = axis + ndim if axis < 0 else axis
        output_y = mapper_helper.split_helper(
            graph,
            node.input('X'),
            axis=axis,
            split=[1] * len(node.output('Y')),
            outputs=len(node.output('Y')))

        if isinstance(output_y, six.string_types):
            output_y = [output_y]

        for i in range(len(output_y)):
            mapper_helper.squeeze_helper(graph, output_y[i], [axis],
                                         node.output('Y', i))


@op_mapper('expand_as_v2')
class ExpandAsV2():
    support_opset_version_range = (8, 15)

    @classmethod
    def opset_8(cls, graph, node, **kw):
        target_shape = node.attr('target_shape')
        if node.input('target_tensor', 0) is not None:
            target_shape = graph.make_node(
                'Shape', inputs=[node.input('target_tensor', 0)])
        elif target_shape is not None:
            target_shape = graph.make_node(
                'Constant',
                attrs={'dtype': dtypes.ONNX.INT64,
                       'value': target_shape})
        else:
            raise Exception(
                "Not find attribute: 'target_shape' or tensor 'target_tensor'")
        node = graph.make_node(
            'Expand',
            inputs=[node.input('X', 0), target_shape],
            outputs=node.output('Out'))


@op_mapper('expand_v2')
class ExpandV2():
    support_opset_version_range = (8, 15)

    @classmethod
    def opset_8(cls, graph, node, **kw):
        expand_shape, _ = mapper_helper.get_node_attr_value(
            graph,
            node,
            'shape',
            'Shape',
            'expand_shapes_tensor',
            dtype=dtypes.ONNX.INT64)

        input_shape = node.input_shape('X', 0)
        input_shape_node = graph.make_node('Shape', inputs=node.input('X', 0))

        node_shape = node.attr('shape')
        node_shape_tensor = node.input('Shape')
        node_shape_tensor_list = node.input('expand_shapes_tensor')
        if node_shape_tensor is not None and len(node_shape_tensor) > 0:
            diff = node.input_shape('Shape', 0)[0] - len(input_shape)
        elif node_shape_tensor_list is not None and \
                len(node_shape_tensor_list) > 0:
            diff = len(node_shape_tensor_list) - len(input_shape)
        elif node_shape is not None and len(node_shape) > 0:
            diff = len(node_shape) - len(input_shape)
            expand_shape = graph.make_node(
                'Constant', dtype=dtypes.ONNX.INT64, value=expand_shape)

        if diff > 0:
            one_node = graph.make_node(
                'Constant',
                attrs={'dtype': dtypes.ONNX.INT64,
                       'value': [1] * diff})
            input_shape_node = graph.make_node(
                'Concat', inputs=[one_node, input_shape_node], axis=0)

        if graph.opset_version < 12:
            input_shape_node = graph.make_node(
                'Cast', inputs=[input_shape_node], to=dtypes.ONNX.FLOAT)
            expand_shape = graph.make_node(
                'Cast', inputs=[expand_shape], to=dtypes.ONNX.FLOAT)
            shape = graph.make_node(
                'Max', inputs=[input_shape_node, expand_shape])
            shape = graph.make_node(
                'Cast', inputs=[shape], to=dtypes.ONNX.INT64)
        else:
            shape = graph.make_node(
                'Max', inputs=[input_shape_node, expand_shape])
        node = graph.make_node(
            'Expand',
            inputs=[node.input('X', 0), shape],
            outputs=node.output('Out'))


@op_mapper('shape')
class Shape():
    support_opset_version_range = (6, 15)

    @classmethod
    def opset_6(cls, graph, node, **kw):
        shape_node = graph.make_node('Shape', inputs=node.input('Input'))
        graph.make_node(
            'Cast',
            inputs=[shape_node],
            outputs=node.output('Out'),
            to=dtypes.ONNX.INT32)


@op_mapper('size')
class Numel():
    supports_opset_version_range = (1, 15)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        size_node = graph.make_node('Size', inputs=node.input('Input'))
        mapper_helper.unsqueeze_helper(graph, size_node, [0],
                                       node.output('Out'))


@op_mapper('split')
class Split():
    support_opset_version_range = (1, 15)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        sections = node.attr('sections')
        axis = cls.get_axis(graph, node)
        if isinstance(sections, list) and len(sections) == 1:
            graph.make_node(
                'Identity', inputs=node.input('X'), outputs=node.output('Out'))
        else:
            if len(sections) > 0:
                input_shape = node.block.vars[node.input('X')[0]].shape
                section_index = [
                    i for i, val in enumerate(sections) if val == -1
                ]
                if input_shape[axis] != -1 and len(section_index) == 1:
                    sections[section_index[0]] = input_shape[axis] - sum(
                        sections) - 1
                mapper_helper.split_helper(
                    graph,
                    node.input('X'),
                    axis=axis,
                    split=sections,
                    outputs=node.output('Out'))
            else:
                graph.make_node(
                    'Split',
                    inputs=node.input('X'),
                    outputs=node.output('Out'),
                    axis=axis)

    @classmethod
    def get_axis(cls, graph, node):
        if len(node.input('AxisTensor')) > 0:
            axis_node = node.input('AxisTensor')[0]
            # When axis is tensor, only int32 and int64 are supported
            if axis_node not in graph.parameters:
                raise Exception(
                    "Currently does not support the axis parameter as input tensor!"
                )
            else:
                axis = graph.parameters[axis_node].attribute[0].t.int32_data
                if axis is None or len(axis) < 1:
                    axis = graph.parameters[axis_node].attribute[
                        0].t.int64_data[0]
        else:
            axis = node.attr('axis')
        return axis


@op_mapper(['roll'])
class Roll():
    support_opset_version_range = (4, 15)

    @classmethod
    def roll(cls, graph, node, input_x, dims, shifts):
        for i in range(len(dims)):
            if graph.opset_version >= 10 and isinstance(shifts,
                                                        six.string_types):
                to_dtype = dtypes.DTYPE_PADDLE_ONNX_MAP[node.input_dtype(
                    'ShiftsTensor', 0)]
                const_i = graph.make_node('Constant', dtype=to_dtype, value=i)
                const_0 = graph.make_node('Constant', dtype=to_dtype, value=0)
                shift_node = graph.make_node(
                    'Gather', inputs=[shifts, const_i], axis=0)
                shift_node = graph.make_node(
                    "Sub", inputs=[const_0, shift_node])
                shift_node = mapper_helper.unsqueeze_helper(graph, shift_node,
                                                            [0])
            elif graph.opset_version < 10 and isinstance(shifts,
                                                         six.string_types):
                raise Exception(
                    "shifts of roll is Tensor, please try with higher onnx opset_version>=10."
                )
            else:
                shift_node = [-shifts[i]]
                to_dtype = dtypes.ONNX.INT64
            shapes = []
            shape = mapper_helper.slice_helper(
                graph, input_x, [dims[i]], shift_node, [60000], dtype=to_dtype)
            shapes.append(shape)
            shape = mapper_helper.slice_helper(
                graph, input_x, [dims[i]], [0], shift_node, dtype=to_dtype)
            shapes.append(shape)
            input_x = graph.make_node('Concat', inputs=shapes, axis=dims[i])
        return input_x

    @classmethod
    def flatten(cls, graph, node):
        dims = len(node.input_shape('X', 0))
        start_axis = 0
        end_axis = dims - 1
        shape_node = graph.make_node('Shape', inputs=node.input('X'))
        if end_axis < dims - 1:
            slice1 = mapper_helper.slice_helper(
                graph, shape_node, axes=[0], starts=[0], ends=[start_axis])
            slice3 = mapper_helper.slice_helper(
                graph, shape_node, axes=[0], starts=[end_axis + 1],
                ends=[dims])
            slices = [
                slice1, graph.make_node(
                    'Constant', value=[-1], dtype=dtypes.ONNX.INT64), slice3
            ]
        else:
            slice1 = mapper_helper.slice_helper(
                graph, shape_node, axes=[0], starts=[0], ends=[start_axis])
            slices = [
                slice1, graph.make_node(
                    'Constant', value=[-1], dtype=dtypes.ONNX.INT64)
            ]
        final_shape = graph.make_node('Concat', inputs=slices, axis=0)
        output = graph.make_node(
            'Reshape', inputs=[node.input('X')[0], final_shape])
        return output

    @classmethod
    def opset_4(cls, graph, node, **kw):
        dims = node.attr('axis')
        shifts = node.attr('shifts')
        input_x = node.input('X')[0]
        input_shape = node.input_shape('X', 0)
        shifts_node = node.input('ShiftsTensor')
        if len(dims) > 0:
            axes = [
                axis + len(input_shape) if axis < 0 else axis
                for i, axis in enumerate(dims)
            ]
            if shifts_node is not None and len(shifts_node) > 0:
                shifts = shifts_node[0]
            else:
                for i in range(0, len(axes)):
                    if input_shape[axes[i]] > 0:
                        assert -input_shape[axes[i]] <= shifts[i] <= input_shape[axes[i]], \
                            "the value of shifts in axis is less than the value of input_shape in axis."

            input_x = cls.roll(graph, node, input_x, axes, shifts)
            graph.make_node(
                'Identity', inputs=[input_x], outputs=node.output('Out'))
        else:
            if shifts_node is not None and len(shifts_node) > 0:
                shifts = shifts_node[0]
            input_x = cls.flatten(graph, node)
            input_x = cls.roll(graph, node, input_x, [0], shifts)
            shape_node = graph.make_node(
                'Constant',
                attrs={'dtype': dtypes.ONNX.INT64,
                       'value': list(input_shape)})
            graph.make_node(
                'Reshape',
                inputs=[input_x, shape_node],
                outputs=node.output('Out'))


@op_mapper(['slice', 'strided_slice'])
class Slice():
    support_opset_version_range = (1, 15)

    @classmethod
    def decrease_axis(cls, node):
        # tensor[i,:] will decrease rank of origin input, example:
        # paddle.slice() will not decrease rank of origin input
        # if input shape is [2, 3], input[0, :] will generate output with shape [3], not [1, 3].
        # paddle.slice(input, 0, 1, 0) will  generate output with shape [1, 3], not [3].

        decrease_axis = node.attr('decrease_axis')
        if len(decrease_axis) == 0:
            return None
        if node.output_shape('Out', 0) == [0]:
            return decrease_axis
        if len(node.input_shape('Input', 0)) > len(node.output_shape('Out', 0)):
            return decrease_axis
        return None

    @classmethod
    def opset_1(cls, graph, node, **kw):
        axes = node.attr('axes')
        strides, strides_is_tensor = mapper_helper.get_node_attr_value(
            graph, node, 'strides', 'StridesTensor', 'StridesTensorList', True)
        strides = [1] * len(axes) if strides is None else strides
        steps = [i for i, val in enumerate(strides) if val == 1]
        assert len(steps) == len(axes), \
            "Slice in onnx(opset<10) not support attribute 'step', Try converting with opset_version >=10"

        starts, start_is_tensor = mapper_helper.get_node_attr_value(
            graph, node, 'starts', 'StartsTensor', 'StartsTensorList', True)
        ends, end_is_tensor = mapper_helper.get_node_attr_value(
            graph, node, 'ends', 'EndsTensor', 'EndsTensorList', True)

        assert not strides_is_tensor and not start_is_tensor and not end_is_tensor, \
            "Slice in onnx(opset<10) not support attribute 'steps','starts' or 'ends' which have tensor value, " \
            "Try converting with opset_version >=10 "

        decrease_axis = cls.decrease_axis(node)
        if decrease_axis is None:
            graph.make_node(
                "Slice",
                inputs=[node.input('Input')[0]],
                outputs=node.output('Out'),
                axes=axes,
                starts=starts,
                ends=ends)
        else:
            sliced = graph.make_node(
                "Slice",
                inputs=[node.input('Input')[0]],
                axes=axes,
                starts=starts,
                ends=ends)
            mapper_helper.squeeze_helper(graph, sliced, decrease_axis,
                                         node.output('Out'))

    @classmethod
    def opset_10(cls, graph, node, **kw):
        axes = node.attr('axes')
        strides, _ = mapper_helper.get_node_attr_value(
            graph,
            node,
            'strides',
            'StridesTensor',
            'StridesTensorList',
            dtype=dtypes.ONNX.INT64)
        strides = [1] * len(axes) if strides is None else strides

        starts, _ = mapper_helper.get_node_attr_value(
            graph,
            node,
            'starts',
            'StartsTensor',
            'StartsTensorList',
            dtype=dtypes.ONNX.INT64)
        ends, _ = mapper_helper.get_node_attr_value(
            graph,
            node,
            'ends',
            'EndsTensor',
            'EndsTensorList',
            dtype=dtypes.ONNX.INT64)

        if isinstance(starts, list):
            starts_node = graph.make_node(
                'Constant',
                attrs={'dtype': dtypes.ONNX.INT64,
                       'value': starts})
        else:
            starts_node = starts
        if isinstance(ends, list):
            ends_node = graph.make_node(
                'Constant', attrs={'dtype': dtypes.ONNX.INT64,
                                   'value': ends})
        else:
            ends_node = ends

        if isinstance(strides, list):
            strides_node = graph.make_node(
                'Constant',
                attrs={'dtype': dtypes.ONNX.INT64,
                       'value': strides})
        else:
            strides_node = strides

        steps_node = strides_node
        axes_node = graph.make_node(
            'Constant', attrs={'dtype': dtypes.ONNX.INT64,
                               'value': axes})

        decrease_axis = cls.decrease_axis(node)
        if decrease_axis is None:
            sliced = graph.make_node(
                "Slice",
                inputs=[
                    node.input('Input')[0], starts_node, ends_node, axes_node,
                    steps_node
                ],
                outputs=node.output('Out'))
        else:
            sliced = graph.make_node(
                "Slice",
                inputs=[
                    node.input('Input')[0], starts_node, ends_node, axes_node,
                    steps_node
                ])
            mapper_helper.squeeze_helper(graph, sliced, decrease_axis,
                                         node.output('Out'))


@op_mapper(['sequence_expand'])
class SequenceExpand():
    support_opset_version_range = ()

    @classmethod
    def opset_1(cls, graph, node, **kw):
        graph.make_node(
            'Identity', inputs=node.input('X'), outputs=node.output('Out'))


@op_mapper(['expand'])
class Expand():
    support_opset_version_range = (6, 15)

    @classmethod
    def opset_6(cls, graph, node, **kw):
        expand_times, _ = mapper_helper.get_node_attr_value(
            graph,
            node,
            'expand_times',
            'ExpandTimes',
            'expand_times_tensor',
            dtype=dtypes.ONNX.INT64)

        if isinstance(expand_times, list):
            expand_times = graph.make_node(
                'Constant',
                attrs={'dtype': dtypes.ONNX.INT64,
                       'value': expand_times})

        graph.make_node(
            "Tile",
            inputs=[node.input('X', 0), expand_times],
            outputs=node.output('Out'))


@op_mapper(['tile'])
class Tile():
    support_opset_version_range = (6, 15)

    @classmethod
    def opset_6(cls, graph, node, **kw):
        repeat_times, _ = mapper_helper.get_node_attr_value(
            graph,
            node,
            'repeat_times',
            'RepeatTimes',
            'repeat_times_tensor',
            dtype=dtypes.ONNX.INT64)

        if isinstance(repeat_times, list):
            repeat_times = graph.make_node(
                'Constant',
                attrs={'dtype': dtypes.ONNX.INT64,
                       'value': repeat_times})

        graph.make_node(
            "Tile",
            inputs=[node.input('X', 0), repeat_times],
            outputs=node.output('Out'))


@op_mapper('range')
class Range():
    support_opset_version_range = (11, 15)

    @classmethod
    def opset_11(cls, graph, node, **kw):
        start = node.input('Start', 0)
        end = node.input('End', 0)
        step = node.input('Step', 0)
        start_t = mapper_helper.squeeze_helper(graph, start, [0])
        end_t = mapper_helper.squeeze_helper(graph, end, [0])
        step_t = mapper_helper.squeeze_helper(graph, step, [0])
        graph.make_node(
            "Range",
            inputs=[start_t, end_t, step_t],
            outputs=node.output('Out'))


@op_mapper('fill_constant')
class Constant():
    support_opset_version_range = (1, 15)

    @classmethod
    def check_int_type(cls, dtype):
        if dtype in [dtypes.ONNX.INT16, dtypes.ONNX.INT32, dtypes.ONNX.INT64]:
            return True
        return False

    @classmethod
    def opset_1(cls, graph, node, **kw):
        value = node.attr('value')
        dtype = node.attr('dtype')
        value_is_scalar_tensor = False
        if 'ValueTensor' in node.inputs and len(node.input('ValueTensor')) > 0:
            rank = len(node.input_shape("ValueTensor", 0))
            if rank == 1 and node.input_shape("ValueTensor", 0)[0] == 1:
                value_is_scalar_tensor = True
                value = node.input("ValueTensor")[0]
            else:
                raise Exception(
                    "paddle.full with tensor value parameter is not supported yet."
                )

        shape, is_shape_tensor = mapper_helper.get_node_attr_value(
            graph,
            node,
            'shape',
            'ShapeTensor',
            'ShapeTensorList',
            dtype=dtypes.ONNX.INT64)

        if graph.opset_version >= 9 and (is_shape_tensor or
                                         value_is_scalar_tensor):
            if not is_shape_tensor:
                shape = graph.make_node(
                    'Constant', dtype=dtypes.ONNX.INT64, value=shape)
            input_dtype = dtypes.DTYPE_PADDLE_ONNX_MAP[dtype]
            if not value_is_scalar_tensor and cls.check_int_type(input_dtype):
                to_dtype = dtypes.ONNX.DOUBLE
                outputs = None
            else:
                to_dtype = input_dtype
                outputs = node.output('Out')

            if value_is_scalar_tensor:
                base_value = graph.make_node(
                    'ConstantOfShape',
                    inputs=shape,
                    attrs={'dims': [1],
                           'dtype': to_dtype,
                           'value': 0})
                node2 = graph.make_node(
                    "Add", inputs=[base_value, value], outputs=outputs)
            else:
                node2 = graph.make_node(
                    'ConstantOfShape',
                    inputs=shape,
                    outputs=outputs,
                    attrs={'dims': [1],
                           'dtype': to_dtype,
                           'value': value})

            if not value_is_scalar_tensor and cls.check_int_type(input_dtype):
                graph.make_node(
                    'Cast',
                    inputs=node2,
                    outputs=node.output('Out'),
                    attrs={'to': input_dtype})
        else:
            assert not is_shape_tensor and not value_is_scalar_tensor, \
                "Currently op ['fill_constant'] does not support in onnx(opset<9) when 'shape' or 'fill_value' has " \
                "tensor, Try converting with opset_version >=9 "

            value = np.ones(shape) * value
            value = value.astype(dtypes.DTYPE_PADDLE_NUMPY_MAP[dtype])
            value = value.flatten().tolist()

            graph.make_node(
                'Constant',
                inputs=[],
                outputs=node.output('Out'),
                attrs={
                    'dims': shape,
                    'dtype': dtypes.DTYPE_PADDLE_ONNX_MAP[dtype],
                    'value': value
                })


@op_mapper(['lookup_table_v2', 'lookup_table'])
class Embedding():
    support_opset_version_range = (1, 15)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        ids = node.input('Ids', 0)
        if node.type == 'lookup_table' and node.input_shape('Ids', 0)[-1] == 1:
            ids = mapper_helper.squeeze_helper(graph,
                                               node.input('Ids', 0), [-1])
        padding_idx = node.attr('padding_idx')
        input_shape = node.input_shape('W', 0)
        if padding_idx != -1:
            key = node.input('W', 0)
            if -1 in input_shape:
                assert False, "opset version < 11 do not support padding_idx !=-1 and weight is tensor with dynamic shape, please set opset version > 11 or use input_spec to set input shape"
            else:
                data = np.ones(shape=input_shape, dtype=np.float32)
                data[padding_idx] = 0.0
                dtype = dtypes.DTYPE_PADDLE_ONNX_MAP[node.input_dtype('W', 0)]
                constant = graph.make_node(
                    'Constant',
                    dtype=dtype,
                    dims=input_shape,
                    value=data.flatten().tolist())
                weight_node = graph.make_node(
                    'Mul', inputs=[node.input('W', 0), constant])
                graph.make_node(
                    'Gather',
                    inputs=[weight_node, ids],
                    outputs=node.output('Out'))
        else:
            graph.make_node(
                'Gather',
                inputs=[node.input('W', 0), ids],
                outputs=node.output('Out'))

    @classmethod
    def opset_11(cls, graph, node, **kw):
        ids = node.input('Ids', 0)
        if node.type == 'lookup_table' and node.input_shape('Ids', 0)[-1] == 1:
            ids = mapper_helper.squeeze_helper(graph,
                                               node.input('Ids', 0), [-1])

        padding_idx = node.attr('padding_idx')
        input_shape = node.input_shape('W', 0)
        if padding_idx != -1:
            if -1 in input_shape:
                replace_shape = list(copy.copy(input_shape))
                del (replace_shape[0])
                replace_data = graph.make_node(
                    'Constant',
                    dtype=dtypes.DTYPE_PADDLE_ONNX_MAP[node.input_dtype('W',
                                                                        0)],
                    dims=replace_shape,
                    value=[0.0] * np.prod(replace_shape))
                index = graph.make_node(
                    'Constant', dtype=dtypes.ONNX.INT64, value=[padding_idx])
                Scatter_node = graph.make_node(
                    'ScatterND',
                    inputs=[node.input('W', 0), index, replace_data])
                graph.make_node(
                    'Gather',
                    inputs=[Scatter_node, ids],
                    outputs=node.output('Out'))
            else:
                data = np.ones(shape=input_shape, dtype=np.float32)
                data[padding_idx] = 0.0
                dtype = dtypes.DTYPE_PADDLE_ONNX_MAP[node.input_dtype('W', 0)]
                constant = graph.make_node(
                    'Constant',
                    dtype=dtype,
                    dims=input_shape,
                    value=data.flatten().tolist())
                weight_node = graph.make_node(
                    'Mul', inputs=[node.input('W', 0), constant])
                graph.make_node(
                    'Gather',
                    inputs=[weight_node, ids],
                    outputs=node.output('Out'))
        else:
            graph.make_node(
                'Gather',
                inputs=[node.input('W', 0), ids],
                outputs=node.output('Out'))


@op_mapper('fill_constant_batch_size_like')
class FillConstantBatchSizeLike():
    support_opset_version_range = (9, 12)

    @classmethod
    def opset_10(cls, graph, node, **kw):
        out_shape = node.attr('shape')
        input_dim_idx = node.attr('input_dim_idx')
        output_dim_idx = node.attr('output_dim_idx')

        del out_shape[output_dim_idx]
        out_shape.insert(0, 1)

        dtype = dtypes.DTYPE_PADDLE_ONNX_MAP[node.attr('dtype')]
        if node.attr("str_value") is not None and node.attr("str_value") != "":
            value = eval(node.attr("str_value"))
        else:
            value = node.attr('value')
        input_shape = node.input_shape('Input', 0)
        constant = graph.make_node(
            'Constant',
            dtype=dtype,
            dims=out_shape,
            value=[value] * np.prod(out_shape))

        shape = graph.make_node('Shape', inputs=node.input('Input'))
        start = graph.make_node(
            'Constant', dtype=dtypes.ONNX.INT64, value=[input_dim_idx])
        end = graph.make_node(
            'Constant', dtype=dtypes.ONNX.INT64, value=[input_dim_idx + 1])
        batch = graph.make_node('Slice', inputs=[shape, start, end])
        repeat = batch
        if len(out_shape) > 1:
            repeat = graph.make_node(
                'Constant',
                dtype=dtypes.ONNX.INT64,
                value=[1] * (len(out_shape) - 1))
            repeat = graph.make_node('Concat', inputs=[batch, repeat], axis=-1)
        if output_dim_idx == 0:
            graph.make_node(
                'Tile', inputs=[constant, repeat], outputs=node.output('Out'))
        else:
            out = graph.make_node('Tile', inputs=[constant, repeat])
            perm = list(range(len(out_shape)))
            del perm[0]
            perm.insert(output_dim_idx, 0)
            graph.make_node(
                'Transpose',
                inputs=[out],
                perm=perm,
                outputs=node.output('Out'))


@op_mapper('fill_any_like')
class FullLike():
    '''
    fill_any_like is kernel for paddle op::full_like & ones_like
    '''
    support_opset_version_range = (9, 15)

    @classmethod
    def opset_9(cls, graph, node, **kw):
        shape_node = graph.make_node('Shape', inputs=node.input('X'))
        value = node.attr('value')
        dtype = node.attr('dtype')
        input_dtype = node.input_dtype('X', 0)
        if dtype is None:
            dtype = input_dtype
        np_dtype = dtypes.DTYPE_PADDLE_STR_MAP[dtype]
        onnx_dtype = dtypes.DTYPE_PADDLE_ONNX_MAP[dtype]
        graph.make_node(
            'ConstantOfShape',
            inputs=[shape_node],
            outputs=node.output('Out'),
            dims=[1],
            dtype=onnx_dtype,
            value=np.array(value).astype(np_dtype).tolist())


@op_mapper('fill_zeros_like')
class FullZeroLike():
    '''
    fill_zeros_like is kernel for paddle op::zeros_like
    '''
    support_opset_version_range = (9, 15)

    @classmethod
    def opset_9(cls, graph, node, **kw):
        shape_node = graph.make_node('Shape', inputs=node.input('X'))
        value = 0
        dtype = node.attr('dtype')
        input_dtype = node.input_dtype('X', 0)
        if dtype is None:
            dtype = input_dtype
        np_dtype = dtypes.DTYPE_PADDLE_STR_MAP[dtype]
        onnx_dtype = dtypes.DTYPE_PADDLE_ONNX_MAP[dtype]
        graph.make_node(
            'ConstantOfShape',
            inputs=[shape_node],
            outputs=node.output('Out'),
            dims=[1],
            dtype=onnx_dtype,
            value=np.array(value).astype(np_dtype).tolist())


@op_mapper('gather_nd')
class Gather_nd():
    support_opset_version_range = (11, 15)

    @classmethod
    def opset_11(cls, graph, node, **kw):
        data = node.input('X', 0)
        index = node.input('Index', 0)
        index_dtype = node.input_dtype('Index', 0)
        index_node = None
        if index_dtype != paddle.int64:
            index_node = graph.make_node(
                'Cast', inputs=[node.input('Index', 0)], to=dtypes.ONNX.INT64)
        else:
            index_node = index
        graph.make_node(
            'GatherND', inputs=[data, index_node], outputs=node.output('Out'))


@op_mapper('gather')
class Gather():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        axis = node.attr('axis')
        if node.input('Axis', 0) != None:
            axis_node = node.input('Axis', 0)
            try:
                axis = mapper_helper.get_value_from_parameters(graph,
                                                               axis_node)[0]
            except Exception as e:
                raise Exception(
                    "Currently does not support the axis parameter as input tensor"
                    + str(e))
        if axis is None:
            axis = 0
        if len(node.input_shape('Index', 0)) == 1:
            # gather
            graph.make_node(
                'Gather',
                inputs=[node.input('X', 0), node.input('Index', 0)],
                outputs=node.output('Out'),
                attrs={'axis': axis})
        else:
            raise Exception(
                "please try to convert OP:gather(indices's rank >1) with opset_version >= 11."
            )

    @classmethod
    def opset_11(cls, graph, node, **kw):
        axis = node.attr('axis')
        if node.input('Axis', 0) != None:
            axis_node = node.input('Axis', 0)
            try:
                axis = mapper_helper.get_value_from_parameters(graph,
                                                               axis_node)[0]
            except Exception as e:
                raise Exception(
                    "Currently does not support the axis parameter as input tensor"
                    + str(e))
        if axis is None:
            axis = 0
        if len(node.input_shape('Index', 0)) == 1:
            # gather
            graph.make_node(
                'Gather',
                inputs=[node.input('X', 0), node.input('Index', 0)],
                outputs=node.output('Out'),
                attrs={'axis': axis})
        else:
            # gather_nd
            index_dtype = node.input_dtype('Index', 0)
            if index_dtype != paddle.int64:
                index_node = graph.make_node(
                    'Cast',
                    inputs=[node.input('Index', 0)],
                    to=dtypes.ONNX.INT64)
                graph.make_node(
                    'GatherND',
                    inputs=[node.input('X', 0), index_node],
                    outputs=node.output('Out'))
            else:
                graph.make_node(
                    'GatherND',
                    inputs=[node.input('X', 0), node.input('Index', 0)],
                    outputs=node.output('Out'))


@op_mapper('squeeze2')
class Squeeze():
    support_opset_version_range = (1, 15)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        shape = node.input_shape('X', 0)
        ret = [i for i, val in enumerate(shape) if val > 1]
        if len(ret) == len(shape):
            graph.make_node(
                'Identity', inputs=node.input('X'), outputs=node.output('Out'))
        else:
            axes = cls.compute_axes(graph, node)
            if len(axes) > 0:
                axes.sort()
                mapper_helper.squeeze_helper(graph,
                                             node.input('X', 0), axes,
                                             node.output('Out'))
            else:
                graph.make_node(
                    'Squeeze',
                    inputs=[node.input('X', 0)],
                    outputs=node.output('Out'))

    @classmethod
    def compute_axes(cls, graph, node):
        shape = node.input_shape('X', 0)
        axes = node.attr('axes')
        if len(axes) > 0:
            axes = [
                axis + len(shape) if axis < 0 else axis
                for i, axis in enumerate(axes)
            ]
        return axes


@op_mapper('assign_value')
class Assign():
    support_opset_version_range = (1, 15)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        if len(node.input_names) > 0:
            graph.make_node(
                'Identity', inputs=node.input('X'), outputs=node.output('Out'))
        else:
            parameters = {}
            value = np.array(node.attr('fp32_values'))
            if value is None or value.size < 1:
                value = np.array(node.attr('int32_values'))
            if value is None or value.size < 1:
                value = np.array(node.attr('int64_values'))
            parameter = {
                'data': value,
                'dtype': node.output_dtype("Out", 0),
                'shape': node.attr('shape')
            }
            parameters[node.output('Out', 0)] = parameter
            graph.build_parameters(parameters)


@op_mapper('transpose2')
class Transpose():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_7(cls, graph, node, **kw):
        graph.make_node(
            'Transpose',
            inputs=node.input('X'),
            outputs=node.output('Out'),
            perm=node.attr('axis'))


@op_mapper('flatten2')
class Flatten():
    support_opset_version_range = (1, 15)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        input_dtype = dtypes.DTYPE_PADDLE_ONNX_MAP[node.input_dtype('X', 0)]
        if input_dtype in [dtypes.ONNX.INT32, dtypes.ONNX.INT64
                           ] and graph.opset_version < 9:
            raise Exception(
                "int32 or int64 not supported in onnx <9, please try with higher onnx opset_version>=9."
            )

        graph.make_node(
            'Flatten',
            inputs=node.input('X'),
            outputs=node.output('Out'),
            axis=node.attr('axis'))


@op_mapper('flatten_contiguous_range')
class FlattenContiguousRange():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_7(cls, graph, node, **kw):
        dims = len(node.input_shape('X', 0))
        start_axis = node.attr('start_axis')
        end_axis = node.attr('stop_axis')
        shape_node = graph.make_node('Shape', inputs=node.input('X'))
        if start_axis < 0:
            start_axis += dims
        if end_axis < 0:
            end_axis += dims
        if start_axis == 0 and end_axis == dims - 1:
            final_shape = graph.make_node(
                'Constant', value=[-1], dtype=dtypes.ONNX.INT64)
        elif start_axis == 0:
            slice_end = mapper_helper.slice_helper(
                graph, shape_node, axes=[0], starts=[end_axis + 1],
                ends=[dims])
            slices = [
                graph.make_node(
                    'Constant', value=[-1], dtype=dtypes.ONNX.INT64), slice_end
            ]
            final_shape = graph.make_node('Concat', inputs=slices, axis=0)
        elif end_axis == dims - 1:
            slice_start = mapper_helper.slice_helper(
                graph, shape_node, axes=[0], starts=[0], ends=[start_axis])
            slices = [
                slice_start, graph.make_node(
                    'Constant', value=[-1], dtype=dtypes.ONNX.INT64)
            ]
            final_shape = graph.make_node('Concat', inputs=slices, axis=0)
        else:
            slice_start = mapper_helper.slice_helper(
                graph, shape_node, axes=[0], starts=[0], ends=[start_axis])
            slice_end = mapper_helper.slice_helper(
                graph, shape_node, axes=[0], starts=[end_axis + 1],
                ends=[dims])
            slices = [
                slice_start, graph.make_node(
                    'Constant', value=[-1], dtype=dtypes.ONNX.INT64), slice_end
            ]
            final_shape = graph.make_node('Concat', inputs=slices, axis=0)
        graph.make_node(
            'Reshape',
            inputs=[node.input('X')[0], final_shape],
            outputs=node.output('Out'))


@op_mapper('reshape2')
class Reshape():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_7(cls, graph, node, **kw):
        shape_name = 'ShapeTensor'
        if shape_name not in node.inputs or len(node.input(shape_name)) == 0:
            shape_name = 'Shape'
        if shape_name not in node.inputs or len(node.input(shape_name)) == 0:
            if node.attr('shape') is None or len(node.attr('shape')) == 0:
                raise Exception("shape tensor and shape attrubite all unkown.")
        if len(node.input(shape_name)) > 1:
            dims = []
            for i in range(len(node.input(shape_name))):
                dim = node.input(shape_name)[i]
                dim = graph.make_node(
                    'Cast', inputs=[dim], to=dtypes.ONNX.INT64)
                dims.append(dim)
            shape = graph.make_node('Concat', inputs=dims, axis=-1)
            graph.make_node(
                'Reshape',
                inputs=[node.input('X')[0], shape],
                outputs=node.output('Out'))
        elif len(node.input(shape_name)) == 1:
            cast_shape_node = graph.make_node(
                'Cast', inputs=node.input(shape_name), to=dtypes.ONNX.INT64)
            graph.make_node(
                'Reshape',
                inputs=[node.input('X')[0], cast_shape_node],
                outputs=node.output('Out'))
        elif node.attr('shape') is not None and len(node.attr('shape')) > 0:
            shape_node = graph.make_node(
                'Constant',
                attrs={
                    'dtype': dtypes.ONNX.INT64,
                    'value': node.attr('shape')
                })
            reshape_node = graph.make_node(
                'Reshape',
                inputs=[node.input('X')[0], shape_node],
                outputs=node.output('Out'))


@op_mapper('unsqueeze2')
class Unsqueeze():
    support_opset_version_range = (1, 15)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        axes = cls.get_axes(graph, node)
        mapper_helper.unsqueeze_helper(graph,
                                       node.input('X'), axes,
                                       node.output('Out'))

    @classmethod
    def opset_13(cls, graph, node, **kw):
        axes_node = cls.get_axes(graph, node, return_node=True)
        graph.make_node(
            'Unsqueeze',
            inputs=node.input('X') + [axes_node],
            outputs=node.output('Out'))

    @classmethod
    def get_axes(cls, graph, node, return_node=False):
        axes_node = None
        ndim = node.block.vars[node.input('X')[0]].ndim
        if len(node.attr('axes')) > 0:
            axes = node.attr('axes')
        else:
            axes_node = node.input('AxesTensor')[0]
            if axes_node is not None and graph.opset_version > 12 and return_node:
                return axes_node
            try:
                axes = mapper_helper.get_value_from_parameters(graph, axes_node)
            except Exception as e:
                raise Exception(
                    "Currently does not support the axes parameter as input tensor in onnx(opset<13), "
                    "Try converting with opset_version >=13 " + str(e))
        # axes is list of non-negative integers
        axes = [
            axis + ndim + i + 1 if axis < 0 else axis
            for i, axis in enumerate(axes)
        ]

        axes_copy = axes.copy()
        assert sorted(
            axes) == axes_copy, "axes must be arranged in the following order"
        assert len(set(axes)) == len(axes), "axes have duplicate axis"

        if return_node:
            if axes_node is None:
                axes_node = graph.make_node(
                    'Constant',
                    attrs={'dtype': dtypes.ONNX.INT64,
                           'value': axes})
            return axes_node
        return axes


@op_mapper('reciprocal')
class Reciprocal():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        graph.make_node(
            'Reciprocal', inputs=node.input('X'), outputs=node.output('Out'))


@op_mapper('cast')
class Cast():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        graph.make_node(
            'Cast',
            inputs=node.input('X'),
            outputs=node.output('Out'),
            to=dtypes.DTYPE_PADDLE_ONNX_MAP[node.attr('out_dtype')])


@op_mapper('linspace')
class Linspace():
    support_opset_version_range = (9, 15)

    @classmethod
    def opset_9(cls, graph, node, **kw):
        start = node.input('Start', 0)
        stop = node.input('Stop', 0)
        num = node.input('Num', 0)
        dtype = node.attr('dtype')

        start = graph.make_node('Cast', inputs=[start], to=dtypes.ONNX.FLOAT)
        stop = graph.make_node('Cast', inputs=[stop], to=dtypes.ONNX.FLOAT)

        sub_a_node = graph.make_node('Sub', inputs=[stop, start])

        one_node = graph.make_node(
            'Constant',
            dtype=dtypes.DTYPE_PADDLE_ONNX_MAP[node.input_dtype('Num', 0)],
            value=[1])

        sub_b_node = graph.make_node('Sub', inputs=[num, one_node])

        sub_b_float_node = graph.make_node(
            'Cast', inputs=[sub_b_node], to=dtypes.ONNX.FLOAT)

        step = graph.make_node('Div', inputs=[sub_a_node, sub_b_float_node])

        range_tensor = graph.make_node(
            'Cast', inputs=[num], to=dtypes.ONNX.INT64)

        one_like_node = graph.make_node(
            'ConstantOfShape',
            inputs=[range_tensor],
            dtype=dtypes.ONNX.FLOAT,
            value=[1])

        none_zero_node = graph.make_node('NonZero', inputs=[one_like_node])

        trans_none_zero_node = graph.make_node(
            'Transpose', inputs=[none_zero_node], perm=[1, 0])

        trans_squeeze = mapper_helper.squeeze_helper(graph,
                                                     trans_none_zero_node, [1])

        trans_squeeze = graph.make_node(
            'Cast', inputs=[trans_squeeze], to=dtypes.ONNX.FLOAT)

        mul_node = graph.make_node('Mul', inputs=[trans_squeeze, step])

        add_node = graph.make_node('Add', inputs=[mul_node, start])
        graph.make_node(
            'Cast',
            inputs=[add_node],
            outputs=node.output('Out'),
            to=dtypes.DTYPE_PADDLE_ONNX_MAP[node.input_dtype('Start', 0)])


@op_mapper('clip')
class Clip():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        min_value = node.attr('min')
        max_value = node.attr('max')
        if node.input('Max', 0) is None or len(node.input('Max')) == 0:
            max_ = max_value
        else:
            max_ = node.input('Max', 0)
        if node.input('Min', 0) is None or len(node.input('Min')) == 0:
            min_ = min_value
        else:
            min_ = node.input('Min', 0)
        mapper_helper.clip_helper(graph, node,
                                  node.input('X', 0), max_, min_,
                                  node.output('Out', 0))


@op_mapper(['pad2d', 'pad3d'])
class Pad():
    support_opset_version_range = (1, 12)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        if node.attr('mode') == 'replicate':
            mode = 'edge'
        elif node.attr('mode') == 'circular':
            raise Exception("The padding mode = circular is not supported, " \
                            "Please try the other three ways")
        else:
            mode = node.attr('mode')
        pads = cls.convert_padding(node, **kw)
        if pads is None:
            key = node.input('Paddings', 0)
            padding = None
            if key in graph.parameters.keys():
                paddings = graph.parameters[key].attribute[0].t.int32_data
                if node.attr('data_format') == 'NCHW':
                    pads = [
                        0, 0, paddings[0], paddings[2], 0, 0, paddings[1],
                        paddings[3]
                    ]
                elif node.attr('data_format') == 'NHWC':
                    pads = [
                        0, paddings[0], paddings[2], 0, 0, paddings[1],
                        paddings[3], 0
                    ]
                elif node.attr('data_format') == 'NCDHW':
                    pads = [
                        0, 0, paddings[4], paddings[2], paddings[0], 0, 0,
                        paddings[5], paddings[3], paddings[1]
                    ]
                elif node.attr('data_format') == 'NDHWC':
                    pads = [
                        0, paddings[4], paddings[2], paddings[0], 0, 0,
                        paddings[5], paddings[3], paddings[1], 0
                    ]
            else:
                raise Exception("In Pad op, padding can not be tensor" \
                                "Please set opset version >= 11")

        value = None
        if node.attr('pad_value') is not None:
            value = node.attr('pad_value')
        elif node.attr('value') is not None:
            value = node.attr('value')
        graph.make_node(
            'Pad',
            inputs=node.input('X'),
            outputs=node.output('Out'),
            mode=mode,
            value=value,
            pads=pads)

    @classmethod
    def opset_11(cls, graph, node, **kw):
        pads = cls.convert_padding(node, **kw)
        if node.attr('mode') == 'replicate':
            mode = 'edge'
        elif node.attr('mode') == 'circular':
            raise Exception("The padding mode = circular is not supported, " \
                            "Please try the other three ways")
        else:
            mode = node.attr('mode')
        pads_node = None
        if isinstance(pads, list):
            pads_node = graph.make_node(
                'Constant', attrs={'dtype': dtypes.ONNX.INT64,
                                   'value': pads})
        else:
            key = node.input('Paddings', 0)
            padding = None
            if key in graph.parameters.keys():
                paddings = graph.parameters[key].attribute[0].t.int32_data
                onnx_paddings = None
                if node.attr('data_format') == 'NCHW':
                    onnx_paddings = [
                        0, 0, paddings[0], paddings[2], 0, 0, paddings[1],
                        paddings[3]
                    ]
                elif node.attr('data_format') == 'NHWC':
                    onnx_paddings = [
                        0, paddings[0], paddings[2], 0, 0, paddings[1],
                        paddings[3], 0
                    ]
                elif node.attr('data_format') == 'NCDHW':
                    onnx_paddings = [
                        0, 0, paddings[4], paddings[2], paddings[0], 0, 0,
                        paddings[5], paddings[3], paddings[1]
                    ]
                elif node.attr('data_format') == 'NDHWC':
                    onnx_paddings = [
                        0, paddings[4], paddings[2], paddings[0], 0, 0,
                        paddings[5], paddings[3], paddings[1], 0
                    ]

                pads_node = graph.make_node(
                    'Constant',
                    attrs={'dtype': dtypes.ONNX.INT64,
                           'value': onnx_paddings})
            else:
                padding_node = node.input('Paddings', 0)
                casted_padding_node = graph.make_node(
                    'Cast', inputs=[padding_node], to=dtypes.ONNX.FLOAT)
                zero_node = None
                if node.attr('data_format') == 'NCHW' or node.attr(
                        'data_format') == 'NHWC':
                    zero_node = graph.make_node(
                        'Constant', dtype=dtypes.ONNX.FLOAT, value=[0] * 8)
                else:
                    zero_node = graph.make_node(
                        'Constant', dtype=dtypes.ONNX.FLOAT, value=[0] * 10)
                index = None
                if node.attr('data_format') == 'NCHW':
                    index = graph.make_node(
                        'Constant', dtype=dtypes.ONNX.INT32,
                        value=[2, 6, 3, 7])
                elif node.attr('data_format') == 'NHWC':
                    index = graph.make_node(
                        'Constant', dtype=dtypes.ONNX.INT32,
                        value=[1, 5, 2, 6])
                elif node.attr('data_format') == 'NCDHW':
                    index = graph.make_node(
                        'Constant',
                        dtype=dtypes.ONNX.INT32,
                        value=[4, 9, 3, 8, 2, 7])
                elif node.attr('data_format') == 'NDHWC':
                    index = graph.make_node(
                        'Constant',
                        dtype=dtypes.ONNX.INT32,
                        value=[3, 8, 2, 7, 1, 6])

                float_paddle_node = graph.make_node(
                    'ScatterElements',
                    inputs=[zero_node, index, casted_padding_node])
                paddle_node = graph.make_node(
                    'Cast', inputs=[float_paddle_node], to=dtypes.ONNX.INT64)
                pads_node = paddle_node

        value = None
        if node.attr('pad_value') is not None:
            value = node.attr('pad_value')
        elif node.attr('value') is not None:
            value = node.attr('value')
        value_node = graph.make_node(
            'Constant',
            attrs={
                'dtype': dtypes.DTYPE_PADDLE_ONNX_MAP[node.input_dtype('X', 0)],
                'value': value
            })

        graph.make_node(
            'Pad',
            inputs=node.input('X') + [pads_node, value_node],
            outputs=node.output('Out'),
            mode=mode)

    @classmethod
    def convert_padding(cls, node, **kw):
        x_shape = node.input_shape('X', 0)
        paddings = node.attr('paddings')
        if paddings == []:
            return None
        onnx_paddings = None
        if node.attr('data_format') == 'NCHW':
            onnx_paddings = [
                0, 0, paddings[0], paddings[2], 0, 0, paddings[1], paddings[3]
            ]
        elif node.attr('data_format') == 'NHWC':
            onnx_paddings = [
                0, paddings[0], paddings[2], 0, 0, paddings[1], paddings[3], 0
            ]
        elif node.attr('data_format') == 'NCDHW':
            onnx_paddings = [
                0, 0, paddings[4], paddings[2], paddings[0], 0, 0, paddings[5],
                paddings[3], paddings[1]
            ]
        elif node.attr('data_format') == 'NDHWC':
            onnx_paddings = [
                0, paddings[4], paddings[2], paddings[0], 0, 0, paddings[5],
                paddings[3], paddings[1], 0
            ]
        return onnx_paddings


@op_mapper('gaussian_random')
class GaussianRandom():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_7(cls, graph, node, **kw):
        shape_input_list = node.input('ShapeTensorList')
        shape_input = None
        if len(shape_input_list) == 0:
            shape_input = node.input('ShapeTensor')
        else:
            shape_input = graph.make_node(
                "Concat", inputs=node.input('ShapeTensorList'), axis=0)
        if shape_input is None or len(shape_input) == 0:
            graph.make_node(
                'RandomNormal',
                dtype=dtypes.DTYPE_PADDLE_ONNX_MAP[node.attr('dtype')],
                outputs=node.output('Out'),
                shape=node.attr('shape'),
                seed=float(node.attr('seed')),
                mean=node.attr('mean'),
                scale=node.attr('std'))
        else:
            cast_input_shape = graph.make_node(
                'Cast', inputs=shape_input, to=dtypes.ONNX.INT64)
            zero_like_node = graph.make_node(
                'ConstantOfShape',
                inputs=cast_input_shape,
                dims=[1],
                dtype=dtypes.ONNX.FLOAT,
                value=[0])
            graph.make_node(
                'RandomNormalLike',
                dtype=dtypes.DTYPE_PADDLE_ONNX_MAP[node.attr('dtype')],
                outputs=node.output('Out'),
                inputs=zero_like_node,
                seed=float(node.attr('seed')),
                mean=node.attr('mean'),
                scale=node.attr('std'))


@op_mapper('uniform_random_batch_size_like')
class UniformRandom():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        graph.make_node(
            'RandomUniformLike',
            inputs=node.input('Input'),
            outputs=node.output('Out'),
            high=node.attr('max'),
            dtype=dtypes.DTYPE_PADDLE_ONNX_MAP[node.attr('dtype')],
            low=node.attr('min'),
            seed=float(node.attr('seed')), )


@op_mapper('uniform_random')
class UniformRandom():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_7(cls, graph, node, **kw):
        shape_input_list = node.input('ShapeTensorList')
        shape_input = None
        if len(shape_input_list) == 0:
            shape_input = node.input('ShapeTensor')
        else:
            shape_input = graph.make_node(
                "Concat", inputs=node.input('ShapeTensorList'), axis=0)
        if shape_input is None or len(shape_input) == 0:
            graph.make_node(
                'RandomUniform',
                dtype=dtypes.DTYPE_PADDLE_ONNX_MAP[node.attr('dtype')],
                outputs=node.output('Out'),
                shape=node.attr('shape'),
                seed=float(node.attr('seed')),
                low=node.attr('min'),
                high=node.attr('max'))
        else:
            cast_input_shape = graph.make_node(
                'Cast', inputs=shape_input, to=dtypes.ONNX.INT64)
            zero_like_node = graph.make_node(
                'ConstantOfShape',
                inputs=cast_input_shape,
                dtype=dtypes.ONNX.FLOAT,
                value=[0])
            graph.make_node(
                'RandomUniformLike',
                dtype=dtypes.DTYPE_PADDLE_ONNX_MAP[node.attr('dtype')],
                outputs=node.output('Out'),
                inputs=zero_like_node,
                seed=float(node.attr('seed')),
                low=node.attr('min'),
                high=node.attr('max'))


# 'bilinear_interp', 'nearest_interp', scale only support 2, 4, 6, 8, 10
@op_mapper(
    [
        'bilinear_interp', 'nearest_interp', 'bilinear_interp_v2',
        'nearest_interp_v2', 'bicubic_interp_v2', 'linear_interp_v2',
        'trilinear_interp_v2', 'trilinear_interp', 'linear_interp'
    ],
    mapper_dict={
        'bilinear_interp': 'linear',
        'nearest_interp': 'nearest',
        'bilinear_interp_v2': 'linear',
        'nearest_interp_v2': 'nearest',
        'bicubic_interp_v2': 'cubic',
        'linear_interp_v2': 'linear',
        'trilinear_interp_v2': 'linear',
        'trilinear_interp': 'linear',
        'linear_interp': 'linear',
    },
    opset_op_dict={
        9: 'Upsample',
        10: 'Resize',
    })
class Resize():
    support_opset_version_range = (9, 15)

    @classmethod
    def opset_9(cls, graph, node, **kw):
        inputs = [node.input('X')[0]]
        resize_type = kw['mapper_dict'][node.type]
        cls.waringInfo(graph, node, resize_type)
        if len(node.input('OutSize')) > 0 or len(node.input('SizeTensor')) > 0:
            output_node = cls.compute_outsize_node(
                graph, node, return_scale=True)
        elif 'Scale' in node.inputs and len(node.input('Scale')) > 0:
            output_node = cls.compute_scale_node(graph, node)
        else:
            output_node = cls.compute_attrs_node(graph, node, return_scale=True)

        inputs = inputs + output_node
        op = kw['opset_op_dict'][graph.opset_version]
        graph.make_node(
            op, inputs=inputs, outputs=node.output('Out'), mode=resize_type)

    @classmethod
    def opset_11(cls, graph, node, **kw):
        inputs = [node.input('X')[0]]
        resize_type = kw['mapper_dict'][node.type]
        cls.waringInfo(graph, node, resize_type)
        if node.attr('align_corners'):
            coordinate_transformation_mode = 'align_corners'
        elif node.attr('align_mode') == 1 and resize_type is not 'cubic':
            coordinate_transformation_mode = 'asymmetric'
        elif resize_type == 'nearest':
            coordinate_transformation_mode = 'asymmetric'
        else:
            coordinate_transformation_mode = 'half_pixel'
        roi_node = graph.make_node(
            'Constant',
            attrs={
                'dtype': dtypes.ONNX.FLOAT,
                'value': [1, 1, 1, 1, 1, 1, 1, 1]
            })

        inputs.append(roi_node)
        if len(node.input('OutSize')) > 0 or len(node.input('SizeTensor')) > 0:
            output_node = cls.compute_outsize_node(graph, node)
        elif 'Scale' in node.inputs and len(node.input('Scale')) > 0:
            output_node = cls.compute_scale_node(graph, node)
        else:
            output_node = cls.compute_attrs_node(graph, node)
        inputs = inputs + output_node
        attrs = {
            'mode': resize_type,
            'coordinate_transformation_mode': coordinate_transformation_mode
        }
        if resize_type == 'nearest' and coordinate_transformation_mode == 'asymmetric':
            attrs['nearest_mode'] = 'floor'
        graph.make_node(
            'Resize', inputs=inputs, outputs=node.output('Out'), attrs=attrs)

    @classmethod
    def compute_outsize_node(cls, graph, node, return_scale=False):
        dtype = dtypes.ONNX.INT64
        if return_scale:
            dtype = dtypes.ONNX.FLOAT
        input_shape_node = graph.make_node('Shape', inputs=node.input('X'))
        if dtype != dtypes.ONNX.INT64:
            input_shape_node = graph.make_node(
                'Cast', inputs=[input_shape_node], to=dtype)
        shape_pre_node = mapper_helper.slice_helper(
            graph, input_shape_node, axes=[], starts=[0], ends=[2])

        out_size = [node.attr('out_d'), node.attr('out_h'), node.attr('out_w')]
        out_size = [val for val in out_size if val > 0]
        use_tensor = False
        if len(node.input('OutSize')) > 0 or len(node.input('SizeTensor')) > 0:
            use_tensor = True
        if len(out_size) > 0 and not use_tensor:
            out_size_node = graph.make_node(
                'Constant', attrs={'dtype': dtype,
                                   'value': out_size})
        else:
            out_size_node, _ = mapper_helper.get_node_attr_value(
                graph, node, None, 'OutSize', 'SizeTensor', dtype=dtype)
        out_size_node = graph.make_node(
            'Concat', inputs=[shape_pre_node, out_size_node], axis=0)

        if return_scale:
            scale_node = graph.make_node(
                'Div', inputs=[out_size_node, input_shape_node])
            return [scale_node]

        scale_empty_node = graph.make_node(
            'Constant', attrs={'dtype': dtypes.ONNX.FLOAT,
                               'value': []})
        return [scale_empty_node, out_size_node]

    @classmethod
    def compute_scale_node(cls, graph, node):
        cast_scale = graph.make_node(
            'Cast', inputs=node.input('Scale'), to=dtypes.ONNX.FLOAT)
        inputs_cocat = []
        const_node = graph.make_node(
            'Constant', attrs={'dtype': dtypes.ONNX.FLOAT,
                               'value': [1, 1]})
        inputs_cocat.append(const_node)
        scale = node.attr('scale')
        if isinstance(scale, (float, int)):
            cast_scale = [cast_scale] * (len(node.input_shape('X', 0)) - 2)
            inputs_cocat = inputs_cocat + cast_scale
        else:
            inputs_cocat = inputs_cocat + [cast_scale]
        scale_node = graph.make_node('Concat', inputs=inputs_cocat, axis=0)
        return [scale_node]

    @classmethod
    def compute_attrs_node(cls, graph, node, return_scale=False):
        out_size = [node.attr('out_d'), node.attr('out_h'), node.attr('out_w')]
        scale = node.attr('scale')
        if isinstance(scale, (float, int)):
            scale = [scale] * (len(node.input_shape('X', 0)) - 2)

        out_size = [val for val in out_size if val > 0]
        if len(out_size) > 0:
            output_node = cls.compute_outsize_node(
                graph, node, return_scale=return_scale)
            return output_node

        assert len(scale) > 0, Exception("scale size should > 0!")
        scale_node = graph.make_node(
            'Constant',
            attrs={'dtype': dtypes.ONNX.FLOAT,
                   'value': [1, 1] + scale})
        return [scale_node]

    @classmethod
    def waringInfo(cls, graph, node, resize_type):
        assert node.attrs['data_layout'] == 'NCHW', \
            "The conv data layout should be 'NCHW' , but received data format " \
            "is %s." % node.attrs['data_format']

        if graph.opset_version < 11:
            if node.attr('align_corners') or resize_type in ["cubic"]:
                raise Exception(
                    "When align_corners is true or resize_type is 'cubic', the case isn't supported in onnx(opset<=10), "
                    "Try converting with opset_version>= 11 ")
            if node.attr('align_mode') == 0 and resize_type in [
                    "bilinear", "linear", "trilinear"
            ]:
                raise Exception(
                    "When align_mode == 0 and resize_type is 'bilinear' or 'linear or 'trilinear', the case isn't "
                    "supported in onnx(opset<=10), Try converting with opset_version>= 11 "
                )


@op_mapper('pixel_shuffle')
class PixelShuffle():
    support_opset_version_range = (11, 15)

    @classmethod
    def opset_11(cls, graph, node, **kw):
        upscale_factor = node.attr('upscale_factor')

        node = graph.make_node(
            'DepthToSpace',
            inputs=node.input('X'),
            outputs=node.output('Out'),
            blocksize=upscale_factor,
            mode='CRD')


@op_mapper('scatter')
class Scatter():
    support_opset_version_range = (11, 15)

    @classmethod
    def opset_11(cls, graph, node, **kw):
        ids = node.input('Ids', 0)
        input_dtype = dtypes.DTYPE_PADDLE_ONNX_MAP[node.input_dtype('Ids', 0)]
        if input_dtype != dtypes.ONNX.INT64:
            ids = graph.make_node('Cast', inputs=[ids], to=dtypes.ONNX.INT64)

        shape = graph.make_node(
            'Constant',
            value=[node.input_shape('Ids', 0)[0], 1],
            dtype=dtypes.ONNX.INT64)
        reshape_index = graph.make_node('Reshape', inputs=[ids, shape])
        if not node.attr('overwrite'):
            raise Exception("overwrite = False not support yet.")
        else:
            graph.make_node(
                'ScatterND',
                inputs=[
                    node.input('X', 0), reshape_index, node.input('Updates', 0)
                ],
                outputs=node.output('Out'))


@op_mapper('scatter_nd_add')
class ScatterndAdd():
    support_opset_version_range = (11, 12)

    @classmethod
    def opset_11(cls, graph, node, **kw):
        shape = graph.make_node('Shape', inputs=node.input('X', 0))
        zero_like_node = graph.make_node(
            'ConstantOfShape',
            inputs=[shape],
            dims=[1],
            dtype=dtypes.ONNX.FLOAT,
            value=[0])
        add_node = graph.make_node(
            'ScatterND',
            inputs=[
                zero_like_node, node.input('Index', 0), node.input('Updates', 0)
            ], )
        graph.make_node(
            'Add',
            inputs=[node.input('X', 0), add_node],
            outputs=node.output('Out'))


@op_mapper('meshgrid')
class Meshgrid():
    support_opset_version_range = (8, 15)

    @classmethod
    def opset_8(cls, graph, node, **kw):
        tensors = [t for t in list(node.input('X'))]
        tensors_shape = [graph.make_node('Shape', inputs=t) for t in tensors]
        out_shape = graph.make_node('Concat', inputs=tensors_shape, axis=0)
        out = []
        for i, t in enumerate(tensors):
            shape_i = [
                graph.make_node(
                    'Constant',
                    attrs={'dtype': dtypes.ONNX.INT64,
                           'value': [1]})
            ] * len(tensors)
            shape_i[i] = tensors_shape[i]
            t_reshaped = graph.make_node(
                'Reshape',
                inputs=[t, graph.make_node(
                    'Concat', inputs=shape_i, axis=0)])
            out.append(
                graph.make_node(
                    'Expand',
                    inputs=[t_reshaped, out_shape],
                    outputs=node.output('Out')[i]))


@op_mapper('flip')
class Flip():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_7(cls, graph, node, **kw):
        inputs = node.input('X')
        x_dtype = node.input_dtype('X', 0)
        if x_dtype == paddle.bool or x_dtype == paddle.float64:
            inputs = [
                graph.make_node(
                    "Cast", inputs=inputs, to=dtypes.ONNX.FLOAT)
            ]
        axes = node.attr("axis")
        if not isinstance(axes, list):
            axes = [axes]
        input_shape = node.input_shape('X', 0)

        for i, axis in enumerate(axes):
            if axis < 0:
                axes[i] += len(input_shape)
            assert input_shape[
                axis] > 0, "The dimension in axis of input must be fixed for flip operator, but now the input shape({}) in axis({}) is unknow.".format(
                    input_shape, axis)

        temp_input = inputs[0]
        for i, axis in enumerate(axes):
            if input_shape[axis] == 1:
                if i != len(axes) - 1:
                    continue
                else:
                    if x_dtype == paddle.bool or x_dtype == paddle.float64:
                        graph.make_node(
                            "Cast",
                            inputs=[temp_input],
                            outputs=node.output("Out"),
                            to=dtypes.DTYPE_PADDLE_ONNX_MAP[x_dtype])
                    else:
                        graph.make_node(
                            "Identity",
                            inputs=[temp_input],
                            outputs=node.output("Out"))
            else:
                splits = graph.make_node(
                    "Split",
                    inputs=[temp_input],
                    outputs=input_shape[axis],
                    axis=axis,
                    split=[1] * input_shape[axis])
                reversed_splits = splits[::-1]
                if i != len(axes) - 1:
                    temp_input = graph.make_node(
                        "Concat", inputs=reversed_splits, axis=axis)
                else:
                    if x_dtype == paddle.bool or x_dtype == paddle.float64:
                        out = graph.make_node(
                            "Concat", inputs=reversed_splits, axis=axis)
                        graph.make_node(
                            "Cast",
                            inputs=[out],
                            outputs=node.output("Out"),
                            to=dtypes.DTYPE_PADDLE_ONNX_MAP[x_dtype])
                    else:
                        graph.make_node(
                            "Concat",
                            inputs=reversed_splits,
                            outputs=node.output("Out"),
                            axis=axis)

    @classmethod
    def opset_13(cls, graph, node, **kw):
        inputs = node.input('X')
        x_dtype = node.input_dtype('X', 0)
        if x_dtype == paddle.bool or x_dtype == paddle.float64:
            inputs = [
                graph.make_node(
                    "Cast", inputs=inputs, to=dtypes.ONNX.FLOAT)
            ]
        axes = node.attr("axis")
        if not isinstance(axes, list):
            axes = [axes]
        input_shape = node.input_shape('X', 0)

        for i, axis in enumerate(axes):
            if axis < 0:
                axes[i] += len(input_shape)
            assert input_shape[
                axis] > 0, "The dimension in axis of input must be fixed for flip operator, but now the input shape({}) in axis({}) is unknow.".format(
                    input_shape, axis)

        temp_input = inputs[0]
        for i, axis in enumerate(axes):
            if input_shape[axis] == 1:
                if i != len(axes) - 1:
                    continue
                else:
                    if x_dtype == paddle.bool or x_dtype == paddle.float64:
                        graph.make_node(
                            "Cast",
                            inputs=[temp_input],
                            outputs=node.output("Out"),
                            to=dtypes.DTYPE_PADDLE_ONNX_MAP[x_dtype])
                    else:
                        graph.make_node(
                            "Identity",
                            inputs=[temp_input],
                            outputs=node.output("Out"))
            else:
                split = graph.make_node(
                    'Constant',
                    attrs={
                        'dtype': dtypes.ONNX.INT64,
                        'value': [1] * input_shape[axis]
                    })
                splits = graph.make_node(
                    "Split",
                    inputs=[temp_input, split],
                    outputs=input_shape[axis],
                    axis=axis)
                reversed_splits = splits[::-1]
                if i != len(axes) - 1:
                    temp_input = graph.make_node(
                        "Concat", inputs=reversed_splits, axis=axis)
                else:
                    if x_dtype == paddle.bool or x_dtype == paddle.float64:
                        out = graph.make_node(
                            "Concat", inputs=reversed_splits, axis=axis)
                        graph.make_node(
                            "Cast",
                            inputs=[out],
                            outputs=node.output("Out"),
                            to=dtypes.DTYPE_PADDLE_ONNX_MAP[x_dtype])
                    else:
                        graph.make_node(
                            "Concat",
                            inputs=reversed_splits,
                            outputs=node.output("Out"),
                            axis=axis)
