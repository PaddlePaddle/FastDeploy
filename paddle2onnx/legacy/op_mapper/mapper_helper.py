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

import paddle.fluid.core as core
import six
import copy
from paddle2onnx.legacy.constant import dtypes
import paddle
from onnx import TensorProto


def is_static_shape(shape):
    if len(shape) > 1 and shape[1:].count(-1) > 0:
        raise Exception(
            "Converting this model to ONNX need with static input shape," \
            " please fix input shape of this model, see doc Q2 in" \
            " https://github.com/PaddlePaddle/paddle2onnx/blob/develop/docs/en/FAQ.md."
        )


def shape_helper(graph, input, dim=None):
    if dim is None:
        shape_node = graph.make_node('Shape', inputs=[input])
        return shape_node
    full_shape = graph.make_node('Shape', inputs=[input])
    shape_node = slice_helper(graph, full_shape, [0], [dim], [dim + 1])
    return shape_node


def unsqueeze_helper(graph, input, axes, outputs=None):
    inputs = []
    if not isinstance(input, list):
        input = [input]
    inputs.append(input[0])
    if not isinstance(axes, list):
        axes = [axes]
    if outputs is not None and isinstance(outputs, six.string_types):
        outputs = [outputs]

    if graph.opset_version < 13:
        unsqueeze_node = graph.make_node(
            "Unsqueeze", inputs=inputs, outputs=outputs, axes=axes)
        return unsqueeze_node
    else:
        axes_node = graph.make_node(
            'Constant', dtype=dtypes.ONNX.INT64, value=axes)
        inputs = inputs + [axes_node]
        unsqueeze_node = graph.make_node(
            "Unsqueeze", inputs=inputs, outputs=outputs)
        return unsqueeze_node


def split_helper(graph, input, axis=0, split=None, outputs=None):
    assert outputs is not None, "outputs can not be None in split_helper."
    inputs = []
    if not isinstance(input, list):
        input = [input]
    inputs.append(input[0])
    if split is not None and not isinstance(split, list):
        split = [split]
    if split is None:
        split_node = graph.make_node(
            "Split", inputs=inputs, outputs=outputs, axis=axis)
        return split_node
    if graph.opset_version < 13:
        split_node = graph.make_node(
            "Split", inputs=inputs, outputs=outputs, axis=axis, split=split)
        return split_node
    else:
        split = graph.make_node(
            'Constant', dtype=dtypes.ONNX.INT64, value=split)
        inputs = inputs + [split]
        split_node = graph.make_node(
            "Split", inputs=inputs, axis=axis, outputs=outputs)
        return split_node
    
    
def slice_helper(graph,
                 input,
                 axes,
                 starts,
                 ends,
                 outputs=None,
                 dtype=dtypes.ONNX.INT64):
    inputs = []
    if not isinstance(input, list):
        input = [input]
    inputs.append(input[0])
    if axes is not None and not isinstance(axes, list):
        axes = [axes]
    if starts is not None and not isinstance(starts, (list, six.string_types)):
        starts = [starts]
    if ends is not None and not isinstance(ends, (list, six.string_types)):
        ends = [ends]

    if graph.opset_version < 10:
        attrs = {
            'starts': starts,
            'ends': ends,
        }
        if axes not in [None, []]:
            attrs['axes'] = axes
        slice_node = graph.make_node(
            "Slice", inputs=inputs, outputs=outputs, attrs=attrs)
        return slice_node
    else:
        if not isinstance(starts, six.string_types):
            starts = graph.make_node('Constant', dtype=dtype, value=starts)
        if not isinstance(ends, six.string_types):
            ends = graph.make_node('Constant', dtype=dtype, value=ends)
        inputs = inputs + [starts, ends]
        if axes not in [None, []]:
            axes_node = graph.make_node('Constant', dtype=dtype, value=axes)
            inputs.append(axes_node)
        slice_node = graph.make_node("Slice", inputs=inputs, outputs=outputs)
        return slice_node


def squeeze_helper(graph, input, axes=None, outputs=None):
    inputs = []
    if not isinstance(input, list):
        input = [input]
    inputs.append(input[0])
    if axes is not None and not isinstance(axes, list):
        axes = [axes]
    if graph.opset_version < 13:
        squeeze_node = graph.make_node(
            "Squeeze", inputs=inputs, axes=axes, outputs=outputs)
        return squeeze_node
    else:
        if axes is not None:
            axes_node = graph.make_node(
                'Constant', dtype=dtypes.ONNX.INT64, value=axes)
            inputs.append(axes_node)
        squeeze_node = graph.make_node(
            "Squeeze", inputs=inputs, outputs=outputs)
        return squeeze_node


def unsqueeze_helper(graph, input, axes, outputs=None):
    inputs = []
    if isinstance(input, list):
        input = input[0]
    inputs.append(input)
    if not isinstance(axes, list):
        axes = [axes]
    if graph.opset_version < 13:
        unsqueeze_node = graph.make_node(
            'Unsqueeze', inputs=inputs, axes=axes, outputs=outputs)
    else:
        axes_node = graph.make_node(
            'Constant', attrs={'dtype': dtypes.ONNX.INT64,
                               'value': axes})
        inputs.append(axes_node)
        unsqueeze_node = graph.make_node(
            'Unsqueeze', inputs=inputs, outputs=outputs)
    return unsqueeze_node


def split_helper(graph, inputs, outputs, axis, split, dtype=paddle.float32):
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]

    if not isinstance(outputs, int) and not isinstance(outputs, (list, tuple)):
        outputs = [outputs]

    if dtype == paddle.float64:
        cast_inputs = []
        for i in range(len(inputs)):
            one = graph.make_node(
                'Cast', inputs=[inputs[i]], to=TensorProto.FLOAT)
            cast_inputs.append(one)
        if graph.opset_version < 13:
            split_node = graph.make_node(
                "Split",
                inputs=cast_inputs,
                outputs=outputs,
                axis=axis,
                split=split)
        else:
            split_const = graph.make_node(
                'Constant', dtype=dtypes.ONNX.INT64, value=split)
            split_node = graph.make_node(
                "Split",
                inputs=cast_inputs + [split_const],
                outputs=outputs,
                axis=axis)
        casted_output = []
        for i in range(len(outputs)):
            one = graph.make_node(
                'Cast',
                inputs=[split_node[i]],
                outputs=[outputs[i]],
                to=TensorProto.DOUBLE)
            casted_output.append(one)
        return casted_output
    else:
        if graph.opset_version < 13:
            split_node = graph.make_node(
                "Split", inputs=inputs, outputs=outputs, axis=axis, split=split)
        else:
            split_const = graph.make_node(
                'Constant', dtype=dtypes.ONNX.INT64, value=split)
            split_node = graph.make_node(
                "Split",
                inputs=inputs + [split_const],
                outputs=outputs,
                axis=axis)
        return split_node


def constant_helper(graph, dtype, value, shape=None, outputs=[]):
    constant = graph.make_node(
        'Constant',
        inputs=[],
        outputs=outputs,
        attrs={
            'dims': shape,
            'dtype': dtypes.DTYPE_PADDLE_ONNX_MAP[dtype],
            'value': value
        })
    return constant


def clip_helper(graph, node, input, max, min, output=[]):
    x_dtype = node.input_dtype('X', 0)
    if (isinstance(min, six.string_types) or
            isinstance(max, six.string_types)) and graph.opset_version < 11:
        raise Exception(
            "min or max of Clip is Tensor, please try with higher onnx opset_version."
        )
    if graph.opset_version < 11:
        if x_dtype != paddle.float32:
            input = graph.make_node(
                'Cast', inputs=[input], to=dtypes.ONNX.FLOAT)
            clip = graph.make_node('Clip', inputs=input, max=max, min=min)
            clip = graph.make_node(
                'Cast',
                inputs=[clip],
                to=dtypes.DTYPE_PADDLE_ONNX_MAP[x_dtype],
                outputs=output)
        else:
            clip = graph.make_node(
                'Clip', inputs=input, max=max, min=min, outputs=output)
    else:
        if x_dtype != paddle.float32:
            input = graph.make_node(
                'Cast', inputs=[input], to=dtypes.ONNX.FLOAT)

        if not isinstance(min, six.string_types):
            min = graph.make_node(
                'Constant',
                attrs={
                    'dtype': dtypes.DTYPE_PADDLE_ONNX_MAP[paddle.float32],
                    'value': min
                })
        else:
            if node.input_dtype('Min', 0) != paddle.float32:
                min = graph.make_node(
                    'Cast',
                    inputs=min,
                    attrs={'to': dtypes.DTYPE_PADDLE_ONNX_MAP[paddle.float32]})
            min = graph.make_node('Squeeze', min)

        if not isinstance(max, six.string_types):
            max = graph.make_node(
                'Constant',
                attrs={
                    'dtype': dtypes.DTYPE_PADDLE_ONNX_MAP[paddle.float32],
                    'value': max
                })
        else:
            if node.input_dtype('Max', 0) != paddle.float32:
                max = graph.make_node(
                    'Cast',
                    inputs=max,
                    attrs={'to': dtypes.DTYPE_PADDLE_ONNX_MAP[paddle.float32]})
            max = graph.make_node('Squeeze', max)
        if x_dtype != paddle.float32:
            clip_pre = graph.make_node('Clip', inputs=[input, min, max])
            clip = graph.make_node(
                'Cast',
                inputs=[clip_pre],
                outputs=output,
                to=dtypes.DTYPE_PADDLE_ONNX_MAP[x_dtype])
        else:
            clip = graph.make_node(
                'Clip', inputs=[input, min, max], outputs=output)
    return clip


def dtype_alignment(graph, nodes, node_dtypes, to=None):
    assert len(nodes) == len(
        node_dtypes), "Length of nodes and node_dtypes should be equal."
    dtype_order = [
        core.VarDesc.VarType.BOOL,
        core.VarDesc.VarType.INT16,
        core.VarDesc.VarType.INT32,
        core.VarDesc.VarType.INT64,
        core.VarDesc.VarType.FP16,
        core.VarDesc.VarType.FP32,
        core.VarDesc.VarType.FP64,
    ]
    max_index = -1
    for dtype in node_dtypes:
        index = dtype_order.index(dtype)
        if index > max_index:
            max_index = index

    if max_index < 0:
        return nodes

    casted_nodes = list()
    cast_dtype = dtype_order[max_index]
    cast_dtype = dtypes.DTYPE_PADDLE_ONNX_MAP[cast_dtype]
    for i, dtype in enumerate(node_dtypes):
        index = dtype_order.index(dtype)
        if to is not None:
            cast_dtype = to
            condition = dtypes.DTYPE_PADDLE_ONNX_MAP[index] != cast_dtype
        else:
            condition = index != max_index
        if condition:
            cast_node = graph.make_node(
                'Cast', inputs=[nodes[i]], to=cast_dtype)
            casted_nodes.append(cast_node)
        else:
            casted_nodes.append(nodes[i])
    return casted_nodes


def cast(graph, input, origin_dtype, target_dtype):
    if not isinstance(origin_dtype, six.string_types):
        origin_dtype = dtypes.DTYPE_PADDLE_STR_MAP[origin_dtype]
    if origin_dtype != target_dtype:
        cast_node = graph.make_node(
            'Cast', inputs=input, to=dtypes.DTYPE_ONNX_STR_MAP[target_dtype])
        return cast_node
    return input


def shape_alignment(graph, nodes, node_shapes):
    assert len(nodes) == len(
        node_shapes), "Length of nodes and node_shapes should be equal."
    max_dim = -1
    for shape in node_shapes:
        dim = len(shape)
        if dim > max_dim:
            max_dim = dim

    if max_dim < 0:
        return nodes

    assert max_dim == 1 or max_dim == 0, "max_dim is only supported when max_dim is 1 or 0."
    max_dim = 1 if max_dim == 0 else max_dim
    unsqueeze_nodes = list()
    for i, shape in enumerate(node_shapes):
        dim = len(shape)
        if dim != max_dim:
            unsqueeze_node = nodes[i]
            for j in range(max_dim - dim):
                unsqueeze_node = unsqueeze_helper(graph, unsqueeze_node, [0])
            unsqueeze_nodes.append(unsqueeze_node)
        else:
            unsqueeze_nodes.append(nodes[i])
    return unsqueeze_nodes


def get_tensor_list_node(graph, node, name, dtype=None):
    node_list = node.input(name)
    node_dtypes = [node.input_dtype(name, i) for i in range(len(node_list))]
    node_list = dtype_alignment(graph, node_list, node_dtypes, dtype)

    node_shapes = [node.input_shape(name, i) for i in range(len(node_list))]
    node_list = shape_alignment(graph, node_list, node_shapes)
    node = graph.make_node("Concat", inputs=node_list, axis=0)
    return node


def get_value_from_parameters(graph, input_node):
    assert input_node in graph.parameters, "{} is not in graph.parameters".format(
        input_node)
    data = graph.parameters[input_node].attribute[0].t.int32_data
    if data is None or len(data) < 1:
        data = graph.parameters[input_node].attribute[0].t.int64_data
    value = [val for _, val in enumerate(data)]
    return value


# return value
# arg1: attr_value
# arg2: attr_value is tensor or not
def get_node_attr_value(graph,
                        node,
                        attr_name=None,
                        attr_tensor_name=None,
                        attr_tensor_list_name=None,
                        return_list=False,
                        dtype=None):
    attr_tensor = node.input(attr_tensor_name)
    attr_tensor_list = node.input(attr_tensor_list_name)
    if attr_tensor is not None and len(attr_tensor) > 0:
        value = node.input(attr_tensor_name)[0]
        if return_list:
            try:
                value = get_value_from_parameters(graph, value)
                return value, False  # value, is_tensor
            except Exception:
                return value, True
        else:
            input_dtype = dtypes.DTYPE_PADDLE_ONNX_MAP[node.input_dtype(
                attr_tensor_name, 0)]
            if input_dtype != dtype:
                value = graph.make_node('Cast', inputs=[value], to=dtype)
            return value, True
    elif attr_tensor_list is not None and len(attr_tensor_list) > 0:
        value = get_tensor_list_node(graph, node, attr_tensor_list_name, dtype)
        return value, True
    else:
        value = node.attr(attr_name)
        return value, False
