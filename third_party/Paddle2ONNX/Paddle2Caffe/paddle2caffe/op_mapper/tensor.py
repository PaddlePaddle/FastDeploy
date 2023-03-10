#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from paddle2caffe.constant.op_mapping_status import *
from paddle2caffe.op_mapper import OpMapper as op_mapper
from paddle2caffe.op_mapper import mapper_helper

from paddle2caffe.graph import Graph, Node
from paddle2caffe.utils import logging


def invert_minus(input_num, boarder, is_left=True):
    if is_left:
        if input_num < 0:
            return input_num + boarder
        else:
            return input_num
    else:
        if input_num > boarder:
            return boarder
        else:
            return input_num


@op_mapper('concat')
class Concat:
    have_custom = False

    @classmethod
    def standard_mapping(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, **kw):
        # transfer info: input blob: 1, output blob: 1, node: 1, extra_blob: 0, extra_node: 0
        concat_attrs = {'axis': source_node.attrs.get('axis', 1)}
        input_num = len(source_node.input_blobs_names)
        caffe_node = mapper_helper.convert_identity_operation(caffe_graph, source_node, source_graph,
                                                              caffe_type="Concat", attrs=concat_attrs,
                                                              input_blob_num=input_num)

        return OP_MAPPING_IDENTITY, [caffe_node.name]


@op_mapper('shape')
class Shape:
    have_custom = False

    @classmethod
    def standard_mapping(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, **kw):
        raise NotImplementedError('To be continued')


@op_mapper('size')
class Size:
    have_custom = False

    @classmethod
    def standard_mapping(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, **kw):
        raise NotImplementedError('To be continued')


@op_mapper(['slice', 'split'])
class Slice:
    have_custom = False

    @classmethod
    def _slice_single_dim(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, **kw):
        # transfer info: input blob: 1, output blob: 1, node: 1, extra_blob: 0, extra_node: 0
        if kw.get('points') is not None:
            points = kw['points']
        else:
            start = kw['starts'][0]
            end = kw['ends'][0]
            if start == 0:
                points = [end]
            else:
                points = [start, end]
        axis = kw['axes'][0]
        slice_attrs = {
            'slice_point': points,
            'axis': axis
        }
        output_num = len(source_node.output_blobs_names)
        slice_node = mapper_helper.convert_identity_operation(caffe_graph, source_node, source_graph,
                                                              'Slice', slice_attrs, 1, output_num)
        # paddle model may ignore final blob which caffe do not support
        if output_num != len(points) + 1:
            extra_blob_name = source_node.output_blobs_names[-1] + '_extra_dummy'
            # without creat blob instance since graph wont use it anymore
            slice_node.output_blobs_names.append(extra_blob_name)

        return OP_MAPPING_IDENTITY, [slice_node.name]

    @classmethod
    def _slice_hw_dim(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, **kw):
        # transfer info: input blob: 1, output blob: 1, node: 0, extra_blob: 0, extra_node: 1
        start_h, start_w = kw['starts']
        end_h, end_w = kw['ends']
        input_h, input_w = kw['input_shape'][2:4]
        output_h, output_w = kw['output_shape'][2:4]
        chn = kw['input_shape'][1]

        start_h = invert_minus(start_h, input_h)
        start_w = invert_minus(start_w, input_w)
        end_h = invert_minus(end_h, input_h)
        end_h = invert_minus(end_h, input_h, False)
        end_w = invert_minus(end_w, input_w)
        end_w = invert_minus(end_w, input_w, False)

        if start_h * start_w > 0 and start_h + output_h == end_h and start_w + output_w == end_w:
            # this means crop input from left and top, which can be replace by dummy_dw as crop
            input_blob_name = caffe_graph.transfer_op_input(source_node, source_graph)
            input_blob_name = input_blob_name[0]
            slice_node = mapper_helper.creat_dummy_dwconv(caffe_graph, source_node, source_graph,
                                                          input_blob_name, source_node.name, [chn, chn, None, None],
                                                          offset=[start_h, start_w], is_right_bottom=False)
        elif start_h * start_w == 0 and end_h == output_h and end_w == output_w:
            # this means crop input from right and bottom, which can be replace by dummy_dw as crop
            input_blob_name = caffe_graph.transfer_op_input(source_node, source_graph)
            input_blob_name = input_blob_name[0]
            slice_node = mapper_helper.creat_dummy_dwconv(caffe_graph, source_node, source_graph,
                                                          input_blob_name, source_node.name, [chn, chn, None, None],
                                                          offset=[input_h - output_h, input_w - output_w])
        else:
            print(start_h, start_w)
            print(end_h, end_w)
            raise NotImplementedError

        return OP_MAPPING_IDENTITY, [slice_node.name]

    @classmethod
    def standard_mapping(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, **kw):
        """
        https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/slice_cn.html#slice
        https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/split_cn.html#split
        """
        input_shape = source_graph.blob_map[source_node.input(0)].shape
        output_shape = source_graph.blob_map[source_node.output(0)].shape
        axes, starts, ends, points = [], [], [], []
        output_num = len(source_node.output_blobs_names)

        if source_node.op_type == 'split':  # change split params into slice like
            axes = source_node.attrs.get('axis', 1)
            if axes < 0:
                axes += len(input_shape)
            axes = [axes]
            sections = source_node.attr('sections')
            points = []
            if isinstance(sections, int):
                # if sections, stands for same cut
                for idx in range(output_num):
                    if idx == 0:
                        points.append(sections)
                    else:
                        points.append(sections + points[idx - 1])
            elif len(sections) == 0:
                # if sections is null, split equally according to output num
                slice_chn = input_shape[axes[0]]
                for idx in range(1, output_num):
                    points.append(int(idx * slice_chn / output_num))
            else:
                for idx in range(len(sections)):
                    if idx == 0:
                        points.append(sections[idx])
                    else:
                        points.append(sections[idx] + points[idx - 1])
                if points[-1] == input_shape[axes[0]]:  # final section is not needed
                    points.pop(-1)

            # need to split equally from output num manually
            if len(points) == 0:
                output_num = len(source_node.output_blobs_names)
                assert input_shape[axes[0]] % output_num == 0
                point_single = input_shape[axes[0]] // output_num
                for idx in range(1, output_num):
                    points.append(point_single * idx)
        else:
            axes = source_node.attr('axes')
            starts = source_node.attr('starts')
            ends = source_node.attr('ends')

        if len(axes) == 1:
            if len(starts) == 1 and len(ends) == 1:
                return cls._slice_single_dim(caffe_graph, source_node, source_graph,
                                             input_shape=input_shape, output_shape=output_shape,
                                             axes=axes, starts=starts, ends=ends, points=None)
            else:
                return cls._slice_single_dim(caffe_graph, source_node, source_graph,
                                             input_shape=input_shape, output_shape=output_shape,
                                             axes=axes, starts=None, ends=None, points=points)
        elif len(axes) == 2 and axes == [2, 3]:
            assert len(starts) == 2 and len(ends) == 2
            return cls._slice_hw_dim(caffe_graph, source_node, source_graph,
                                     input_shape=input_shape, output_shape=output_shape,
                                     axes=axes, starts=starts, ends=ends, points=None)
        else:
            raise NotImplementedError


@op_mapper(['tile'])
class Tile:
    have_custom = False

    @classmethod
    def standard_mapping(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, **kw):
        raise NotImplementedError('To be continued')


@op_mapper('range')
class Range:
    have_custom = False

    @classmethod
    def standard_mapping(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, **kw):
        raise NotImplementedError('To be continued')


@op_mapper('fill_constant')
class Constant:
    have_custom = False

    @classmethod
    def standard_mapping(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, **kw):
        raise NotImplementedError('To be continued')


@op_mapper('fill_constant_batch_size_like')
class FillConstantBatchSizeLike:
    have_custom = False

    @classmethod
    def standard_mapping(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, **kw):
        raise NotImplementedError('To be continued')


@op_mapper('gather')
class Gather:
    have_custom = False

    @classmethod
    def standard_mapping(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, **kw):
        raise NotImplementedError('To be continued')


@op_mapper('squeeze2')
class Squeeze:
    have_custom = False

    @classmethod
    def _map_to_reshape(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, **kw):
        shape = list(source_graph.blob_map[source_node.output(0)].shape)
        shape[0] = 0  # keep batch dim the same
        caffe_node = mapper_helper.convert_identity_operation(caffe_graph, source_node, source_graph,
                                                              'Reshape', {'shape': shape})
        return OP_MAPPING_IDENTITY, [caffe_node.name]

    @classmethod
    def _map_to_flatten(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, **kw):
        input_shape = list(source_graph.blob_map[source_node.output(0)].shape)
        axis = source_node.attr('axes')[0]
        axis += len(input_shape) if axis < 0 else 0
        caffe_node = mapper_helper.convert_identity_operation(caffe_graph, source_node, source_graph,
                                                              'Flatten', {'axis': axis})
        return OP_MAPPING_IDENTITY, [caffe_node.name]

    @classmethod
    def standard_mapping(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, **kw):
        """
        https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/squeeze_cn.html#squeeze
        use flatten or reshape instead in caffe
        """
        input_shape = source_graph.blob_map[source_node.input(0)].shape
        axes = source_node.attr('axes')
        if axes is None:
            return cls._map_to_reshape(caffe_graph, source_node, source_graph)
        else:
            use_flatten = True
            for dim in axes:
                dim += len(input_shape) if dim < 0 else 0
                if input_shape[dim] != 1:
                    use_flatten = False
                    break

            if use_flatten:
                return cls._map_to_flatten(caffe_graph, source_node, source_graph)
            else:
                return cls._map_to_reshape(caffe_graph, source_node, source_graph)


@op_mapper('transpose2')
class Transpose:
    have_custom = True

    @classmethod
    def custom_mapping(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, **kw):
        """https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/transpose_cn.html#transpose"""
        orders = source_node.attr('axis')
        caffe_node = mapper_helper.convert_identity_operation(caffe_graph, source_node, source_graph,
                                                              caffe_type="Permute", attrs={'order': orders})
        return OP_MAPPING_IDENTITY, [caffe_node.name]


@op_mapper(['flatten_contiguous_range', 'flatten2'])
class Flatten:
    have_custom = False

    @classmethod
    def standard_mapping(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, **kw):
        if source_node.op_type == 'flatten_contiguous_range':
            axis = source_node.attr('start_axis')
            end_axis = source_node.attr('stop_axis')
        else:
            axis = source_node.attr('axis')
            end_axis = -1
        ori_input_shape = source_graph.get_blob(source_node.input(0)).shape
        if end_axis > 0 and end_axis == len(ori_input_shape) - 1:
            end_axis = -1
        # hide default attrs
        flatten_attrs = dict()
        if axis != 1:
            flatten_attrs['axis'] = axis
        if end_axis != -1:
            flatten_attrs['end_axis'] = end_axis
        caffe_node = mapper_helper.convert_identity_operation(caffe_graph, source_node, source_graph,
                                                              caffe_type="Flatten", attrs=flatten_attrs)
        return OP_MAPPING_IDENTITY, [caffe_node.name]


@op_mapper('reshape2')
class Reshape:
    have_custom = False

    @classmethod
    def standard_mapping(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, **kw):
        if len(source_node.input_blobs_names) == 1:
            shape = source_node.attr('shape')
            if shape[0] == -1:
                shape[0] = 0
            caffe_node = mapper_helper.convert_identity_operation(caffe_graph, source_node, source_graph,
                                                                  caffe_type="Reshape", attrs={'shape': shape})

            return OP_MAPPING_IDENTITY, [caffe_node.name]
        else:
            raise NotImplementedError


@op_mapper('unsqueeze2')
class Unsqueeze:
    have_custom = False

    @classmethod
    def standard_mapping(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, **kw):
        raise NotImplementedError('To be continued')


@op_mapper('cast')
class Cast:
    have_custom = False

    @classmethod
    def standard_mapping(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, **kw):
        """https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/cast_cn.html#cast"""
        # caffe do not have dtype difference, just amend this node
        # (change into Dropout, will remove in optimizer)
        caffe_node = mapper_helper.convert_identity_operation(caffe_graph, source_node, source_graph,
                                                              caffe_type="Dropout", attrs={})
        return OP_MAPPING_AMMEND, [caffe_node.name]


@op_mapper('clip')
class Clip:
    have_custom = False

    @classmethod
    def standard_mapping(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, **kw):
        """https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/clip_cn.html#clip"""
        min = source_node.attr('min')
        max = source_node.attr('max')
        assert min is not None and max is not None, 'caffe do not support clip with no min or max setting'
        # hide default attrs
        clip_attrs = {'min': min, 'max': max}
        caffe_node = mapper_helper.convert_identity_operation(caffe_graph, source_node, source_graph,
                                                              caffe_type="Clip", attrs=clip_attrs)

        return OP_MAPPING_IDENTITY, [caffe_node.name]


@op_mapper(['pad2d', 'pad3d'])
class Pad:
    have_custom = False

    @classmethod
    def standard_mapping(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, **kw):
        raise NotImplementedError('To be continued')


@op_mapper(
    ['bilinear_interp', 'nearest_interp', 'bilinear_interp_v2', 'nearest_interp_v2'],
    mapper_dict={
        'bilinear_interp': 'linear',
        'nearest_interp': 'nearest',
        'bilinear_interp_v2': 'linear',
        'nearest_interp_v2': 'nearest'
    })
class Resize:
    have_custom = True

    @classmethod
    def custom_mapping(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, **kw):
        """https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/interpolate_cn.html#interpolate"""
        input_shape = source_graph.blob_map[source_node.input(0)].shape
        input_shape = input_shape[-2:]  # get h and w
        output_shape = source_graph.blob_map[source_node.output(0)].shape
        output_shape = output_shape[-2:]  # get h and w

        mode = source_node.attrs.get('mode', 'nearest')
        if mode == 'nearest':
            mode = 'nearest_neighbor'
        elif mode == 'bilinear':
            pass
        else:
            logging.warning('Unsupport resize type, use nearest_neighbor instead')
            mode = 'nearest_neighbor'
        align_corners = source_node.attrs.get('align_corners', True)
        scales = source_node.attrs.get('scale_factor')
        size = [source_node.attrs.get('out_h'), source_node.attrs.get('out_w')]
        size = None if None in size else size

        if align_corners is False and source_node.attrs.get('align_mode', 0) == 0:
            size_ext, scales_ext = mapper_helper.resize_shape_helper(input_shape, output_shape, size, scales, True)
        else:
            size_ext, scales_ext = mapper_helper.resize_shape_helper(input_shape, output_shape, size, scales)

        # if scales is integer and equal in h/w use Upsample, otherwise use Interp
        if isinstance(scales_ext[0], int) and isinstance(scales_ext[1], int) and scales_ext[0] == scales_ext[1]:
            upsample_attrs = {
                'scale': scales_ext[0],
                'upsample_type': mode
            }
            # transfer info: input blob: 1, output blob: 1, node: 1, extra_blob: 0, extra_node: 0
            upsample_node = mapper_helper.convert_identity_operation(
                caffe_graph, source_node, source_graph, 'Upsample', upsample_attrs)
            return OP_MAPPING_IDENTITY, [upsample_node.name]
        else:
            interp_attrs = {
                'height': size_ext[0],
                'width': size_ext[1],
                'interp_type': mode
            }
            # transfer info: input blob: 1, output blob: 1, node: 1, extra_blob: 0, extra_node: 0
            interp_node = mapper_helper.convert_identity_operation(
                caffe_graph, source_node, source_graph, 'Interp', interp_attrs)
            return OP_MAPPING_IDENTITY, [interp_node.name]


@op_mapper('set_value')
class SetValue:
    have_custom = False

    @classmethod
    def standard_mapping(cls, caffe_graph: Graph, source_node: Node, source_graph: Graph, **kw):
        """
        https://www.paddlepaddle.org.cn/documentation/docs/zh/1.8/api_cn/fluid_cn/Variable_cn.html#set-value
        use slice + concat instead in caffe
            concat(x[:, :ends, :] = y, x[:, ends:, :])
        """
        assert len(source_node.input_blobs_names) == 2
        input_x_shape = source_graph.blob_map[source_node.input(0)].shape
        input_y_shape = source_graph.blob_map[source_node.input(1)].shape
        assert len(input_x_shape) == len(input_y_shape)
        # check offset
        for idx in range(2, len(input_x_shape)):
            assert input_x_shape[idx] == input_y_shape[idx], 'currently only valid in channel dim'

        input_x_blob = caffe_graph.get_blob(source_node.input(0))
        input_y_blob = caffe_graph.get_blob(source_node.input(1))
        output_blob_name = source_node.output(0)

        extra_blob_name_slice_0 = input_x_blob.name + '_extra_dummy'
        extra_blob_name_slice_1 = input_x_blob.name + '_extra_slice_1'

        extra_slice_name = source_node.name + '_extra_slice'
        source_name = source_node.name  # final concat will use source name

        # connect input
        assert extra_slice_name not in input_x_blob.dst_nodes_names
        input_x_blob.dst_nodes_names.append(extra_slice_name)
        # add slice
        slice_attrs = {'slice_point': [input_y_shape[1]], 'axis': 1}
        slice_node = caffe_graph.make_node('Slice', source_node.raw_name + '_extra_slice', extra_slice_name,
                                           [input_x_blob.name], [extra_blob_name_slice_0, extra_blob_name_slice_1],
                                           attrs=slice_attrs, do_insert=True)
        # without creat blob instance since graph wont use it anymore
        # extra_blob_name_slice_0 = caffe_graph.make_blob(input_y_shape, extra_blob_name_slice_0, extra_blob_name_slice_0,
        #                                                 slice_node.name, [], do_insert=True)
        slice_shape = list(input_x_shape)
        slice_shape[1] = input_x_shape[1] - input_y_shape[1]
        extra_blob_name_slice_1 = caffe_graph.make_blob(slice_shape, extra_blob_name_slice_1, extra_blob_name_slice_1,
                                                        slice_node.name, [source_node.name], do_insert=True)
        # add concat
        concat_node = caffe_graph.make_node('Concat', source_node.raw_name, source_node.name,
                                            [input_y_blob.name, extra_blob_name_slice_1], [output_blob_name],
                                            attrs={'axis': 1}, do_insert=True)
        # connect output
        _ = caffe_graph.transfer_op_output(source_node, source_graph)

        return OP_MAPPING_WITH_EXTRA, [slice_node.name, concat_node.name]
