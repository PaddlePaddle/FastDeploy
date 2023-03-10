#   Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys
import uuid
from importlib import import_module
from caffe import layers as L
from caffe import params as P
import numpy as np
import six

from paddle2caffe.utils import logging
from paddle2caffe.caffe_helper.caffe_pb2 import *


class CaffeEmitter(object):

    def __init__(self, graph=None):
        self.phase = None
        self.use_default_name = True  # do not creat layer name, only blob name
        self.caffe_graph = graph
        self.body_codes = ''
        self.params_dict = dict()

    def export(self, save_file, tmp_dir='output', do_clean=True):
        export_code = self.gen_code()
        export_weight = self.gen_params()
        random_mark = uuid.uuid4().hex[:16]
        temp_filename = 'p2c-' + random_mark

        if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)
        tmp_weight_path = os.path.join(tmp_dir, temp_filename + '.npy')
        tmp_code_path = os.path.join(tmp_dir, temp_filename + '.py')

        np.save(tmp_weight_path, export_weight, allow_pickle=True)
        with open(tmp_code_path, 'w') as f:
            f.write(export_code)

        sys.path.append(os.path.dirname(os.path.abspath(tmp_code_path)))
        MainModel = import_module(temp_filename)

        dump_net = str(save_file + '.prototxt')
        dump_weight = str(save_file + '.caffemodel')
        if os.path.isfile(dump_net):
            logging.warning('will remove exist file')
            os.remove(dump_net)
        if os.path.isfile(dump_weight):
            logging.warning('will remove exist file')
            os.remove(dump_weight)

        MainModel.make_net(dump_net)
        MainModel.gen_weight(tmp_weight_path, dump_weight, dump_net)
        logging.info('Caffe model files are saved as [{}] and [{}].'.format(dump_net, dump_weight))
        if do_clean:
            os.remove(tmp_weight_path)
            os.remove(tmp_code_path)

    def gen_params(self):
        for node in self.caffe_graph.node_map.values():
            if len(node.params) > 0:
                if self.use_default_name:
                    # use blob0 name as default layer name
                    self.params_dict[node.output_blobs_names[0]] = dict()
                    for k, name in node.params.items():
                        self.params_dict[node.output_blobs_names[0]][k] = self.caffe_graph.parameters.get(name)
                else:
                    # use layer name directly
                    self.params_dict[node.name] = dict()
                    for k, name in node.params.items():
                        self.params_dict[node.name][k] = self.caffe_graph.parameters.get(name)

        return self.params_dict

    def add_body(self, indent, codes):
        if isinstance(codes, six.string_types):
            codes = [codes]
        for code in codes:
            self.body_codes += ("    " * indent) + code + '\n'

    @property
    def header_code(self):
        return """
from __future__ import print_function
import numpy as np
import sys, argparse
from six import text_type as _text_type
import caffe
from caffe import layers as L


_weights_dict = dict()


def load_weights(weight_file):
    if weight_file is None:
        return

    try:
        weights_dict = np.load(weight_file, allow_pickle=True).item()
    except:
        weights_dict = np.load(weight_file, allow_pickle=True, encoding='bytes').item()

    return weights_dict


def KitModel(weight_file=None):
    n = caffe.NetSpec()
"""

    @property
    def end_code(self):
        return """
    return n


def make_net(prototxt):
    n = KitModel()
    with open(prototxt, 'w') as fpb:
        print(n.to_proto(), file=fpb)


def gen_weight(weight_file, model, prototxt):
    global _weights_dict
    _weights_dict = load_weights(weight_file)

    net = caffe.Net(prototxt, caffe.TRAIN)

    for key in _weights_dict:
        if 'weights' in _weights_dict[key]:
            net.params[key][0].data.flat = _weights_dict[key]['weights']
        elif 'mean' in _weights_dict[key]:
            net.params[key][0].data.flat = _weights_dict[key]['mean']
            net.params[key][1].data.flat = _weights_dict[key]['variance']
            if 'scale' in _weights_dict[key]:
                net.params[key][2].data.flat = _weights_dict[key]['scale']
            else:
                # to see is to believe
                net.params[key][2].data.flat = np.array(1.)
        elif 'scale' in _weights_dict[key]:
            net.params[key][0].data.flat = _weights_dict[key]['scale']
            
        if 'bias' in _weights_dict[key]:
            net.params[key][1].data.flat = _weights_dict[key]['bias']
            
        if 'gamma' in _weights_dict[key]: # used for prelu, not sure if other layers use this too
            net.params[key][0].data.flat = _weights_dict[key]['gamma']

    net.save(model)
    return net


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate caffe model and prototxt')
    parser.add_argument('--weight_file', '-w', type=_text_type, default='p2c-[uuid16].npy')
    parser.add_argument('--prototxt', '-p', type=_text_type, default='caffe_converted.prototxt')
    parser.add_argument('--model', '-m', type=_text_type, default='caffe_converted.caffemodel')
    args = parser.parse_args()
    # For some reason argparser gives us unicode, so we need to conver to str first
    make_net(str(args.prototxt))
    gen_weight(str(args.weight_file), str(args.model), str(args.prototxt))

"""

    def gen_input(self):
        for node in self.caffe_graph.input_nodes:
            input_shape = list(self.caffe_graph.blob_map[node.output_blobs_names[0]].shape)
            if input_shape[0] < 1:
                input_shape[0] = 1  # fix batch shape
            self.add_body(1, '{} = L.Input(shape=[dict(dim={})]{}, ntop=1)'.format(
                'n.{}'.format(node.output_blobs_names[0]),
                input_shape,
                ', name=\'{}\''.format(node.name) if not self.use_default_name else ''
            ))

    def gen_code(self, phase='test', use_default_name=True):
        self.phase = phase
        self.use_default_name = use_default_name
        self.add_body(0, self.header_code)
        self.gen_input()
        for node_name, current_node in self.caffe_graph.node_map.items():
            node_type = str.lower(current_node.op_type)  # function name prefers lower format

            if hasattr(self, "emit_" + node_type):
                func = getattr(self, "emit_" + node_type)
                func(current_node)
            else:
                logging.warning("CaffeEmitter has not supported operator \'{}\', "
                                "use default emit method".format(node_type))
                self.emit_default(current_node)

        self.add_body(0, self.end_code)

        return self.body_codes

    def emit_convolution(self, node):
        """https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto#L592"""
        if node.attrs['dilations'][0] != node.attrs['dilations'][1]:
            logging.warning('dilations h and w is not equal, which will ignored in caffe')
        if node.attrs['paddings'][0] != node.attrs['paddings'][1] or node.attrs['paddings'][2] != node.attrs['paddings'][3]:
            logging.warning('paddings h/w begin and end is not equal, which will ignored in caffe')
        self.add_body(1, '{} = L.Convolution({}{}, num_output={}, bias_term={}, dilation={}, group={}'
                         ', stride_h={}, stride_w={}, kernel_h={}, kernel_w={}, pad_h={}, pad_w={}, ntop={})'
                      .format(', '.join(['n.{}'.format(opt) for opt in node.output_blobs_names]),
                              ', '.join(['n.{}'.format(ipt) for ipt in node.input_blobs_names]),
                              ', name=\'{}\''.format(node.name) if not self.use_default_name else '',
                              node.attrs['num_output'],
                              node.attrs['bias_term'],
                              node.attrs['dilations'][0],
                              node.attrs['group'],
                              node.attrs['strides'][0],
                              node.attrs['strides'][1],
                              node.attrs['kernels'][0],
                              node.attrs['kernels'][1],
                              node.attrs['paddings'][0],
                              node.attrs['paddings'][2],
                              len(node.output_blobs_names)))

    def emit_pooling(self, node):
        """https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto#L919"""
        if node.attrs['paddings'][0] != node.attrs['paddings'][1] or node.attrs['paddings'][2] != node.attrs['paddings'][3]:
            logging.warning('paddings h/w begin and end is not equal, which will ignored in caffe')
        if not node.attrs.get('ceil_mode', True):
            logging.warning('pooling with ceil_mode is not support in standard caffe')
            ceil_mode_str = ', ceil_mode=False'
        else:
            ceil_mode_str = ''

        pooling_type = node.attrs.get('pooling_type')
        if str.upper(pooling_type) == 'MAX':
            pooling_type = P.Pooling.MAX
        elif str.upper(pooling_type) == 'AVG':
            pooling_type = P.Pooling.AVE
        elif str.upper(pooling_type) == 'STOCHASTIC':
            pooling_type = P.Pooling.STOCHASTIC
        else:
            raise ValueError('unknown pooling type for caffe pooling')

        if node.attrs.get('global_pooling', False) is True:
            self.add_body(1, '{} = L.Pooling({}{}, pool={}, global_pooling=True, ntop={})'.format(
                ', '.join(['n.{}'.format(opt) for opt in node.output_blobs_names]),
                ', '.join(['n.{}'.format(ipt) for ipt in node.input_blobs_names]),
                ', name=\'{}\''.format(node.name) if not self.use_default_name else '',
                pooling_type,
                len(node.output_blobs_names)
            ))
        else:
            self.add_body(1, '{} = L.Pooling({}{}, pool={}, stride_h={}, stride_w={}'
                             ', kernel_h={}, kernel_w={}, pad_h={}, pad_w={}{}, ntop={})'
                          .format(', '.join(['n.{}'.format(opt) for opt in node.output_blobs_names]),
                                  ', '.join(['n.{}'.format(ipt) for ipt in node.input_blobs_names]),
                                  ', name=\'{}\''.format(node.name) if not self.use_default_name else '',
                                  pooling_type,
                                  node.attrs['strides'][0],
                                  node.attrs['strides'][1],
                                  node.attrs['kernels'][0],
                                  node.attrs['kernels'][1],
                                  node.attrs['paddings'][0],
                                  node.attrs['paddings'][2],
                                  ceil_mode_str,
                                  len(node.output_blobs_names)))

    def emit_batchnorm(self, node):
        """https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto#L526"""
        self.add_body(1, '{} = L.BatchNorm({}{}, eps={}, use_global_stats={}, ntop={})'.format(
            ', '.join(['n.{}'.format(opt) for opt in node.output_blobs_names]),
            ', '.join(['n.{}'.format(ipt) for ipt in node.input_blobs_names]),
            ', name=\'{}\''.format(node.name) if not self.use_default_name else '',
            node.attrs.get('eps', 1e-5),
            self.phase == 'test',
            len(node.output_blobs_names)
        ))

    def emit_scale(self, node):
        """https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto#L1101"""
        self.add_body(1, '{} = L.Scale({}{}, axis={}, num_axes={}, bias_term={}, ntop={})'.format(
            ', '.join(['n.{}'.format(opt) for opt in node.output_blobs_names]),
            ', '.join(['n.{}'.format(ipt) for ipt in node.input_blobs_names]),
            ', name=\'{}\''.format(node.name) if not self.use_default_name else '',
            node.attrs.get('axis', 1),
            node.attrs.get('num_axes', 1),
            True if 'bias' in node.params.keys() else False,
            len(node.output_blobs_names)
        ))

    def emit_relu(self, node):
        """https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto#L1022"""
        self.emit_default(node)

    def emit_prelu(self, node):
        """https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto#L1441"""
        self.emit_default(node)

    def emit_power(self, node):
        """https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto#L954"""
        self.emit_default(node)

    def emit_absval(self, node):
        """do not have layer params"""
        self.emit_default(node)

    def emit_exp(self, node):
        """https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto#L757"""
        self.emit_default(node)

    def emit_argmax(self, node):
        """https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto#L498"""
        self.emit_default(node)

    def emit_clip(self, node):
        """https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto#L510"""
        self.emit_default(node)

    def emit_softmax(self, node):
        """https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto#L1159"""
        self.emit_default(node)

    def emit_sigmoid(self, node):
        """https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto#L1138"""
        self.emit_default(node)

    def emit_slice(self, node):
        """https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto#L1147"""
        self.emit_default(node)

    def emit_eltwise(self, node):
        """https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto#L1022"""
        operation = node.attrs.get('operation', 'sum')
        if str.upper(operation) == 'SUM':
            operation = P.Eltwise.SUM
        elif str.upper(operation) == 'PROD':
            operation = P.Eltwise.PROD
        elif str.upper(operation) == 'MAX':
            operation = P.Eltwise.MAX
        else:
            raise ValueError('unknown operation type for caffe eltwise')
        self.add_body(1, '{} = L.Eltwise({}{}, operation={}, ntop={})'.format(
            ', '.join(['n.{}'.format(opt) for opt in node.output_blobs_names]),
            ', '.join(['n.{}'.format(ipt) for ipt in node.input_blobs_names]),
            ', name=\'{}\''.format(node.name) if not self.use_default_name else '',
            operation,
            len(node.output_blobs_names)
        ))

    def emit_flatten(self, node):
        """https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto#L767"""
        self.emit_default(node)

    def emit_dropout(self, node):
        """https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto#L697"""
        self.emit_default(node)

    def emit_concat(self, node):
        """https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto#L515"""
        self.add_body(1, '{} = L.Concat({}{}, axis={}, ntop={})'.format(
            ', '.join(['n.{}'.format(opt) for opt in node.output_blobs_names]),
            ', '.join(['n.{}'.format(ipt) for ipt in node.input_blobs_names]),
            ', name=\'{}\''.format(node.name) if not self.use_default_name else '',
            node.attrs.get('axis', 1),
            len(node.output_blobs_names)
        ))

    def emit_innerproduct(self, node):
        """https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto#L767"""
        self.emit_default(node)

    def emit_reshape(self, node):
        """https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto#L1037"""
        dim_str = "'dim': {}".format(node.attrs.get('shape'))
        dim_str = " reshape_param={'shape': { " + dim_str + '} }'
        self.add_body(1, '{} = L.Reshape({}{}, {}, ntop={})'.format(
            ', '.join(['n.{}'.format(opt) for opt in node.output_blobs_names]),
            ', '.join(['n.{}'.format(ipt) for ipt in node.input_blobs_names]),
            ', name=\'{}\''.format(node.name) if not self.use_default_name else '',
            dim_str,
            len(node.output_blobs_names)
        ))

    def emit_matmul(self, node):
        """custom op"""
        self.add_body(1, '{} = L.MatMul({}{}, ntop={})'.format(
            ', '.join(['n.{}'.format(opt) for opt in node.output_blobs_names]),
            ', '.join(['n.{}'.format(ipt) for ipt in node.input_blobs_names]),
            ', name=\'{}\''.format(node.name) if not self.use_default_name else '',
            len(node.output_blobs_names)
        ))

    def emit_upsample(self, node):
        """custom op"""
        self.add_body(1, '{} = L.Upsample({}{}, scale={}, upsample_type=\'{}\', ntop={})'.format(
            ', '.join(['n.{}'.format(opt) for opt in node.output_blobs_names]),
            ', '.join(['n.{}'.format(ipt) for ipt in node.input_blobs_names]),
            ', name=\'{}\''.format(node.name) if not self.use_default_name else '',
            node.attrs['scale'],
            node.attrs['upsample_type'],
            len(node.output_blobs_names)
        ))

    def emit_interp(self, node):
        """custom op"""
        self.add_body(1, '{} = L.Interp({}{}, height={}, width={}, interp_type=\'{}\', ntop={})'.format(
            ', '.join(['n.{}'.format(opt) for opt in node.output_blobs_names]),
            ', '.join(['n.{}'.format(ipt) for ipt in node.input_blobs_names]),
            ', name=\'{}\''.format(node.name) if not self.use_default_name else '',
            node.attrs['height'],
            node.attrs['width'],
            node.attrs['interp_type'],
            len(node.output_blobs_names)
        ))

    def emit_relu6(self, node):
        """custom op"""
        self.emit_default(node)

    def emit_permute(self, node):
        """custom op"""
        self.emit_default(node)

    def emit_priorbox(self, node):
        """custom op"""
        not_required_str = ''
        if len(node.attr('min_size')) > 0:
            not_required_str += ', min_size={}'.format(node.attrs['min_size'])
        if len(node.attr('max_size')) > 0:
            not_required_str += ', max_size={}'.format(node.attrs['max_size'])
        if len(node.attr('aspect_ratio')) > 0:
            not_required_str += ', aspect_ratio={}'.format(node.attrs['aspect_ratio'])
        if len(node.attr('variance')) > 0:
            not_required_str += ', variance={}'.format(node.attrs['variance'])
        if node.attr('fixed_size') is not None and len(node.attrs['fixed_sizes']) > 0:
            not_required_str += ', fixed_sizes={}'.format(node.attrs['fixed_sizes'])
        if node.attr('fixed_ratios') is not None and len(node.attrs['fixed_ratios']) > 0:
            not_required_str += ', fixed_ratios={}'.format(node.attrs['fixed_ratios'])
        if node.attr('offset') is not None:
            not_required_str += ', offset={}'.format(node.attrs['offset'])

        self.add_body(1, '{} = L.PriorBox({}{}, clip={}, flip={}, step_w={}, step_h={}{}, ntop={})'.format(
            ', '.join(['n.{}'.format(opt) for opt in node.output_blobs_names]),
            ', '.join(['n.{}'.format(ipt) for ipt in node.input_blobs_names]),
            ', name=\'{}\''.format(node.name) if not self.use_default_name else '',
            node.attrs['clip'],
            node.attrs['flip'],
            node.attrs['step_w'],
            node.attrs['step_h'],
            not_required_str,
            len(node.output_blobs_names)
        ))

    def emit_detectionoutput(self, node):
        """custom op"""
        # code type
        code_type = ''
        if node.attr('code_type') == 'CORNER':
            code_type = P.PriorBox.CORNER
        if node.attr('code_type') == 'CENTER_SIZE':
            code_type = P.PriorBox.CENTER_SIZE
        if node.attr('code_type') == 'CORNER_SIZE':
            code_type = P.PriorBox.CORNER_SIZE
        # nms param
        nms_param = {}
        if node.attr('nms_param.nms_threshold') and node.attr('nms_param.top_k'):
            nms_param['nms_threshold'] = node.attr('nms_param.nms_threshold')
            nms_param['top_k'] = node.attr('nms_param.top_k')

        self.add_body(1, '{} = L.DetectionOutput({}{}, num_classes={}, share_location={}, '
                         'background_label_id={}, code_type={}, keep_top_k={}, confidence_threshold={}, '
                         'nms_param={}, ntop={})'.format(
                            ', '.join(['n.{}'.format(opt) for opt in node.output_blobs_names]),
                            ', '.join(['n.{}'.format(ipt) for ipt in node.input_blobs_names]),
                            ', name=\'{}\''.format(node.name) if not self.use_default_name else '',
                            node.attrs['num_classes'],
                            node.attrs['share_location'],
                            node.attrs['background_label_id'],
                            code_type,
                            node.attrs['keep_top_k'],
                            node.attrs['confidence_threshold'],
                            nms_param,
                            len(node.output_blobs_names)))

    def emit_normalize(self, node):
        """custom op"""
        self.emit_default(node)

    def emit_yolov3detectionoutput(self, node):
        """custom op"""
        self.emit_default(node)

    def emit_default(self, node):
        attr_str = ''
        for key, value in node.attrs.items():
            attr_str += ', {}={}'.format(key, value)
        self.add_body(1, '{} = L.{}({}{}{}, ntop={})'.format(
            ', '.join(['n.{}'.format(opt) for opt in node.output_blobs_names]),
            node.op_type,
            ', '.join(['n.{}'.format(ipt) for ipt in node.input_blobs_names]),
            ', name=\'{}\''.format(node.name) if not self.use_default_name else '',
            attr_str,
            len(node.output_blobs_names)
        ))
