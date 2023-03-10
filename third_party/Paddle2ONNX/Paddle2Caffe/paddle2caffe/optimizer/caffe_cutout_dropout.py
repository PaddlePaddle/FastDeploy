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
from paddle2caffe.utils import logging

from paddle2caffe.graph import CaffeGraph, CaffeNode


class OptCaffeCutoutDropout:
    """
    OptCaffeCutoutDropout
    """
    def __init__(self, graph: CaffeGraph):
        self.graph = graph
        self.process_node_list = []
        self.process_blob_list = []

    def match(self):
        """
        check whether this optimizer pass is valid in current graph
        """
        match = True
        if not isinstance(self.graph, CaffeGraph):
            logging.warning('input graph is not match with opt type')
            match = False
        if 'Dropout' not in self.graph.op_type_count.keys() or \
                self.graph.op_type_count['Dropout'] == 0:
            logging.info('pass opt_caffe_cutout_dropout is skipped')
            match = False

        return match

    def apply(self):
        """
        apply this optimizer pass in current graph
        """
        graph = self.graph
        process_node_list = self.process_node_list
        process_blob_list = self.process_blob_list

        logging.info('pass opt_caffe_cutout_dropout processed...')

        dropout_node = []
        for _, node in graph.node_map.items():
            if node.op_type == 'Dropout':
                dropout_node.append(node)

        for node in dropout_node:
            input_blob = graph.get_blob(node.input_blobs_names[0])
            output_blob = graph.get_blob(node.output_blobs_names[0])

            # modify connection
            # 1. connect output_blob's dst node into input_blob
            for node_name in output_blob.dst_nodes_names:
                if node_name not in input_blob.dst_nodes_names:
                    input_blob.dst_nodes_names.append(node_name)
                    son_node = graph.get_node(node_name)
                    # 2. reconnect dst node's input blobs into input_blob
                    for idx, blob_name in enumerate(son_node.input_blobs_names):
                        if blob_name == output_blob.name:
                            # inplace change to protect connection
                            son_node.input_blobs_names[idx] = input_blob.name
                else:
                    raise NotImplementedError
            # 3. delete output_blob and node
            graph.remove_blob(output_blob.name)
            process_blob_list.append(output_blob.name)
            graph.remove_node(node.name)
            process_node_list.append(node.name)

        logging.info('\tcurrent pass process total nodes_num:{}'.format(len(process_node_list)))
        logging.debug('\ttotal nodes name:{}'.format(process_node_list))
        logging.info('\tcurrent pass process total blobs_num:{}'.format(len(process_blob_list)))
        logging.debug('\ttotal blobs name:{}'.format(process_blob_list))

        return self.graph
