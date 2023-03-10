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

from .caffe_cutout_dropout import OptCaffeCutoutDropout
# from .paddle_fold_const import OptPaddleFoldConst


class OptManager(object):
    """
    OptManager
    """
    strategy_map = dict()

    @classmethod
    def register(cls, graph_type, pass_name, pass_cls):
        """

        :param graph_type:
        :param pass_name:
        :param pass_cls:
        :return:
        """
        if graph_type not in cls.strategy_map:
            cls.strategy_map[graph_type] = dict()

        if pass_name in cls.strategy_map[graph_type].keys():
            logging.warning('register a same pass_name:{} as former passes'.format(pass_name))

        cls.strategy_map[graph_type][pass_name] = pass_cls

    @classmethod
    def pick(cls, graph_type, pass_name_list='all'):
        """

        :param graph_type:
        :param pass_name_list:
        :return:
        """
        assert cls.strategy_map.get(graph_type, None)
        if pass_name_list is 'all':
            pass_cls_list = []
            for pass_name, pass_cls in cls.strategy_map[graph_type].items():
                pass_cls_list.append(pass_cls)
            return pass_cls_list
        else:
            return [pass_cls for pass_cls in cls.strategy_map[graph_type][pass_name_list]]


# register
OptManager.register('CAFFE', 'opt_caffe_cutout_dropout', OptCaffeCutoutDropout)
# OptManager.register('PADDLE', 'opt_paddle_cutout_dropout', OptPaddleFoldConst)

