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

import inspect


class PassManager(object):
    PASSES = {}

    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs

    def __call__(self, cls):
        for k, v in inspect.getmembers(cls, inspect.ismethod):
            if k == 'run_pass':
                self.PASSES[self.name] = (v, self.kwargs)

    @staticmethod
    def run_pass(graph, custom_pass_list):
        for pass_name in custom_pass_list:
            try:
                pass_func, kw = PassManager.PASSES[pass_name]
                pass_func(graph, **kw)
            except:
                raise Exception("Error happened when excute pass: {}".format(
                    pass_name))
        return graph
