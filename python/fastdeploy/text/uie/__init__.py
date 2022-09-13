# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import logging
from ... import Frontend
from ... import RuntimeOption
from ... import c_lib_wrap as C


class SchemaNode(object):
    def __init__(self, name, children=[]):
        schema_node_children = []
        if isinstance(children, str):
            children = [children]
        for child in children:
            if isinstance(child, str):
                schema_node_children += [C.text.SchemaNode(child, [])]
            elif isinstance(child, dict):
                for key, val in child.items():
                    schema_node_child = SchemaNode(key, list(val))
                    schema_node_children += [schema_node_child._schema_node]
            else:
                assert "The type of child of SchemaNode should be str or dict."
        self._schema_node = C.text.SchemaNode(name, schema_node_children)
        self._schema_node_children = schema_node_children


class UIEModel(object):
    def __init__(self,
                 model_file,
                 params_file,
                 vocab_file,
                 position_prob=0.5,
                 max_length=128,
                 schema=[],
                 runtime_option=RuntimeOption(),
                 model_format=Frontend.PADDLE):
        if isinstance(schema, list):
            schema = SchemaNode("", schema)._schema_node_children
        elif isinstance(schema, dict):
            schema_tmp = []
            for key, val in schema.items():
                schema_tmp += [SchemaNode(key, val)._schema_node]
            schema = schema_tmp
        else:
            assert "The type of schema should be list or dict."
        self._model = C.text.UIEModel(model_file, params_file, vocab_file,
                                      position_prob, max_length, schema,
                                      runtime_option._option, model_format)

    def set_schema(self, schema):
        if isinstance(schema, list):
            schema = SchemaNode("", schema)._schema_node_children
        elif isinstance(schema, dict):
            schema_tmp = []
            for key, val in schema.items():
                schema_tmp += [SchemaNode(key, val)._schema_node]
            schema = schema_tmp
        self._model.set_schema(schema)

    def predict(self, texts):
        return self._model.predict(texts)
