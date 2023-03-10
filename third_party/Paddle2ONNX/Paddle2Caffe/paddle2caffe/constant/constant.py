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

PRODUCER = 'PaddlePaddle'

PADDLE_VERSION = '2.1.0'
CAFFE_CUSTOM_VERSION = '2022.04.13'


class NodeDomain(object):
    PADDLE = 'paddle'

    CAFFE_CUSTOM = 'caffe_custom'
    CAFFE = 'caffe'

    NONE = 'none'


class DataDomain(object):
    FLOAT32 = 'float32'
    FLOAT64 = 'float64'
    INT16 = 'int16'
    INT32 = 'int32'
    INT64 = 'int64'
    BOOL = 'bool'

    NONE = 'none'
