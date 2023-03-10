#   Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import paddle.fluid.core as core

from paddle2caffe.caffe_helper.caffe_pb2 import BlobProto

# numpy
DTYPE_STR_NUMPY_MAP = {
    np.float32: 'float32',
    np.float64: 'float64',
    np.int16: 'int16',
    np.int32: 'int32',
    np.int64: 'int64',
    np.bool: 'bool',
    'float32': np.float32,
    'float64': np.float64,
    'int16': np.int16,
    'int32': np.int32,
    'int64': np.int64,
    'bool': np.bool
}

# paddle
DTYPE_PADDLE_NUMPY_MAP = {
    np.float32: core.VarDesc.VarType.FP32,
    np.float64: core.VarDesc.VarType.FP64,
    np.int16: core.VarDesc.VarType.INT16,
    np.int32: core.VarDesc.VarType.INT32,
    np.int64: core.VarDesc.VarType.INT64,
    np.bool: core.VarDesc.VarType.BOOL,
    core.VarDesc.VarType.FP32: np.float,
    core.VarDesc.VarType.FP64: np.float64,
    core.VarDesc.VarType.INT16: np.int16,
    core.VarDesc.VarType.INT32: np.int32,
    core.VarDesc.VarType.INT64: np.int64,
    core.VarDesc.VarType.BOOL: np.bool
}

DTYPE_PADDLE_STR_MAP = {
    core.VarDesc.VarType.FP32: 'float32',
    core.VarDesc.VarType.FP64: 'float64',
    core.VarDesc.VarType.INT16: 'int16',
    core.VarDesc.VarType.INT32: 'int32',
    core.VarDesc.VarType.INT64: 'int64',
    core.VarDesc.VarType.BOOL: 'bool',
    'float32': core.VarDesc.VarType.FP32,
    'float64': core.VarDesc.VarType.FP64,
    'int16': core.VarDesc.VarType.INT16,
    'int32': core.VarDesc.VarType.INT32,
    'int64': core.VarDesc.VarType.INT64,
    'bool': core.VarDesc.VarType.BOOL
}

# caffe
# without map since only have float32 and float64 data type
CAFFE_BLOB = BlobProto
