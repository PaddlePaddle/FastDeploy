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
from onnx import helper
from onnx import TensorProto

ONNX = TensorProto

DTYPE_PADDLE_ONNX_MAP = {
    TensorProto.FLOAT16: core.VarDesc.VarType.FP16,
    TensorProto.FLOAT: core.VarDesc.VarType.FP32,
    TensorProto.DOUBLE: core.VarDesc.VarType.FP64,
    TensorProto.INT16: core.VarDesc.VarType.INT16,
    TensorProto.INT32: core.VarDesc.VarType.INT32,
    TensorProto.INT64: core.VarDesc.VarType.INT64,
    TensorProto.BOOL: core.VarDesc.VarType.BOOL,
    TensorProto.UINT8: core.VarDesc.VarType.UINT8,
    core.VarDesc.VarType.FP16: TensorProto.FLOAT16,
    core.VarDesc.VarType.FP32: TensorProto.FLOAT,
    core.VarDesc.VarType.FP64: TensorProto.DOUBLE,
    core.VarDesc.VarType.INT16: TensorProto.INT16,
    core.VarDesc.VarType.INT32: TensorProto.INT32,
    core.VarDesc.VarType.INT64: TensorProto.INT64,
    core.VarDesc.VarType.BOOL: TensorProto.BOOL,
    core.VarDesc.VarType.UINT8: TensorProto.UINT8,
}

DTYPE_PADDLE_NUMPY_MAP = {
    np.float32: core.VarDesc.VarType.FP32,
    np.float64: core.VarDesc.VarType.FP64,
    np.int16: core.VarDesc.VarType.INT16,
    np.int32: core.VarDesc.VarType.INT32,
    np.int64: core.VarDesc.VarType.INT64,
    np.bool_: core.VarDesc.VarType.BOOL,
    core.VarDesc.VarType.FP32: np.float32,
    core.VarDesc.VarType.FP64: np.float64,
    core.VarDesc.VarType.INT16: np.int16,
    core.VarDesc.VarType.INT32: np.int32,
    core.VarDesc.VarType.INT64: np.int64,
    core.VarDesc.VarType.BOOL: np.bool_
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

DTYPE_ONNX_STR_MAP = {
    TensorProto.FLOAT: 'float32',
    TensorProto.DOUBLE: 'float64',
    TensorProto.INT16: 'int16',
    TensorProto.INT32: 'int32',
    TensorProto.INT64: 'int64',
    TensorProto.BOOL: 'bool',
    'float32': TensorProto.FLOAT,
    'float64': TensorProto.DOUBLE,
    'int16': TensorProto.INT16,
    'int32': TensorProto.INT32,
    'int64': TensorProto.INT64,
    'bool': TensorProto.BOOL,
}
