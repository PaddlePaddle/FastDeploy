# Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
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

from paddle2onnx.utils import logging
from . import command
from .convert import dygraph2onnx
from .convert import program2onnx
from .version import version
from .version import git_version

__version__ = version
__commit_id__ = git_version


def run_convert(model, input_shape_dict=None, scope=None, opset_version=9):
    logging.warning(
        "[Deprecated] `paddle2onnx.run_convert` will be deprecated in the future version, the recommended usage is `paddle2onnx.export`"
    )
    from paddle2onnx.legacy import run_convert
    return run_convert(model, input_shape_dict, scope, opset_version)


def export(model_file,
           params_file="",
           save_file=None,
           opset_version=11,
           auto_upgrade_opset=True,
           verbose=True,
           enable_onnx_checker=True,
           enable_experimental_op=True,
           enable_optimize=True,
           custom_op_info=None,
           deploy_backend="onnxruntime",
           calibration_file="",
           external_file=""):
    import paddle2onnx.paddle2onnx_cpp2py_export as c_p2o
    deploy_backend = deploy_backend.lower()
    if custom_op_info is None:
        onnx_model_str = c_p2o.export(
            model_file, params_file, opset_version, auto_upgrade_opset, verbose,
            enable_onnx_checker, enable_experimental_op, enable_optimize, {},
            deploy_backend, calibration_file, external_file)
    else:
        onnx_model_str = c_p2o.export(
            model_file, params_file, opset_version, auto_upgrade_opset, verbose,
            enable_onnx_checker, enable_experimental_op, enable_optimize,
            custom_op_info, deploy_backend, calibration_file, external_file)
    if save_file is not None:
        with open(save_file, "wb") as f:
            f.write(onnx_model_str)
    else:
        return onnx_model_str
