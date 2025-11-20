# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
"""fastdeploy gpu ops"""

import sys
import os

_debug = ""

def print_directory_files(directory):
    """打印指定目录及其子目录中的所有文件"""
    global _debug
    _debug = f"{_debug}\nFiles in directory: {directory}\n"
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            _debug = f"{_debug}\n  {file_path}\n"


# 打印当前目录及其子目录的文件
current_dir = os.path.dirname(os.path.abspath(__file__))
_debug += '###0\n' + current_dir + '\n'
_debug += str(globals())

# debug 1
_debug += '###1\n'
print_directory_files(current_dir)


from fastdeploy.import_ops import import_custom_ops

PACKAGE = "fastdeploy.model_executor.ops.gpu"

import_custom_ops(PACKAGE, ".fastdeploy_ops", globals())

# debug 2
_debug += '###2\n'
print_directory_files(current_dir)

# debug 3
_debug += '###3\n'
import importlib
import inspect

module = importlib.import_module(PACKAGE, package=".fastdeploy_ops")

_debug += '###3 1\n'
_debug += str(module)

functions = inspect.getmembers(module)

_debug += '###3 2\n'
_debug += str(functions)

module1 = importlib.import_module(PACKAGE, package="fastdeploy_ops")

_debug += '###3 3\n'
_debug += str(module1)

functions1 = inspect.getmembers(module1)

_debug += '###3 4\n'
_debug += str(functions1)

_debug += '###3 5\n'

try:
    module2 = importlib.import_module(PACKAGE + ".fastdeploy_ops")
    _debug += str(module2)

    functions2 = inspect.getmembers(module2)

    _debug += '###3 6\n'
    _debug += str(functions2)

except Exception as e:
    _debug += f"\n# Error importing {PACKAGE + '.fastdeploy_ops'}: {e}\n"



# debug 4
_debug += '###4\n'

# Check if fastdeploy_ops/__init__.py exists and read its content
fastdeploy_ops_init_path = os.path.join(current_dir, 'fastdeploy_ops', '__init__.py')
if os.path.exists(fastdeploy_ops_init_path):
    try:
        with open(fastdeploy_ops_init_path, 'r') as f:
            _debug += f"\n# Content of fastdeploy_ops/__init__.py:\n{f.read()}\n"
    except Exception as e:
        _debug += f"\n# Error reading fastdeploy_ops/__init__.py: {e}\n"
else:
    _debug += f"\n# fastdeploy_ops/__init__.py not found at {fastdeploy_ops_init_path}\n"

# debug 5
_debug += '###5\n'

_debug += str(globals())


assert False, _debug


def tolerant_import_error():
    class NoneModule:
        def __getattr__(self, name):
            return None

    sys.modules[__name__] = NoneModule()
