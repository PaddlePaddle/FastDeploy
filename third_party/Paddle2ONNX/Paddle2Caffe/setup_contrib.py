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

from collections import namedtuple
import os
import subprocess
import setuptools


TOP_DIR = os.path.realpath(os.path.dirname(__file__))
SRC_DIR = os.path.join(TOP_DIR, 'paddle2caffe')
packages = setuptools.find_packages()

################################################################################
# Version
################################################################################
try:
    git_version = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=TOP_DIR).decode('ascii').strip()
except (OSError, subprocess.CalledProcessError):
    git_version = None
with open(os.path.join(TOP_DIR, 'VERSION_NUMBER')) as version_file:
    VersionInfo = namedtuple('VersionInfo', ['version', 'git_version'])(
        version=version_file.read().strip(), git_version=git_version)


################################################################################
# Setup
################################################################################
setuptools.setup(
    name="paddle2caffe",
    version=VersionInfo.version,
    description="Export PaddlePaddle to ONNX",
    # ext_modules=ext_modules,
    # cmdclass=cmdclass,
    packages=packages,
    include_package_data=True,
    # setup_requires=setup_requires,
    # extras_require=extras_require,
    author='paddle-infer',
    author_email='paddle-infer@baidu.com',
    url='https://github.com/PaddlePaddle/Paddle2ONNX.git',
    # install_requires=['six', 'protobuf', 'caffe'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    license='Apache 2.0',
    entry_points={'console_scripts': ['paddle2caffe=paddle2caffe.command:main']}
)
