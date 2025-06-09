"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
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
"""

import setuptools
import os

long_description = "FastDeploy: Large Language Model Serving.\n\n"
long_description += "GitHub: https://github.com/PaddlePaddle/FastDeploy\n"
long_description += "Email: dltp@baidu.com"


def load_requirements():
    """加载requirements.txt中的依赖"""
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    with open(requirements_path, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setuptools.setup(
    name="fastdeploy",
    version="2.0.0-alpha",
    author="PaddlePaddle",
    author_email="dltp@baidu.com",
    description="FastDeploy: Large Language Model Serving.",
    long_description=long_description,
    long_description_content_type="text/plain",
    url="https://github.com/PaddlePaddle/FastDeploy",
    packages=setuptools.find_packages(),
    package_dir={"fastdeploy": "fastdeploy/"},
    package_data={
        "fastdeploy": [
            "model_executor/ops/gpu/*",
            "model_executor/ops/gpu/deep_gemm/include/**/*",
            "model_executor/ops/cpu/*",
            "model_executor/ops/xpu/*",
            "model_executor/ops/npu/*",
            "model_executor/ops/base/*",
            "model_executor/models/*",
            "model_executor/layers/*",
            "input/mm_processor/utils/*"
        ]
    },
    install_requires=load_requirements(),
    classifiers=[
        "Programming Language :: Python :: 3", 
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],  
    license='Apache 2.0',
) 
