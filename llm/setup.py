# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import setuptools

setuptools.setup(
    name="fastdeploy-llm",
    version="0.0.7",
    author="fastdeploy",
    author_email="fastdeploy@baidu.com",
    description="FastDeploy for Large Language Model",
    long_description_content_type="text/plain",
    url="https://github.com/PaddlePaddle/FastDeploy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires=["colorlog"],
    extras_require={"client": ['grpcio', 'tritonclient']},
    license='Apache 2.0')
