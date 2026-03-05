# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""
FastDeploy logging module.

Provides a unified logging interface for all FastDeploy components.

Usage::

    from fastdeploy.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Hello from FastDeploy")
"""

from fastdeploy.logger.logger import FastDeployLogger


def get_logger(name=None):
    """Get a FastDeploy logger with the given name.

    The name will be prefixed with 'fastdeploy.' automatically
    if it is not already in the fastdeploy namespace.

    Args:
        name: Logger name. If None, returns the root fastdeploy logger.

    Returns:
        logging.Logger instance
    """
    return FastDeployLogger().get_logger(name)


__all__ = ["get_logger", "FastDeployLogger"]
