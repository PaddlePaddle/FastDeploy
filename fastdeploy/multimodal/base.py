"""
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

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

_T = TypeVar("_T")


class MediaIO(ABC, Generic[_T]):
    @abstractmethod
    def load_bytes(self, data: bytes) -> _T:
        """
            将字节数据加载为对象，并返回该对象。
        如果加载失败，则抛出异常。

        Args:
            data (bytes): 要加载的字节数据。

        Raises:
            NotImplementedError: 当前类未实现此方法。

        Returns:
            _T: 加载后的对象。
        """
        raise NotImplementedError

    @abstractmethod
    def load_base64(self, media_type: str, data: str) -> _T:
        """
        List of media types:
        https://www.iana.org/assignments/media-types/media-types.xhtml
        """
        raise NotImplementedError

    @abstractmethod
    def load_file(self, filepath: Path) -> _T:
        """
            加载文件，返回解析后的数据。

        Args:
            filepath (Path): 文件路径，必须是一个绝对路径。

        Raises:
            NotImplementedError: 当前方法未被实现。

        Returns:
            _T: 任意类型，表示解析后的数据。
        """
        raise NotImplementedError
