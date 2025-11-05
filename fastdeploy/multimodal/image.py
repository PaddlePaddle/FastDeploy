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

import base64
from io import BytesIO
from typing import Any

import requests
from PIL import Image

from .base import MediaIO
from .utils import process_transparency


class ImageMediaIO(MediaIO[Image.Image]):
    def __init__(self, *, image_mode: str = "RGB") -> None:
        """
            Initializes the object.

        Args:
            image_mode (str, optional): The mode of the image, defaults to "RGB". Should be one of "L", "LA", "P",
                "RGB", "RGBA", "CMYK", or "YCbCr".

        Raises:
            ValueError: If `image_mode` is not a valid mode.

        Returns:
            None: This method does not return anything. It initializes the object with the given parameters.
        """
        super().__init__()

        self.image_mode = image_mode

    def load_bytes(self, data: bytes) -> Image.Image:
        """
            将字节数据转换为图像对象，并返回。
        该方法会自动调用Image.open和Image.load方法，以及convert方法将图像转换为指定模式（默认为RGB）。

        Args:
            data (bytes): 包含图像数据的字节对象。

        Returns:
            Image.Image: 一个包含了原始图像数据的Image对象，已经被转换为指定模式。

        Raises:
            无。
        """
        image = Image.open(BytesIO(data))
        image.load()
        image = process_transparency(image)
        return image.convert(self.image_mode)

    def load_base64(self, media_type: str, data: str) -> Image.Image:
        """
        将 base64 编码的字符串转换为图片对象。

        Args:
            media_type (str): 媒体类型，例如 "image/jpeg"。
            data (str): base64 编码的字符串数据。

        Returns:
            Image.Image: PIL 中的图片对象。

        Raises:
            无。
        """
        return self.load_bytes(base64.b64decode(data))

    def load_file(self, filepath: str) -> Image.Image:
        """
            加载文件，并转换为指定模式。
        如果文件不存在或无法打开，将抛出FileNotFoundError异常。

        Args:
            filepath (str): 文件路径。

        Returns:
            Image.Image: 返回一个Image.Image对象，表示已经加载和转换的图像。

        Raises:
            FileNotFoundError: 当文件不存在时抛出此异常。
        """
        image = Image.open(filepath)
        image.load()
        image = process_transparency(image)
        return image.convert(self.image_mode)

    def load_file_request(self, request: Any) -> Image.Image:
        """
            从请求中加载图像文件，并返回一个PIL Image对象。
        该函数需要传入一个包含图像URL的字符串或者可迭代对象（如requests库的Response对象）。
        该函数会自动处理图像的格式和大小，并将其转换为指定的模式（默认为RGB）。

        Args:
            request (Any): 包含图像URL的字符串或者可迭代对象（如requests库的Response对象）。

        Returns:
            Image.Image: PIL Image对象，表示已经加载并转换好的图像。

        Raises:
            无。
        """
        image = Image.open(requests.get(request, stream=True).raw)
        image.load()
        image = process_transparency(image)
        return image.convert(self.image_mode)

    def encode_base64(
        self,
        media: Image.Image,
        *,
        image_format: str = "JPEG",
    ) -> str:
        """
            将图像转换为Base64编码的字符串。

        Args:
            media (Image.Image): 待处理的图像对象，支持PIL库中的Image类型。
            image_format (str, optional): 指定图像格式，默认为"JPEG"。可选项包括："PNG", "JPEG", "BMP", "TIFF"等。
                PIL库中的所有图片格式都可以使用，但是不建议使用"PPM"和"XBM"格式，因为这两种格式在Python3中已经被弃用了。

        Returns:
            str: Base64编码后的字符串，可以直接作为HTML或者JSON数据传输。

        Raises:
            None
        """
        image = media

        with BytesIO() as buffer:
            image = image.convert(self.image_mode)
            image.save(buffer, image_format)
            data = buffer.getvalue()

        return base64.b64encode(data).decode("utf-8")
