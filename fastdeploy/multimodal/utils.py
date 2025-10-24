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

import base64
import io
import ipaddress
import mimetypes
import os
import socket
import subprocess
import tempfile
from urllib.parse import urlparse

import cairosvg
import pyheif
import requests
from pdf2image import convert_from_path
from PIL import Image, ImageOps

from fastdeploy.utils import data_processor_logger


def process_image_data(image_data, mime_type, url):
    """处理不同类型的图像数据并返回 PIL 图像对象"""

    if mime_type in ["image/heif", "image/heic"] or url.lower().endswith(".heif") or url.lower().endswith(".heic"):
        heif_file = pyheif.read(image_data)
        pil_image = Image.frombytes(
            heif_file.mode,
            heif_file.size,
            heif_file.data,
            "raw",
            heif_file.mode,
            heif_file.stride,
        )
    elif mime_type == "application/pdf" or url.lower().endswith(".pdf"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(image_data.getvalue())
            temp_pdf_path = temp_pdf.name
        images = convert_from_path(temp_pdf_path)
        pil_image = images[0]
        os.remove(temp_pdf_path)
    elif mime_type == "image/svg+xml" or url.lower().endswith(".svg"):
        png_data = cairosvg.svg2png(bytestring=image_data.getvalue())
        pil_image = Image.open(io.BytesIO(png_data))
    elif mime_type in [
        "application/postscript",
        "application/illustrator",
    ] or url.lower().endswith(".ai"):
        with (
            tempfile.NamedTemporaryFile(delete=False, suffix=".ai") as ai_temp,
            tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as pdf_temp,
        ):
            ai_temp_path = ai_temp.name
            pdf_temp_path = pdf_temp.name
            ai_temp.write(image_data.getvalue())
            ai_temp.close()
            subprocess.run(
                ["inkscape", ai_temp_path, "--export-pdf=" + pdf_temp_path],
                check=True,
            )
            images = convert_from_path(pdf_temp_path)
            pil_image = images[0]
            os.remove(ai_temp_path)
            os.remove(pdf_temp_path)

    elif mime_type == "image/gif" or url.lower().endswith(".gif"):
        pil_image = Image.open(image_data)
    else:
        pil_image = Image.open(image_data)

    return pil_image


def http_to_pil_image(url):
    """http_to_pil_image"""

    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to download the image from URL.")
    image_data = io.BytesIO(response.content)

    mime_type = response.headers.get("Content-Type")
    if mime_type is None:
        mime_type, _ = mimetypes.guess_type(url)

    data_processor_logger.info(f"Detected MIME type: {mime_type}")  # 调试信息
    pil_image = process_image_data(image_data, mime_type, url)

    return pil_image


def base64_to_pil_image(base64_string):
    """base64_to_pil_image"""
    image_bytes = base64.b64decode(base64_string)
    buffer = io.BytesIO(image_bytes)
    pil_image = Image.open(buffer)
    return pil_image


def is_public_url(url):
    """判断是否公网url"""
    try:
        # 解析URL
        parsed_url = urlparse(url)
        hostname = parsed_url.hostname
        if hostname is None:
            return False
        # 尝试将域名解析为IP地址
        ip_address = socket.gethostbyname(hostname)
        # 转换为IP地址对象
        ip_obj = ipaddress.ip_address(ip_address)
        # 判断是否为私有IP或保留IP地址
        if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local or ip_obj.is_reserved:
            return False
        else:
            return True
    except Exception as e:
        print(f"Error checking URL: {e}")
        return False


def process_transparency(image):
    """process transparency."""

    def _is_transparent(image):
        # 检查图片是否有alpha通道
        if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
            # 获取alpha通道
            alpha = image.convert("RGBA").split()[-1]
            # 如果alpha通道中存在0，说明图片有透明部分
            if alpha.getextrema()[0] < 255:
                return True
        return False

    def _convert_transparent_paste(image):
        width, height = image.size
        new_image = Image.new("RGB", (width, height), (255, 255, 255))  # 生成一张白色底图
        new_image.paste(image, (0, 0), image)
        return new_image

    try:
        if _is_transparent(image):  # Check and fix transparent images
            data_processor_logger.info("Image has transparent background, adding white background.")
            image = _convert_transparent_paste(image)
    except:
        pass

    return ImageOps.exif_transpose(image)
