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
import os
import random

import socket
from urllib.parse import urlparse
import ipaddress

import requests
from PIL import Image, ImageOps
from fastdeploy.utils import data_processor_logger

import pyheif
from pdf2image import convert_from_path
import cairosvg
import subprocess
import tempfile
import mimetypes

def process_image_data(image_data, mime_type, url):
    """处理不同类型的图像数据并返回 PIL 图像对象"""

    if mime_type in ['image/heif', 'image/heic'] or url.lower().endswith('.heif') or url.lower().endswith('.heic'):
        heif_file = pyheif.read(image_data)
        pil_image = Image.frombytes(
            heif_file.mode, heif_file.size, heif_file.data,
            "raw", heif_file.mode, heif_file.stride
        )
    elif mime_type == 'application/pdf' or url.lower().endswith('.pdf'):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
            temp_pdf.write(image_data.getvalue())
            temp_pdf_path = temp_pdf.name
        images = convert_from_path(temp_pdf_path)
        pil_image = images[0]
        os.remove(temp_pdf_path)
    elif mime_type == 'image/svg+xml' or url.lower().endswith('.svg'):
        png_data = cairosvg.svg2png(bytestring=image_data.getvalue())
        pil_image = Image.open(io.BytesIO(png_data))
    elif mime_type in ['application/postscript', 'application/illustrator'] or url.lower().endswith('.ai'):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.ai') as ai_temp, tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as pdf_temp:
            ai_temp_path = ai_temp.name
            pdf_temp_path = pdf_temp.name
            ai_temp.write(image_data.getvalue())
            ai_temp.close()
            subprocess.run(['inkscape', ai_temp_path, '--export-pdf=' + pdf_temp_path], check=True)
            images = convert_from_path(pdf_temp_path)
            pil_image = images[0]
            os.remove(ai_temp_path)
            os.remove(pdf_temp_path)

    elif mime_type == 'image/gif' or url.lower().endswith('.gif'):
        pil_image = Image.open(image_data)
    else:
        pil_image = Image.open(image_data)

    return pil_image

def http_to_pil_image(url):
    """http_to_pil_image"""
    if is_public_url(url) and int(os.getenv("DOWNLOAD_WITH_TP_SERVER", "0")):
        return http_to_pil_image_with_tp_server(url)

    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to download the image from URL.")
    image_data = io.BytesIO(response.content)

    mime_type = response.headers.get('Content-Type')
    if mime_type is None:
        mime_type, _ = mimetypes.guess_type(url)

    data_processor_logger.info(f"Detected MIME type: {mime_type}")  # 调试信息
    pil_image = process_image_data(image_data, mime_type, url)

    return pil_image

def http_to_pil_image_with_tp_server(url, retry_time=6):
    """cnap平台没有外网访问权限，需要使用tp服务下载图片"""
    proxies = [{"http": "http://10.229.197.142:8807"}, {"http": "http://10.229.197.161:8804"},
               {"http": "http://10.229.198.143:8804"}, {"http": "http://10.122.108.164:8807"},
               {"http": "http://10.122.108.165:8807"}, {"http": "http://10.122.108.166:8807"},
               {"http": "http://10.122.108.168:8801"}, {"http": "http://10.122.150.146:8802"},
               {"http": "http://10.122.150.158:8802"}, {"http": "http://10.122.150.164:8801"},
               {"http": "http://10.143.51.38:8813"}, {"http": "http://10.143.103.42:8810"},
               {"http": "http://10.143.194.45:8804"}, {"http": "http://10.143.226.25:8801"},
               {"http": "http://10.143.236.12:8807"}, {"http": "http://10.143.238.36:8807"},
               {"http": "http://10.144.71.30:8807"}, {"http": "http://10.144.73.16:8804"},
               {"http": "http://10.144.138.36:8801"}, {"http": "http://10.144.152.40:8810"},
               {"http": "http://10.144.199.29:8810"}, {"http": "http://10.144.251.29:8813"},
               ]
    headers = {
        "X-Tp-Authorization": "Basic RVJOSUVMaXRlVjpFUk5JRUxpdGVWXzFxYXo0cmZ2M2VkYzV0Z2Iyd3N4LWJmZS10cA==",
        "scheme": "https"
        }

    new_url = url.replace("https://", "http://") if url.startswith("https://") else url

    # 代理可能不稳定，需要重试
    for idx in range(retry_time):
        try:
            response = requests.get(new_url, headers=headers, proxies=random.choice(proxies))
            if response.status_code == 200:
                image_data = io.BytesIO(response.content)

                mime_type = response.headers.get('Content-Type')
                if mime_type is None:
                    mime_type, _ = mimetypes.guess_type(url)

                data_processor_logger.info(f"Detected MIME type: {mime_type}")  # 调试信息
                pil_image = process_image_data(image_data, mime_type, url)

                return pil_image
        except Exception as e:
            data_processor_logger.error(f"Failed to download the image, idx: {idx}, URL: {url}, error: {e}")

    raise Exception(f"Failed to download the image from URL: {url}")



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
    """ process transparency. """
    def _is_transparent(image):
        # 检查图片是否有alpha通道
        if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
            # 获取alpha通道
            alpha = image.convert('RGBA').split()[-1]
            # 如果alpha通道中存在0，说明图片有透明部分
            if alpha.getextrema()[0] < 255:
                return True
        return False


    def _convert_transparent_paste(image):
        width, height = image.size
        new_image = Image.new("RGB", (width, height), (255, 255, 255)) # 生成一张白色底图
        new_image.paste(image, (0, 0), image)
        return new_image

    try:
        if _is_transparent(image):  # Check and fix transparent images
            data_processor_logger.info("Image has transparent background, adding white background.")
            image = _convert_transparent_paste(image)
    except:
        pass

    return ImageOps.exif_transpose(image)
