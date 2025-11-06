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

from PIL import Image, ImageOps

from fastdeploy.utils import data_processor_logger


def process_transparency(image):
    """process transparency."""

    def _is_transparent(image):
        # Check if image has alpha channel
        if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
            # Get alpha channel
            alpha = image.convert("RGBA").split()[-1]
            # If alpha channel contains 0, image has transparent part
            if alpha.getextrema()[0] < 255:
                return True
        return False

    def _convert_transparent_paste(image):
        width, height = image.size
        new_image = Image.new("RGB", (width, height), (255, 255, 255))  # Generate an image with white background
        new_image.paste(image, (0, 0), image)
        return new_image

    try:
        if _is_transparent(image):  # Check and fix transparent images
            data_processor_logger.info("Image has transparent background, adding white background.")
            image = _convert_transparent_paste(image)
    except:
        pass

    return ImageOps.exif_transpose(image)
