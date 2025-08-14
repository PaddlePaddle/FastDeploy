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

import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

cur_directory = Path(__file__).parent.absolute()
FONT_PATH = os.path.join(cur_directory, "Roboto-Regular.ttf")


def render_single_image_with_timestamp(image: Image, number: str, rate: float, font_path: str = FONT_PATH):
    """
    函数功能: 给pil.image的图片渲染时间戳
    时间戳的大小为 min(width, height)的rate
    字体的颜色为黑色, 轮廓是白色, 轮廓的大小是字体的10%
    返回一个 Image 对象
    """
    draw = ImageDraw.Draw(image)  # 创建一个可绘制对象
    width, height = image.size  # 获取图片大小
    font_size = int(min(width, height) * rate)  # 设置字体大小
    outline_size = int(font_size * 0.1)  # 设置轮廓大小
    font = ImageFont.truetype(font_path, font_size)  # 加载字体文件, 设置字体大小
    x = 0
    y = 0  # 文本的x坐标, y坐标

    # 绘制黑色的时间戳，白色的边框
    draw.text(
        (x, y),
        number,
        font=font,
        fill=(0, 0, 0),
        stroke_width=outline_size,
        stroke_fill=(255, 255, 255),
    )

    return image


def timestamp_converting(time_stamp_in_seconds):
    """
    convert timestamp format from seconds to hr:min:sec
    """
    # get hours
    hours = 0
    while time_stamp_in_seconds >= 3600:
        hours += 1
        time_stamp_in_seconds -= 3600
    # get minutes
    mins = 0
    while time_stamp_in_seconds >= 60:
        mins += 1
        time_stamp_in_seconds -= 60
    time_hours = f"{int(hours):02d}"
    time_mins = f"{int(mins):02d}"
    time_secs = f"{time_stamp_in_seconds:05.02f}"
    fi_time_stamp = time_hours + ":" + time_mins + ":" + time_secs

    return fi_time_stamp


def get_timestamp_for_uniform_frame_extraction(num_frames, frame_id, duration):
    """
    function: get the timestamp of a frame, 在均匀抽帧时用。

    num_frames: 总帧数
    frameid_list: 被抽帧的帧的索引
    duration: 视频的总时长
    return: timestamp; xx:xx:xx (str)
    """
    time_stamp = duration * 1.0 * frame_id / num_frames

    return time_stamp


def render_frame_timestamp(frame, timestamp, font_rate=0.1):
    """
    函数功能, 给frame, 按照顺序将 index 渲染上去
    逻辑思路: 把index渲染到图片的左上方

    frame: 帧，PIL.Image object
    timestamp: 时间戳，单位是秒
    font_rate: 字体大小占 min(wi, hei)的比率
    """

    time_stamp = "time: " + timestamp_converting(timestamp)
    new_frame = render_single_image_with_timestamp(frame, time_stamp, font_rate)

    return new_frame
