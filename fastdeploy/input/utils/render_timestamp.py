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

"""Render timestamps onto video frames."""

import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_PATH = os.path.join(Path(__file__).parent.absolute(), "Roboto-Regular.ttf")


def render_single_image_with_timestamp(image: Image, number: str, rate: float, font_path: str = FONT_PATH):
    """Render a timestamp string onto a PIL Image.

    The font size is ``min(width, height) * rate``.
    Text is drawn in black with a white outline (10% of font size).
    """
    draw = ImageDraw.Draw(image)
    width, height = image.size
    font_size = int(min(width, height) * rate)
    outline_size = int(font_size * 0.1)
    font = ImageFont.truetype(font_path, font_size)
    x = 0
    y = 0

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
    """Convert timestamp from seconds to ``HH:MM:SS.ss`` format."""
    hours = 0
    while time_stamp_in_seconds >= 3600:
        hours += 1
        time_stamp_in_seconds -= 3600
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
    """Get the timestamp of a frame during uniform extraction.

    Returns the timestamp in seconds.
    """
    time_stamp = duration * 1.0 * frame_id / num_frames

    return time_stamp


def render_frame_timestamp(frame, timestamp, font_rate=0.1):
    """Render a timestamp onto a video frame.

    Parameters
    ----------
    frame : PIL.Image
        The video frame.
    timestamp : float
        Timestamp in seconds.
    font_rate : float
        Font size as a fraction of ``min(width, height)``.
    """
    time_stamp = "time: " + timestamp_converting(timestamp)
    new_frame = render_single_image_with_timestamp(frame, time_stamp, font_rate)

    return new_frame
