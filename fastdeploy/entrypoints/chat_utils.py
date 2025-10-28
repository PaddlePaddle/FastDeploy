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

import os
import time
import uuid
from pathlib import Path
from typing import List, Literal, Optional, Union
from urllib.parse import urlparse

import requests
from openai.types.chat import (
    ChatCompletionContentPartParam as OpenAIChatCompletionContentPartParam,
)
from openai.types.chat import (
    ChatCompletionMessageParam as OpenAIChatCompletionMessageParam,
)
from openai.types.chat.chat_completion_content_part_image_param import ImageURL
from typing_extensions import Required, TypeAlias, TypedDict

from fastdeploy.multimodal.image import ImageMediaIO
from fastdeploy.multimodal.video import VideoMediaIO
from fastdeploy.utils import api_server_logger


class CustomChatCompletionContentPartImageParam(TypedDict, total=False):
    """Custom Image URL object"""

    type: Required[Literal["image_url"]]
    """The type of the content part."""

    image_url: Optional[ImageURL]

    uuid: Optional[str]


class VideoURL(TypedDict, total=False):
    """Video URL object"""

    url: Required[str]
    """Either a URL of the video or the base64 encoded video data"""


class CustomChatCompletionContentPartVideoParam(TypedDict, total=False):
    """Custom Video URL object"""

    type: Required[Literal["video_url"]]
    """The type of the content part."""

    video_url: Optional[VideoURL]

    uuid: Optional[str]


CustomChatCompletionContentPartParam: TypeAlias = Union[
    OpenAIChatCompletionContentPartParam,
    CustomChatCompletionContentPartImageParam,
    CustomChatCompletionContentPartVideoParam,
]


class CustomChatCompletionMessageParam(TypedDict, total=False):
    """Custom User chat message parameter."""

    content: Required[Union[str, List[CustomChatCompletionContentPartParam]]]
    """The contents of the user message"""

    role: Required[str]
    """The role of the messages author, in this case `user`."""

    name: str
    """An optional name for the participant

    Provides the model information to differentiate between participants of the same role.
    """


ChatCompletionMessageParam = Union[OpenAIChatCompletionMessageParam, CustomChatCompletionMessageParam]


class MultimodalPartParser:
    """Multi Modal Part parser"""

    def __init__(self):
        self.image_io = ImageMediaIO()
        self.video_io = VideoMediaIO()

    def parse_image(self, image_url):
        """ "Parse Image"""
        return self.load_from_url(image_url, self.image_io)

    def parse_video(self, video_url):
        """Parse Video"""
        return self.load_from_url(video_url, self.video_io)

    def http_get_with_retry(self, url, max_retries=3, retry_delay=1, backoff_factor=2):
        """HTTP GET retry"""

        retry_cnt = 0
        delay = retry_delay

        while retry_cnt < max_retries:
            try:
                response = requests.get(url)
                response.raise_for_status()
                return response.content
            except Exception as e:
                retry_cnt += 1
                if retry_cnt >= max_retries:
                    api_server_logger.error(f"HTTP GET failed: {e}. Max retries reached")
                    raise
                api_server_logger.info(f"HTTP GET failed: {e}. Start retry {retry_cnt}")
                time.sleep(delay)
                delay *= backoff_factor

    def load_from_url(self, url, media_io):
        """Load media from URL"""

        parsed = urlparse(url)
        if parsed.scheme.startswith("http"):
            media_bytes = self.http_get_with_retry(url)
            return media_io.load_bytes(media_bytes)

        if parsed.scheme.startswith("data"):
            data_spec, data = parsed.path.split(",", 1)
            media_type, data_type = data_spec.split(";", 1)
            return media_io.load_base64(media_type, data)

        if parsed.scheme.startswith("file"):
            localpath = parsed.path
            return media_io.load_file(localpath)


def parse_content_part(mm_parser, part):
    """only support openai compatible format for now"""

    part_type = part.get("type", None)

    if part_type == "text":
        return part

    if part_type == "image_url":
        if not part.get("image_url", None) and not part.get("uuid", None):
            raise ValueError("Both image_url and uuid are missing")

        if part.get("image_url", None):
            url = part["image_url"]["url"]
            image = mm_parser.parse_image(url)
        else:
            image = None

        parsed = {}
        parsed["type"] = "image"
        parsed["data"] = image
        parsed["uuid"] = part.get("uuid", None)

        return parsed
    if part_type == "video_url":
        if not part.get("video_url", None) and not part.get("uuid", None):
            raise ValueError("Both video_url and uuid are missing")

        if part.get("video_url", None):
            url = part["video_url"]["url"]
            video = mm_parser.parse_video(url)
        else:
            video = None

        parsed = {}
        parsed["type"] = "video"
        parsed["data"] = video
        parsed["uuid"] = part.get("uuid", None)

        return parsed

    raise ValueError(f"Unknown content part type: {part_type}")


# TODO async
def parse_chat_messages(messages: List[ChatCompletionMessageParam]):
    """Parse chat messages to [dict]"""

    mm_parser = MultimodalPartParser()

    conversation = []
    for message in messages:
        role = message["role"]
        content = message["content"]

        parsed_content = []
        if content is None:
            parsed_content = []
        elif isinstance(content, str):
            parsed_content = [{"type": "text", "text": content}]
        else:
            parsed_content = [parse_content_part(mm_parser, part) for part in content]

        conversation.append({"role": role, "content": parsed_content})
    return conversation


def load_chat_template(
    chat_template: Union[Path, str],
    model_path: Path = None,
    is_literal: bool = False,
) -> Optional[str]:
    if chat_template is None:
        if model_path:
            chat_template_file = os.path.join(model_path, "chat_template.jinja")
            if os.path.exists(chat_template_file):
                with open(chat_template_file) as f:
                    return f.read()
        return None
    if is_literal:
        if isinstance(chat_template, Path):
            raise TypeError("chat_template is expected to be read directly " "from its value")

        return chat_template

    try:
        with open(chat_template) as f:
            return f.read()
    except OSError as e:
        if isinstance(chat_template, Path):
            raise
        JINJA_CHARS = "{}\n"
        if not any(c in chat_template for c in JINJA_CHARS):
            msg = (
                f"The supplied chat template ({chat_template}) "
                f"looks like a file path, but it failed to be "
                f"opened. Reason: {e}"
            )
            raise ValueError(msg) from e

        # If opening a file fails, set chat template to be args to
        # ensure we decode so our escape are interpreted correctly
        return load_chat_template(chat_template, is_literal=True)


def random_tool_call_id() -> str:
    return f"chatcmpl-tool-{str(uuid.uuid4().hex)}"
