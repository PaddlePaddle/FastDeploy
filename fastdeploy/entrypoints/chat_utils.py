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

from typing import Literal, Union, List
from typing_extensions import Required, TypedDict, TypeAlias

from openai.types.chat import ChatCompletionContentPartParam as OpenAIChatCompletionContentPartParam
from openai.types.chat import ChatCompletionMessageParam as OpenAIChatCompletionMessageParam

from urllib.parse import urlparse
import requests
from copy import deepcopy

from fastdeploy.input.multimodal.video import VideoMediaIO
from fastdeploy.input.multimodal.image import ImageMediaIO

class VideoURL(TypedDict, total=False):
    """Video URL object"""
    url: Required[str]
    """Either a URL of the video or the base64 encoded video data"""

class CustomChatCompletionContentPartVideoParam(TypedDict, total=False):
    """Custom Video URL object"""
    video_url: Required[VideoURL]

    type: Required[Literal["video_url"]]
    """The type of the content type."""

CustomChatCompletionContentPartParam: TypeAlias = Union[
    OpenAIChatCompletionContentPartParam, CustomChatCompletionContentPartVideoParam
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


class MultiModalPartParser(object):
    """Multi Modal Part parser"""
    def __init__(self):
        self.image_io = ImageMediaIO()
        self.video_io = VideoMediaIO()

    def parse_image(self, image_url):
        """"Parse Image"""
        return self.load_from_url(image_url, self.image_io)

    def parse_video(self, video_url):
        """Parse Video"""
        return self.load_from_url(video_url, self.video_io)

    def load_from_url(self, url, media_io):
        """Load media from URL"""

        parsed = urlparse(url)
        if parsed.scheme.startswith("http"):
            media_bytes = requests.get(url).content
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
        content = part.get("image_url", {}).get("url", None)
        image = mm_parser.parse_image(content)
        parsed = deepcopy(part)
        del parsed["image_url"]["url"]
        parsed["image"] = image
        parsed["type"] = "image"
        return parsed

    if part_type == "video_url":
        content = part.get("video_url", {}).get("url", None)
        video = mm_parser.parse_video(content)
        parsed = deepcopy(part)
        del parsed["video_url"]["url"]
        parsed["video"] = video
        parsed["type"] = "video"
        return parsed

    raise ValueError(f"Unknown content part type: {part_type}")

#TODO async
#def parse_chat_messages(messages: List[ChatCompletionMessageParam]):
def parse_chat_messages(messages):
    """Parse chat messages to [dict]"""

    mm_parser = MultiModalPartParser()

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