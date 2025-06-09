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
    """
    Represents a video URL or base64 encoded video data.
    
    Attributes:
        url: Required string containing either a URL or base64 encoded video data
    """
    url: Required[str]
    """Either a URL of the video or the base64 encoded video data"""

class CustomChatCompletionContentPartVideoParam(TypedDict, total=False):
    """
    Custom video content part parameter for chat completion.
    
    Attributes:
        video_url: Required VideoURL object containing video data
        type: Required literal string "video_url" indicating content type
    """
    video_url: Required[VideoURL]

    type: Required[Literal["video_url"]]
    """The type of the content type."""

CustomChatCompletionContentPartParam: TypeAlias = Union[
    OpenAIChatCompletionContentPartParam, CustomChatCompletionContentPartVideoParam
]

class CustomUserChatCompletionMessageParam(TypedDict, total=False):
    """
    Custom user chat message parameter for chat completion.
    
    Attributes:
        content: Required content of the message (string or list of content parts)
        role: Required string indicating the role of message author (should be 'user')
        name: Optional name to differentiate between participants of same role
    """

    content: Required[Union[str, List[CustomChatCompletionContentPartParam]]]
    """The contents of the user message"""

    role: Required[str]
    """The role of the messages author, in this case `user`."""

    name: str
    """An optional name for the participant

    Provides the model information to differentiate between participants of the same role.
    """

ChatCompletionMessageParam = Union[OpenAIChatCompletionMessageParam, CustomUserChatCompletionMessageParam]


class MultiModalPartParser(object):
    """
    Parser for handling multi-modal content parts (images, videos, etc.)
    
    Attributes:
        image_io: ImageMediaIO instance for handling image operations
        video_io: VideoMediaIO instance for handling video operations
    """
    def __init__(self):
        self.image_io = ImageMediaIO()
        self.video_io = VideoMediaIO(self.image_io)

    def parse_image(self, image_url):
        """
        Parse an image from given URL.
        
        Args:
            image_url: URL or base64 string of the image
            
        Returns:
            Parsed image data
        """
        # image_io = ImageMediaIO()
        return self.load_from_url(image_url, self.image_io)

    def parse_video(self, video_url):
        """
        Parse a video from given URL.
        
        Args:
            video_url: URL or base64 string of the video
            
        Returns:
            Parsed video data
        """
        # video_io = VideoMediaIO()
        return self.get_bytes(video_url)


    def load_from_url(self, url, media_io):
        """
        Load media content from URL or base64 string.
        
        Args:
            url: URL or base64 string of the media
            media_io: MediaIO instance for handling the specific media type
            
        Returns:
            Loaded media data
        """

        parsed = urlparse(url)
        if parsed.scheme.startswith("http"):
            media_bytes = self.get_bytes(url)
            return media_io.load_bytes(media_bytes)
        
        if parsed.scheme.startswith("data"):
            data_spec, data = parsed.path.split(",", 1)
            media_type, data_type = data_spec.split(";", 1)
            return media_io.load_base64(media_type, data)

        if parsed.scheme.startswith("file"):
            localpath = parsed.path
            return media_io.load_file(localpath)

    def get_bytes(self, url):
        """
        Fetch raw bytes from a URL.
        
        Args:
            url: URL to fetch data from
            
        Returns:
            bytes: Raw content from the URL
        """
        # TODO: Add error handling and timeout
        return requests.get(url).content


def parse_content_part(mm_parser, part):
    """
    Parse a single content part (text, image or video).
    Currently supports OpenAI-compatible formats.
    
    Args:
        mm_parser: MultiModalPartParser instance
        part: Content part to parse
        
    Returns:
        dict: Parsed content part
        
    Raises:
        ValueError: If content part type is unknown
    """

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
    """
    Parse a list of chat messages into standardized format.
    
    Args:
        messages: List of chat messages to parse
        
    Returns:
        list: Parsed conversation in standardized format
    """

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
