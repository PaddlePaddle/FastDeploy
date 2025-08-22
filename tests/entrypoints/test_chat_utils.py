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

import os
import tempfile
import unittest
import uuid
from pathlib import Path
from copy import deepcopy
from urllib.parse import urlparse


# Standalone implementations for testing (copied from source)
def random_tool_call_id() -> str:
    return f"chatcmpl-tool-{str(uuid.uuid4().hex)}"


def load_chat_template(chat_template, is_literal=False):
    if chat_template is None:
        return None
    if is_literal:
        if isinstance(chat_template, Path):
            raise TypeError("chat_template is expected to be read directly from its value")
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


class MockMediaIO:
    def load_bytes(self, data):
        return f"media_from_bytes({len(data)})"

    def load_base64(self, media_type, data):
        return f"media_from_base64({media_type}, {data})"

    def load_file(self, path):
        return f"media_from_file({path})"


class MultiModalPartParser:
    def __init__(self):
        self.image_io = MockMediaIO()
        self.video_io = MockMediaIO()

    def parse_image(self, image_url):
        return self.load_from_url(image_url, self.image_io)

    def parse_video(self, video_url):
        return self.load_from_url(video_url, self.video_io)

    def load_from_url(self, url, media_io):
        parsed = urlparse(url)
        if parsed.scheme.startswith("http"):
            media_bytes = b"mock_http_data"  # Mock HTTP response
            return media_io.load_bytes(media_bytes)

        if parsed.scheme.startswith("data"):
            data_spec, data = parsed.path.split(",", 1)
            media_type, data_type = data_spec.split(";", 1)
            return media_io.load_base64(media_type, data)

        if parsed.scheme.startswith("file"):
            localpath = parsed.path
            return media_io.load_file(localpath)


def parse_content_part(mm_parser, part):
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


def parse_chat_messages(messages):
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


class TestChatUtils(unittest.TestCase):
    """Test chat utility functions"""

    def test_random_tool_call_id(self):
        """Test random tool call ID generation"""
        tool_id = random_tool_call_id()

        # Should start with expected prefix
        self.assertTrue(tool_id.startswith("chatcmpl-tool-"))

        # Should contain a UUID hex string
        uuid_part = tool_id.replace("chatcmpl-tool-", "")
        self.assertEqual(len(uuid_part), 32)  # UUID hex is 32 chars

        # Should be different each time
        tool_id2 = random_tool_call_id()
        self.assertNotEqual(tool_id, tool_id2)

    def test_load_chat_template_literal(self):
        """Test loading chat template as literal string"""
        template = "Hello {{name}}"
        result = load_chat_template(template, is_literal=True)
        self.assertEqual(result, template)

    def test_load_chat_template_literal_with_path_object(self):
        """Test loading chat template with Path object in literal mode should raise error"""
        template_path = Path("/some/path")
        with self.assertRaises(TypeError):
            load_chat_template(template_path, is_literal=True)

    def test_load_chat_template_from_file(self):
        """Test loading chat template from file"""
        template_content = "Hello {{name}}, how are you?"

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write(template_content)
            temp_path = f.name

        try:
            result = load_chat_template(temp_path)
            self.assertEqual(result, template_content)
        finally:
            os.unlink(temp_path)

    def test_load_chat_template_file_not_found(self):
        """Test loading chat template from non-existent file"""
        # Test with path-like string that looks like a file path
        with self.assertRaises(ValueError) as cm:
            load_chat_template("/non/existent/path.txt")

        self.assertIn("looks like a file path", str(cm.exception))

    def test_load_chat_template_fallback_to_literal(self):
        """Test fallback to literal when file doesn't exist but contains jinja chars"""
        template = "Hello {{name}}\nHow are you?"
        result = load_chat_template(template)
        self.assertEqual(result, template)

    def test_load_chat_template_none(self):
        """Test loading None template"""
        result = load_chat_template(None)
        self.assertIsNone(result)

    def test_parse_chat_messages_text_only(self):
        """Test parsing chat messages with text content only"""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]

        result = parse_chat_messages(messages)

        expected = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hi there!"}]}
        ]

        self.assertEqual(result, expected)

    def test_parse_chat_messages_none_content(self):
        """Test parsing chat messages with None content"""
        messages = [{"role": "user", "content": None}]
        result = parse_chat_messages(messages)

        expected = [{"role": "user", "content": []}]
        self.assertEqual(result, expected)

    def test_parse_content_part_text(self):
        """Test parsing text content part"""
        parser = MultiModalPartParser()
        part = {"type": "text", "text": "Hello world"}

        result = parse_content_part(parser, part)
        self.assertEqual(result, part)

    def test_parse_content_part_image_url(self):
        """Test parsing image URL content part"""
        parser = MultiModalPartParser()
        part = {
            "type": "image_url",
            "image_url": {"url": "http://example.com/image.jpg"}
        }

        result = parse_content_part(parser, part)

        expected = {
            "type": "image",
            "image_url": {},
            "image": "media_from_bytes(14)"  # Mock HTTP response data
        }
        self.assertEqual(result, expected)

    def test_parse_content_part_video_url(self):
        """Test parsing video URL content part"""
        parser = MultiModalPartParser()
        part = {
            "type": "video_url",
            "video_url": {"url": "http://example.com/video.mp4"}
        }

        result = parse_content_part(parser, part)

        expected = {
            "type": "video",
            "video_url": {},
            "video": "media_from_bytes(14)"  # Mock HTTP response data
        }
        self.assertEqual(result, expected)

    def test_parse_content_part_unknown_type(self):
        """Test parsing unknown content part type"""
        parser = MultiModalPartParser()
        part = {"type": "unknown", "data": "test"}

        with self.assertRaises(ValueError) as cm:
            parse_content_part(parser, part)

        self.assertIn("Unknown content part type: unknown", str(cm.exception))

    def test_multimodal_part_parser_data_url(self):
        """Test parsing data URL"""
        parser = MultiModalPartParser()
        result = parser.load_from_url("data:image/jpeg;base64,SGVsbG8gV29ybGQ=", parser.image_io)
        self.assertEqual(result, "media_from_base64(image/jpeg, SGVsbG8gV29ybGQ=)")

    def test_multimodal_part_parser_file_url(self):
        """Test parsing file URL"""
        parser = MultiModalPartParser()
        result = parser.load_from_url("file:///path/to/image.jpg", parser.image_io)
        self.assertEqual(result, "media_from_file(/path/to/image.jpg)")


if __name__ == "__main__":
    unittest.main()