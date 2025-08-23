"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
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
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastdeploy.entrypoints.chat_utils import (
    MultiModalPartParser,
    load_chat_template,
    parse_chat_messages,
    parse_content_part,
    random_tool_call_id,
)


class TestChatUtils(unittest.TestCase):
    """Test case for chat utils functionality"""

    def setUp(self):
        """Set up test environment"""
        self.parser = MultiModalPartParser()

    def test_random_tool_call_id(self):
        """Test random tool call ID generation"""
        tool_id = random_tool_call_id()
        self.assertTrue(tool_id.startswith("chatcmpl-tool-"))
        self.assertEqual(len(tool_id), len("chatcmpl-tool-") + 32)  # UUID hex is 32 chars
        
        # Test uniqueness
        tool_id2 = random_tool_call_id()
        self.assertNotEqual(tool_id, tool_id2)

    def test_load_chat_template_literal(self):
        """Test loading chat template as literal string"""
        template = "Hello {{ name }}"
        result = load_chat_template(template, is_literal=True)
        self.assertEqual(result, template)

    def test_load_chat_template_literal_with_path_raises_error(self):
        """Test that passing Path object with is_literal=True raises TypeError"""
        template_path = Path("/some/path")
        with self.assertRaises(TypeError):
            load_chat_template(template_path, is_literal=True)

    def test_load_chat_template_from_file(self):
        """Test loading chat template from file"""
        template_content = "Hello {{ name }}!\nHow are you?"
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write(template_content)
            temp_path = f.name
        
        try:
            result = load_chat_template(temp_path)
            self.assertEqual(result, template_content)
        finally:
            os.unlink(temp_path)

    def test_load_chat_template_file_not_found_with_jinja_chars(self):
        """Test loading non-existent file that looks like a template"""
        template = "Hello {{ name }}"  # Contains jinja chars
        result = load_chat_template(template)  # Should fallback to literal
        self.assertEqual(result, template)

    def test_load_chat_template_file_not_found_without_jinja_chars(self):
        """Test loading non-existent file that doesn't look like a template"""
        template = "/nonexistent/path/template.txt"
        with self.assertRaises(ValueError) as context:
            load_chat_template(template)
        
        self.assertIn("looks like a file path", str(context.exception))

    def test_load_chat_template_none_input(self):
        """Test loading None template"""
        result = load_chat_template(None)
        self.assertIsNone(result)

    def test_parse_content_part_text(self):
        """Test parsing text content part"""
        part = {"type": "text", "text": "Hello world"}
        result = parse_content_part(self.parser, part)
        self.assertEqual(result, part)

    @patch('requests.get')
    def test_parse_content_part_image_url_http(self, mock_get):
        """Test parsing image URL content part with HTTP URL"""
        mock_response = MagicMock()
        mock_response.content = b"fake_image_data"
        mock_get.return_value = mock_response
        
        with patch.object(self.parser.image_io, 'load_bytes', return_value="parsed_image") as mock_load:
            part = {
                "type": "image_url",
                "image_url": {"url": "http://example.com/image.jpg"}
            }
            
            result = parse_content_part(self.parser, part)
            
            self.assertEqual(result["type"], "image")
            self.assertEqual(result["image"], "parsed_image")
            self.assertNotIn("image_url", result)
            mock_get.assert_called_once_with("http://example.com/image.jpg")
            mock_load.assert_called_once_with(b"fake_image_data")

    def test_parse_content_part_image_url_data(self):
        """Test parsing image URL content part with data URL"""
        with patch.object(self.parser.image_io, 'load_base64', return_value="parsed_image") as mock_load:
            part = {
                "type": "image_url", 
                "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD"}
            }
            
            result = parse_content_part(self.parser, part)
            
            self.assertEqual(result["type"], "image")
            self.assertEqual(result["image"], "parsed_image")
            mock_load.assert_called_once_with("image/jpeg", "/9j/4AAQSkZJRgABAQAAAQABAAD")

    def test_parse_content_part_image_url_file(self):
        """Test parsing image URL content part with file URL"""
        with patch.object(self.parser.image_io, 'load_file', return_value="parsed_image") as mock_load:
            part = {
                "type": "image_url",
                "image_url": {"url": "file:///path/to/image.jpg"}
            }
            
            result = parse_content_part(self.parser, part)
            
            self.assertEqual(result["type"], "image")
            self.assertEqual(result["image"], "parsed_image")
            mock_load.assert_called_once_with("/path/to/image.jpg")

    @patch('requests.get')
    def test_parse_content_part_video_url(self, mock_get):
        """Test parsing video URL content part"""
        mock_response = MagicMock()
        mock_response.content = b"fake_video_data"
        mock_get.return_value = mock_response
        
        with patch.object(self.parser.video_io, 'load_bytes', return_value="parsed_video") as mock_load:
            part = {
                "type": "video_url",
                "video_url": {"url": "http://example.com/video.mp4"}
            }
            
            result = parse_content_part(self.parser, part)
            
            self.assertEqual(result["type"], "video")
            self.assertEqual(result["video"], "parsed_video")
            self.assertNotIn("video_url", result)
            mock_get.assert_called_once_with("http://example.com/video.mp4")
            mock_load.assert_called_once_with(b"fake_video_data")

    def test_parse_content_part_unknown_type(self):
        """Test parsing unknown content part type raises ValueError"""
        part = {"type": "unknown_type", "data": "some_data"}
        
        with self.assertRaises(ValueError) as context:
            parse_content_part(self.parser, part)
        
        self.assertIn("Unknown content part type: unknown_type", str(context.exception))

    def test_parse_chat_messages_string_content(self):
        """Test parsing chat messages with string content"""
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

    @patch('requests.get')
    def test_parse_chat_messages_multimodal_content(self, mock_get):
        """Test parsing chat messages with multimodal content"""
        mock_response = MagicMock()
        mock_response.content = b"fake_image_data"
        mock_get.return_value = mock_response
        
        with patch.object(MultiModalPartParser, 'parse_image', return_value="parsed_image"):
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": {"url": "http://example.com/image.jpg"}}
                ]
            }]
            
            result = parse_chat_messages(messages)
            
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["role"], "user")
            self.assertEqual(len(result[0]["content"]), 2)
            self.assertEqual(result[0]["content"][0]["type"], "text")
            self.assertEqual(result[0]["content"][1]["type"], "image")

    def test_multimodal_part_parser_init(self):
        """Test MultiModalPartParser initialization"""
        parser = MultiModalPartParser()
        self.assertIsNotNone(parser.image_io)
        self.assertIsNotNone(parser.video_io)

    @patch('requests.get')
    def test_multimodal_part_parser_parse_image(self, mock_get):
        """Test MultiModalPartParser parse_image method"""
        mock_response = MagicMock()
        mock_response.content = b"fake_image_data"
        mock_get.return_value = mock_response
        
        with patch.object(self.parser.image_io, 'load_bytes', return_value="parsed_image") as mock_load:
            result = self.parser.parse_image("http://example.com/image.jpg")
            self.assertEqual(result, "parsed_image")
            mock_get.assert_called_once_with("http://example.com/image.jpg")
            mock_load.assert_called_once_with(b"fake_image_data")

    @patch('requests.get')
    def test_multimodal_part_parser_parse_video(self, mock_get):
        """Test MultiModalPartParser parse_video method"""
        mock_response = MagicMock()
        mock_response.content = b"fake_video_data"
        mock_get.return_value = mock_response
        
        with patch.object(self.parser.video_io, 'load_bytes', return_value="parsed_video") as mock_load:
            result = self.parser.parse_video("http://example.com/video.mp4")
            self.assertEqual(result, "parsed_video")
            mock_get.assert_called_once_with("http://example.com/video.mp4")
            mock_load.assert_called_once_with(b"fake_video_data")


if __name__ == "__main__":
    unittest.main()