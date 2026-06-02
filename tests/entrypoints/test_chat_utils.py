"""
# Copyright (c) 2026  PaddlePaddle Authors. All Rights Reserved.
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
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastdeploy.entrypoints.chat_utils import (
    MultimodalPartParser,
    load_chat_template,
    parse_chat_messages,
    parse_content_part,
    random_tool_call_id,
)


class TestMultimodalPartParserInit(unittest.TestCase):
    """Test MultimodalPartParser.__init__."""

    @patch("fastdeploy.entrypoints.chat_utils.VideoMediaIO")
    @patch("fastdeploy.entrypoints.chat_utils.ImageMediaIO")
    def test_init_creates_media_ios(self, mock_image_io_cls, mock_video_io_cls):
        """__init__ creates ImageMediaIO and VideoMediaIO instances."""
        parser = MultimodalPartParser()
        mock_image_io_cls.assert_called_once()
        mock_video_io_cls.assert_called_once()
        self.assertIs(parser.image_io, mock_image_io_cls.return_value)
        self.assertIs(parser.video_io, mock_video_io_cls.return_value)


class TestMultimodalPartParserParseImage(unittest.TestCase):
    """Test MultimodalPartParser.parse_image."""

    @patch("fastdeploy.entrypoints.chat_utils.VideoMediaIO")
    @patch("fastdeploy.entrypoints.chat_utils.ImageMediaIO")
    def test_parse_image_calls_load_from_url(self, mock_image_io_cls, mock_video_io_cls):
        """parse_image delegates to load_from_url with image_io."""
        parser = MultimodalPartParser()
        with patch.object(parser, "load_from_url", return_value="parsed_image") as mock_load:
            result = parser.parse_image("http://example.com/img.png")
            mock_load.assert_called_once_with("http://example.com/img.png", parser.image_io)
            self.assertEqual(result, "parsed_image")


class TestMultimodalPartParserParseVideo(unittest.TestCase):
    """Test MultimodalPartParser.parse_video."""

    @patch("fastdeploy.entrypoints.chat_utils.VideoMediaIO")
    @patch("fastdeploy.entrypoints.chat_utils.ImageMediaIO")
    def test_parse_video_calls_load_from_url(self, mock_image_io_cls, mock_video_io_cls):
        """parse_video delegates to load_from_url with video_io."""
        parser = MultimodalPartParser()
        with patch.object(parser, "load_from_url", return_value="parsed_video") as mock_load:
            result = parser.parse_video("http://example.com/vid.mp4")
            mock_load.assert_called_once_with("http://example.com/vid.mp4", parser.video_io)
            self.assertEqual(result, "parsed_video")


class TestMultimodalPartParserHttpGetWithRetry(unittest.TestCase):
    """Test MultimodalPartParser.http_get_with_retry."""

    @patch("fastdeploy.entrypoints.chat_utils.VideoMediaIO")
    @patch("fastdeploy.entrypoints.chat_utils.ImageMediaIO")
    def setUp(self, mock_image_io_cls, mock_video_io_cls):
        self.parser = MultimodalPartParser()

    @patch("fastdeploy.entrypoints.chat_utils.time.sleep")
    @patch("fastdeploy.entrypoints.chat_utils.requests.get")
    def test_success_first_try(self, mock_get, mock_sleep):
        """Returns content on first successful request."""
        mock_response = MagicMock()
        mock_response.content = b"image_data"
        mock_get.return_value = mock_response

        result = self.parser.http_get_with_retry("http://example.com/img.png")

        self.assertEqual(result, b"image_data")
        mock_get.assert_called_once_with("http://example.com/img.png")
        mock_response.raise_for_status.assert_called_once()
        mock_sleep.assert_not_called()

    @patch("fastdeploy.entrypoints.chat_utils.time.sleep")
    @patch("fastdeploy.entrypoints.chat_utils.requests.get")
    def test_retry_then_success(self, mock_get, mock_sleep):
        """Retries on failure and returns content on subsequent success."""
        mock_fail_response = MagicMock()
        mock_fail_response.raise_for_status.side_effect = Exception("500 error")
        mock_get.side_effect = [Exception("connection error"), mock_fail_response]

        mock_success_response = MagicMock()
        mock_success_response.content = b"data"
        mock_get.side_effect = [Exception("connection error"), mock_success_response]

        result = self.parser.http_get_with_retry("http://example.com/img.png", max_retries=3, retry_delay=1)

        self.assertEqual(result, b"data")
        self.assertEqual(mock_get.call_count, 2)
        mock_sleep.assert_called_once_with(1)

    @patch("fastdeploy.entrypoints.chat_utils.time.sleep")
    @patch("fastdeploy.entrypoints.chat_utils.requests.get")
    def test_all_retries_exhausted_raises(self, mock_get, mock_sleep):
        """Raises exception after all retries exhausted."""
        mock_get.side_effect = Exception("connection error")

        with self.assertRaises(Exception) as ctx:
            self.parser.http_get_with_retry(
                "http://example.com/img.png", max_retries=3, retry_delay=1, backoff_factor=2
            )

        self.assertIn("connection error", str(ctx.exception))
        self.assertEqual(mock_get.call_count, 3)
        # Sleep called with backoff: 1, 2
        self.assertEqual(mock_sleep.call_count, 2)
        mock_sleep.assert_any_call(1)
        mock_sleep.assert_any_call(2)

    @patch("fastdeploy.entrypoints.chat_utils.time.sleep")
    @patch("fastdeploy.entrypoints.chat_utils.requests.get")
    def test_raise_for_status_failure_triggers_retry(self, mock_get, mock_sleep):
        """raise_for_status() failure triggers retry."""
        mock_response_fail = MagicMock()
        mock_response_fail.raise_for_status.side_effect = Exception("404")

        mock_response_ok = MagicMock()
        mock_response_ok.content = b"ok"

        mock_get.side_effect = [mock_response_fail, mock_response_ok]

        result = self.parser.http_get_with_retry("http://example.com/img.png", max_retries=3, retry_delay=2)

        self.assertEqual(result, b"ok")
        self.assertEqual(mock_get.call_count, 2)
        mock_sleep.assert_called_once_with(2)


class TestMultimodalPartParserLoadFromUrl(unittest.TestCase):
    """Test MultimodalPartParser.load_from_url."""

    @patch("fastdeploy.entrypoints.chat_utils.VideoMediaIO")
    @patch("fastdeploy.entrypoints.chat_utils.ImageMediaIO")
    def setUp(self, mock_image_io_cls, mock_video_io_cls):
        self.parser = MultimodalPartParser()
        self.mock_media_io = MagicMock()

    def test_http_url_calls_http_get_and_load_bytes(self):
        """HTTP URL fetches bytes and calls media_io.load_bytes."""
        with patch.object(self.parser, "http_get_with_retry", return_value=b"img_bytes") as mock_http:
            self.mock_media_io.load_bytes.return_value = "loaded_image"
            result = self.parser.load_from_url("http://example.com/img.png", self.mock_media_io)

            mock_http.assert_called_once_with("http://example.com/img.png")
            self.mock_media_io.load_bytes.assert_called_once_with(b"img_bytes")
            self.assertEqual(result, "loaded_image")

    def test_https_url_calls_http_get_and_load_bytes(self):
        """HTTPS URL fetches bytes and calls media_io.load_bytes."""
        with patch.object(self.parser, "http_get_with_retry", return_value=b"data") as mock_http:
            self.mock_media_io.load_bytes.return_value = "loaded"
            result = self.parser.load_from_url("https://example.com/img.png", self.mock_media_io)

            mock_http.assert_called_once_with("https://example.com/img.png")
            self.mock_media_io.load_bytes.assert_called_once_with(b"data")
            self.assertEqual(result, "loaded")

    def test_data_url_calls_load_base64(self):
        """data: URL extracts media type and base64 data."""
        url = "data:image/png;base64,iVBORw0KGgo="
        self.mock_media_io.load_base64.return_value = "base64_image"

        result = self.parser.load_from_url(url, self.mock_media_io)

        self.mock_media_io.load_base64.assert_called_once_with("image/png", "iVBORw0KGgo=")
        self.assertEqual(result, "base64_image")

    def test_file_url_calls_load_file(self):
        """file: URL calls media_io.load_file with path."""
        url = "file:///tmp/image.png"
        self.mock_media_io.load_file.return_value = "file_image"

        result = self.parser.load_from_url(url, self.mock_media_io)

        self.mock_media_io.load_file.assert_called_once_with("/tmp/image.png")
        self.assertEqual(result, "file_image")

    def test_unknown_scheme_returns_none(self):
        """Unknown URL scheme returns None."""
        result = self.parser.load_from_url("ftp://example.com/img.png", self.mock_media_io)
        self.assertIsNone(result)


class TestParseContentPart(unittest.TestCase):
    """Test parse_content_part function."""

    def setUp(self):
        self.mm_parser = MagicMock()

    def test_text_part_returned_as_is(self):
        """Text part is returned unchanged."""
        part = {"type": "text", "text": "hello"}
        result = parse_content_part(self.mm_parser, part)
        self.assertEqual(result, part)

    def test_image_url_with_url(self):
        """image_url part with URL calls parse_image."""
        self.mm_parser.parse_image.return_value = "parsed_img"
        part = {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}}

        result = parse_content_part(self.mm_parser, part)

        self.mm_parser.parse_image.assert_called_once_with("http://example.com/img.png")
        self.assertEqual(result["type"], "image")
        self.assertEqual(result["data"], "parsed_img")
        self.assertIsNone(result["uuid"])

    def test_image_url_with_uuid_only(self):
        """image_url part with uuid only sets data to None."""
        part = {"type": "image_url", "uuid": "abc-123"}

        result = parse_content_part(self.mm_parser, part)

        self.mm_parser.parse_image.assert_not_called()
        self.assertEqual(result["type"], "image")
        self.assertIsNone(result["data"])
        self.assertEqual(result["uuid"], "abc-123")

    def test_image_url_missing_both_raises(self):
        """image_url part missing both image_url and uuid raises ValueError."""
        part = {"type": "image_url"}

        with self.assertRaises(ValueError) as ctx:
            parse_content_part(self.mm_parser, part)
        self.assertIn("Both image_url and uuid are missing", str(ctx.exception))

    def test_video_url_with_url(self):
        """video_url part with URL calls parse_video."""
        self.mm_parser.parse_video.return_value = "parsed_vid"
        part = {"type": "video_url", "video_url": {"url": "http://example.com/vid.mp4"}}

        result = parse_content_part(self.mm_parser, part)

        self.mm_parser.parse_video.assert_called_once_with("http://example.com/vid.mp4")
        self.assertEqual(result["type"], "video")
        self.assertEqual(result["data"], "parsed_vid")
        self.assertIsNone(result["uuid"])

    def test_video_url_with_uuid_only(self):
        """video_url part with uuid only sets data to None."""
        part = {"type": "video_url", "uuid": "vid-456"}

        result = parse_content_part(self.mm_parser, part)

        self.mm_parser.parse_video.assert_not_called()
        self.assertEqual(result["type"], "video")
        self.assertIsNone(result["data"])
        self.assertEqual(result["uuid"], "vid-456")

    def test_video_url_missing_both_raises(self):
        """video_url part missing both video_url and uuid raises ValueError."""
        part = {"type": "video_url"}

        with self.assertRaises(ValueError) as ctx:
            parse_content_part(self.mm_parser, part)
        self.assertIn("Both video_url and uuid are missing", str(ctx.exception))

    def test_unknown_type_raises(self):
        """Unknown part type raises ValueError."""
        part = {"type": "audio_url"}

        with self.assertRaises(ValueError) as ctx:
            parse_content_part(self.mm_parser, part)
        self.assertIn("Unknown content part type: audio_url", str(ctx.exception))

    def test_none_type_raises(self):
        """Missing type key raises ValueError."""
        part = {"text": "hello"}

        with self.assertRaises(ValueError) as ctx:
            parse_content_part(self.mm_parser, part)
        self.assertIn("Unknown content part type: None", str(ctx.exception))

    def test_image_url_with_url_and_uuid(self):
        """image_url part with both URL and uuid parses image and returns uuid."""
        self.mm_parser.parse_image.return_value = "img_data"
        part = {"type": "image_url", "image_url": {"url": "http://img.png"}, "uuid": "u1"}

        result = parse_content_part(self.mm_parser, part)

        self.assertEqual(result["data"], "img_data")
        self.assertEqual(result["uuid"], "u1")


class TestParseChatMessages(unittest.TestCase):
    """Test parse_chat_messages function."""

    @patch("fastdeploy.entrypoints.chat_utils.MultimodalPartParser")
    def test_string_content(self, mock_parser_cls):
        """String content is wrapped in text part."""
        messages = [{"role": "user", "content": "Hello"}]
        result = parse_chat_messages(messages)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["role"], "user")
        self.assertEqual(result[0]["content"], [{"type": "text", "text": "Hello"}])

    @patch("fastdeploy.entrypoints.chat_utils.MultimodalPartParser")
    def test_none_content(self, mock_parser_cls):
        """None content results in empty list."""
        messages = [{"role": "assistant", "content": None}]
        result = parse_chat_messages(messages)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["role"], "assistant")
        self.assertEqual(result[0]["content"], [])

    @patch("fastdeploy.entrypoints.chat_utils.parse_content_part")
    @patch("fastdeploy.entrypoints.chat_utils.MultimodalPartParser")
    def test_list_content_calls_parse_content_part(self, mock_parser_cls, mock_parse_part):
        """List content calls parse_content_part for each part."""
        mock_parse_part.side_effect = lambda parser, part: {"type": "text", "text": part["text"]}
        messages = [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]

        result = parse_chat_messages(messages)

        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]["content"]), 2)
        self.assertEqual(mock_parse_part.call_count, 2)

    @patch("fastdeploy.entrypoints.chat_utils.MultimodalPartParser")
    def test_multiple_messages(self, mock_parser_cls):
        """Multiple messages are all parsed."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]

        result = parse_chat_messages(messages)

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["role"], "system")
        self.assertEqual(result[1]["role"], "user")
        self.assertEqual(result[2]["role"], "assistant")


class TestLoadChatTemplate(unittest.TestCase):
    """Test load_chat_template function."""

    def test_none_template_no_model_path_returns_none(self):
        """None template with no model_path returns None."""
        result = load_chat_template(None)
        self.assertIsNone(result)

    def test_none_template_model_path_with_jinja_file(self):
        """None template with model_path loads chat_template.jinja."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jinja_path = os.path.join(tmpdir, "chat_template.jinja")
            with open(jinja_path, "w") as f:
                f.write("{{ message }}")

            result = load_chat_template(None, model_path=tmpdir)
            self.assertEqual(result, "{{ message }}")

    def test_none_template_model_path_without_jinja_file(self):
        """None template with model_path but no jinja file returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = load_chat_template(None, model_path=tmpdir)
            self.assertIsNone(result)

    def test_is_literal_returns_string(self):
        """is_literal=True returns the template string directly."""
        result = load_chat_template("{{ content }}", is_literal=True)
        self.assertEqual(result, "{{ content }}")

    def test_is_literal_with_path_raises_type_error(self):
        """is_literal=True with Path raises TypeError."""
        with self.assertRaises(TypeError) as ctx:
            load_chat_template(Path("/some/path.jinja"), is_literal=True)
        self.assertIn("expected to be read directly", str(ctx.exception))

    def test_file_path_reads_template(self):
        """String file path reads and returns template content."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jinja", delete=False) as f:
            f.write("{% for m in messages %}{{ m }}{% endfor %}")
            f.flush()
            tmppath = f.name

        try:
            result = load_chat_template(tmppath)
            self.assertEqual(result, "{% for m in messages %}{{ m }}{% endfor %}")
        finally:
            os.unlink(tmppath)

    def test_path_object_reads_template(self):
        """Path object reads and returns template content."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jinja", delete=False) as f:
            f.write("template_content")
            f.flush()
            tmppath = f.name

        try:
            result = load_chat_template(Path(tmppath))
            self.assertEqual(result, "template_content")
        finally:
            os.unlink(tmppath)

    def test_nonexistent_path_object_raises(self):
        """Non-existent Path object raises OSError."""
        with self.assertRaises(OSError):
            load_chat_template(Path("/nonexistent/path/template.jinja"))

    def test_nonexistent_string_without_jinja_chars_raises_value_error(self):
        """Non-existent string path without jinja chars raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            load_chat_template("/nonexistent/path/template.jinja")
        self.assertIn("looks like a file path", str(ctx.exception))

    def test_nonexistent_string_with_jinja_chars_returns_literal(self):
        """Non-existent string with jinja chars is treated as literal template."""
        template = "{{ message.content }}"
        result = load_chat_template(template)
        self.assertEqual(result, template)


class TestRandomToolCallId(unittest.TestCase):
    """Test random_tool_call_id function."""

    def test_returns_string_with_prefix(self):
        """Returns string with chatcmpl-tool- prefix."""
        result = random_tool_call_id()
        self.assertTrue(result.startswith("chatcmpl-tool-"))

    def test_returns_unique_ids(self):
        """Returns unique IDs on each call."""
        ids = {random_tool_call_id() for _ in range(100)}
        self.assertEqual(len(ids), 100)

    def test_id_has_expected_length(self):
        """ID has expected format: prefix + 32 hex chars."""
        result = random_tool_call_id()
        # "chatcmpl-tool-" is 14 chars, uuid hex is 32 chars
        self.assertEqual(len(result), 14 + 32)


if __name__ == "__main__":
    unittest.main()
