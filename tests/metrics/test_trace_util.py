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

import unittest
from unittest.mock import MagicMock, patch

from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from fastdeploy.metrics.trace_util import FilteringSpanProcessor, lable_span


class TestFilteringSpanProcessor(unittest.TestCase):
    """Test cases for FilteringSpanProcessor class"""

    def setUp(self):
        """Set up test fixtures"""
        self.exporter = ConsoleSpanExporter()
        self.processor = FilteringSpanProcessor(self.exporter)

    def test_initialization(self):
        """Test that FilteringSpanProcessor is properly initialized"""
        self.assertIsInstance(self.processor._processor, BatchSpanProcessor)
        self.assertEqual(self.processor._processor.span_exporter, self.exporter)

    def test_on_start_with_parent_span(self):
        """Test on_start method with parent span containing stream attribute"""
        # Mock span and parent context
        mock_span = MagicMock()
        mock_parent_span = MagicMock()
        mock_parent_span.is_recording.return_value = True
        mock_parent_span.attributes.get.return_value = "test_stream"

        # Mock trace.get_current_span to return parent span
        with patch("fastdeploy.metrics.trace_util.trace.get_current_span", return_value=mock_parent_span):
            with patch.object(self.processor._processor, "on_start") as mock_parent_on_start:
                self.processor.on_start(mock_span, parent_context=None)

                # Verify stream attribute is set on child span
                mock_span.set_attribute.assert_called_once_with("stream", "test_stream")
                mock_parent_on_start.assert_called_once_with(mock_span, None)

    def test_on_start_without_parent_span(self):
        """Test on_start method without parent span"""
        mock_span = MagicMock()

        # Mock trace.get_current_span to return None
        with patch("fastdeploy.metrics.trace_util.trace.get_current_span", return_value=None):
            with patch.object(self.processor._processor, "on_start") as mock_parent_on_start:
                self.processor.on_start(mock_span, parent_context=None)

                # Verify no attributes are set
                mock_span.set_attribute.assert_not_called()
                mock_parent_on_start.assert_called_once_with(mock_span, None)

    def test_on_start_with_non_recording_parent_span(self):
        """Test on_start method with non-recording parent span"""
        mock_span = MagicMock()
        mock_parent_span = MagicMock()
        mock_parent_span.is_recording.return_value = False

        with patch("fastdeploy.metrics.trace_util.trace.get_current_span", return_value=mock_parent_span):
            with patch.object(self.processor._processor, "on_start") as mock_parent_on_start:
                self.processor.on_start(mock_span, parent_context=None)

                # Verify no attributes are set
                mock_span.set_attribute.assert_not_called()
                mock_parent_on_start.assert_called_once_with(mock_span, None)

    def test_on_end_filter_stream_http_response(self):
        """Test on_end method filters out stream http response spans"""
        mock_span = MagicMock()
        mock_span.attributes.get.side_effect = lambda key: {
            "asgi.event.type": "http.response.body",
            "stream": "true",
        }.get(key)
        mock_span.name = "http send request"

        with patch.object(self.processor._processor, "on_end") as mock_parent_on_end:
            self.processor.on_end(mock_span)

            # Verify parent on_end is NOT called (span is filtered out)
            mock_parent_on_end.assert_not_called()

    def test_on_end_keep_non_stream_spans(self):
        """Test on_end method keeps non-stream spans"""
        mock_span = MagicMock()
        mock_span.attributes.get.side_effect = lambda key: {"asgi.event.type": "http.request", "stream": None}.get(key)
        mock_span.name = "http receive request"

        with patch.object(self.processor._processor, "on_end") as mock_parent_on_end:
            self.processor.on_end(mock_span)

            # Verify parent on_end is called
            mock_parent_on_end.assert_called_once_with(mock_span)

    def test_on_end_keep_spans_without_http_send(self):
        """Test on_end method keeps spans without 'http send' in name"""
        mock_span = MagicMock()
        mock_span.attributes.get.side_effect = lambda key: {
            "asgi.event.type": "http.response.body",
            "stream": "true",
        }.get(key)
        mock_span.name = "other operation"

        with patch.object(self.processor._processor, "on_end") as mock_parent_on_end:
            self.processor.on_end(mock_span)

            # Verify parent on_end is called
            mock_parent_on_end.assert_called_once_with(mock_span)

    def test_shutdown(self):
        """Test shutdown method"""
        with patch.object(self.processor._processor, "shutdown") as mock_shutdown:
            self.processor.shutdown()
            mock_shutdown.assert_called_once()

    def test_force_flush(self):
        """Test force_flush method"""
        with patch.object(self.processor._processor, "force_flush") as mock_force_flush:
            self.processor.force_flush(timeout_millis=5000)
            mock_force_flush.assert_called_once_with(5000)


class TestLableSpan(unittest.TestCase):
    """Test cases for lable_span function"""

    def test_lable_span_with_stream_request(self):
        """Test lable_span function with streaming request"""
        mock_request = MagicMock()
        mock_request.stream = True

        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        with patch("fastdeploy.metrics.trace_util.trace.get_current_span", return_value=mock_span):
            lable_span(mock_request)

            # Verify stream attribute is set
            mock_span.set_attribute.assert_called_once_with("stream", "true")

    def test_lable_span_without_stream_request(self):
        """Test lable_span function with non-streaming request"""
        mock_request = MagicMock()
        mock_request.stream = False

        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        with patch("fastdeploy.metrics.trace_util.trace.get_current_span", return_value=mock_span):
            lable_span(mock_request)

            # Verify no attributes are set
            mock_span.set_attribute.assert_not_called()

    def test_lable_span_without_current_span(self):
        """Test lable_span function when no current span exists"""
        mock_request = MagicMock()
        mock_request.stream = True

        with patch("fastdeploy.metrics.trace_util.trace.get_current_span", return_value=None):
            # Should not raise any exception
            lable_span(mock_request)

    def test_lable_span_with_non_recording_span(self):
        """Test lable_span function with non-recording span"""
        mock_request = MagicMock()
        mock_request.stream = True

        mock_span = MagicMock()
        mock_span.is_recording.return_value = False

        with patch("fastdeploy.metrics.trace_util.trace.get_current_span", return_value=mock_span):
            lable_span(mock_request)

            # Verify no attributes are set
            mock_span.set_attribute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
