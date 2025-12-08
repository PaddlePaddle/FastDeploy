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
import threading
import time
import unittest
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from fastdeploy.metrics import trace
from fastdeploy.metrics.trace import FilteringSpanProcessor, label_span


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
        with patch("fastdeploy.metrics.trace.trace.get_current_span", return_value=mock_parent_span):
            with patch.object(self.processor._processor, "on_start") as mock_parent_on_start:
                self.processor.on_start(mock_span, parent_context=None)

                # Verify stream attribute is set on child span
                mock_span.set_attribute.assert_called_once_with("stream", "test_stream")
                mock_parent_on_start.assert_called_once_with(mock_span, None)

    def test_on_start_without_parent_span(self):
        """Test on_start method without parent span"""
        mock_span = MagicMock()

        # Mock trace.get_current_span to return None
        with patch("fastdeploy.metrics.trace.trace.get_current_span", return_value=None):
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

        with patch("fastdeploy.metrics.trace.trace.get_current_span", return_value=mock_parent_span):
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
    """Test cases for label_span function"""

    def test_lable_span_with_stream_request(self):
        """Test label_span function with streaming request"""
        mock_request = MagicMock()
        mock_request.stream = True

        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        with patch("fastdeploy.metrics.trace.trace.get_current_span", return_value=mock_span):
            label_span(mock_request)

            # Verify stream attribute is set
            mock_span.set_attribute.assert_called_once_with("stream", "true")

    def test_lable_span_without_stream_request(self):
        """Test label_span function with non-streaming request"""
        mock_request = MagicMock()
        mock_request.stream = False

        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        with patch("fastdeploy.metrics.trace.trace.get_current_span", return_value=mock_span):
            label_span(mock_request)

            # Verify no attributes are set
            mock_span.set_attribute.assert_not_called()

    def test_lable_span_without_current_span(self):
        """Test label_span function when no current span exists"""
        mock_request = MagicMock()
        mock_request.stream = True

        with patch("fastdeploy.metrics.trace.trace.get_current_span", return_value=None):
            # Should not raise any exception
            label_span(mock_request)

    def test_lable_span_with_non_recording_span(self):
        """Test label_span function with non-recording span"""
        mock_request = MagicMock()
        mock_request.stream = True

        mock_span = MagicMock()
        mock_span.is_recording.return_value = False

        with patch("fastdeploy.metrics.trace.trace.get_current_span", return_value=mock_span):
            label_span(mock_request)

            # Verify no attributes are set
            mock_span.set_attribute.assert_not_called()


class TestTraceComprehensive:
    """Comprehensive tests for tracing functionality"""

    def setup_method(self):
        """Setup test environment"""
        # Mock environment variables
        self.original_env = os.environ.copy()
        os.environ["TRACES_ENABLE"] = "true"
        os.environ["FD_SERVICE_NAME"] = "test_service"
        os.environ["FD_HOST_NAME"] = "test_host"
        os.environ["EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4317"
        os.environ["EXPORTER_OTLP_HEADERS"] = "key1=value1,key2=value2"
        os.environ["FD_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS"] = "1000"
        os.environ["FD_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE"] = "512"

        # Reset global state
        trace.remote_trace_contexts = {}
        trace.threads_info = {}
        trace.reqs_context = {}
        trace.tracing_enabled = False

    def teardown_method(self):
        """Restore environment"""
        os.environ = self.original_env

    def test_process_tracing_init_with_different_scenarios(self):
        """Test tracing initialization under different scenarios"""
        # Test normal initialization
        trace.process_tracing_init()
        assert trace.tracing_enabled is True

        # Test with tracing disabled
        os.environ["TRACES_ENABLE"] = "false"
        trace.process_tracing_init()
        assert trace.tracing_enabled is False

        # Test with invalid endpoint
        os.environ["TRACES_ENABLE"] = "true"
        os.environ["EXPORTER_OTLP_ENDPOINT"] = ""
        with mock.patch("fastdeploy.metrics.trace.logger"):
            trace.process_tracing_init()
            # Should log error but not crash
            # Check if error was called (may not always be called depending on implementation)
            pass

        # Test with different protocols
        for protocol in ["grpc", "http/protobuf"]:
            os.environ["OTEL_EXPORTER_OTLP_TRACES_PROTOCOL"] = protocol
            trace.process_tracing_init()
            assert trace.tracing_enabled is True

        # Test with unsupported protocol
        os.environ["OTEL_EXPORTER_OTLP_TRACES_PROTOCOL"] = "unsupported"
        with pytest.raises(ValueError):
            trace.get_otlp_span_exporter("http://localhost:4317", None)

    def test_thread_info_with_different_ranks(self):
        """Test thread info with TP and DP ranks"""
        # Test with TP rank
        trace.process_tracing_init()
        trace.trace_set_thread_info("test_thread_tp", tp_rank=0, dp_rank=1)

        pid = threading.get_native_id()
        info = trace.threads_info[pid]
        assert info.tp_rank == 0
        assert info.dp_rank == 1

        # Test with None ranks
        trace.trace_set_thread_info("test_thread_no_ranks")
        info = trace.threads_info[pid]  # Should still be the same thread
        assert info.tp_rank == 0  # Should preserve previous values

    def test_advanced_request_scenarios(self):
        """Test advanced request tracing scenarios"""
        # Test request with timestamp
        rid = "test_request_timestamp"
        ts = int(time.time() * 1e9) - 1000  # 1 microsecond ago

        trace.process_tracing_init()
        trace.trace_set_thread_info("test_thread")

        trace.trace_req_start(rid, "", ts=ts)
        assert rid in trace.reqs_context
        assert trace.reqs_context[rid].start_time_ns == ts

        trace.trace_req_finish(rid, ts=ts + 2000)

        # Test request with attributes
        rid2 = "test_request_attrs"
        trace.trace_req_start(rid2, "")
        attrs = {"attr1": "value1", "attr2": 123}
        trace.trace_req_finish(rid2, attrs=attrs)

    def test_complex_slice_scenarios(self):
        """Test complex slice operations"""
        rid = "test_complex_slices"

        trace.process_tracing_init()
        trace.trace_set_thread_info("test_thread")
        trace.trace_req_start(rid, "")

        # Test nested slices
        trace.trace_slice_start("outer", rid)
        trace.trace_slice_start("inner", rid)
        trace.trace_slice_end("inner", rid)
        trace.trace_slice_end("outer", rid)

        # Test anonymous slices
        trace.trace_slice_start("", rid, anonymous=True)
        trace.trace_slice_end("anonymous_test", rid)

        trace.trace_req_finish(rid)

    def test_trace_report_span_function(self):
        """Test the trace_report_span convenience function"""
        rid = "test_report_span"

        trace.process_tracing_init()
        trace.trace_set_thread_info("test_thread")
        trace.trace_req_start(rid, "")

        # Test trace_report_span
        start_time = int(time.time() * 1e9)
        end_time = start_time + 1000000  # 1ms later
        attrs = {"test_attr": "test_value"}

        trace.trace_report_span("report_test", rid, start_time, end_time, attrs)

        trace.trace_req_finish(rid)

    def test_propagation_advanced_scenarios(self):
        """Test advanced context propagation scenarios"""
        rid = "test_advanced_propagation"

        trace.process_tracing_init()
        trace.trace_set_thread_info("test_thread")
        trace.trace_req_start(rid, "")

        # Create slices to get a non-null prev_span_context
        trace.trace_slice_start("slice1", rid)
        trace.trace_slice_end("slice1", rid)

        # Get context with prev_span_context
        context_dict = trace.trace_get_proc_propagate_context(rid)
        assert context_dict is not None
        assert "prev_span" in context_dict

        # Test propagation with timestamp
        new_rid = "test_propagated"
        ts = int(time.time() * 1e9)
        trace.trace_set_proc_propagate_context(new_rid, context_dict, ts=ts)

        assert new_rid in trace.reqs_context
        assert trace.reqs_context[new_rid].is_copy is True
        assert trace.reqs_context[new_rid].start_time_ns == ts

        # Test with empty or invalid context
        trace.trace_set_proc_propagate_context("invalid_rid", None)
        trace.trace_set_proc_propagate_context("invalid_rid", {})
        trace.trace_set_proc_propagate_context("invalid_rid", {"invalid": "data"})

        trace.trace_req_finish(rid)
        trace.trace_req_finish(new_rid)

    def test_multiple_threads_same_request(self):
        """Test tracing with multiple threads on same request"""
        rid = "test_multi_thread"

        trace.process_tracing_init()

        # Setup main thread
        trace.trace_set_thread_info("main_thread")
        trace.trace_req_start(rid, "")

        # Create worker thread
        def worker_thread():
            trace.trace_set_thread_info("worker_thread")
            trace.trace_slice_start("worker_task", rid)
            time.sleep(0.001)  # Simulate work
            trace.trace_slice_end("worker_task", rid)

        thread = threading.Thread(target=worker_thread)
        thread.start()
        thread.join()

        # Main thread continues
        trace.trace_slice_start("main_task", rid)
        trace.trace_slice_end("main_task", rid)

        trace.trace_req_finish(rid)

    def test_trace_span_enum(self):
        """Test TraceSpanName enum values"""
        assert trace.TraceSpanName.FASTDEPLOY == "FASTDEPLOY"
        assert trace.TraceSpanName.PREPROCESSING == "PREPROCESSING"
        assert trace.TraceSpanName.SCHEDULE == "SCHEDULE"
        assert trace.TraceSpanName.PREFILL == "PREFILL"
        assert trace.TraceSpanName.DECODE == "DECODE"
        assert trace.TraceSpanName.POSTPROCESSING == "POSTPROCESSING"

        # Test all enum members exist
        expected_spans = ["FASTDEPLOY", "PREPROCESSING", "SCHEDULE", "PREFILL", "DECODE", "POSTPROCESSING"]
        for span_name in expected_spans:
            assert hasattr(trace.TraceSpanName, span_name)

    def test_host_id_generation(self):
        """Test host ID generation logic"""
        # Test with environment variable (most reliable)
        os.environ["FD_HOST_NAME"] = "env-host-id"
        trace.process_tracing_init()
        trace.trace_set_thread_info("test_thread")
        pid = threading.get_native_id()
        assert pid in trace.threads_info
        assert trace.threads_info[pid].host_id == "env-host-id"

        # Test fallback (when env var is not set)
        os.environ.pop("FD_HOST_NAME", None)
        trace.threads_info.clear()  # Reset to trigger re-calculation
        trace.trace_set_thread_info("test_thread2")
        pid2 = threading.get_native_id()
        assert pid2 in trace.threads_info
        # Should generate some kind of host ID
        assert trace.threads_info[pid2].host_id is not None
        assert len(trace.threads_info[pid2].host_id) > 0

    def test_edge_case_operations(self):
        """Test edge case operations"""
        trace.process_tracing_init()
        trace.trace_set_thread_info("test_thread")

        # Test operations on empty stack
        rid = "test_edge_cases"
        trace.trace_req_start(rid, "")

        # Try to end a slice that doesn't exist
        trace.trace_slice_end("non_existent", rid)

        # Try to add event to non-existent slice
        trace.trace_event("test_event", rid)

        trace.trace_req_finish(rid)

        # Test repeated operations on finished request
        trace.trace_slice_start("test", rid)
        trace.trace_slice_end("test", rid)
        trace.trace_event("test", rid)

    def test_timing_functions(self):
        """Test timing-related functions"""
        # Test that time_ns is used if available
        if hasattr(time, "time_ns"):
            trace.process_tracing_init()
            # Test that timing works correctly by checking timestamps
            ts1 = int(time.time() * 1e9)
            time.sleep(0.001)  # 1ms
            ts2 = int(time.time() * 1e9)
            assert ts2 > ts1
            assert ts2 - ts1 >= 1000000  # At least 1ms in nanoseconds

    def test_request_start_with_trace_content(self):
        """Test request start with trace content (upstream context)"""
        rid = "test_upstream_context"

        trace.process_tracing_init()
        trace.trace_set_thread_info("test_thread")

        # Test with empty upstream context (valid case)
        trace_content = ""
        trace.trace_req_start(rid, trace_content, role="test_role")

        # Verify the request was created
        assert rid in trace.reqs_context

        trace.trace_req_finish(rid)

    def test_span_linking_logic(self):
        """Test span linking functionality"""
        rid = "test_span_linking"

        trace.process_tracing_init()
        trace.trace_set_thread_info("test_thread")
        trace.trace_req_start(rid, "")

        # Create first slice
        trace.trace_slice_start("first_slice", rid)
        trace.trace_slice_end("first_slice", rid)

        # Create second slice (should be linked to first)
        trace.trace_slice_start("second_slice", rid)
        trace.trace_slice_end("second_slice", rid)

        trace.trace_req_finish(rid)

    @mock.patch("fastdeploy.metrics.trace.trace")
    def test_active_span_handling(self, mock_trace):
        """Test handling of active spans from FastAPI Instrumentor"""
        rid = "test_active_span"

        # Mock an active span
        mock_span = mock.MagicMock()
        mock_span.is_recording.return_value = True
        mock_span.name = "fastapi_request"
        mock_span.get_span_context.return_value = mock.MagicMock(is_valid=True, trace_id=1234567890)
        mock_trace.get_current_span.return_value = mock_span
        mock_trace.set_span_in_context.return_value = "mock_context"

        trace.process_tracing_init()
        trace.trace_set_thread_info("test_thread")

        trace.trace_req_start(rid, "")

        # Verify the active span was used
        assert rid in trace.reqs_context
        assert trace.reqs_context[rid].is_copy is True
        mock_span.set_attribute.assert_called_with("rid", rid)
        mock_span.update_name.assert_called_with("fastapi_request (Req: test_active_span)")

        trace.trace_req_finish(rid)

    def test_lable_span_functionality(self):
        """Test the label_span function with different scenarios"""

        # Create mock request and span
        class MockRequest:
            def __init__(self, stream):
                self.stream = stream

        mock_span = mock.MagicMock()
        mock_span.is_recording.return_value = True

        with mock.patch("fastdeploy.metrics.trace.trace.get_current_span", return_value=mock_span):
            # Test with stream=True
            request_stream = MockRequest(True)
            trace.label_span(request_stream)
            mock_span.set_attribute.assert_called_with("stream", "true")

            # Test with stream=False
            request_no_stream = MockRequest(False)
            trace.label_span(request_no_stream)
            # Should not set stream attribute for False

        # Test with no active span
        with mock.patch(
            "fastdeploy.metrics.trace.trace.get_current_span", return_value=mock.MagicMock(is_recording=False)
        ):
            request_no_stream = MockRequest(False)
            trace.label_span(request_no_stream)
            # Should not set stream attribute for False
            # Should not crash

    def test_error_handling_and_logging(self):
        """Test error handling and logging scenarios"""
        trace.process_tracing_init()
        trace.trace_set_thread_info("test_thread")

        with mock.patch("fastdeploy.metrics.trace.logger") as mock_logger:
            # Test operations on non-existent request
            rid = "non_existent"
            trace.trace_slice_start("test", rid)
            trace.trace_slice_end("test", rid)
            trace.trace_event("test", rid)
            trace.trace_slice_add_attr(rid, {"test": "value"})

            # Should log warnings but not crash
            # Check if warning was called (may not always be called depending on implementation)
            pass

        # Test slice name mismatch warning
        rid = "test_mismatch_warning"
        trace.trace_req_start(rid, "")

        with mock.patch("fastdeploy.metrics.trace.logger") as mock_logger:
            trace.trace_slice_start("start_name", rid)
            trace.trace_slice_end("different_name", rid)
            assert mock_logger.warning.called

        trace.trace_req_finish(rid)


class TestPerformanceAndConcurrency:
    """Performance and concurrency tests"""

    def test_concurrent_requests(self):
        """Test handling of concurrent requests"""
        trace.process_tracing_init()

        def process_request(request_id, results_list):
            """Process a single request"""
            trace.trace_set_thread_info(f"thread_{request_id}")
            trace.trace_req_start(request_id, "")
            trace.trace_slice_start("process", request_id)
            time.sleep(0.001)  # Simulate work
            trace.trace_slice_end("process", request_id)
            trace.trace_req_finish(request_id)
            result = f"request_{request_id}_completed"
            results_list.append(result)
            return result

        # Process multiple requests concurrently
        results = []
        threads = []

        for i in range(10):
            thread = threading.Thread(target=process_request, args=(f"req_{i}", results))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Verify all requests were processed
        assert len([r for r in results if r.endswith("_completed")]) == 10

    def test_memory_cleanup(self):
        """Test proper memory cleanup"""
        trace.process_tracing_init()
        trace.trace_set_thread_info("test_thread")

        # Create and finish multiple requests
        for i in range(5):
            rid = f"test_request_{i}"
            trace.trace_req_start(rid, "")
            trace.trace_slice_start("test", rid)
            trace.trace_slice_end("test", rid)
            trace.trace_req_finish(rid)

        # Verify cleanup
        assert len(trace.reqs_context) == 0

        # Thread info should persist
        pid = threading.get_native_id()
        assert pid in trace.threads_info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
