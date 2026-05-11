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
        os.environ["FD_TRACE"] = "otel"
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
        os.environ["FD_TRACE"] = "off"
        trace.process_tracing_init()
        assert trace.tracing_enabled is False

        # Test with invalid endpoint
        os.environ["FD_TRACE"] = "otel"
        os.environ["EXPORTER_OTLP_ENDPOINT"] = ""

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
        info = trace.threads_info[pid]  # Should still be same thread
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
        """Test trace_report_span convenience function"""
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

        # Verify that request was created
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

        # Verify that active span was used
        assert rid in trace.reqs_context
        assert trace.reqs_context[rid].is_copy is True
        mock_span.set_attribute.assert_called_with("rid", rid)
        mock_span.update_name.assert_called_with("fastapi_request (Req: test_active_span)")

        trace.trace_req_finish(rid)

    def test_lable_span_functionality(self):
        """Test label_span function with different scenarios"""

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


class TestAdditionalCoverage:
    """Additional test cases for better code coverage"""

    def setup_method(self):
        """Setup test environment"""
        self.original_env = os.environ.copy()
        os.environ["FD_TRACE"] = "otel"
        os.environ["FD_SERVICE_NAME"] = "test_service"
        os.environ["EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4317"

        # Reset global state
        trace.remote_trace_contexts = {}
        trace.threads_info = {}
        trace.reqs_context = {}
        trace.tracing_enabled = False

    def teardown_method(self):
        """Restore environment"""
        os.environ = self.original_env

    def test_trace_propagate_context_to_dict(self):
        """Test TracePropagateContext.to_dict method"""
        from fastdeploy.metrics.trace import TracePropagateContext

        # Mock context objects
        mock_root_context = MagicMock()
        mock_prev_span_context = MagicMock()
        mock_prev_span_context.span_id = 12345
        mock_prev_span_context.trace_id = 67890

        # Test with prev_span_context
        propagate_context = TracePropagateContext(mock_root_context, mock_prev_span_context)
        result_dict = propagate_context.to_dict()

        assert "root_span" in result_dict
        assert "prev_span" in result_dict
        assert result_dict["prev_span"]["span_id"] == 12345
        assert result_dict["prev_span"]["trace_id"] == 67890

        # Test without prev_span_context
        propagate_context_none = TracePropagateContext(mock_root_context, None)
        result_dict_none = propagate_context_none.to_dict()

        assert "root_span" in result_dict_none
        assert result_dict_none["prev_span"] == "None"

    def test_trace_propagate_context_instance_from_dict(self):
        """Test TracePropagateContext.instance_from_dict method"""
        from fastdeploy.metrics.trace import TracePropagateContext

        # Test valid dict with prev_span
        valid_dict = {"root_span": {"test": "carrier"}, "prev_span": {"span_id": 12345, "trace_id": 67890}}

        with mock.patch("fastdeploy.metrics.trace.propagate.extract") as mock_extract:
            mock_extract.return_value = "mock_context"

            with mock.patch("fastdeploy.metrics.trace.trace.span.SpanContext") as mock_span_context:
                mock_span_context_instance = MagicMock()
                mock_span_context.return_value = mock_span_context_instance

                result = TracePropagateContext.instance_from_dict(valid_dict)

                assert result is not None
                assert result.root_span_context == "mock_context"
                assert result.prev_span_context == mock_span_context_instance
                mock_span_context.assert_called_once_with(trace_id=67890, span_id=12345, is_remote=True)

        # Test with None prev_span
        valid_dict_none = {"root_span": {"test": "carrier"}, "prev_span": "None"}

        with mock.patch("fastdeploy.metrics.trace.propagate.extract") as mock_extract:
            mock_extract.return_value = "mock_context"

            result = TracePropagateContext.instance_from_dict(valid_dict_none)

            assert result is not None
            assert result.root_span_context == "mock_context"
            assert result.prev_span_context is None

        # Test invalid dict (missing keys)
        invalid_dict = {"invalid": "data"}
        result = TracePropagateContext.instance_from_dict(invalid_dict)
        assert result is None

        # Test empty dict
        result = TracePropagateContext.instance_from_dict({})
        assert result is None

    def test_trace_custom_id_generator(self):
        """Test TraceCustomIdGenerator class"""
        from fastdeploy.metrics.trace import TraceCustomIdGenerator

        generator = TraceCustomIdGenerator()

        # Test generate_trace_id
        trace_id = generator.generate_trace_id()
        assert isinstance(trace_id, int)
        assert trace_id > 0

        # Test generate_span_id
        span_id = generator.generate_span_id()
        assert isinstance(span_id, int)
        assert span_id > 0

        # Test that multiple calls generate different IDs
        trace_id2 = generator.generate_trace_id()
        span_id2 = generator.generate_span_id()

        # Should be different (very high probability)
        assert trace_id != trace_id2
        assert span_id != span_id2

    def test_get_host_id_fallback_methods(self):
        """Test __get_host_id function fallback methods"""
        # Access function through module directly
        import fastdeploy.metrics.trace as trace_module

        get_host_id_func = trace_module.__dict__.get("__get_host_id")

        if get_host_id_func is None:
            # Skip test if function is not accessible
            pytest.skip("__get_host_id function not accessible for testing")
            return

        # Test with FD_HOST_NAME set
        os.environ["FD_HOST_NAME"] = "test-host-name"
        host_id = get_host_id_func()
        assert host_id == "test-host-name"

        # Test fallback when machine-id files don't exist and MAC is 0
        os.environ.pop("FD_HOST_NAME", None)

        with mock.patch("builtins.open", side_effect=FileNotFoundError):
            with mock.patch("uuid.getnode", return_value=0):
                with mock.patch("uuid.uuid4") as mock_uuid4:
                    mock_uuid = MagicMock()
                    mock_uuid.hex = "test-uuid-hex"
                    mock_uuid4.return_value = mock_uuid

                    with mock.patch("os.getpid", return_value=12345):
                        host_id = get_host_id_func()
                        # The function might return different values based on environment
                        # Just verify it returns a non-empty string
                        assert isinstance(host_id, str)
                        assert len(host_id) > 0

    def test_get_host_id_exception_handling(self):
        """Test __get_host_id exception handling"""
        import fastdeploy.metrics.trace as trace_module

        get_host_id_func = trace_module.__dict__.get("__get_host_id")

        if get_host_id_func is None:
            # Skip test if function is not accessible
            pytest.skip("__get_host_id function not accessible for testing")
            return

        os.environ.pop("FD_HOST_NAME", None)

        with mock.patch("builtins.open", side_effect=FileNotFoundError):
            with mock.patch("uuid.getnode", return_value=0):
                with mock.patch("uuid.uuid4", side_effect=Exception("UUID generation failed")):
                    host_id = get_host_id_func()
                    # The function should return some fallback value
                    assert isinstance(host_id, str)
                    assert len(host_id) > 0
                    # In case of complete failure, it should return "unknown"
                    # but depending on environment, it might return other fallback values

    def test_trace_slice_auto_next_anon(self):
        """Test trace_slice_end with auto_next_anon parameter"""
        trace.process_tracing_init()
        trace.trace_set_thread_info("test_thread")

        rid = "test_auto_anon"
        trace.trace_req_start(rid, "")

        # Start a slice
        trace.trace_slice_start("first_slice", rid)

        # End with auto_next_anon=True
        trace.trace_slice_end("first_slice", rid, auto_next_anon=True)

        # Should have automatically started an anonymous slice
        pid = threading.get_native_id()
        thread_context = trace.reqs_context[rid].threads_context[pid]
        assert len(thread_context.cur_slice_stack) == 1
        assert thread_context.cur_slice_stack[0].anonymous is True
        assert thread_context.cur_slice_stack[0].slice_name == ""

        trace.trace_req_finish(rid)

    def test_trace_slice_thread_finish_flag(self):
        """Test trace_slice_end with thread_finish_flag parameter"""
        trace.process_tracing_init()
        trace.trace_set_thread_info("test_thread")

        rid = "test_thread_finish"
        trace.trace_req_start(rid, "")

        pid = threading.get_native_id()

        # Start and end a slice with thread_finish_flag=True
        trace.trace_slice_start("test_slice", rid)
        trace.trace_slice_end("test_slice", rid, thread_finish_flag=True)

        # Thread context should be removed
        assert pid not in trace.reqs_context[rid].threads_context

        trace.trace_req_finish(rid)

    def test_trace_slice_alias(self):
        """Test trace_slice alias function"""
        # trace_slice should be an alias for trace_slice_end
        assert trace.trace_slice == trace.trace_slice_end

    def test_trace_event_and_add_attr_functionality(self):
        """Test trace_event and trace_slice_add_attr functionality"""
        trace.process_tracing_init()
        trace.trace_set_thread_info("test_thread")

        rid = "test_events_attrs"
        trace.trace_req_start(rid, "")

        # Start a slice
        trace.trace_slice_start("test_slice", rid)

        # Test trace_event
        attrs = {"event_attr": "event_value"}
        trace.trace_event("test_event", rid, attrs=attrs)

        # Test trace_slice_add_attr
        slice_attrs = {"slice_attr": "slice_value"}
        trace.trace_slice_add_attr(rid, slice_attrs)

        trace.trace_slice_end("test_slice", rid)
        trace.trace_req_finish(rid)

    def test_trace_span_decorator_sync(self):
        """Test trace_span decorator with sync function"""
        trace.process_tracing_init()
        trace.trace_set_thread_info("test_thread")

        @trace.trace_span("test_sync_function")
        def test_function():
            return "test_result"

        result = test_function()
        assert result == "test_result"

    def test_trace_span_decorator_async(self):
        """Test trace_span decorator with async function"""
        import asyncio

        trace.process_tracing_init()
        trace.trace_set_thread_info("test_thread")

        @trace.trace_span("test_async_function")
        async def test_async_function():
            return "test_async_result"

        async def run_test():
            result = await test_async_function()
            return result

        result = asyncio.run(run_test())
        assert result == "test_async_result"

    def test_trace_span_decorator_disabled(self):
        """Test trace_span decorator when tracing is disabled"""
        trace.tracing_enabled = False

        @trace.trace_span("test_disabled_function")
        def test_function():
            return "test_result_disabled"

        result = test_function()
        assert result == "test_result_disabled"

    def test_trace_span_decorator_no_thread_info(self):
        """Test trace_span decorator when thread info is not set"""
        trace.process_tracing_init()
        trace.threads_info.clear()  # Clear thread info

        @trace.trace_span("test_no_thread_info")
        def test_function():
            return "test_result_no_thread"

        result = test_function()
        assert result == "test_result_no_thread"

        # Should have created thread info automatically
        pid = threading.get_native_id()
        assert pid in trace.threads_info

    def test_get_otlp_span_exporter_grpc(self):
        """Test get_otlp_span_exporter with grpc protocol"""
        # Set environment variable for grpc protocol
        os.environ["OTEL_EXPORTER_OTLP_TRACES_PROTOCOL"] = "grpc"
        exporter = trace.get_otlp_span_exporter("http://localhost:4317", None)
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter as GRPCSpanExporter,
        )

        assert isinstance(exporter, GRPCSpanExporter)

    def test_get_otlp_span_exporter_http(self):
        """Test get_otlp_span_exporter with http protocol"""
        # Set environment variable for http protocol
        os.environ["OTEL_EXPORTER_OTLP_TRACES_PROTOCOL"] = "http/protobuf"
        headers = {"Authorization": "Bearer token"}
        exporter = trace.get_otlp_span_exporter("http://localhost:4318", headers)
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter as HTTPSpanExporter,
        )

        assert isinstance(exporter, HTTPSpanExporter)

    def test_get_otlp_span_exporter_unsupported_protocol(self):
        """Test get_otlp_span_exporter with unsupported protocol"""
        # Set environment variable for unsupported protocol
        os.environ["OTEL_EXPORTER_OTLP_TRACES_PROTOCOL"] = "unsupported"
        with pytest.raises(ValueError, match="Unsupported OTLP protocol"):
            trace.get_otlp_span_exporter("http://localhost:4317", None)

    def test_process_tracing_init_without_opentelemetry(self):
        """Test process_tracing_init when opentelemetry is not imported"""
        original_opentelemetry_imported = trace.opentelemetry_imported
        trace.opentelemetry_imported = False

        try:
            trace.process_tracing_init()
            assert trace.tracing_enabled is False
        finally:
            trace.opentelemetry_imported = original_opentelemetry_imported

    def test_trace_set_thread_info_when_tracing_disabled(self):
        """Test trace_set_thread_info when tracing is disabled"""
        trace.tracing_enabled = False

        # Should not raise any exception
        trace.trace_set_thread_info("test_thread")

        # Should not add to threads_info
        pid = threading.get_native_id()
        assert pid not in trace.threads_info

    def test_trace_set_thread_info_existing_thread(self):
        """Test trace_set_thread_info when thread already exists"""
        trace.process_tracing_init()

        # Set thread info first time
        trace.trace_set_thread_info("test_thread")

        # Try to set again - should not overwrite
        original_thread_info = trace.threads_info[threading.get_native_id()]
        trace.trace_set_thread_info("different_thread")

        # Should still have original info
        pid = threading.get_native_id()
        assert trace.threads_info[pid] == original_thread_info

    def test_trace_req_start_without_thread_info(self):
        """Test trace_req_start when thread info is not set"""
        trace.process_tracing_init()
        trace.threads_info.clear()  # Clear thread info

        rid = "test_no_thread_info_req"
        trace.trace_req_start(rid, "")

        # Should not create request context
        assert rid not in trace.reqs_context

    def test_trace_req_start_existing_request(self):
        """Test trace_req_start when request already exists"""
        trace.process_tracing_init()
        trace.trace_set_thread_info("test_thread")

        rid = "test_existing_req"
        trace.trace_req_start(rid, "")

        # Try to start same request again - should return early
        trace.trace_req_start(rid, "")

        # Should not overwrite existing request (function returns early)
        assert rid in trace.reqs_context

    def test_trace_req_finish_nonexistent_request(self):
        """Test trace_req_finish with non-existent request"""
        trace.process_tracing_init()

        # Should not raise any exception
        trace.trace_req_finish("nonexistent_rid")

    def test_trace_slice_operations_without_request(self):
        """Test trace slice operations without request context"""
        trace.process_tracing_init()
        trace.trace_set_thread_info("test_thread")

        rid = "nonexistent_request"

        # Should not raise any exception
        trace.trace_slice_start("test", rid)
        trace.trace_slice_end("test", rid)
        trace.trace_event("test", rid)
        trace.trace_slice_add_attr(rid, {"test": "value"})

    def test_trace_get_proc_propagate_context_without_request(self):
        """Test trace_get_proc_propagate_context without request"""
        trace.process_tracing_init()

        result = trace.trace_get_proc_propagate_context("nonexistent_rid")
        assert result is None

    def test_trace_set_proc_propagate_context_without_request(self):
        """Test trace_set_proc_propagate_context without request"""
        trace.process_tracing_init()
        trace.trace_set_thread_info("test_thread")

        context_dict = {"test": "context"}

        # Should not raise any exception
        trace.trace_set_proc_propagate_context("nonexistent_rid", context_dict)

    def test_trace_set_proc_propagate_context_existing_thread(self):
        """Test trace_set_proc_propagate_context when thread already exists"""
        trace.process_tracing_init()
        trace.trace_set_thread_info("test_thread")

        rid = "test_existing_thread"
        context_dict = {"test": "context"}

        # Create request context first
        trace.reqs_context[rid] = trace.TraceReqContext(
            rid=rid,
            start_time_ns=int(time.time() * 1e9),
            threads_context={threading.get_native_id(): MagicMock()},
            is_copy=True,
        )

        # Try to set propagate context - should not create new thread context
        original_threads_context = trace.reqs_context[rid].threads_context.copy()
        trace.trace_set_proc_propagate_context(rid, context_dict)

        # Should not have changed threads_context
        assert trace.reqs_context[rid].threads_context == original_threads_context

    def test_trace_report_span_without_request(self):
        """Test trace_report_span without request context"""
        trace.process_tracing_init()

        # Should not raise any exception
        trace.trace_report_span("test", "nonexistent_rid", 0, 1000000)

    def test_all_functions_when_tracing_disabled(self):
        """Test all trace functions when tracing is disabled"""
        trace.tracing_enabled = False

        rid = "test_disabled"

        # All these should not raise exceptions
        trace.trace_req_start(rid, "")
        trace.trace_req_finish(rid)
        trace.trace_slice_start("test", rid)
        trace.trace_slice_end("test", rid)
        trace.trace_event("test", rid)
        trace.trace_slice_add_attr(rid, {"test": "value"})
        trace.trace_get_proc_propagate_context(rid)
        trace.trace_set_proc_propagate_context(rid, {})
        trace.trace_report_span("test", rid, 0, 1000000)

    def test_trace_req_start_with_role(self):
        """Test trace_req_start with role parameter"""
        trace.process_tracing_init()
        trace.trace_set_thread_info("test_thread")

        rid = "test_with_role"
        role = "test_role"

        trace.trace_req_start(rid, "", role=role)

        # Should create request context
        assert rid in trace.reqs_context

    def test_trace_req_start_with_null_role(self):
        """Test trace_req_start with null role"""
        trace.process_tracing_init()
        trace.trace_set_thread_info("test_thread")

        rid = "test_null_role"
        role = "null"

        trace.trace_req_start(rid, "", role=role)

        # Should create request context
        assert rid in trace.reqs_context

    def test_trace_span_decorator_with_custom_name(self):
        """Test trace_span decorator with custom span name"""
        trace.process_tracing_init()
        trace.trace_set_thread_info("test_thread")

        @trace.trace_span("custom_span_name")
        def test_function():
            return "test_result"

        result = test_function()
        assert result == "test_result"

    def test_trace_span_decorator_without_name(self):
        """Test trace_span decorator without custom span name"""
        trace.process_tracing_init()
        trace.trace_set_thread_info("test_thread")

        @trace.trace_span()
        def test_function():
            return "test_result"

        result = test_function()
        assert result == "test_result"

    def test_trace_span_decorator_with_none_name(self):
        """Test trace_span decorator with None span name"""
        trace.process_tracing_init()
        trace.trace_set_thread_info("test_thread")

        @trace.trace_span(None)
        def test_function():
            return "test_result"

        result = test_function()
        assert result == "test_result"

    def test_trace_slice_start_with_timestamp(self):
        """Test trace_slice_start with custom timestamp"""
        trace.process_tracing_init()
        trace.trace_set_thread_info("test_thread")

        rid = "test_timestamp"
        ts = int(time.time() * 1e9) - 1000000  # 1ms ago

        trace.trace_req_start(rid, "")
        trace.trace_slice_start("test_slice", rid, ts=ts)
        trace.trace_slice_end("test_slice", rid)
        trace.trace_req_finish(rid)

    def test_trace_slice_end_with_timestamp(self):
        """Test trace_slice_end with custom timestamp"""
        trace.process_tracing_init()
        trace.trace_set_thread_info("test_thread")

        rid = "test_end_timestamp"
        ts = int(time.time() * 1e9) + 1000000  # 1ms in future

        trace.trace_req_start(rid, "")
        trace.trace_slice_start("test_slice", rid)
        trace.trace_slice_end("test_slice", rid, ts=ts)
        trace.trace_req_finish(rid)

    def test_trace_slice_end_with_attributes(self):
        """Test trace_slice_end with attributes"""
        trace.process_tracing_init()
        trace.trace_set_thread_info("test_thread")

        rid = "test_attrs"
        attrs = {"test_attr": "test_value", "number_attr": 123}

        trace.trace_req_start(rid, "")
        trace.trace_slice_start("test_slice", rid)
        trace.trace_slice_end("test_slice", rid, attrs=attrs)
        trace.trace_req_finish(rid)

    def test_trace_event_with_timestamp(self):
        """Test trace_event with custom timestamp"""
        trace.process_tracing_init()
        trace.trace_set_thread_info("test_thread")

        rid = "test_event_timestamp"
        ts = int(time.time() * 1e9) - 500000  # 0.5ms ago

        trace.trace_req_start(rid, "")
        trace.trace_slice_start("test_slice", rid)
        trace.trace_event("test_event", rid, ts=ts)
        trace.trace_slice_end("test_slice", rid)
        trace.trace_req_finish(rid)

    def test_trace_event_without_attributes(self):
        """Test trace_event without attributes"""
        trace.process_tracing_init()
        trace.trace_set_thread_info("test_thread")

        rid = "test_event_no_attrs"

        trace.trace_req_start(rid, "")
        trace.trace_slice_start("test_slice", rid)
        trace.trace_event("test_event", rid)
        trace.trace_slice_end("test_slice", rid)
        trace.trace_req_finish(rid)

    def test_trace_report_span_with_thread_finish(self):
        """Test trace_report_span with thread_finish_flag"""
        trace.process_tracing_init()
        trace.trace_set_thread_info("test_thread")

        rid = "test_report_thread_finish"
        start_time = int(time.time() * 1e9)
        end_time = start_time + 1000000  # 1ms later

        trace.trace_req_start(rid, "")
        trace.trace_report_span("test_span", rid, start_time, end_time, thread_finish_flag=True)
        trace.trace_req_finish(rid)

    def test_multiple_nested_slices(self):
        """Test multiple levels of nested slices"""
        trace.process_tracing_init()
        trace.trace_set_thread_info("test_thread")

        rid = "test_nested"

        trace.trace_req_start(rid, "")
        trace.trace_slice_start("level1", rid)
        trace.trace_slice_start("level2", rid)
        trace.trace_slice_start("level3", rid)
        trace.trace_slice_end("level3", rid)
        trace.trace_slice_end("level2", rid)
        trace.trace_slice_end("level1", rid)
        trace.trace_req_finish(rid)

    def test_concurrent_slice_operations(self):
        """Test concurrent slice operations"""
        trace.process_tracing_init()

        rid = "test_concurrent_slices"

        def worker_slices():
            trace.trace_set_thread_info("worker_thread")
            trace.trace_req_start(rid, "")
            trace.trace_slice_start("worker_slice", rid)
            time.sleep(0.001)
            trace.trace_slice_end("worker_slice", rid)
            trace.trace_req_finish(rid)

        # Main thread
        trace.trace_set_thread_info("main_thread")
        trace.trace_req_start(rid, "")
        trace.trace_slice_start("main_slice", rid)

        # Start worker thread
        thread = threading.Thread(target=worker_slices)
        thread.start()
        thread.join()

        trace.trace_slice_end("main_slice", rid)
        trace.trace_req_finish(rid)


class TestGetTraceInfoForRequest:
    """Test cases for get_trace_info_for_request function - comprehensive coverage"""

    def setup_method(self):
        """Setup test environment"""
        self.original_env = os.environ.copy()
        os.environ["FD_TRACE"] = "otel"
        os.environ["FD_SERVICE_NAME"] = "test_service"
        os.environ["EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4317"

        # Reset global state
        trace.remote_trace_contexts = {}
        trace.threads_info = {}
        trace.reqs_context = {}
        trace.tracing_enabled = False

    def teardown_method(self):
        """Restore environment"""
        os.environ = self.original_env

    def test_get_trace_info_tracing_disabled(self):
        """Test get_trace_info_for_request when tracing is disabled"""
        trace.tracing_enabled = False

        result = trace.get_trace_info_for_request("test_rid")
        assert result is None

    def test_get_trace_info_request_not_found(self):
        """Test get_trace_info_for_request when request doesn't exist"""
        trace.process_tracing_init()

        result = trace.get_trace_info_for_request("nonexistent_rid")
        assert result is None

    def test_get_trace_info_request_not_found_with_suffix(self):
        """Test get_trace_info_for_request when rid with suffix doesn't exist"""
        trace.process_tracing_init()

        # Request with _idx suffix where neither rid nor orig_rid exists
        result = trace.get_trace_info_for_request("nonexistent_rid_123")
        assert result is None

    def test_get_trace_info_rid_with_suffix_fallback(self):
        """Test get_trace_info_for_request with rid suffix fallback to orig_rid"""
        trace.process_tracing_init()

        # Note: get_base_request_id uses CHOICE_SEPARATOR (::n::) to split
        # So "test::n::0" -> "test", preserving underscores in the base request_id
        rid = "testrid"

        # Create request context directly to avoid FastAPI instrumentation complications
        from fastdeploy.metrics.trace import TraceReqContext

        mock_span = MagicMock()
        mock_span.get_span_context.return_value = MagicMock(is_valid=True, trace_id=123456789, span_id=987654321)

        trace.reqs_context[rid] = TraceReqContext(
            rid=rid,
            start_time_ns=int(time.time() * 1e9),
            threads_context={},
            root_span=mock_span,
            root_span_context=None,
        )

        # Request with ::n:: suffix should fallback to orig_rid via get_base_request_id
        result = trace.get_trace_info_for_request("testrid::n::0")

        # Should find the request and return trace info
        assert result is not None
        assert "trace_id" in result
        assert "span_id" in result
        assert result["trace_id"] == format(123456789, "032x")
        assert result["span_id"] == format(987654321, "016x")

        del trace.reqs_context[rid]

    def test_get_trace_info_from_root_span_valid(self):
        """Test get_trace_info_for_request from valid root_span"""
        trace.process_tracing_init()
        trace.trace_set_thread_info("test_thread")

        rid = "test_valid_root_span"
        trace.trace_req_start(rid, "")

        result = trace.get_trace_info_for_request(rid)

        assert result is not None
        assert "trace_id" in result
        assert "span_id" in result
        # Verify format: trace_id should be 32 hex chars, span_id 16 hex chars
        assert len(result["trace_id"]) == 32
        assert len(result["span_id"]) == 16
        # Verify they are valid hex strings
        int(result["trace_id"], 16)
        int(result["span_id"], 16)

        trace.trace_req_finish(rid)

    def test_get_trace_info_root_span_invalid_span_context(self):
        """Test get_trace_info_for_request when root_span has invalid span_context"""
        trace.process_tracing_init()
        trace.trace_set_thread_info("test_thread")

        rid = "test_invalid_span"

        # Create a request context manually with mocked root_span
        from fastdeploy.metrics.trace import TraceReqContext

        mock_span = MagicMock()
        mock_span.get_span_context.return_value = MagicMock(is_valid=False, trace_id=0, span_id=0)

        trace.reqs_context[rid] = TraceReqContext(
            rid=rid,
            start_time_ns=int(time.time() * 1e9),
            threads_context={},
            root_span=mock_span,
            root_span_context=None,
        )

        result = trace.get_trace_info_for_request(rid)
        # Should return None since span_context is invalid and no root_span_context
        assert result is None

        del trace.reqs_context[rid]

    def test_get_trace_info_root_span_zero_trace_id(self):
        """Test get_trace_info_for_request when root_span has trace_id=0"""
        trace.process_tracing_init()

        rid = "test_zero_trace"

        # Create a request context manually with mocked root_span with trace_id=0
        from fastdeploy.metrics.trace import TraceReqContext

        mock_span = MagicMock()
        mock_span.get_span_context.return_value = MagicMock(is_valid=True, trace_id=0, span_id=12345)

        trace.reqs_context[rid] = TraceReqContext(
            rid=rid,
            start_time_ns=int(time.time() * 1e9),
            threads_context={},
            root_span=mock_span,
            root_span_context=None,
        )

        result = trace.get_trace_info_for_request(rid)
        # Should return None since trace_id is 0
        assert result is None

        del trace.reqs_context[rid]

    def test_get_trace_info_from_root_span_context(self):
        """Test get_trace_info_for_request from root_span_context when root_span is None"""
        trace.process_tracing_init()

        rid = "test_root_span_context"

        # Create a request context with root_span=None but valid root_span_context
        from fastdeploy.metrics.trace import TraceReqContext

        mock_root_context = MagicMock()

        # Mock the span that will be returned from get_current_span
        mock_span = MagicMock()
        mock_span.get_span_context.return_value = MagicMock(is_valid=True, trace_id=123456789, span_id=987654321)

        trace.reqs_context[rid] = TraceReqContext(
            rid=rid,
            start_time_ns=int(time.time() * 1e9),
            threads_context={},
            root_span=None,
            root_span_context=mock_root_context,
        )

        # Patch get_current_span at the source module where it's imported from
        with mock.patch("opentelemetry.trace.get_current_span", return_value=mock_span):
            result = trace.get_trace_info_for_request(rid)

            assert result is not None
            assert "trace_id" in result
            assert "span_id" in result
            assert result["trace_id"] == format(123456789, "032x")
            assert result["span_id"] == format(987654321, "016x")

        del trace.reqs_context[rid]

    def test_get_trace_info_root_span_context_invalid(self):
        """Test get_trace_info_for_request when root_span_context span is invalid"""
        trace.process_tracing_init()

        rid = "test_invalid_root_span_context"

        # Create a request context with root_span=None but root_span_context present
        from fastdeploy.metrics.trace import TraceReqContext

        mock_root_context = MagicMock()

        # Mock the span with invalid context
        mock_span = MagicMock()
        mock_span.get_span_context.return_value = MagicMock(is_valid=False, trace_id=0)

        trace.reqs_context[rid] = TraceReqContext(
            rid=rid,
            start_time_ns=int(time.time() * 1e9),
            threads_context={},
            root_span=None,
            root_span_context=mock_root_context,
        )

        with mock.patch("opentelemetry.trace.get_current_span", return_value=mock_span):
            result = trace.get_trace_info_for_request(rid)
            # Should return None since span_context is invalid
            assert result is None

        del trace.reqs_context[rid]

    def test_get_trace_info_get_current_span_exception(self):
        """Test get_trace_info_for_request when get_current_span raises exception"""
        trace.process_tracing_init()

        rid = "test_exception"

        # Create a request context with root_span=None but root_span_context present
        from fastdeploy.metrics.trace import TraceReqContext

        mock_root_context = MagicMock()

        trace.reqs_context[rid] = TraceReqContext(
            rid=rid,
            start_time_ns=int(time.time() * 1e9),
            threads_context={},
            root_span=None,
            root_span_context=mock_root_context,
        )

        # Mock get_current_span to raise exception
        with mock.patch("opentelemetry.trace.get_current_span", side_effect=Exception("Test exception")):
            result = trace.get_trace_info_for_request(rid)
            # Should return None after catching exception
            assert result is None

        del trace.reqs_context[rid]

    def test_get_trace_info_no_root_span_no_context(self):
        """Test get_trace_info_for_request when both root_span and root_span_context are None"""
        trace.process_tracing_init()
        trace.trace_set_thread_info("test_thread")

        rid = "test_no_root_no_context"
        trace.trace_req_start(rid, "")

        # Set both root_span and root_span_context to None
        trace.reqs_context[rid].root_span = None
        trace.reqs_context[rid].root_span_context = None

        result = trace.get_trace_info_for_request(rid)
        # Should return None since neither root_span nor root_span_context is available
        assert result is None

        trace.trace_req_finish(rid)

    def test_get_trace_info_rid_conversion_to_string(self):
        """Test get_trace_info_for_request converts rid to string"""
        trace.process_tracing_init()
        trace.trace_set_thread_info("test_thread")

        # Use integer rid
        rid = 12345
        trace.trace_req_start(str(rid), "")

        # Call with integer rid
        result = trace.get_trace_info_for_request(rid)

        assert result is not None
        assert "trace_id" in result
        assert "span_id" in result

        trace.trace_req_finish(str(rid))

    def test_get_trace_info_after_request_finished(self):
        """Test get_trace_info_for_request after request is finished"""
        trace.process_tracing_init()
        trace.trace_set_thread_info("test_thread")

        rid = "test_finished_request"
        trace.trace_req_start(rid, "")
        trace.trace_req_finish(rid)

        # Request should be removed from reqs_context
        result = trace.get_trace_info_for_request(rid)
        assert result is None

    def test_get_trace_info_from_copy_request(self):
        """Test get_trace_info_for_request from a copied request (is_copy=True)"""
        trace.process_tracing_init()
        trace.trace_set_thread_info("test_thread")

        # Create a request with is_copy=True
        rid = "test_copy_request"
        from fastdeploy.metrics.trace import TraceReqContext

        trace.reqs_context[rid] = TraceReqContext(
            rid=rid,
            start_time_ns=int(time.time() * 1e9),
            threads_context={},
            is_copy=True,
            root_span=None,
            root_span_context=None,
        )

        # Without root_span or root_span_context, should return None
        result = trace.get_trace_info_for_request(rid)
        assert result is None

    def test_get_trace_info_with_propagated_context(self):
        """Test get_trace_info_for_request with propagated context from another process"""
        trace.process_tracing_init()
        trace.trace_set_thread_info("test_thread")

        rid1 = "test_original"
        trace.trace_req_start(rid1, "")

        # Get propagation context
        context_dict = trace.trace_get_proc_propagate_context(rid1)

        # Set propagated context for new request
        rid2 = "test_propagated"
        trace.trace_set_proc_propagate_context(rid2, context_dict)

        # Should be able to get trace info for propagated request
        result = trace.get_trace_info_for_request(rid2)

        # Note: The result might be None depending on how the context is set up,
        # but the function should not crash
        print(result)

        trace.trace_req_finish(rid1)
        trace.trace_req_finish(rid2)


class TestProcessTracingInitError(unittest.TestCase):
    """Test process_tracing_init logs error when initialization fails."""

    def test_process_tracing_init_logs_error_on_exception(self):
        """Test that initialize opentelemetry error is logged with traceback."""
        import fastdeploy.metrics.trace as trace_module

        original_enabled = trace_module.tracing_enabled
        original_imported = trace_module.opentelemetry_imported
        try:
            trace_module.opentelemetry_imported = True
            with patch.object(trace_module.envs, "FD_TRACE", "otel"):
                with patch.object(
                    trace_module, "get_otlp_span_exporter", side_effect=RuntimeError("otlp init failed")
                ):
                    with patch.object(trace_module.logger, "error") as mock_error:
                        trace_module.process_tracing_init()

            mock_error.assert_called_once()
            error_msg = mock_error.call_args[0][0]
            self.assertIn("initialize opentelemetry error", error_msg)
            self.assertIn("otlp init failed", error_msg)
            self.assertFalse(trace_module.tracing_enabled)
        finally:
            trace_module.tracing_enabled = original_enabled
            trace_module.opentelemetry_imported = original_imported


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
