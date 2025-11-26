"""
Test cases for fastdeploy/metrics/trace.py
"""

import os
import threading
import time
from unittest import mock

import pytest

from fastdeploy.metrics import trace


class TestTraceBasic:
    """Test basic tracing functionality"""

    def setup_method(self):
        """Setup test environment"""
        # Mock environment variables
        self.original_env = os.environ.copy()
        os.environ["TRACES_ENABLE"] = "true"
        os.environ["FD_SERVICE_NAME"] = "test_service"
        os.environ["EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4317"

        # Reset global state
        trace.remote_trace_contexts = {}
        trace.threads_info = {}
        trace.reqs_context = {}

    def teardown_method(self):
        """Restore environment"""
        os.environ = self.original_env

    def test_process_tracing_init(self):
        """Test tracing initialization"""
        trace.process_tracing_init()
        assert trace.tracing_enabled is True

    def test_thread_info_registration(self):
        """Test thread info registration"""
        trace.trace_set_thread_info("test_thread")
        pid = threading.get_native_id()
        assert pid in trace.threads_info
        assert trace.threads_info[pid].thread_label == "test_thread"

    def test_thread_info_idempotency(self):
        """Test thread info registration is idempotent"""
        trace.trace_set_thread_info("test_thread")
        pid = threading.get_native_id()
        initial_info = trace.threads_info[pid]

        # Register again
        trace.trace_set_thread_info("test_thread_again")
        assert trace.threads_info[pid] is initial_info  # Should not change


class TestRequestTracing:
    """Test request tracing functionality"""

    def setup_method(self):
        """Setup test environment"""
        self.setup_method_basic()

    def setup_method_basic(self):
        """Basic setup for request tracing tests"""
        os.environ["TRACES_ENABLE"] = "true"
        trace.process_tracing_init()
        trace.trace_set_thread_info("test_thread")

    def test_request_start_end(self):
        """Test basic request tracing"""
        rid = "test_request_123"

        # Start request
        trace.trace_req_start(rid)
        assert rid in trace.reqs_context

        # End request
        trace.trace_req_finish(rid)
        assert rid not in trace.reqs_context

    def test_slice_tracing(self):
        """Test slice tracing within a request"""
        rid = "test_request_456"
        slice_name = "test_slice"

        trace.trace_req_start(rid)

        # Start slice
        trace.trace_slice_start(slice_name, rid)

        pid = threading.get_native_id()
        thread_context = trace.reqs_context[rid].threads_context[pid]
        assert len(thread_context.cur_slice_stack) == 1
        assert thread_context.cur_slice_stack[0].slice_name == slice_name

        # End slice
        trace.trace_slice_end(slice_name, rid)
        assert len(thread_context.cur_slice_stack) == 0

        trace.trace_req_finish(rid)

    def test_nested_slices(self):
        """Test nested slice tracing"""
        rid = "test_request_nested"

        trace.trace_req_start(rid)

        # Start multiple slices
        trace.trace_slice_start("outer_slice", rid)
        trace.trace_slice_start("middle_slice", rid)
        trace.trace_slice_start("inner_slice", rid)

        pid = threading.get_native_id()
        thread_context = trace.reqs_context[rid].threads_context[pid]
        assert len(thread_context.cur_slice_stack) == 3

        # End slices in reverse order
        trace.trace_slice_end("inner_slice", rid)
        trace.trace_slice_end("middle_slice", rid)
        trace.trace_slice_end("outer_slice", rid)

        assert len(thread_context.cur_slice_stack) == 0
        trace.trace_req_finish(rid)

    def test_slice_with_attributes(self):
        """Test slice with attributes"""
        rid = "test_request_with_attrs"

        trace.trace_req_start(rid)
        trace.trace_slice_start("test_slice", rid)

        attrs = {"key1": "value1", "key2": 42}
        trace.trace_slice_end("test_slice", rid, attrs=attrs)
        trace.trace_req_finish(rid)

    def test_trace_event(self):
        """Test event tracing"""
        rid = "test_request_event"

        trace.trace_req_start(rid)
        trace.trace_slice_start("parent_slice", rid)

        event_attrs = {"event_type": "test", "count": 5}
        trace.trace_event("test_event", rid, attrs=event_attrs)

        trace.trace_slice_end("parent_slice", rid)
        trace.trace_req_finish(rid)

    def test_slice_add_attr(self):
        """Test adding attributes to current slice"""
        rid = "test_request_add_attr"

        trace.trace_req_start(rid)
        trace.trace_slice_start("test_slice", rid)

        attrs = {"new_attr": "value"}
        trace.trace_slice_add_attr(rid, attrs)

        trace.trace_slice_end("test_slice", rid)
        trace.trace_req_finish(rid)


class TestContextPropagation:
    """Test context propagation functionality"""

    def setup_method(self):
        """Setup test environment"""
        self.setup_method_basic()

    def setup_method_basic(self):
        """Basic setup for context propagation tests"""
        os.environ["TRACES_ENABLE"] = "true"
        trace.process_tracing_init()
        trace.trace_set_thread_info("test_thread")

    def test_context_propagation(self):
        """Test context serialization and deserialization"""
        rid = "test_request_789"
        trace.trace_req_start(rid)

        # Get context
        context_dict = trace.trace_get_proc_propagate_context(rid)
        assert context_dict is not None

        # Simulate propagation to another process
        new_rid = "test_request_789_child"
        trace.trace_set_proc_propagate_context(new_rid, context_dict)

        assert new_rid in trace.reqs_context
        assert trace.reqs_context[new_rid].is_copy is True

        trace.trace_req_finish(rid)
        trace.trace_req_finish(new_rid)

    def test_empty_context_propagation(self):
        """Test propagation with empty context"""
        rid = "test_request_empty"
        trace.trace_req_start(rid)

        # Test with None context
        trace.trace_set_proc_propagate_context(rid, None)

        trace.trace_req_finish(rid)


class TestEdgeCases:
    """Test edge cases and error handling"""

    def setup_method(self):
        """Setup test environment"""
        self.setup_method_basic()

    def setup_method_basic(self):
        """Basic setup for edge case tests"""
        os.environ["TRACES_ENABLE"] = "true"
        trace.process_tracing_init()
        trace.trace_set_thread_info("test_thread")

    def test_double_request_start(self):
        """Test starting same request twice"""
        rid = "test_request_111"
        trace.trace_req_start(rid)
        trace.trace_req_start(rid)  # Should be idempotent
        assert rid in trace.reqs_context
        trace.trace_req_finish(rid)

    def test_missing_thread_info(self):
        """Test operations without thread info"""
        rid = "test_request_222"
        # Clear thread info
        trace.threads_info = {}

        # Should handle gracefully
        trace.trace_req_start(rid)
        assert rid not in trace.reqs_context

    def test_tracing_disabled(self):
        """Test operations when tracing is disabled"""
        os.environ["TRACES_ENABLE"] = "false"
        trace.process_tracing_init()

        rid = "test_request_333"
        trace.trace_req_start(rid)
        assert rid not in trace.reqs_context

    def test_slice_name_mismatch(self):
        """Test slice name mismatch handling"""
        rid = "test_request_mismatch"

        trace.trace_req_start(rid)
        trace.trace_slice_start("start_name", rid)

        # Try to end with different name (should log warning but not crash)
        trace.trace_slice_end("end_name", rid)

        trace.trace_req_finish(rid)

    def test_nonexistent_request(self):
        """Test operations on nonexistent request"""
        rid = "nonexistent_request"

        # All operations should handle gracefully
        trace.trace_slice_start("test", rid)
        trace.trace_slice_end("test", rid)
        trace.trace_event("test", rid)
        trace.trace_slice_add_attr(rid, {"test": "value"})
        trace.trace_req_finish(rid)

    def test_event_without_slice(self):
        """Test event without active slice"""
        rid = "test_request_no_slice"

        trace.trace_req_start(rid)
        # Try to add event without starting a slice (should log warning)
        trace.trace_event("test_event", rid)
        trace.trace_req_finish(rid)

    def test_anonymous_slice(self):
        """Test anonymous slice functionality"""
        rid = "test_request_anonymous"

        trace.trace_req_start(rid)
        trace.trace_slice_start("", rid, anonymous=True)
        trace.trace_slice_end("named_slice", rid)  # Name gets assigned at end
        trace.trace_req_finish(rid)


class TestTraceSpanName:
    """Test TraceSpanName enum"""

    def test_span_name_values(self):
        """Test all span name values"""
        assert trace.TraceSpanName.FASTDEPLOY == "fastdeploy"
        assert trace.TraceSpanName.PREPROCESS == "preprocess"
        assert trace.TraceSpanName.ENCODE == "encode"
        assert trace.TraceSpanName.SUBMIT_TO_INFER_ENGINE == "submit_to_infer_engine"
        assert trace.TraceSpanName.SCHEDULE == "schedule"
        assert trace.TraceSpanName.SCHEDULE_QUEUE_WAIT == "schedule_queue_wait"
        assert trace.TraceSpanName.SCHEDULE_DISPATCH == "schedule_dispatch"
        assert trace.TraceSpanName.SCHEDULER_ALLOCATE_RESOURCE == "scheduler_allocate_resource"
        assert trace.TraceSpanName.PREFILL == "prefill"
        assert trace.TraceSpanName.DECODE_LOOP == "decode_loop"


def test_lable_span_function():
    """Test the lable_span function"""

    # Mock a request object with stream attribute
    class MockRequest:
        def __init__(self, stream_value):
            self.stream = stream_value

    # Test with stream=True
    mock_request_stream = MockRequest(True)

    # Since we don't have actual OpenTelemetry spans in tests,
    # we mainly test that the function doesn't crash
    trace.lable_span(mock_request_stream)

    # Test with stream=False
    mock_request_no_stream = MockRequest(False)
    trace.lable_span(mock_request_no_stream)


def test_get_otlp_span_exporter():
    """Test OTLP span exporter creation"""
    # Test with different protocols
    with mock.patch.dict(os.environ, {trace.OTEL_EXPORTER_OTLP_TRACES_PROTOCOL: "grpc"}):
        exporter = trace.get_otlp_span_exporter("http://localhost:4317", None)
        assert exporter is not None

    with mock.patch.dict(os.environ, {trace.OTEL_EXPORTER_OTLP_TRACES_PROTOCOL: "http/protobuf"}):
        exporter = trace.get_otlp_span_exporter("http://localhost:4318", {"header": "value"})
        assert exporter is not None

    # Test with unsupported protocol
    with mock.patch.dict(os.environ, {trace.OTEL_EXPORTER_OTLP_TRACES_PROTOCOL: "unsupported"}):
        with pytest.raises(ValueError):
            trace.get_otlp_span_exporter("http://localhost:4317", None)


@pytest.mark.skipif(True, reason="Requires OpenTelemetry SDK to be installed")
class TestIntegration:
    """Integration tests with actual OpenTelemetry SDK"""

    def setup_method(self):
        """Setup test environment"""
        self.setup_method_basic()

    def setup_method_basic(self):
        """Basic setup for integration tests"""
        os.environ["TRACES_ENABLE"] = "true"
        trace.process_tracing_init()
        trace.trace_set_thread_info("test_thread")

    def test_end_to_end_tracing(self):
        """Test complete tracing workflow"""
        rid = "test_request_444"

        # Start request
        trace.trace_req_start(rid)

        # Add slice
        trace.trace_slice_start("preprocess", rid)
        time.sleep(0.1)
        trace.trace_slice_end("preprocess", rid)

        # Add event
        trace.trace_event("cache_hit", rid, attrs={"size": 1024})

        # End request
        trace.trace_req_finish(rid)
