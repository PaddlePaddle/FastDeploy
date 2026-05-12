"""
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

# This file is modified from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/tracing/trace.py

from __future__ import annotations

import inspect
import os
import random
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from enum import Enum, unique
from functools import wraps
from typing import Any, Dict, List, Optional

from fastdeploy import envs
from fastdeploy.utils import api_server_logger as logger
from fastdeploy.utils import get_base_request_id

opentelemetry_imported = False
tracing_enabled = False

try:
    from opentelemetry import context, propagate, trace
    from opentelemetry.sdk.environment_variables import (
        OTEL_EXPORTER_OTLP_TRACES_PROTOCOL,
    )
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import SpanProcessor, TracerProvider, id_generator
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter

    opentelemetry_imported = True
except ImportError as e:
    logger.error(f"Failed to import opentelemetry, tracing disabled.{e}")

    class id_generator:
        class IdGenerator:
            pass

    logger.info("opentelemetry package is not installed, tracing disabled")


class FilteringSpanProcessor(SpanProcessor):
    def __init__(self, exporter: SpanExporter, **kwargs):
        self._processor = BatchSpanProcessor(exporter, **kwargs)

    def on_start(self, span, parent_context=None):
        parent_span = trace.get_current_span()
        if parent_span and parent_span.is_recording():
            stream_attr = parent_span.attributes.get("stream")
            if stream_attr is not None:
                span.set_attribute("stream", stream_attr)
        self._processor.on_start(span, parent_context)

    def on_end(self, span):
        # asgi_event_type = span.attributes.get("asgi.event.type")
        # stream = span.attributes.get("stream")
        span_name = span.name or ""

        if "http" in span_name:
            return

        self._processor.on_end(span)

    def shutdown(self):
        self._processor.shutdown()

    def force_flush(self, timeout_millis=None):
        self._processor.force_flush(timeout_millis)


def label_span(request):
    if request.stream:
        span = trace.get_current_span()
        if span is not None and span.is_recording():
            span.set_attribute("stream", "true")


@dataclass
class TraceThreadInfo:
    host_id: str
    pid: int
    thread_label: str
    tp_rank: int
    dp_rank: int
    tracer: trace.Tracer


@dataclass
class TraceSliceContext:
    slice_name: str
    span: Optional[trace.span.Span] = None
    # When True, defers slice_name assignment until trace_slice_end()
    anonymous: bool = False


@dataclass
class TraceThreadContext:
    thread_info: TraceThreadInfo
    cur_slice_stack: List[TraceSliceContext]
    thread_span: Optional[trace.span.Span] = None
    # Record the most recently completed span as the previous span for the next span to be created.
    last_span_context: Optional[trace.span.SpanContext] = None


@dataclass
class TraceReqContext:
    rid: str
    start_time_ns: int
    threads_context: Dict[int, TraceThreadContext]

    # Indicates whether this instance is a replica from the main process.
    # When True, root_span is None and only root_span_context is preserved.
    is_copy: bool = False
    root_span: Optional[trace.span.Span] = None
    root_span_context: Optional[context.Context] = None


@dataclass
class TracePropagateContext:
    root_span_context: context.Context
    prev_span_context: Optional[trace.span.SpanContext]

    def to_dict(self):
        carrier: dict[str, str] = {}
        propagate.inject(carrier, context=self.root_span_context)

        if self.prev_span_context:
            return {
                "root_span": carrier,
                "prev_span": {
                    "span_id": self.prev_span_context.span_id,
                    "trace_id": self.prev_span_context.trace_id,
                },
            }
        else:
            return {"root_span": carrier, "prev_span": "None"}

    @classmethod
    def instance_from_dict(cls, d):
        if "root_span" not in d or "prev_span" not in d:
            return None

        carrier = d["root_span"]
        root_span_context = propagate.extract(carrier)

        if d["prev_span"] == "None":
            prev_span_context = None
        else:
            prev_span_context = trace.span.SpanContext(
                trace_id=d["prev_span"]["trace_id"],
                span_id=d["prev_span"]["span_id"],
                is_remote=True,
            )

        return cls(root_span_context, prev_span_context)


class TraceCustomIdGenerator(id_generator.IdGenerator):
    """
    The default IdGenerator may produce duplicate trace IDs across multiple TP scheduler processes,
    hence a custom IdGenerator is implemented.
    """

    def __init__(self):
        super().__init__()
        self.local_random = random.Random()
        self.local_random.seed(time.time())

    def generate_trace_id(self) -> int:
        return self.local_random.getrandbits(64)

    def generate_span_id(self) -> int:
        return self.local_random.getrandbits(64)


# global variables
remote_trace_contexts: Dict[str, TracePropagateContext] = {}
threads_info: Dict[int, TraceThreadInfo] = {}
reqs_context: Dict[str, TraceReqContext] = {}

__get_cur_time_ns = lambda: int(time.time() * 1e9)


def __get_host_id() -> str:
    """
    In distributed tracing systems, obtain a unique node identifier
    and inject it into all subsequently generated spans
    to prevent PID conflicts between threads on different nodes.
    """
    if envs.FD_HOST_NAME:
        return envs.FD_HOST_NAME
    paths = ["/etc/machine-id", "/var/lib/dbus/machine-id"]
    for path in paths:
        try:
            with open(path, "r") as f:
                val = f.read().strip()
                if val:
                    return val
        except Exception:
            continue

    mac = uuid.getnode()
    if mac != 0:
        return uuid.UUID(int=mac).hex

    try:
        unique_id = uuid.uuid4().hex + "-" + str(os.getpid())
        return unique_id
    except Exception:
        return "unknown"


# Should be called by each tracked process.
def process_tracing_init():
    global tracing_enabled
    global __get_cur_time_ns
    tracing_enabled = envs.FD_TRACE in ("otel", "all")

    if not tracing_enabled:
        logger.warning("Opentelemetry is DISABLED.")
        return

    if not opentelemetry_imported:
        tracing_enabled = False
        return

    try:
        # --- read env ---
        service_name = envs.FD_SERVICE_NAME
        host_name = envs.FD_HOST_NAME
        resource_attributes = {"service.name": service_name}
        if host_name:
            resource_attributes["host.name"] = host_name
        resource = Resource(attributes=resource_attributes)
        endpoint = envs.EXPORTER_OTLP_ENDPOINT
        headers = envs.EXPORTER_OTLP_HEADERS
        headers = dict(item.split("=") for item in headers.split(",")) if headers else None

        otlp_exporter = get_otlp_span_exporter(endpoint, headers)

        schedule_delay_millis = envs.FD_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS
        max_export_batch_size = envs.FD_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE
        processor = FilteringSpanProcessor(
            otlp_exporter,
            schedule_delay_millis=schedule_delay_millis,
            max_export_batch_size=max_export_batch_size,
        )
        tracer_provider = TracerProvider(resource=resource, id_generator=TraceCustomIdGenerator())

        tracer_provider.add_span_processor(processor)
        # tracer_provider.add_span_processor(
        #     SimpleSpanProcessor(ConsoleSpanExporter())
        # )
        trace.set_tracer_provider(tracer_provider)
    except Exception as e:
        logger.error(f": initialize opentelemetry error:{e}, {traceback.format_exc()}")
        logger.warning("please set correct otlp endpoint")
        tracing_enabled = False
        return

    if hasattr(time, "time_ns"):
        __get_cur_time_ns = lambda: int(time.time_ns())

    tracing_enabled = True


def get_otlp_span_exporter(endpoint, headers):
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter as GRPCSpanExporter,
    )
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter as HTTPSpanExporter,
    )

    protocol = os.environ.get(OTEL_EXPORTER_OTLP_TRACES_PROTOCOL, "http/protobuf")
    supported_protocols = {"grpc", "http/protobuf"}

    if protocol not in supported_protocols:
        raise ValueError(
            f"Unsupported OTLP protocol '{protocol}' configured. "
            f"Supported protocols are: {', '.join(sorted(supported_protocols))}"
        )

    if protocol == "grpc":
        return GRPCSpanExporter(endpoint=endpoint, insecure=True)
    elif protocol == "http/protobuf":
        return HTTPSpanExporter(endpoint=endpoint, headers=headers)


# Should be called by each tracked thread.
def trace_set_thread_info(thread_label: str, tp_rank: Optional[int] = None, dp_rank: Optional[int] = None):
    if not tracing_enabled:
        return

    pid = threading.get_native_id()
    if pid in threads_info:
        return

    threads_info[pid] = TraceThreadInfo(
        host_id=__get_host_id(),
        pid=pid,
        thread_label=thread_label,
        tp_rank=tp_rank,
        dp_rank=dp_rank,
        tracer=trace.get_tracer("fastdeploy server"),
    )


def __create_thread_context(pid, req_span_context, ts: Optional[int] = None):
    if pid not in threads_info:
        trace_set_thread_info("unknown")

    thread_info = threads_info[pid]
    thread_context = TraceThreadContext(
        thread_info=thread_info,
        cur_slice_stack=[],
    )

    thread_name = f"{thread_info.thread_label}"
    if thread_info.tp_rank is not None:
        thread_name += f" [TP {thread_info.tp_rank}] "
    thread_name += f"(host:{thread_info.host_id} | pid:{pid})"
    ts = ts or __get_cur_time_ns()
    thread_context.thread_span = thread_context.thread_info.tracer.start_span(
        name=thread_name,
        start_time=ts,
        context=req_span_context,
    )

    if thread_info.tp_rank is not None:
        thread_context.thread_span.set_attributes({"tp_rank": thread_info.tp_rank})

    thread_context.thread_span.set_attributes(
        {
            "host_id": thread_info.host_id,
            "pid": thread_info.pid,
            "thread_label": thread_info.thread_label,
        }
    )

    return thread_context


def trace_get_proc_propagate_context(rid) -> Optional[Dict[str, Any]]:
    if not tracing_enabled:
        return None

    rid = str(rid)
    if rid not in reqs_context or not reqs_context[rid].root_span_context:
        return None

    pid = threading.get_native_id()
    prev_span_context = None
    thread_context = reqs_context[rid].threads_context[pid]
    if thread_context.cur_slice_stack:
        cur_slice_info = thread_context.cur_slice_stack[0]
        prev_span_context = cur_slice_info.span.get_span_context()
    elif thread_context.last_span_context:
        prev_span_context = thread_context.last_span_context

    root_span_context = reqs_context[rid].root_span_context

    trace_context = TracePropagateContext(root_span_context, prev_span_context)
    return trace_context.to_dict()


def trace_set_proc_propagate_context(rid, trace_context: Optional[Dict[str, Any]], ts: Optional[int] = None):
    if not tracing_enabled:
        return
    if not trace_context:
        return

    trace_context = TracePropagateContext.instance_from_dict(trace_context)
    if not trace_context:
        return

    rid = str(rid)
    # Create a copy of the request context
    if rid not in reqs_context:
        reqs_context[rid] = TraceReqContext(
            rid=rid,
            start_time_ns=ts or __get_cur_time_ns(),
            threads_context={},
            root_span_context=trace_context.root_span_context,
            is_copy=True,
        )

    pid = threading.get_native_id()

    if pid in reqs_context[rid].threads_context:
        return

    # Create new thread context.
    reqs_context[rid].threads_context[pid] = __create_thread_context(
        pid,
        trace_context.root_span_context,
        reqs_context[rid].start_time_ns,
    )

    reqs_context[rid].threads_context[pid].last_span_context = trace_context.prev_span_context


def trace_req_start(
    rid: str,
    trace_content: str,
    ts: Optional[int] = None,
    role: Optional[str] = "null",
):
    if not tracing_enabled:
        return

    rid = str(rid)

    ts = ts or __get_cur_time_ns()

    pid = threading.get_native_id()
    if pid not in threads_info:
        return

    tracer = threads_info[pid].tracer

    upstream_context = trace_content

    # 1. Check if there is already an active Span (from FastAPI Instrumentor)
    active_span = trace.get_current_span()
    if active_span is not None and active_span.is_recording():
        active_span.set_attribute("rid", rid)
        new_span_name = active_span.name + f" (Req: {rid})"
        active_span.update_name(new_span_name)

    active_span_context = active_span.get_span_context()

    if active_span_context.is_valid and active_span_context.trace_id != 0:
        # Scenario: FastAPIInstrumentor has created the top-level Span

        if rid in reqs_context:
            return

        logger.info(f"Using existing active span from context as root for RID: {rid}")

        # Inject the FastAPI Span Context as the root Span Context into the internal structure
        reqs_context[rid] = TraceReqContext(
            rid=rid,
            start_time_ns=ts,
            threads_context={},
            root_span=active_span,
            root_span_context=context.get_current(),
            is_copy=True,
        )
        # Thread context is necessary so that trace_slice_start can find the tracer
        if pid not in reqs_context[rid].threads_context:
            reqs_context[rid].threads_context[pid] = __create_thread_context(
                pid,
                context.get_current(),
                ts,
            )
        # No need to manually end req/bootstrap room span, this is handled by FastAPIInstrumentor
        return

    parent_context = None

    use_upstream = False
    if upstream_context:
        ctx_span = trace.get_current_span(upstream_context)
        if ctx_span.get_span_context().is_valid:
            use_upstream = True

    if use_upstream:
        logger.info(f"Continuing upstream trace for RID={rid}")
        parent_context = upstream_context

        reqs_context[rid] = TraceReqContext(
            rid=rid,
            start_time_ns=ts,
            threads_context={},
            is_copy=True,
        )

    else:
        reqs_context[rid] = TraceReqContext(
            rid=rid,
            start_time_ns=ts,
            threads_context={},
            is_copy=False,
        )

    orig_rid = get_base_request_id(rid)
    role = "" if role == "null" else role
    attrs = {"rid": orig_rid}

    root_span = tracer.start_span(
        name=f"{role} Req {orig_rid}".strip(),
        start_time=ts,
        context=parent_context,
        kind=trace.SpanKind.SERVER,
        attributes=attrs,
    )

    root_span.set_attributes(
        {
            "rid": rid,
        }
    )

    # Consistently populate the Root Span information in reqs_context
    reqs_context[rid].root_span = root_span
    reqs_context[rid].root_span_context = trace.set_span_in_context(root_span)

    # create thread context and thread span
    reqs_context[rid].threads_context[pid] = __create_thread_context(
        pid,
        reqs_context[rid].root_span_context,
        ts,
    )


def trace_req_finish(rid: str, ts: Optional[int] = None, attrs: Optional[Dict[str, Any]] = None):
    if not tracing_enabled:
        return

    rid = str(rid)
    if rid not in reqs_context:
        return

    req_context = reqs_context[rid]
    ts = ts or __get_cur_time_ns()

    # End all unclosed thread spans.
    for thread_context in req_context.threads_context.values():
        thread_context.thread_span.end(end_time=ts)

    # Only end the root_span if it was manually created
    if req_context.root_span:
        if attrs:
            req_context.root_span.set_attributes(attrs)
        req_context.root_span.end(end_time=ts)

    del reqs_context[rid]


def trace_slice_start(
    name: str,
    rid: str,
    ts: Optional[int] = None,
    anonymous: bool = False,
):
    if not tracing_enabled:
        return

    rid = str(rid)
    if rid not in reqs_context:
        return

    pid = threading.get_native_id()
    if pid not in reqs_context[rid].threads_context:
        return

    thread_context = reqs_context[rid].threads_context[pid]

    ts = ts or __get_cur_time_ns()

    slice_info = TraceSliceContext(
        slice_name=name,
        anonymous=anonymous,
    )

    # find prev slice
    prev_span_context = None
    if not thread_context.cur_slice_stack:
        if thread_context.last_span_context:
            prev_span_context = thread_context.last_span_context

    parent_span = thread_context.thread_span
    if thread_context.cur_slice_stack:
        parent_span = thread_context.cur_slice_stack[-1].span

    parent_span_context = trace.set_span_in_context(parent_span)
    span = thread_context.thread_info.tracer.start_span(
        name=slice_info.slice_name,
        start_time=ts,
        context=parent_span_context,
    )

    if prev_span_context:
        span.add_link(prev_span_context)

    slice_info.span = span

    thread_context.cur_slice_stack.append(slice_info)


def trace_slice_end(
    name: str,
    rid: str,
    ts: Optional[int] = None,
    attrs: Optional[Dict[str, Any]] = None,
    auto_next_anon: bool = False,
    thread_finish_flag: bool = False,
):
    if not tracing_enabled:
        return

    rid = str(rid)
    if rid not in reqs_context:
        return

    pid = threading.get_native_id()
    if pid not in reqs_context[rid].threads_context:
        return

    thread_context = reqs_context[rid].threads_context[pid]

    if not thread_context.cur_slice_stack:
        logger.warning(f"No matching with the SLICE_START event{name} is required.")
        return

    ts = ts or __get_cur_time_ns()
    slice_info = thread_context.cur_slice_stack[-1]
    span = slice_info.span

    if slice_info.anonymous:
        span.update_name(name)
    else:
        span = slice_info.span
        if slice_info.slice_name != name:
            span.set_status(trace.Status(trace.StatusCode.ERROR))
            logger.warning(f"Slice name mismatch: {name} != {slice_info.slice_name}")

    if attrs:
        span.set_attributes(attrs)

    span.end(end_time=ts)

    thread_context.cur_slice_stack.pop()
    if len(thread_context.cur_slice_stack) == 0:
        thread_context.last_span_context = span.get_span_context()

    # If this is the last slice in the thread,
    # release the thread context and check whether to release the request context.
    if thread_finish_flag:
        thread_context.thread_span.end(end_time=ts)
        del reqs_context[rid].threads_context[pid]
        # Note: Don't delete reqs_context[rid] here, let trace_req_finish do it
        # to ensure trace info is available for the entire request lifecycle.
        return

    if auto_next_anon:
        trace_slice_start("", rid, ts, True)


# alias
trace_slice = trace_slice_end


def trace_report_span(
    name: str,
    rid: str,
    start_time_ns: int,
    end_time_ns: int,
    attrs: Dict[str, Any] = None,
    thread_finish_flag: bool = False,
):
    if not tracing_enabled:
        return
    trace_slice_start(name, rid, start_time_ns)
    trace_slice_end(name, rid, end_time_ns, attrs, False, thread_finish_flag)


# Add event to the current slice on the same thread with the same rid.
def trace_event(name: str, rid: str, ts: Optional[int] = None, attrs: Dict[str, Any] = None):
    if not tracing_enabled:
        return

    rid = str(rid)
    if rid not in reqs_context:
        return

    pid = threading.get_native_id()
    if pid not in reqs_context[rid].threads_context:
        return

    thread_context = reqs_context[rid].threads_context[pid]

    if not thread_context.cur_slice_stack:
        logger.warning("No slice is currently being traced.")
        return

    ts = ts or __get_cur_time_ns()

    slice_info = thread_context.cur_slice_stack[-1]
    slice_info.span.add_event(name=name, timestamp=ts, attributes=attrs)


# Add attrs to the current slice on the same thread with the same rid.
def trace_slice_add_attr(rid: str, attrs: Dict[str, Any]):
    if not tracing_enabled:
        return

    rid = str(rid)
    if rid not in reqs_context:
        return

    pid = threading.get_native_id()
    if pid not in reqs_context[rid].threads_context:
        return

    thread_context = reqs_context[rid].threads_context[pid]

    if not thread_context.cur_slice_stack:
        logger.warning("No slice is currently being traced.")
        return

    slice_info = thread_context.cur_slice_stack[-1]
    slice_info.span.set_attributes(attrs)


def trace_span(span_name: str = None):

    def decorator(func):
        if not tracing_enabled:
            return func

        pid = threading.get_native_id()
        if pid not in threads_info:
            trace_set_thread_info("FastDeploy")

        tracer = threads_info[pid].tracer

        name = span_name or func.__name__

        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                with tracer.start_as_current_span(name):
                    return await func(*args, **kwargs)

            return async_wrapper

        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                with tracer.start_as_current_span(name):
                    return func(*args, **kwargs)

            return sync_wrapper

    return decorator


def get_trace_info_for_request(rid: str) -> Optional[Dict[str, str]]:
    """Get trace_id and span_id for the specified request

    Args:
        rid: Request ID

    Returns:
        Dictionary containing trace_id and span_id, returns None if not found
    """
    if not tracing_enabled:
        return None
    rid = str(rid)
    if rid not in reqs_context:
        # Try using original rid (remove choice index suffix)
        orig_rid = get_base_request_id(rid)
        if orig_rid not in reqs_context:
            return None
        rid = orig_rid

    req_context = reqs_context[rid]

    # First try to get from root_span
    if req_context.root_span:
        span_context = req_context.root_span.get_span_context()
        if span_context.is_valid and span_context.trace_id != 0:
            return {
                "trace_id": format(span_context.trace_id, "032x"),
                "span_id": format(span_context.span_id, "016x"),
            }

    # If restored from other process context, get trace_id from root_span_context
    if req_context.root_span_context:
        # Extract span from Context
        from opentelemetry.trace import get_current_span

        try:
            span = get_current_span(req_context.root_span_context)
            span_context = span.get_span_context()
            if span_context.is_valid and span_context.trace_id != 0:
                return {
                    "trace_id": format(span_context.trace_id, "032x"),
                    "span_id": format(span_context.span_id, "016x"),
                }
        except:
            pass

    return None


@unique
class TraceSpanName(str, Enum):

    FASTDEPLOY = "FASTDEPLOY"
    PREPROCESSING = "PREPROCESSING"
    SCHEDULE = "SCHEDULE"
    PREFILL = "PREFILL"
    DECODE = "DECODE"
    POSTPROCESSING = "POSTPROCESSING"
