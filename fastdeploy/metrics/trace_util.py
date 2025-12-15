import json
import os

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.propagate import extract, inject
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SpanExporter,
)

from fastdeploy import envs
from fastdeploy.utils import llm_logger

# OpenTelemetry Trace context store in metadata
TRACE_CARRIER = "trace_carrier"

traces_enable = False
tracer = trace.get_tracer(__name__)


class FilteringSpanProcessor(SpanProcessor):
    def __init__(self, exporter: SpanExporter):
        self._processor = BatchSpanProcessor(exporter)

    # 父span属性继承逻辑
    def on_start(self, span, parent_context=None):
        parent_span = trace.get_current_span()
        if parent_span and parent_span.is_recording():
            stream_attr = parent_span.attributes.get("stream")
            if stream_attr is not None:
                span.set_attribute("stream", stream_attr)
        self._processor.on_start(span, parent_context)

    # span导出时的过滤逻辑
    def on_end(self, span):
        asgi_event_type = span.attributes.get("asgi.event.type")
        stream = span.attributes.get("stream")
        span_name = span.name or ""

        if stream and asgi_event_type == "http.response.body" and "http send" in span_name:
            return

        self._processor.on_end(span)

    def shutdown(self):
        self._processor.shutdown()

    def force_flush(self, timeout_millis=None):
        self._processor.force_flush(timeout_millis)


# 标记函数
def lable_span(request):
    if request.stream:
        span = trace.get_current_span()
        if span is not None and span.is_recording():
            span.set_attribute("stream", "true")


def set_up():
    try:
        # when TRACES_ENABLED=true start trace
        global traces_enable
        traces_enable = envs.TRACES_ENABLE.lower() == "true"
        if not traces_enable:
            llm_logger.warning("Opentelemetry is DISABLED.")
            return

        llm_logger.info("Opentelemetry is ENABLED, configuring...")
        # --- read env ---
        service_name = envs.FD_SERVICE_NAME
        host_name = envs.FD_HOST_NAME
        # --- set attributes (Service Name, Host Name, etc.) ---
        resource_attributes = {"service.name": service_name}
        if host_name:
            resource_attributes["host.name"] = host_name

        resource = Resource(attributes=resource_attributes)

        # --- set Exporter ---
        exporter_type = envs.TRACES_EXPORTER.lower()
        if exporter_type == "otlp":
            endpoint = envs.EXPORTER_OTLP_ENDPOINT  # should be set
            headers = envs.EXPORTER_OTLP_HEADERS  # e.g., "Authentication=***,k2=v2"

            otlp_exporter = OTLPSpanExporter(
                endpoint=endpoint,
                headers=(dict(item.split("=") for item in headers.split(",")) if headers else None),
            )
            processor = FilteringSpanProcessor(otlp_exporter)
            llm_logger.info(f"Using OTLP Exporter, sending to {endpoint} with headers {headers}")
        else:  # default console
            processor = FilteringSpanProcessor(ConsoleSpanExporter())
            llm_logger.info("Using Console Exporter.")

        # --- set Tracer Provider ---
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        global tracer
        tracer = trace.get_tracer(__name__)
    except:
        llm_logger.error("set_up failed")
        pass


def instrument(app: FastAPI):
    try:
        set_up()
        if traces_enable:
            llm_logger.info("Applying instrumentors...")
            FastAPIInstrumentor.instrument_app(app)
            try:
                LoggingInstrumentor().instrument(set_logging_format=True)
            except Exception:
                pass
    except:
        llm_logger.info("instrument failed")
        pass


def inject_to_metadata(request, metadata_attr="metadata"):
    """
    Inject OpenTelemetry trace context into the metadata field of the request.

    Parameters:
    request: can be a dict or object, with metadata attributes or fields.
    metadata_attr: the field name of metadata, default is 'metadata'.

    Operation:
    - If metadata does not exist, create a new one and mount it on the request.
    - Inject the current trace context as a JSON string and store it in metadata.
    - Use the key TRACE_CARRIER to store the injected content.

    Note:
    - This function is a non-blocking operation, and errors are silently ignored.
    - If there is no metadata attribute in the request, an empty dict will be created for it as its attribute
    """
    try:
        if request is None or not traces_enable:
            return

        metadata = request.get(metadata_attr) if isinstance(request, dict) else getattr(request, metadata_attr, None)
        if metadata is None:
            metadata = {}
            if isinstance(request, dict):
                request[metadata_attr] = metadata
            else:
                setattr(request, metadata_attr, metadata)

        trace_carrier = {}
        inject(trace_carrier)
        trace_carrier_json_string = json.dumps(trace_carrier)
        metadata[TRACE_CARRIER] = trace_carrier_json_string
    except:
        pass


def extract_from_metadata(request, metadata_attr="metadata"):
    """
    Extract trace context from metadata of request object (dict or class instance).

    Parameters:
    request: can be a dictionary or any object, containing metadata attributes or fields.
    metadata_attr: metadata field name, default is 'metadata'.

    Returns:
    - Extraction success: returns OpenTelemetry context object (Context)
    - Extraction failure or exception: returns None
    """
    try:
        metadata = request.get(metadata_attr) if isinstance(request, dict) else getattr(request, metadata_attr, None)
        if metadata is None:
            return None

        trace_carrier_json_string = metadata.get(TRACE_CARRIER)
        if trace_carrier_json_string is None:
            return None

        trace_carrier = json.loads(trace_carrier_json_string)
        ctx = extract(trace_carrier)
        return ctx
    except:
        return None


def extract_from_request(request):
    """
    Extract trace context from trace_carrier of request object (dict or class instance).

    Parameters:
    request: can be a dictionary or any object, containing metadata attributes or fields.
    metadata_attr: metadata field name, default is 'metadata'.

    Returns:
    - Extraction success: returns OpenTelemetry context object (Context)
    - Extraction failure or exception: returns None
    """
    try:
        trace_carrier_info = getattr(request, TRACE_CARRIER, None)

        if trace_carrier_info is None:
            return None

        trace_carrier = json.loads(trace_carrier_info)
        ctx = extract(trace_carrier)
        return ctx
    except:
        return None


def start_span(span_name, request, kind=trace.SpanKind.CLIENT):
    """
    just start a new span in request trace context
    """
    try:
        if not traces_enable:
            return
        # extract Trace context from request.metadata.trace_carrier
        ctx = extract_from_metadata(request)
        with tracer.start_as_current_span(span_name, context=ctx, kind=kind) as span:
            span.set_attribute("job_id", os.getenv("FD_JOB_ID", default="null"))
            pass
    except:
        pass


def fd_start_span(span_name, kind=trace.SpanKind.CLIENT):
    """
    when fd start, start a new span show start success
    """
    try:
        if not traces_enable:
            return
        with tracer.start_as_current_span(span_name, kind=kind) as span:
            span.set_attribute("job_id", os.getenv("FD_JOB_ID", default="null"))
            pass
    except:
        pass


def start_span_request(span_name, request, kind=trace.SpanKind.CLIENT):
    """
    just start a new span in request trace context
    """
    try:
        if not traces_enable:
            return
        # extract Trace context from request.metadata.trace_carrier
        ctx = extract_from_request(request)
        with tracer.start_as_current_span(span_name, context=ctx, kind=kind) as span:
            span.set_attribute("job_id", os.getenv("FD_JOB_ID", default="null"))
            pass
    except:
        pass
