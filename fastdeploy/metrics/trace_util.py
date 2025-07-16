from opentelemetry.propagate import inject, extract
from opentelemetry import trace

import json
import os

# create global OpenTelemetry tracer
tracer = trace.get_tracer(__name__)

# OpenTelemetry Trace context store in metadata
TRACE_CARRIER = "trace_carrier"

def inject_to_metadata(request, metadata_attr='metadata'):
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
        if request is None:
            return
        if is_opentelemetry_instrumented() == False:
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

def extract_from_metadata(request, metadata_attr='metadata'):
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
        if is_opentelemetry_instrumented() == False:
            return
        # extract Trace context from request.metadata.trace_carrier
        ctx = extract_from_metadata(request)
        with tracer.start_as_current_span(span_name, context=ctx, kind=kind) as span:
            pass
    except:
        pass

def start_span_request(span_name, request, kind=trace.SpanKind.CLIENT):
    """
        just start a new span in request trace context
    """
    try:
        if is_opentelemetry_instrumented() == False:
            return
        # extract Trace context from request.metadata.trace_carrier
        ctx = extract_from_request(request)
        with tracer.start_as_current_span(span_name, context=ctx, kind=kind) as span:
            pass
    except:
        pass

def is_opentelemetry_instrumented() -> bool:
    """
        check OpenTelemetry is start or not
    """
    try:
        return (
            os.getenv("OTEL_PYTHONE_DISABLED_INSTRUMENTATIONS") is not None
            or os.getenv("OTEL_SERVICE_NAME") is not None
            or os.getenv("OTEL_TRACES_EXPORTER") is not None
        )
    except Exception:
        return False
    
    
