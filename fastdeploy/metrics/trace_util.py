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
       将 OpenTelemetry 的 trace context 注入到 request 的 metadata 字段中。

       参数:
           request: 可为 dict 或对象，需有 metadata 属性或字段。
           metadata_attr: metadata 的字段名，默认是 'metadata'。

       操作:
           - 若 metadata 不存在，则新建并挂载到 request 上。
           - 将当前 trace context 注入为 JSON 字符串形式存储到 metadata 中。
           - 使用键 TRACE_CARRIER 存储注入内容。

       注意:
           - 此函数为非阻塞操作，出错时静默忽略。
           - 如果 request 中没有 metadata 属性，会给它创建一个空dict作为它的属性
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
        从 request 对象(dict 或类实例)的 metadata 中提取 trace context。

        参数:
            request: 可以是字典或任意对象，包含 metadata 属性或字段。
            metadata_attr: metadata 字段名，默认是 'metadata'。

        返回:
            - 提取成功：返回 OpenTelemetry 上下文对象（Context）
            - 提取失败或异常：返回 None
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
    
    
