"""OpenTelemetry tracing and metrics initialization."""
from typing import Any

_tracer = None
_meter = None


def init_telemetry(
    service_name: str = "moleculeforge",
    endpoint: str = "http://localhost:4317",
):
    """Initialize OpenTelemetry tracing and metrics.

    Configures OTLP exporters for both traces and metrics. Falls back
    to no-op implementations if the OpenTelemetry SDK is not installed.

    Args:
        service_name: Name of the service for telemetry attribution.
        endpoint: OTLP collector endpoint.
    """
    global _tracer, _meter
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
            OTLPMetricExporter,
        )

        # Tracing setup
        provider = TracerProvider()
        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer(service_name)

        # Metrics setup
        metric_provider = MeterProvider()
        _meter = metric_provider.get_meter(service_name)
    except ImportError:
        _tracer = _NoopTracer()
        _meter = _NoopMeter()


def get_tracer() -> Any:
    """Get the current OpenTelemetry tracer.

    Initializes telemetry with defaults if not already configured.

    Returns:
        Tracer instance (real or no-op).
    """
    if _tracer is None:
        init_telemetry()
    return _tracer


def get_meter() -> Any:
    """Get the current OpenTelemetry meter.

    Initializes telemetry with defaults if not already configured.

    Returns:
        Meter instance (real or no-op).
    """
    if _meter is None:
        init_telemetry()
    return _meter


class _NoopTracer:
    """No-op tracer for when OpenTelemetry is not installed."""

    def start_span(self, *args, **kwargs):
        return _NoopSpan()

    def start_as_current_span(self, *args, **kwargs):
        return _NoopSpan()


class _NoopSpan:
    """No-op span that implements context manager protocol."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None  # No-op: do not suppress exceptions

    def set_attribute(self, *args):
        return self  # noop: no tracing backend

    def add_event(self, *args):
        return self  # noop: no tracing backend


class _NoopMeter:
    """No-op meter for when OpenTelemetry is not installed."""

    def create_counter(self, *args, **kwargs):
        return _NoopCounter()

    def create_histogram(self, *args, **kwargs):
        return _NoopHistogram()


class _NoopCounter:
    """No-op counter."""

    def add(self, *args, **kwargs):
        return self  # noop: no metrics backend


class _NoopHistogram:
    """No-op histogram."""

    def record(self, *args, **kwargs):
        return self  # noop: no metrics backend
