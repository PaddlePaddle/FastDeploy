package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
)

func init() {
	prometheus.MustRegister(TotalRequests)
	prometheus.MustRegister(InferenceRequests)
	prometheus.MustRegister(RequestDuration)
	prometheus.MustRegister(RouterCacheHitTotal)
	prometheus.MustRegister(RouterCacheRequestTotal)
}

// TotalRequests tracks the total number of HTTP requests
var TotalRequests = prometheus.NewCounterVec(
	prometheus.CounterOpts{
		Name: "http_requests_total",
		Help: "Total number of HTTP requests",
	},
	[]string{"method", "endpoint", "status_code"},
)

// InferenceRequests tracks the total number of inference requests
var InferenceRequests = prometheus.NewCounterVec(
	prometheus.CounterOpts{
		Name: "inference_requests_total",
		Help: "Total number of inference requests",
	},
	[]string{"mixed_worker", "prefill_worker", "decode_worker", "status_code"},
)

// RequestDuration tracks the response latency of HTTP requests
var RequestDuration = prometheus.NewSummaryVec(
	prometheus.SummaryOpts{
		Name:       "http_request_duration_seconds",
		Help:       "Summary of the response latency (seconds) of HTTP requests",
		Objectives: map[float64]float64{0.95: 0.01, 0.99: 0.01}, // Objectives define the required quantiles
	},
	[]string{"method", "endpoint"},
)

// RouterCacheHitTotal tracks the cumulative number of cache hits (session routed to same prefill worker)
var RouterCacheHitTotal = prometheus.NewCounter(
	prometheus.CounterOpts{
		Name: "router_cache_hit_total",
		Help: "Cumulative number of cache hits (same session_id routed to same prefill worker)",
	},
)

// RouterCacheRequestTotal tracks the cumulative number of cache-aware requests with session_id
var RouterCacheRequestTotal = prometheus.NewCounter(
	prometheus.CounterOpts{
		Name: "router_cache_request_total",
		Help: "Cumulative number of cache-aware scheduling requests with session_id",
	},
)
