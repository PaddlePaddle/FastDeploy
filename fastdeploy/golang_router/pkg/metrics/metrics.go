package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
)

func init() {
	prometheus.MustRegister(TotalRequests)
	prometheus.MustRegister(InferenceRequests)
	prometheus.MustRegister(RequestDuration)
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
