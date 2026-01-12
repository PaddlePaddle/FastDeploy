package metrics

import (
	"strings"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
)

func TestMetricsInitialization(t *testing.T) {
	// Verify all metrics are registered
	metrics := []prometheus.Collector{
		TotalRequests,
		InferenceRequests,
		RequestDuration,
	}

	for _, metric := range metrics {
		if err := prometheus.Register(metric); err == nil {
			t.Errorf("Metric %T should already be registered", metric)
		}
	}
}

func TestMetricsHelpText(t *testing.T) {
	tests := []struct {
		name     string
		metric   prometheus.Collector
		expected string
	}{
		{"TotalRequests", TotalRequests, "Total number of HTTP requests"},
		{"InferenceRequests", InferenceRequests, "Total number of inference requests"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			desc := make(chan *prometheus.Desc, 1)
			tt.metric.Describe(desc)
			d := <-desc
			if !strings.Contains(d.String(), tt.expected) {
				t.Errorf("Expected help text to contain '%s', got '%s'", tt.expected, d.String())
			}
		})
	}
}
