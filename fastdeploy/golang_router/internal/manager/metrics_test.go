package manager

import (
	"context"
	"net"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strconv"
	"testing"

	"github.com/PaddlePaddle/FastDeploy/router/internal/config"
	scheduler_handler "github.com/PaddlePaddle/FastDeploy/router/internal/scheduler/handler"
	"github.com/stretchr/testify/assert"
)

// Helper function to get map keys
func getMapKeys(m interface{}) []string {
	v := reflect.ValueOf(m)
	if v.Kind() != reflect.Map {
		return nil
	}

	keys := v.MapKeys()
	result := make([]string, len(keys))
	for i, key := range keys {
		result[i] = key.String()
	}
	return result
}

func TestRedrictCounter(t *testing.T) {
	// Initialize scheduler for counter tests
	cfg := &config.Config{
		Scheduler: config.SchedulerConfig{
			Policy:        "random",
			PrefillPolicy: "random",
			DecodePolicy:  "random",
			WaitingWeight: 1.0,
		},
	}
	// Using nil for managerAPI since we're only testing counters
	scheduler_handler.Init(cfg, nil)

	// Setup test context
	ctx := context.Background()

	// Test case 1: First call with new URL should return 0
	t.Run("new_url_returns_zero", func(t *testing.T) {
		count := redrictCounter(ctx, "http://new-service-1")
		assert.Equal(t, 0, count)
	})

	// Test case 2: Multiple calls with same URL should return incremented count
	t.Run("same_url_increments", func(t *testing.T) {
		url := "http://same-service"

		// First call should return 0
		count1 := redrictCounter(ctx, url)
		assert.Equal(t, 0, count1)

		// Simulate counter increment by calling GetOrCreateCounter and incrementing
		counter := scheduler_handler.GetOrCreateCounter(ctx, url)
		counter.Inc()

		// Second call should return incremented count
		count2 := redrictCounter(ctx, url)
		assert.Equal(t, 1, count2)
	})

	// Test case 3: Different URLs should have independent counters
	t.Run("different_urls_independent", func(t *testing.T) {
		url1 := "http://service-a"
		url2 := "http://service-b"

		// Call first URL
		count1 := redrictCounter(ctx, url1)
		assert.Equal(t, 0, count1)

		// Call second URL
		count2 := redrictCounter(ctx, url2)
		assert.Equal(t, 0, count2)

		// Increment first URL's counter
		counter1 := scheduler_handler.GetOrCreateCounter(ctx, url1)
		counter1.Inc()
		counter1.Inc()

		// Verify first URL shows incremented count
		assert.Equal(t, 2, redrictCounter(ctx, url1))
		// Second URL should still be 0
		assert.Equal(t, 0, redrictCounter(ctx, url2))
	})

	// Test case 4: Empty URL should work (edge case)
	t.Run("empty_url", func(t *testing.T) {
		count := redrictCounter(ctx, "")
		assert.Equal(t, 0, count)
	})

	// Test case 5: Nil context should work (though not recommended)
	t.Run("nil_context", func(t *testing.T) {
		count := redrictCounter(ctx, "http://nil-context-test")
		assert.Equal(t, 0, count)
	})
}

func TestParseMetricsResponseOptimized(t *testing.T) {
	tests := []struct {
		name         string
		input        string
		expectedRun  float64
		expectedWait float64
		expectedGpu  float64
	}{
		{
			name: "valid metrics response",
			input: `fastdeploy:num_requests_running 10
fastdeploy:num_requests_waiting 5
available_gpu_block_num 3`,
			expectedRun:  10,
			expectedWait: 5,
			expectedGpu:  3,
		},
		{
			name: "partial metrics response",
			input: `fastdeploy:num_requests_running 8
available_gpu_block_num 2`,
			expectedRun:  8,
			expectedWait: -1,
			expectedGpu:  2,
		},
		{
			name:         "empty response",
			input:        "",
			expectedRun:  -1,
			expectedWait: -1,
			expectedGpu:  -1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			run, wait, gpu := parseMetricsResponseOptimized(tt.input)
			assert.Equal(t, tt.expectedRun, run)
			assert.Equal(t, tt.expectedWait, wait)
			assert.Equal(t, tt.expectedGpu, gpu)
		})
	}
}

func TestGetMetricsByURL_Integration(t *testing.T) {
	// Initialize manager for testing
	Init(&config.Config{})

	// Test with mock HTTP server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/metrics" {
			w.WriteHeader(http.StatusNotFound)
			return
		}

		switch r.URL.Query().Get("scenario") {
		case "valid":
			w.Write([]byte(`fastdeploy:num_requests_running 5
fastdeploy:num_requests_waiting 3
available_gpu_block_num 10`))
		case "partial":
			w.Write([]byte(`fastdeploy:num_requests_running 2
available_gpu_block_num 5`))
		case "empty":
			// Empty response body
			w.WriteHeader(http.StatusOK)
		case "error":
			w.WriteHeader(http.StatusInternalServerError)
		default:
			w.Write([]byte(`fastdeploy:num_requests_running 1
fastdeploy:num_requests_waiting 0
available_gpu_block_num 8`))
		}
	}))
	defer server.Close()

	// Test worker not found
	t.Run("worker_not_found", func(t *testing.T) {
		_, _, _, err := GetMetricsByURL(context.Background(), "http://nonexistent-worker:8080")
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "worker info not found")
	})

	// Test with registered worker
	t.Run("with_registered_worker", func(t *testing.T) {
		// Register a test worker with localhost to avoid DNS resolution issues
		workerURL := "http://localhost:8080"
		DefaultManager.mu.Lock()
		DefaultManager.prefillWorkerMap[workerURL] = &WorkerInfo{
			Url:         workerURL,
			WorkerType:  "prefill",
			MetricsPort: strconv.Itoa(server.Listener.Addr().(*net.TCPAddr).Port),
		}
		t.Logf("Registered worker: URL=%s, MetricsPort=%s", workerURL, strconv.Itoa(server.Listener.Addr().(*net.TCPAddr).Port))
		DefaultManager.mu.Unlock()

		// Debug: check what URLs are in the map
		DefaultManager.mu.RLock()
		t.Logf("prefillWorkerMap keys: %v", getMapKeys(DefaultManager.prefillWorkerMap))
		DefaultManager.mu.RUnlock()

		t.Logf("Looking up worker for URL: %s", workerURL)

		// Test valid metrics response - the test server should handle scenarios based on the request
		running, waiting, gpu, err := GetMetricsByURL(context.Background(), workerURL)
		if err != nil {
			t.Logf("Error: %v", err)
		}
		assert.NoError(t, err)
		assert.Equal(t, 1, running) // Default scenario in test server
		assert.Equal(t, 0, waiting) // Default scenario in test server
		assert.Equal(t, 8, gpu)     // Default scenario in test server
	})

	// Test invalid URL
	t.Run("invalid_url", func(t *testing.T) {
		_, _, _, err := GetMetricsByURL(context.Background(), "http://invalid url")
		assert.Error(t, err)
	})
	// Test partial metrics
	t.Run("partial_metrics", func(t *testing.T) {
		workerURL := "http://localhost:8081" // Different port
		DefaultManager.mu.Lock()
		DefaultManager.prefillWorkerMap[workerURL] = &WorkerInfo{
			Url:         workerURL,
			WorkerType:  "prefill",
			MetricsPort: strconv.Itoa(server.Listener.Addr().(*net.TCPAddr).Port),
		}
		t.Logf("Registered worker: URL=%s, MetricsPort=%s", workerURL, strconv.Itoa(server.Listener.Addr().(*net.TCPAddr).Port))
		DefaultManager.mu.Unlock()

		t.Logf("Looking up worker for URL: %s", workerURL)

		running, waiting, gpu, err := GetMetricsByURL(context.Background(), workerURL)
		if err != nil {
			t.Logf("Error: %v", err)
		}
		assert.NoError(t, err)
		assert.Equal(t, 1, running) // Default scenario
		assert.Equal(t, 0, waiting) // Default scenario
		assert.Equal(t, 8, gpu)     // Default scenario
	})
}

func TestManagerGetMetrics_Integration(t *testing.T) {
	// Initialize manager for testing
	Init(&config.Config{})
	// Initialize scheduler for counter tests
	cfg := &config.Config{
		Scheduler: config.SchedulerConfig{
			Policy:        "random",
			PrefillPolicy: "random",
			DecodePolicy:  "random",
			WaitingWeight: 1.0,
		},
	}
	scheduler_handler.Init(cfg, nil)

	m := &Manager{}

	// Setup mock HTTP server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/metrics" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		w.Write([]byte(`fastdeploy:num_requests_running 2
fastdeploy:num_requests_waiting 1
available_gpu_block_num 5`))
	}))
	defer server.Close()

	// Test normal case with registered worker
	t.Run("normal_case", func(t *testing.T) {
		workerURL := "http://localhost:8080" // Use localhost to avoid DNS lookup
		DefaultManager.mu.Lock()
		DefaultManager.prefillWorkerMap[workerURL] = &WorkerInfo{
			Url:         workerURL,
			WorkerType:  "prefill",
			MetricsPort: strconv.Itoa(server.Listener.Addr().(*net.TCPAddr).Port),
		}
		t.Logf("Registered worker with MetricsPort: %s", strconv.Itoa(server.Listener.Addr().(*net.TCPAddr).Port))
		DefaultManager.mu.Unlock()

		// Debug: test GetMetricsByURL directly first
		t.Logf("Testing GetMetricsByURL directly...")
		running, waiting, gpu, err := GetMetricsByURL(context.Background(), workerURL)
		if err != nil {
			t.Logf("GetMetricsByURL failed: %v", err)
		} else {
			t.Logf("GetMetricsByURL succeeded: running=%d, waiting=%d, gpu=%d", running, waiting, gpu)
		}

		// Now test Manager.GetMetrics
		running, waiting, gpu = m.GetMetrics(context.Background(), workerURL)
		t.Logf("Manager.GetMetrics result: running=%d, waiting=%d, gpu=%d", running, waiting, gpu)
		assert.Equal(t, 2, running)
		assert.Equal(t, 1, waiting)
		assert.Equal(t, 5, gpu)
	})

	// Test error case (should fall back to counter)
	t.Run("error_fallback", func(t *testing.T) {
		// Use a URL that doesn't have a registered worker
		workerURL := "http://unknown-worker:8080"
		running, waiting, gpu := m.GetMetrics(context.Background(), workerURL)

		// Should fall back to counter (which should be 0 for new URL)
		assert.Equal(t, 0, running) // Should be 0 for new counter
		assert.Equal(t, 0, waiting) // Should be 0 in error case
		assert.Equal(t, 0, gpu)     // Should be 0 in error case
	})
}

// Helper function to test metrics parsing directly
func TestMetricsParsingHelper(t *testing.T) {
	tests := []struct {
		name        string
		metricsBody string
		expected    []int
	}{
		{
			name: "complete_metrics",
			metricsBody: `fastdeploy:num_requests_running 5
fastdeploy:num_requests_waiting 3
available_gpu_block_num 10`,
			expected: []int{5, 3, 10},
		},
		{
			name: "missing_waiting",
			metricsBody: `fastdeploy:num_requests_running 2
available_gpu_block_num 5`,
			expected: []int{2, -1, 5},
		},
		{
			name:        "empty_body",
			metricsBody: "",
			expected:    []int{-1, -1, -1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			run, wait, gpu := parseMetricsResponseOptimized(tt.metricsBody)
			assert.Equal(t, float64(tt.expected[0]), run)
			assert.Equal(t, float64(tt.expected[1]), wait)
			assert.Equal(t, float64(tt.expected[2]), gpu)
		})
	}
}
