package manager

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/PaddlePaddle/FastDeploy/router/internal/config"
	"github.com/PaddlePaddle/FastDeploy/router/pkg/logger"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
)

func init() {
	// Initialize logger for all tests
	logger.Init("info", "stdout")
}

func TestCheckServiceHealth(t *testing.T) {
	// Setup test server
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer ts.Close()

	t.Run("healthy service", func(t *testing.T) {
		healthy := CheckServiceHealth(context.Background(), ts.URL)
		assert.True(t, healthy)
	})

	t.Run("unhealthy service", func(t *testing.T) {
		unhealthyServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusInternalServerError)
		}))
		defer unhealthyServer.Close()

		healthy := CheckServiceHealth(context.Background(), unhealthyServer.URL)
		assert.False(t, healthy)
	})

	t.Run("empty baseURL", func(t *testing.T) {
		healthy := CheckServiceHealth(context.Background(), "")
		assert.False(t, healthy)
	})

	t.Run("timeout", func(t *testing.T) {
		slowServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			time.Sleep(100 * time.Millisecond)
			w.WriteHeader(http.StatusOK)
		}))
		defer slowServer.Close()

		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
		defer cancel()

		healthy := CheckServiceHealth(ctx, slowServer.URL)
		assert.False(t, healthy)
	})
}

func TestCheckWorkerHealth(t *testing.T) {
	// Setup test data
	Init(&config.Config{
		Manager: config.ManagerConfig{
			HealthFailureThreshold: 1,
			HealthSuccessThreshold: 1,
		},
	})

	// Setup test server
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer ts.Close()

	t.Run("healthy worker", func(t *testing.T) {
		healthy := CheckWorkerHealth(context.Background(), ts.URL)
		assert.True(t, healthy)
	})

	t.Run("unhealthy worker", func(t *testing.T) {
		unhealthyServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusInternalServerError)
		}))
		defer unhealthyServer.Close()

		healthy := CheckWorkerHealth(context.Background(), unhealthyServer.URL)
		assert.False(t, healthy)
	})
}

func TestHealthGenerate(t *testing.T) {
	// Setup test server
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer ts.Close()

	// Setup test data
	Init(&config.Config{})
	DefaultManager.prefillWorkerMap = map[string]*WorkerInfo{
		"worker1": {Url: ts.URL},
	}
	DefaultManager.decodeWorkerMap = map[string]*WorkerInfo{
		"worker2": {Url: ts.URL},
	}

	// Test Gin handler
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	// Set up a valid HTTP request for the context
	c.Request = httptest.NewRequest("GET", "/health", nil)
	HealthGenerate(c)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Contains(t, w.Body.String(), "Health check complete")
}

func TestMonitorInstanceHealthCore(t *testing.T) {
	// Setup test server
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer ts.Close()

	// Setup test data
	Init(&config.Config{})
	DefaultManager.prefillWorkerMap = map[string]*WorkerInfo{
		"worker1": {Url: ts.URL, WorkerType: "prefill"},
	}
	DefaultManager.decodeWorkerMap = map[string]*WorkerInfo{
		"worker2": {Url: ts.URL, WorkerType: "decode"},
	}

	MonitorInstanceHealthCore(context.Background())

	// Verify workers still exist (since they're healthy)
	_, exists := DefaultManager.prefillWorkerMap["worker1"]
	assert.True(t, exists)
	_, exists = DefaultManager.decodeWorkerMap["worker2"]
	assert.True(t, exists)
}

func TestReadServers(t *testing.T) {
	// Setup test data
	Init(&config.Config{})
	DefaultManager.prefillWorkerMap = map[string]*WorkerInfo{
		"worker1": {Url: "http://worker1"},
	}
	DefaultManager.decodeWorkerMap = map[string]*WorkerInfo{
		"worker2": {Url: "http://worker2"},
	}

	prefill, decode, mixed := ReadServers(context.Background())
	assert.Equal(t, []string{"http://worker1"}, prefill)
	assert.Equal(t, []string{"http://worker2"}, decode)
	assert.Equal(t, []string{}, mixed)
}
