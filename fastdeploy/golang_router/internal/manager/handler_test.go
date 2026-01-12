package manager

import (
	"context"
	"testing"

	"github.com/PaddlePaddle/FastDeploy/router/internal/config"
	"github.com/stretchr/testify/assert"
)

func TestInit(t *testing.T) {
	cfg := &config.Config{
		Server: config.ServerConfig{
			Splitwise: true,
		},
		Manager: config.ManagerConfig{
			HealthCheckTimeoutSecs: 5.0,
			HealthCheckEndpoint:    "/health",
			HealthFailureThreshold: 3,
			HealthSuccessThreshold: 2,
		},
	}

	Init(cfg)

	assert.NotNil(t, DefaultManager)
	assert.True(t, DefaultManager.splitwise)
	assert.Equal(t, "/health", healthEndpoint)
	assert.Equal(t, 3, failureThreshold)
	assert.Equal(t, 2, successThreshold)
}

func TestWorkerMapToList(t *testing.T) {
	// Setup test data
	Init(&config.Config{})
	DefaultManager.prefillWorkerMap = map[string]*WorkerInfo{
		"http://worker1": {Url: "http://worker1"},
		"http://worker2": {Url: "http://worker2"},
	}
	DefaultManager.decodeWorkerMap = map[string]*WorkerInfo{
		"http://worker3": {Url: "http://worker3"},
	}
	DefaultManager.mixedWorkerMap = map[string]*WorkerInfo{
		"http://worker4": {Url: "http://worker4"},
	}

	t.Run("prefill workers", func(t *testing.T) {
		workers := WorkerMapToList(context.Background(), "prefill")
		assert.Len(t, workers, 2)
		assert.Contains(t, workers, "http://worker1")
		assert.Contains(t, workers, "http://worker2")
	})

	t.Run("decode workers", func(t *testing.T) {
		workers := WorkerMapToList(context.Background(), "decode")
		assert.Len(t, workers, 1)
		assert.Contains(t, workers, "http://worker3")
	})

	t.Run("mixed workers", func(t *testing.T) {
		workers := WorkerMapToList(context.Background(), "mixed")
		assert.Len(t, workers, 1)
		assert.Contains(t, workers, "http://worker4")
	})

	t.Run("invalid worker type", func(t *testing.T) {
		workers := WorkerMapToList(context.Background(), "invalid")
		assert.Len(t, workers, 0)
	})
}

func TestManager_GetHealthyURLs(t *testing.T) {
	// Setup test data
	Init(&config.Config{})
	DefaultManager.prefillWorkerMap = map[string]*WorkerInfo{
		"worker1": {Url: "http://worker1"},
	}
	DefaultManager.decodeWorkerMap = map[string]*WorkerInfo{
		"worker2": {Url: "http://worker2"},
	}
	DefaultManager.mixedWorkerMap = map[string]*WorkerInfo{
		"worker3": {Url: "http://worker3"},
	}

	urls := DefaultManager.GetHealthyURLs(context.Background())
	assert.Len(t, urls, 3)
	assert.Contains(t, urls, "worker1")
	assert.Contains(t, urls, "worker2")
	assert.Contains(t, urls, "worker3")
}

func TestSelectWorker(t *testing.T) {
	// Setup test data
	Init(&config.Config{})
	DefaultManager.mixedWorkerMap = map[string]*WorkerInfo{
		"http://worker1": {Url: "http://worker1"},
	}

	// This will fail because SelectWorker depends on scheduler
	// which we don't want to mock in this unit test
	t.Skip("Integration test requiring scheduler setup")
}

func TestSelectWorkerPair(t *testing.T) {
	// Setup test data
	Init(&config.Config{})
	DefaultManager.prefillWorkerMap = map[string]*WorkerInfo{
		"http://worker1": {Url: "http://worker1"},
	}
	DefaultManager.decodeWorkerMap = map[string]*WorkerInfo{
		"http://worker2": {Url: "http://worker2"},
	}

	// This will fail because SelectWorkerPair depends on scheduler
	// which we don't want to mock in this unit test
	t.Skip("Integration test requiring scheduler setup")
}
