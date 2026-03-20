package handler

import (
	"context"
	"testing"
	"time"

	"github.com/PaddlePaddle/FastDeploy/router/internal/config"
	"github.com/stretchr/testify/assert"
)

type mockManagerAPI struct{}

func (m *mockManagerAPI) GetHealthyURLs(ctx context.Context) []string {
	return []string{"worker1", "worker2"}
}

func (m *mockManagerAPI) GetMetrics(ctx context.Context, url string) (int, int, int) {
	return 0, 0, 0 // 返回默认值用于测试
}

func (m *mockManagerAPI) GetRemoteMetrics(ctx context.Context, url string) (int, int, int) {
	return 0, 0, 0 // 返回默认值用于测试
}

func TestSchedulerInit(t *testing.T) {
	cfg := &config.Config{
		Scheduler: config.SchedulerConfig{
			Policy:        "random",
			PrefillPolicy: "process_tokens",
			DecodePolicy:  "request_num",
		},
	}

	Init(cfg, &mockManagerAPI{})

	assert.NotNil(t, DefaultScheduler)
	assert.Equal(t, "random", DefaultScheduler.policy)
	assert.Equal(t, "process_tokens", DefaultScheduler.prefillPolicy)
	assert.Equal(t, "request_num", DefaultScheduler.decodePolicy)
}

func TestSelectWorker(t *testing.T) {
	ctx := context.Background()
	workers := []string{"worker1", "worker2", "worker3"}

	Init(&config.Config{
		Scheduler: config.SchedulerConfig{
			Policy:        "random",
			PrefillPolicy: "process_tokens",
			DecodePolicy:  "request_num",
		},
	}, &mockManagerAPI{})

	t.Run("prefill worker selection", func(t *testing.T) {
		// Set up token counts
		tc1 := GetOrCreateTokenCounter(ctx, "worker1")
		tc1.Add(100)
		tc2 := GetOrCreateTokenCounter(ctx, "worker2")
		tc2.Add(50) // Should be selected
		tc3 := GetOrCreateTokenCounter(ctx, "worker3")
		tc3.Add(200)

		selected, err := SelectWorker(ctx, workers, "test message", "prefill")
		assert.NoError(t, err)
		assert.Equal(t, "http://worker2", selected)
	})

	t.Run("decode worker selection", func(t *testing.T) {
		// Set up request counts
		c1 := GetOrCreateCounter(ctx, "worker1")
		c1.Inc()
		c1.Inc()                                 // count = 2
		c2 := GetOrCreateCounter(ctx, "worker2") // count = 0 (should be selected)
		c3 := GetOrCreateCounter(ctx, "worker3")
		c3.Inc() // count = 1

		// Verify counts
		assert.Equal(t, uint64(2), c1.Get())
		assert.Equal(t, uint64(0), c2.Get())
		assert.Equal(t, uint64(1), c3.Get())

		selected, err := SelectWorker(ctx, workers, "test", "decode")
		assert.NoError(t, err)
		assert.Equal(t, "http://worker2", selected)
	})
}

func TestCounterOperations(t *testing.T) {
	ctx := context.Background()
	Init(&config.Config{}, nil)

	t.Run("counter increment", func(t *testing.T) {
		counter := GetOrCreateCounter(ctx, "test")
		assert.Equal(t, uint64(0), counter.Get())

		counter.Inc()
		assert.Equal(t, uint64(1), counter.Get())

		ok := counter.Dec()
		assert.True(t, ok)
		assert.Equal(t, uint64(0), counter.Get())
	})

	t.Run("counter underflow protection", func(t *testing.T) {
		counter := GetOrCreateCounter(ctx, "test-underflow")
		assert.Equal(t, uint64(0), counter.Get())
		ok := counter.Dec()
		assert.False(t, ok)
		assert.Equal(t, uint64(0), counter.Get())
	})

	t.Run("token counter operations", func(t *testing.T) {
		tc := GetOrCreateTokenCounter(ctx, "test")
		assert.Equal(t, uint64(0), tc.Get())

		tc.Add(100)
		assert.Equal(t, uint64(100), tc.Get())

		tc.Sub(50)
		assert.Equal(t, uint64(50), tc.Get())
	})
}

func TestCleanupInvalidCounters(t *testing.T) {
	ctx := context.Background()
	Init(&config.Config{}, &mockManagerAPI{})

	t.Run("idle invalid counter deleted", func(t *testing.T) {
		// Add some counters
		c1 := GetOrCreateCounter(ctx, "worker1")
		c1.Inc()
		GetOrCreateCounter(ctx, "invalid-worker") // idle, should be cleaned up

		tc1 := GetOrCreateTokenCounter(ctx, "worker1")
		tc1.Add(100)
		GetOrCreateTokenCounter(ctx, "invalid-worker") // idle, should be cleaned up

		CleanupInvalidCounters(ctx)

		// Healthy worker counters remain
		_, exists := GetCounter(ctx, "worker1")
		assert.True(t, exists)
		_, exists = GetTokenCounter(ctx, "worker1")
		assert.True(t, exists)

		// Idle invalid worker counters deleted
		_, exists = GetCounter(ctx, "invalid-worker")
		assert.False(t, exists)
		_, exists = GetTokenCounter(ctx, "invalid-worker")
		assert.False(t, exists)
	})

	t.Run("inflight invalid counter preserved", func(t *testing.T) {
		Init(&config.Config{}, &mockManagerAPI{})

		inflightCounter := GetOrCreateCounter(ctx, "inflight-invalid-worker")
		inflightCounter.Inc() // simulate inflight request
		inflightTC := GetOrCreateTokenCounter(ctx, "inflight-invalid-worker")
		inflightTC.Add(50)

		CleanupInvalidCounters(ctx)

		// Inflight invalid worker counters preserved
		_, exists := GetCounter(ctx, "inflight-invalid-worker")
		assert.True(t, exists)
		_, exists = GetTokenCounter(ctx, "inflight-invalid-worker")
		assert.True(t, exists)
		assert.Equal(t, uint64(1), inflightCounter.Get())
		assert.Equal(t, uint64(50), inflightTC.Get())
	})
}

func TestEstimateTokens(t *testing.T) {
	tests := []struct {
		input    string
		expected uint64
	}{
		{"", 0},
		{"hello", 10}, // 5 chars * 2
		{"你好", 4},     // 2 chars * 2 (Chinese characters count as 1 char each)
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			assert.Equal(t, tt.expected, estimateTokens(tt.input))
		})
	}
}

func TestReleasePrefillTokens(t *testing.T) {
	ctx := context.Background()
	Init(&config.Config{}, nil)

	t.Run("valid release", func(t *testing.T) {
		tc := GetOrCreateTokenCounter(ctx, "worker1")
		tc.Add(100)
		ReleasePrefillTokens(ctx, "worker1", "hello") // 5 chars * 2 = 10 tokens
		assert.Equal(t, uint64(90), tc.Get())
	})

	t.Run("empty url or message", func(t *testing.T) {
		tc := GetOrCreateTokenCounter(ctx, "worker2")
		tc.Add(100)
		ReleasePrefillTokens(ctx, "", "hello")   // no-op
		ReleasePrefillTokens(ctx, "worker2", "") // no-op
		assert.Equal(t, uint64(100), tc.Get())
	})
}

func TestCleanupUnhealthyCounter(t *testing.T) {
	ctx := context.Background()
	Init(&config.Config{}, nil)

	t.Run("counter preserved when inflight requests exist", func(t *testing.T) {
		c := GetOrCreateCounter(ctx, "unhealthy-worker-inflight")
		c.Inc()
		tc := GetOrCreateTokenCounter(ctx, "unhealthy-worker-inflight")
		tc.Add(100)

		CleanupUnhealthyCounter(ctx, "unhealthy-worker-inflight")

		// Counter should be preserved (inflight requests)
		_, exists := GetCounter(ctx, "unhealthy-worker-inflight")
		assert.True(t, exists)
		_, exists = GetTokenCounter(ctx, "unhealthy-worker-inflight")
		assert.True(t, exists)
		assert.Equal(t, uint64(1), c.Get())
		assert.Equal(t, uint64(100), tc.Get())
	})

	t.Run("counter deleted when no inflight requests", func(t *testing.T) {
		GetOrCreateCounter(ctx, "unhealthy-worker-idle")
		GetOrCreateTokenCounter(ctx, "unhealthy-worker-idle")

		CleanupUnhealthyCounter(ctx, "unhealthy-worker-idle")

		// Counter should be deleted (no inflight requests)
		_, exists := GetCounter(ctx, "unhealthy-worker-idle")
		assert.False(t, exists)
		_, exists = GetTokenCounter(ctx, "unhealthy-worker-idle")
		assert.False(t, exists)
	})
}

func TestStartBackupCleanupTask(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	Init(&config.Config{}, &mockManagerAPI{})

	// Add invalid counter
	GetOrCreateCounter(ctx, "invalid-worker")

	// Start cleanup task with short interval
	go StartBackupCleanupTask(ctx, 0.1) // 0.1 second interval

	// Wait for cleanup
	time.Sleep(200 * time.Millisecond)
	cancel()

	// Verify cleanup
	_, exists := GetCounter(ctx, "invalid-worker")
	assert.False(t, exists)
}

func TestCounterLifecycle_UnhealthyAndReregister(t *testing.T) {
	ctx := context.Background()
	Init(&config.Config{}, &mockManagerAPI{})

	url := "http://10.0.0.1:8080"

	// 1. Simulate request arrival: Inc
	counter := GetOrCreateCounter(ctx, url)
	counter.Inc()
	assert.Equal(t, uint64(1), counter.Get())

	tokenCounter := GetOrCreateTokenCounter(ctx, url)
	tokenCounter.Add(100)
	assert.Equal(t, uint64(100), tokenCounter.Get())

	// 2. Instance becomes unhealthy → CleanupUnhealthyCounter (counter preserved due to inflight)
	CleanupUnhealthyCounter(ctx, url)

	// Counter still exists, value unchanged
	sameCounter := GetOrCreateCounter(ctx, url)
	assert.Equal(t, counter, sameCounter) // same object
	assert.Equal(t, uint64(1), sameCounter.Get())

	// 3. Inflight request completes → Release
	Release(ctx, url)
	assert.Equal(t, uint64(0), counter.Get())

	ReleasePrefillTokens(ctx, url, "dummy message with 10 chars")

	// 4. Another Release does not underflow
	Release(ctx, url)
	assert.Equal(t, uint64(0), counter.Get()) // stays 0, no underflow

	// 5. Instance re-registers → new request Inc
	counter.Inc()
	assert.Equal(t, uint64(1), counter.Get())

	// 6. Request completes → Release
	Release(ctx, url)
	assert.Equal(t, uint64(0), counter.Get()) // back to zero

	// 7. Multiple concurrent requests full cycle
	counter.Inc()
	counter.Inc()
	counter.Inc()
	assert.Equal(t, uint64(3), counter.Get())
	Release(ctx, url)
	Release(ctx, url)
	Release(ctx, url)
	assert.Equal(t, uint64(0), counter.Get()) // back to zero
}

func TestCounterLifecycle_CleanupBeforeRelease(t *testing.T) {
	ctx := context.Background()
	Init(&config.Config{}, &mockManagerAPI{})

	url := "http://10.0.0.2:8080"

	t.Run("cleanup deletes counter then release is no-op", func(t *testing.T) {
		// 1. Request arrives → counter=1
		counter := GetOrCreateCounter(ctx, url)
		counter.Inc()
		assert.Equal(t, uint64(1), counter.Get())

		tc := GetOrCreateTokenCounter(ctx, url)
		tc.Add(200)

		// 2. Request finishes → Release → counter=0
		Release(ctx, url)
		assert.Equal(t, uint64(0), counter.Get())

		// 3. Cleanup runs, sees counter=0, deletes it
		CleanupUnhealthyCounter(ctx, url)
		_, exists := GetCounter(ctx, url)
		assert.False(t, exists) // counter deleted

		// 4. A late/duplicate Release after cleanup should NOT create ghost counter
		Release(ctx, url)

		// Verify no ghost counter was created
		_, exists = GetCounter(ctx, url)
		assert.False(t, exists, "Release should not create ghost counter after cleanup")
	})

	t.Run("cleanup deletes token counter then ReleasePrefillTokens is no-op", func(t *testing.T) {
		Init(&config.Config{}, &mockManagerAPI{})
		tokenURL := "http://10.0.0.3:8080"

		tc := GetOrCreateTokenCounter(ctx, tokenURL)
		tc.Add(200)

		// Sub all tokens so counter=0
		tc.Sub(200)
		assert.Equal(t, uint64(0), tc.Get())

		// Cleanup deletes the token counter
		CleanupUnhealthyCounter(ctx, tokenURL)
		_, exists := GetTokenCounter(ctx, tokenURL)
		assert.False(t, exists)

		// Late ReleasePrefillTokens should not create ghost token counter
		ReleasePrefillTokens(ctx, tokenURL, "hello world")
		_, exists = GetTokenCounter(ctx, tokenURL)
		assert.False(t, exists, "ReleasePrefillTokens should not create ghost token counter after cleanup")
	})
}
