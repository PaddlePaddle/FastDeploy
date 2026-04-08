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

		counter.Dec()
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

	// Add some counters
	c1 := GetOrCreateCounter(ctx, "worker1")
	c1.Inc()
	GetOrCreateCounter(ctx, "invalid-worker") // Should be cleaned up

	tc1 := GetOrCreateTokenCounter(ctx, "worker1")
	tc1.Add(100)
	GetOrCreateTokenCounter(ctx, "invalid-worker") // Should be cleaned up

	CleanupInvalidCounters(ctx)

	// Verify counters
	_, exists := GetCounter(ctx, "worker1")
	assert.True(t, exists)
	_, exists = GetCounter(ctx, "invalid-worker")
	assert.False(t, exists)

	// Verify token counters
	_, exists = GetTokenCounter(ctx, "worker1")
	assert.True(t, exists)
	_, exists = GetTokenCounter(ctx, "invalid-worker")
	assert.False(t, exists)
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

	// Add counters
	c := GetOrCreateCounter(ctx, "unhealthy-worker")
	c.Inc()
	tc := GetOrCreateTokenCounter(ctx, "unhealthy-worker")
	tc.Add(100)

	CleanupUnhealthyCounter(ctx, "unhealthy-worker")

	// Verify cleanup
	_, exists := GetCounter(ctx, "unhealthy-worker")
	assert.False(t, exists)
	_, exists = GetTokenCounter(ctx, "unhealthy-worker")
	assert.False(t, exists)
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
