package handler

import (
	"context"
	"testing"

	"github.com/PaddlePaddle/FastDeploy/router/internal/config"
	"github.com/stretchr/testify/assert"
)

func TestProcessTokensSelectWorker(t *testing.T) {
	ctx := context.Background()

	// Setup test data
	workers := []string{"worker1", "worker2", "worker3"}

	// Initialize scheduler and token counters
	Init(&config.Config{
		Scheduler: config.SchedulerConfig{
			Policy: "process_tokens",
		},
	}, nil)

	t.Run("select worker with least tokens", func(t *testing.T) {
		// Set up token counts
		tc1 := GetOrCreateTokenCounter(ctx, "worker1")
		tc1.Add(100)
		tc2 := GetOrCreateTokenCounter(ctx, "worker2")
		tc2.Add(50) // Should be selected
		tc3 := GetOrCreateTokenCounter(ctx, "worker3")
		tc3.Add(200)

		selected, err := ProcessTokensSelectWorker(ctx, workers, "test message")
		assert.NoError(t, err)
		assert.Equal(t, "worker2", selected)
	})

	t.Run("empty workers list", func(t *testing.T) {
		selected, err := ProcessTokensSelectWorker(ctx, []string{}, "test")
		assert.NoError(t, err)
		assert.Equal(t, "", selected)
	})
}

func TestRequestNumSelectWorker(t *testing.T) {
	ctx := context.Background()
	workers := []string{"worker1", "worker2", "worker3"}

	Init(&config.Config{
		Scheduler: config.SchedulerConfig{
			Policy: "request_num",
		},
	}, nil)

	t.Run("select worker with least requests", func(t *testing.T) {
		// Set up request counts
		c1 := GetOrCreateCounter(ctx, "worker1")
		c1.Inc()
		c1.Inc()                                 // count = 2
		c2 := GetOrCreateCounter(ctx, "worker2") // count = 0 (should be selected)
		c3 := GetOrCreateCounter(ctx, "worker3")
		c3.Inc() // count = 1

		// Verify counts (use variables to avoid "declared and not used" error)
		assert.Equal(t, uint64(2), c1.Get())
		assert.Equal(t, uint64(0), c2.Get())
		assert.Equal(t, uint64(1), c3.Get())

		selected, err := RequestNumSelectWorker(ctx, workers, "test")
		assert.NoError(t, err)
		assert.Equal(t, "worker2", selected)
	})

	t.Run("empty workers list", func(t *testing.T) {
		selected, err := RequestNumSelectWorker(ctx, []string{}, "test")
		assert.NoError(t, err)
		assert.Equal(t, "", selected)
	})
}
