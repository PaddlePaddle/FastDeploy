package handler

import (
	"context"
	"math"
)

// ProcessTokensSelectWorker selects the instance with the smallest number of tokens currently being processed for Prefill nodes.
func ProcessTokensSelectWorker(ctx context.Context, workers []string, message string) (string, error) {
	if len(workers) == 0 {
		return "", nil
	}

	var (
		selected  string
		minTokens uint64 = math.MaxUint64
	)

	for _, w := range workers {
		tc := GetOrCreateTokenCounter(ctx, w)
		load := tc.Get()
		if load < minTokens {
			minTokens = load
			selected = w
		}
	}

	return selected, nil
}

// RequestNumSelectWorker selects the instance with the smallest number of current requests for Decode nodes.
func RequestNumSelectWorker(ctx context.Context, workers []string, message string) (string, error) {
	if len(workers) == 0 {
		return "", nil
	}

	var (
		selected string
		minCount uint64 = math.MaxUint64
	)

	for _, w := range workers {
		c := GetOrCreateCounter(ctx, w)
		load := c.Get()
		if load < minCount {
			minCount = load
			selected = w
		}
	}

	return selected, nil
}
