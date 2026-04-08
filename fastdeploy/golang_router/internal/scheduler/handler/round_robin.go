package handler

import (
	"context"
)

func RoundRobinSelectWorker(ctx context.Context, workers []string, message string) (string, error) {
	if len(workers) == 0 {
		return "", nil
	}

	var count uint64
	if DefaultCounterPolicy.workerType == "prefill" {
		count = DefaultCounterPolicy.prefillCounter.Add(1) - 1
	} else {
		count = DefaultCounterPolicy.counter.Add(1) - 1
	}

	selectedNum := count % uint64(len(workers))
	return workers[selectedNum], nil
}
