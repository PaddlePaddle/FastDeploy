package handler

import (
	"context"

	"github.com/PaddlePaddle/FastDeploy/router/pkg/logger"
)

func RoundRobinSelectWorker(ctx context.Context, workers []string, message string) (string, error) {
	if len(workers) == 0 {
		return "", nil
	}

	count := DefaultRoundRobinPolicy.counter.Load()
	DefaultRoundRobinPolicy.counter.Add(1)

	selectedNum := count % uint64(len(workers))
	logger.Info("selectedNum: %d, workersURL: %s", selectedNum, workers[selectedNum])
	return workers[selectedNum], nil
}
