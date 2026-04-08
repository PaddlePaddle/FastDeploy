package handler

import (
	"context"
	"math/rand"
)

func RandomSelectWorker(ctx context.Context, workers []string, message string) (string, error) {
	if len(workers) == 0 {
		return "", nil
	}

	randomNum := rand.Intn(len(workers))
	return workers[randomNum], nil
}
