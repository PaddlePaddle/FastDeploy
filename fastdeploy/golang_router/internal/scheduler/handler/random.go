package handler

import (
	"context"
	"math/rand"
	"time"
)

var (
	randomSource = rand.NewSource(time.Now().UnixNano())
)

func RandomSelectWorker(ctx context.Context, workers []string, message string) (string, error) {
	if len(workers) == 0 {
		return "", nil
	}

	r := rand.New(randomSource)
	randomNum := r.Intn(len(workers))
	return workers[randomNum], nil
}
