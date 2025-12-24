package handler

import (
	"context"
	"math/rand"
	"time"
)

func RandomSelectWorker(ctx context.Context, workers []string, message string) (string, error) {
	if len(workers) == 0 {
		return "", nil
	}

	source := rand.NewSource(time.Now().UnixNano())
	r := rand.New(source)
	randomNum := r.Intn(len(workers))
	return workers[randomNum], nil
}
