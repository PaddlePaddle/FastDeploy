package handler

import (
	"context"
	"math/rand"
	"time"

	"github.com/PaddlePaddle/FastDeploy/router/pkg/logger"
)

func PowerOfTwoSelectWorker(ctx context.Context, workers []string, message string) (string, error) {
	if len(workers) == 0 {
		return "", nil
	}
	if len(workers) == 1 {
		return workers[0], nil
	}

	source := rand.NewSource(time.Now().UnixNano())
	r := rand.New(source)
	length := len(workers)
	randomNum1 := r.Intn(length)
	randomNum2 := r.Intn(length)

	for randomNum2 == randomNum1 {
		randomNum2 = r.Intn(length)
	}

	worker1 := workers[randomNum1]
	worker2 := workers[randomNum2]

	counter1 := GetOrCreateCounter(ctx, worker1)
	counter2 := GetOrCreateCounter(ctx, worker2)
	load1 := counter1.Get()
	load2 := counter2.Get()

	var selectedURL string
	if load1 <= load2 {
		selectedURL = worker1
	} else {
		selectedURL = worker2
	}

	logger.Info("Power-of-two selection:%s=%d vs %s=%d -> selected %s", worker1, load1, worker2, load2, selectedURL)

	return selectedURL, nil
}
