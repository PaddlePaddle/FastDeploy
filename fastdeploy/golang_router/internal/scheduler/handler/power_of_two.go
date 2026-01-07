package handler

import (
	"context"
	"math/rand"
)

func PowerOfTwoSelectWorker(ctx context.Context, workers []string, message string) (string, error) {
	if len(workers) == 0 {
		return "", nil
	}
	if len(workers) == 1 {
		return workers[0], nil
	}

	length := len(workers)
	randomNum1 := rand.Intn(length)
	randomNum2 := rand.Intn(length)

	for randomNum2 == randomNum1 {
		randomNum2 = rand.Intn(length)
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
	return selectedURL, nil
}
