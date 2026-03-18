package handler

import (
	"context"
	"math"
)

func computeScore(ctx context.Context, runningCnt int, waitingCnt int) float64 {
	score := float64(runningCnt) + float64(waitingCnt)*waitingWeight
	return score
}

func FDRemoteMetricsScoreSelectWorker(ctx context.Context, workers []string, message string) (string, error) {
	if len(workers) == 0 {
		return "", nil
	}

	var (
		selectedURL string  = ""
		minScore    float64 = math.MaxFloat64
	)

	for _, w := range workers {
		runningCnt, waitingCnt, _ := DefaultScheduler.managerAPI.GetRemoteMetrics(ctx, w)
		score := computeScore(ctx, runningCnt, waitingCnt)
		if score < minScore {
			minScore = score
			selectedURL = w
		}
	}
	return selectedURL, nil
}
