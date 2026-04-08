package common

import "context"

type SelectStrategyFunc func(ctx context.Context, workers []string, message string) (string, error)
