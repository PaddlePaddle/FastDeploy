package common

import "context"

type ManagerAPI interface {
	GetHealthyURLs(ctx context.Context) []string
	GetMetrics(ctx context.Context, url string) (int, int, int)
}
