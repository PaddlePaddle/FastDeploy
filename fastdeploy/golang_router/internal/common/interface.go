package common

import "context"

type ManagerAPI interface {
	GetHealthyURLs(ctx context.Context) []string
}
