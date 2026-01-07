package handler

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"

	common "github.com/PaddlePaddle/FastDeploy/router/internal/common"
	"github.com/PaddlePaddle/FastDeploy/router/internal/config"
	scheduler_common "github.com/PaddlePaddle/FastDeploy/router/internal/scheduler/common"
	"github.com/PaddlePaddle/FastDeploy/router/pkg/logger"
)

type Scheduler struct {
	policy        string
	prefillPolicy string
	decodePolicy  string
	IdCounterMap  map[string]*scheduler_common.Counter
	tokenMap      map[string]*scheduler_common.TokenCounter
	managerAPI    common.ManagerAPI
	prefillCache  *prefillCacheStrategy
	mu            sync.RWMutex
}

type CounterPolicy struct {
	counter        atomic.Uint64
	prefillCounter atomic.Uint64
	workerType     string
}

var DefaultScheduler *Scheduler
var DefaultCounterPolicy *CounterPolicy
var waitingWeight float64

// Init initializes the scheduler with the given configuration and manager API
func Init(cfg *config.Config, managerAPI common.ManagerAPI) {
	prefillCfg := &schedulerConfigSnapshot{
		balanceAbsThreshold: cfg.Scheduler.BalanceAbsThreshold,
		balanceRelThreshold: cfg.Scheduler.BalanceRelThreshold,
		hitRatioWeight:      cfg.Scheduler.HitRatioWeight,
		loadBalanceWeight:   cfg.Scheduler.LoadBalanceWeight,
		cacheBlockSize:      cfg.Scheduler.CacheBlockSize,
		tokenizerURL:        cfg.Scheduler.TokenizerURL,
		tokenizerTimeout:    time.Duration(cfg.Scheduler.TokenizerTimeoutSecs * float64(time.Second)),
	}

	scheduler := &Scheduler{
		policy:        cfg.Scheduler.Policy,
		prefillPolicy: cfg.Scheduler.PrefillPolicy,
		decodePolicy:  cfg.Scheduler.DecodePolicy,
		IdCounterMap:  make(map[string]*scheduler_common.Counter),
		tokenMap:      make(map[string]*scheduler_common.TokenCounter),
		managerAPI:    managerAPI,
		prefillCache:  newPrefillCacheStrategy(prefillCfg),
	}
	counterPolicy := &CounterPolicy{}
	DefaultScheduler = scheduler
	DefaultCounterPolicy = counterPolicy
	waitingWeight = cfg.Scheduler.WaitingWeight
}

// SelectWorker selects a worker based on the specified policy and worker type
func SelectWorker(ctx context.Context, workers []string, message string, workerType string) (string, error) {
	if len(workers) == 0 {
		return "", fmt.Errorf("no healthy workers available")
	}

	var policy string
	switch workerType {
	case "prefill":
		policy = DefaultScheduler.prefillPolicy
		DefaultCounterPolicy.workerType = "prefill"
	case "decode":
		policy = DefaultScheduler.decodePolicy
		DefaultCounterPolicy.workerType = "decode"
	default:
		policy = DefaultScheduler.policy
		DefaultCounterPolicy.workerType = "mixed"
	}

	var strategyFunc scheduler_common.SelectStrategyFunc
	switch policy {
	case "random":
		strategyFunc = RandomSelectWorker
	case "round_robin":
		strategyFunc = RoundRobinSelectWorker
	case "power_of_two":
		strategyFunc = PowerOfTwoSelectWorker
	case "process_tokens":
		// Prefill: prioritize the instance with the smallest number of tokens currently being processed
		strategyFunc = ProcessTokensSelectWorker
	case "request_num":
		// Decode/mixed: prioritize the instance with the smallest number of current requests
		strategyFunc = RequestNumSelectWorker
	case "fd_metrics_score":
		strategyFunc = FDMetricsScoreSelectWorker
	case "cache_aware":
		strategyFunc = CacheAwarePrefillSelectWorker
	default:
		strategyFunc = RandomSelectWorker
	}

	selectWorkerURL, err := strategyFunc(ctx, workers, message)
	if err != nil {
		return "", fmt.Errorf("select worker failed [policy: %s]: %w", DefaultScheduler.policy, err)
	}

	if !strings.HasPrefix(selectWorkerURL, "http://") && !strings.HasPrefix(selectWorkerURL, "https://") {
		selectWorkerURL = "http://" + selectWorkerURL
	}

	// 1) All node types: request concurrency count (request_num)
	counter := GetOrCreateCounter(ctx, selectWorkerURL)
	counter.Inc()
	count := counter.Get()

	// 2) Prefill: current token processing count (process_tokens)
	var tokens uint64
	if workerType == "prefill" && message != "" {
		tokenCounter := GetOrCreateTokenCounter(ctx, selectWorkerURL)
		tokenCounter.Add(estimateTokens(message))
		tokens = tokenCounter.Get()
	}

	if workerType == "prefill" {
		logger.Info("select worker (prefill): %s, tokens: %d", selectWorkerURL, tokens)
	} else {
		logger.Info("select worker (%s): %s, count: %d", workerType, selectWorkerURL, count)
	}

	return selectWorkerURL, nil
}

// Release decreases the counter for the specified worker URL
func Release(ctx context.Context, url string) {
	counter := GetOrCreateCounter(ctx, url)
	counter.Dec()
	logger.Info("release worker: %s, count: %d", url, counter.Get())
}

// GetCounter retrieves the counter for the specified root URL
func GetCounter(ctx context.Context, rootURL string) (*scheduler_common.Counter, bool) {
	DefaultScheduler.mu.RLock()
	defer DefaultScheduler.mu.RUnlock()
	counter, exists := DefaultScheduler.IdCounterMap[rootURL]
	return counter, exists
}

// GetOrCreateCounter retrieves an existing counter or creates a new one
func GetOrCreateCounter(ctx context.Context, url string) *scheduler_common.Counter {
	counter, exists := GetCounter(ctx, url)
	if exists {
		return counter
	}
	DefaultScheduler.mu.Lock()
	defer DefaultScheduler.mu.Unlock()
	// Double check: avoid overwriting what other goroutines may have created before acquiring write lock
	if counter, exists = DefaultScheduler.IdCounterMap[url]; exists {
		return counter
	}
	newCounter := &scheduler_common.Counter{}
	DefaultScheduler.IdCounterMap[url] = newCounter
	return newCounter
}

// CleanupUnhealthyCounter removes counters for unhealthy worker URLs
func CleanupUnhealthyCounter(ctx context.Context, unhealthyRootURL string) {
	if unhealthyRootURL == "" {
		return
	}

	if DefaultScheduler == nil {
		return
	}

	DefaultScheduler.mu.Lock()
	defer DefaultScheduler.mu.Unlock()

	delete(DefaultScheduler.IdCounterMap, unhealthyRootURL)
	delete(DefaultScheduler.tokenMap, unhealthyRootURL)
	logger.Info("After cleanup unhealthy counter: %v", DefaultScheduler.IdCounterMap)
}

// CleanupInvalidCounters removes counters for invalid or unreachable workers
func CleanupInvalidCounters(ctx context.Context) {
	if DefaultScheduler == nil {
		return
	}
	if DefaultScheduler.managerAPI == nil {
		return
	}
	healthyURLs := DefaultScheduler.managerAPI.GetHealthyURLs(ctx)
	if len(healthyURLs) == 0 {
		return
	}

	healthyMap := make(map[string]bool)
	for _, rootURL := range healthyURLs {
		healthyMap[rootURL] = true
	}

	DefaultScheduler.mu.Lock()
	defer DefaultScheduler.mu.Unlock()

	for rootURL := range DefaultScheduler.IdCounterMap {
		if _, exists := healthyMap[rootURL]; !exists {
			delete(DefaultScheduler.IdCounterMap, rootURL)
		}
	}

	for rootURL := range DefaultScheduler.tokenMap {
		if _, exists := healthyMap[rootURL]; !exists {
			delete(DefaultScheduler.tokenMap, rootURL)
		}
	}

	logger.Info("After cleanup invalid counters: %v", DefaultScheduler.IdCounterMap)
}

// StartBackupCleanupTask starts a background task for cleaning up invalid counters
func StartBackupCleanupTask(ctx context.Context, interval float64) {
	ticker := time.NewTicker(time.Duration(interval * float64(time.Second)))
	defer ticker.Stop()
	for {
		select {
		// case 1: listen for context cancellation/timeout events → graceful exit
		case <-ctx.Done():
			return // Exit loop, stop cleanup task
		// case 2: listen for timer trigger events → perform cleanup
		case <-ticker.C:
			CleanupInvalidCounters(ctx)
		}
	}
}

// GetTokenCounter gets the TokenCounter for the specified instance
func GetTokenCounter(ctx context.Context, rootURL string) (*scheduler_common.TokenCounter, bool) {
	DefaultScheduler.mu.RLock()
	defer DefaultScheduler.mu.RUnlock()
	counter, exists := DefaultScheduler.tokenMap[rootURL]
	return counter, exists
}

// GetOrCreateTokenCounter gets or creates TokenCounter
func GetOrCreateTokenCounter(ctx context.Context, url string) *scheduler_common.TokenCounter {
	counter, exists := GetTokenCounter(ctx, url)
	if exists {
		return counter
	}
	DefaultScheduler.mu.Lock()
	defer DefaultScheduler.mu.Unlock()
	// Double check to avoid overwriting
	if counter, exists = DefaultScheduler.tokenMap[url]; exists {
		return counter
	}
	newCounter := &scheduler_common.TokenCounter{}
	DefaultScheduler.tokenMap[url] = newCounter
	return newCounter
}

// estimateTokens estimates token count based on character count: character count * 2
func estimateTokens(message string) uint64 {
	if message == "" {
		return 0
	}
	runeCount := utf8.RuneCountInString(message)
	return uint64(runeCount * 2)
}

// ReleasePrefillTokens releases the corresponding token load when request ends
func ReleasePrefillTokens(ctx context.Context, url, message string) {
	if url == "" || message == "" {
		return
	}
	tokenCounter := GetOrCreateTokenCounter(ctx, url)
	tokenCounter.Sub(estimateTokens(message))
	logger.Info("release prefill tokens: %s, tokens: %d", url, tokenCounter.Get())
}
