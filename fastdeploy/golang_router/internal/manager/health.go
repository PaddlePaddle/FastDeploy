package manager

import (
	"context"
	"io"
	"net/http"
	"sync"
	"time"

	scheduler_handler "github.com/PaddlePaddle/FastDeploy/router/internal/scheduler/handler"
	"github.com/PaddlePaddle/FastDeploy/router/pkg/logger"
	"github.com/gin-gonic/gin"
)

type healthCheckResult struct {
	url       string
	isHealthy bool
}

type healthMonitorResult struct {
	id        string
	worker    *WorkerInfo // node information
	isHealthy bool        // check error (nil means healthy)
}

func CheckServiceHealth(ctx context.Context, baseURL string, timeout ...time.Duration) bool {
	// Handle empty baseURL
	if baseURL == "" {
		logger.Error("empty baseURL provided")
		return false
	}

	healthPath := healthEndpoint
	url := baseURL + healthPath
	timeoutToUse := defaultCheckTimeout // Default timeout

	// Override default value if caller provides valid timeout parameter
	if len(timeout) > 0 && timeout[0] > 0 {
		timeoutToUse = timeout[0]
	}

	// Create HTTP request
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		logger.Error("failed to create request: %v", err)
		return false
	}

	// Send request
	client := &http.Client{Timeout: timeoutToUse}
	resp, err := client.Do(req)

	if err != nil {
		logger.Error("failed to send request to %s with error: %v", url, err)
		return false
	}
	defer resp.Body.Close()

	// Read response body
	_, err = io.ReadAll(resp.Body)
	if err != nil {
		logger.Error("failed to read response body: %v", err)
		return false
	}

	// Check response status code
	if resp.StatusCode == http.StatusOK {
		return true
	}
	return false
}

func CheckWorkerHealth(ctx context.Context, baseURL string) bool {
	allServers := GetAllMapServers(ctx)
	_, exists := allServers[baseURL]
	if exists {
		for i := 0; i < failureThreshold; i++ {
			checkOk := CheckServiceHealth(ctx, baseURL)
			if checkOk {
				return true
			}
		}
		return false
	}
	for i := 0; i < successThreshold; i++ {
		checkOk := CheckServiceHealth(ctx, baseURL)
		if !checkOk {
			return false
		}
	}
	return true
}

// Get all URLs of Prefill and Decode servers
func GetAllServerURLs(ctx context.Context) []string {
	DefaultManager.mu.RLock()
	defer DefaultManager.mu.RUnlock()

	totalSeversLength := len(DefaultManager.prefillWorkerMap) + len(DefaultManager.decodeWorkerMap)
	allServerURLs := make([]string, 0, totalSeversLength)

	for _, server := range DefaultManager.prefillWorkerMap {
		allServerURLs = append(allServerURLs, server.Url)
	}
	for _, server := range DefaultManager.decodeWorkerMap {
		allServerURLs = append(allServerURLs, server.Url)
	}
	return allServerURLs
}

func HealthGenerate(c *gin.Context) {
	// The buffer size of this channel equals the total number of tasks, avoids goroutine blocking
	results := make(chan healthCheckResult, len(DefaultManager.prefillWorkerMap)+len(DefaultManager.decodeWorkerMap))
	// Use WaitGroup to wait for all goroutines to complete sending results
	var wg sync.WaitGroup

	allServerURLs := GetAllServerURLs(c.Request.Context())

	for _, s := range allServerURLs {
		wg.Add(1)
		go func(serverURL string) {
			defer wg.Done()
			baseURL := serverURL
			isHealthy := CheckWorkerHealth(c.Request.Context(), baseURL)
			results <- healthCheckResult{
				url:       serverURL,
				isHealthy: isHealthy,
			}
		}(s)
	}

	// Start a goroutine to close the result channel after all check tasks complete
	// Used to notify the range loop can end
	go func() {
		wg.Wait()
		close(results)
	}()

	for res := range results {
		// Process each result
		if !res.isHealthy {
			logger.Warn("Server %s is not healthy", res.url)
		} else {
			logger.Info("Server %s is healthy", res.url)
		}
	}

	c.JSON(http.StatusOK, gin.H{
		"code": 200,
		"msg":  "Health check complete",
	})
}

func RemoveServers(ctx context.Context, prefillToRemove []string, decodeToRemove []string, mixedToRemove []string) {
	DefaultManager.mu.Lock()
	defer DefaultManager.mu.Unlock()

	for _, id := range prefillToRemove {
		if worker, exists := DefaultManager.prefillWorkerMap[id]; exists {
			delete(DefaultManager.prefillWorkerMap, id)
			logger.Info("Removed unhealthy prefill instance: %s", worker.Url)
		}
	}
	for _, id := range decodeToRemove {
		if worker, exists := DefaultManager.decodeWorkerMap[id]; exists {
			delete(DefaultManager.decodeWorkerMap, id)
			logger.Info("Removed unhealthy decode instance: %s", worker.Url)
		}
	}
	for _, id := range mixedToRemove {
		if worker, exists := DefaultManager.mixedWorkerMap[id]; exists {
			delete(DefaultManager.mixedWorkerMap, id)
			logger.Info("Removed unhealthy mixed instance: %s", worker.Url)
		}
	}
}

func ReadServers(ctx context.Context) (prefillInstances, decodeInstances, mixedInstances []string) {
	if DefaultManager == nil {
		logger.Debug("Healthy instances: prefill=[], decode=[], mixed=[] (DefaultManager is nil)")
		return []string{}, []string{}, []string{}
	}

	DefaultManager.mu.RLock()
	defer DefaultManager.mu.RUnlock()

	// Pre-allocate sufficient capacity to avoid multiple expansions
	prefillInstances = make([]string, 0, len(DefaultManager.prefillWorkerMap))
	decodeInstances = make([]string, 0, len(DefaultManager.decodeWorkerMap))
	mixedInstances = make([]string, 0, len(DefaultManager.mixedWorkerMap))

	// Copy data to avoid holding lock for long time
	for _, w := range DefaultManager.prefillWorkerMap {
		prefillInstances = append(prefillInstances, w.Url)
	}
	for _, w := range DefaultManager.decodeWorkerMap {
		decodeInstances = append(decodeInstances, w.Url)
	}
	for _, w := range DefaultManager.mixedWorkerMap {
		mixedInstances = append(mixedInstances, w.Url)
	}
	logger.Debug(
		"Healthy instances: prefill=%v, decode=%v, mixed=%v",
		prefillInstances,
		decodeInstances,
		mixedInstances,
	)
	return prefillInstances, decodeInstances, mixedInstances
}

func MonitorInstanceHealthCore(ctx context.Context) {
	if DefaultManager == nil {
		return
	}
	// Concurrently check health status of all nodes (fix concurrent security issues)
	allServers := GetAllMapServers(ctx)
	length := len(allServers)

	resultCh := make(chan healthMonitorResult, length)
	var wg sync.WaitGroup

	for id, server := range allServers {
		wg.Add(1)
		go func(id string, server *WorkerInfo) {
			defer wg.Done()
			// Execute health check logic
			baseURL := server.Url
			isHealthy := CheckWorkerHealth(ctx, baseURL)
			resultCh <- healthMonitorResult{
				id:        id,
				worker:    server,
				isHealthy: isHealthy,
			}
		}(id, server)
	}

	// Wait for all checks to complete
	go func() {
		wg.Wait()
		close(resultCh)
	}()

	var prefillToRemove, decodeToRemove, mixedToRemove []string

	for res := range resultCh {
		if !res.isHealthy {
			// logger.Warn("Server %s meets error: %v", res.worker.url, res.err)
			switch res.worker.WorkerType {
			case "prefill":
				prefillToRemove = append(prefillToRemove, res.id)
			case "decode":
				decodeToRemove = append(decodeToRemove, res.id)
			case "mixed":
				mixedToRemove = append(mixedToRemove, res.id)
			}
			go scheduler_handler.CleanupUnhealthyCounter(ctx, res.id)
		}
	}

	// Remove unhealthy instances
	RemoveServers(ctx, prefillToRemove, decodeToRemove, mixedToRemove)

	ReadServers(ctx)
}

func MonitorInstanceHealth(ctx context.Context, intervalSecs float64) {
	ticker := time.NewTicker(time.Duration(intervalSecs * float64(time.Second)))
	defer ticker.Stop()

	// Infinite loop: continuously execute health checks
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			go MonitorInstanceHealthCore(ctx)
		}
	}
}
