package manager

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"regexp"
	"strconv"

	scheduler_handler "github.com/PaddlePaddle/FastDeploy/router/internal/scheduler/handler"
)

// Precompile regex to avoid repeated compilation
var (
	runningRequestsRegex      = regexp.MustCompile(`fastdeploy:num_requests_running\s+([0-9.]+)`)
	waitingRequestsRegex      = regexp.MustCompile(`fastdeploy:num_requests_waiting\s+([0-9.]+)`)
	availableGpuBlockNumRegex = regexp.MustCompile(`available_gpu_block_num\s+([0-9.]+)`)
)

// parseMetricsResponseOptimized parses metrics response string and extracts key metrics
func parseMetricsResponseOptimized(response string) (float64, float64, float64) {
	waitingCnt := -1.0
	availableGpuBlockNum := -1.0
	runningCnt := -1.0

	// Find fastdeploy:num_requests_running field using precompiled regex
	if matches := runningRequestsRegex.FindStringSubmatch(response); len(matches) >= 2 {
		if score, err := strconv.ParseFloat(matches[1], 64); err == nil {
			runningCnt = score
		}
	}

	// Find fastdeploy:num_requests_waiting field using precompiled regex
	if matches := waitingRequestsRegex.FindStringSubmatch(response); len(matches) >= 2 {
		if score, err := strconv.ParseFloat(matches[1], 64); err == nil {
			waitingCnt = score
		}
	}

	// Parse available_gpu_block_num field
	if matches := availableGpuBlockNumRegex.FindStringSubmatch(response); len(matches) >= 2 {
		if score, err := strconv.ParseFloat(matches[1], 64); err == nil {
			availableGpuBlockNum = score
		}
	}

	return runningCnt, waitingCnt, availableGpuBlockNum
}

// redrictCounter gets or creates a counter for the given URL and returns current count
func redrictCounter(ctx context.Context, rawURL string) int {
	counter := scheduler_handler.GetOrCreateCounter(ctx, rawURL)
	return int(counter.Get())
}

// GetMetricsByURL retrieves running metrics of a worker by specified URL
func GetMetricsByURL(ctx context.Context, rawURL string) (int, int, int, error) {
	workerInfo := getWorkerInfo(ctx, rawURL)
	if workerInfo == nil {
		return 0, 0, 0, errors.New("worker info not found for URL")
	}
	u, err := url.Parse(rawURL)
	if err != nil {
		return 0, 0, 0, err
	}
	host, _, err := net.SplitHostPort(u.Host)
	if err != nil {
		return 0, 0, 0, err
	}
	u.Host = net.JoinHostPort(host, workerInfo.MetricsPort)
	metricsUrl := fmt.Sprintf("%s/metrics", u.String())

	client := &http.Client{Timeout: defaultCheckTimeout}
	resp, err := client.Get(metricsUrl)
	if err != nil {
		return 0, 0, 0, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0, 0, 0, err
	}

	if len(body) == 0 {
		return 0, 0, 0, errors.New("metrics response is empty")
	}

	// Parse metrics response
	runningCnt, waitingCnt, availableGpuBlockNum := parseMetricsResponseOptimized(string(body))
	if runningCnt < 0 || waitingCnt < 0 || availableGpuBlockNum < 0 {
		return 0, 0, 0, errors.New("failed to parse metrics response")
	}
	return int(runningCnt), int(waitingCnt), int(availableGpuBlockNum), nil
}

// GetMetrics retrieves running metrics of the worker for the specified URL
func (m *Manager) GetMetrics(ctx context.Context, rawURL string) (int, int, int) {
	runningCnt, waitingCnt, availableGpuBlockNum, err := GetMetricsByURL(ctx, rawURL)
	if err != nil {
		runningNewCnt := redrictCounter(ctx, rawURL)
		return runningNewCnt, 0, 0
	}
	return runningCnt, waitingCnt, availableGpuBlockNum
}
