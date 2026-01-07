package manager

import (
	"context"
	"sort"
	"sync"
	"time"

	"github.com/PaddlePaddle/FastDeploy/router/internal/config"
	scheduler_handler "github.com/PaddlePaddle/FastDeploy/router/internal/scheduler/handler"
)

type Manager struct {
	mixedWorkerMap   map[string]*WorkerInfo
	prefillWorkerMap map[string]*WorkerInfo
	decodeWorkerMap  map[string]*WorkerInfo
	splitwise        bool
	mu               sync.RWMutex
}

type WorkerInfo struct {
	Url                   string   `json:"url"`
	WorkerType            string   `json:"worker_type"`
	ConnectorPort         string   `json:"connector_port"`
	EngineWorkerQueuePort string   `json:"engine_worker_queue_port"`
	TransferProtocol      []string `json:"transfer_protocol"`
	RdmaPorts             []string `json:"rdma_ports"`
	DeviceIDs             []string `json:"device_ids"`
	MetricsPort           string   `json:"metrics_port"`
}

var DefaultManager *Manager
var defaultCheckTimeout time.Duration
var healthEndpoint string
var failureThreshold int
var successThreshold int

// Manager module initialization
func Init(cfg *config.Config) {
	manager := &Manager{
		mixedWorkerMap:   make(map[string]*WorkerInfo),
		prefillWorkerMap: make(map[string]*WorkerInfo),
		decodeWorkerMap:  make(map[string]*WorkerInfo),
		splitwise:        cfg.Server.Splitwise,
	}
	DefaultManager = manager
	// Define a default timeout duration
	defaultCheckTimeout = time.Duration(cfg.Manager.HealthCheckTimeoutSecs * float64(time.Second))
	healthEndpoint = cfg.Manager.HealthCheckEndpoint
	failureThreshold = cfg.Manager.HealthFailureThreshold
	successThreshold = cfg.Manager.HealthSuccessThreshold
}

func WorkerMapToList(ctx context.Context, workerType string) []string {
	DefaultManager.mu.RLock()
	defer DefaultManager.mu.RUnlock()

	var workerMap map[string]*WorkerInfo
	switch workerType {
	case "mixed":
		workerMap = DefaultManager.mixedWorkerMap
	case "prefill":
		workerMap = DefaultManager.prefillWorkerMap
	case "decode":
		workerMap = DefaultManager.decodeWorkerMap
	default:
		return []string{}
	}

	if workerMap == nil {
		return []string{}
	}

	// Get all keys and sort them
	keys := make([]string, 0, len(workerMap))
	for key := range workerMap {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	// Build worker list
	workerURLs := make([]string, 0, len(keys))
	for _, key := range keys {
		if workerInfo, exists := workerMap[key]; exists {
			workerURLs = append(workerURLs, workerInfo.Url)
		}
	}
	return workerURLs
}

func (m *Manager) GetHealthyURLs(ctx context.Context) []string {
	if m == nil {
		return []string{}
	}

	m.mu.RLock()
	defer m.mu.RUnlock()

	totalSeversLength := len(m.prefillWorkerMap) + len(m.decodeWorkerMap) + len(m.mixedWorkerMap)
	allServerURLs := make([]string, 0, totalSeversLength)

	for id := range m.prefillWorkerMap {
		allServerURLs = append(allServerURLs, id)
	}
	for id := range m.decodeWorkerMap {
		allServerURLs = append(allServerURLs, id)
	}
	for id := range m.mixedWorkerMap {
		allServerURLs = append(allServerURLs, id)
	}
	return allServerURLs
}

func SelectWorker(ctx context.Context, message string) (string, error) {
	workers := WorkerMapToList(ctx, "mixed")
	selectedWorkerURL, err := scheduler_handler.SelectWorker(ctx, workers, message, "mixed")
	if err != nil {
		return "", err
	}
	return selectedWorkerURL, nil
}

func SelectWorkerPair(ctx context.Context, message string) (string, string, error) {
	prefillWorkers := WorkerMapToList(ctx, "prefill")
	decodeWorkers := WorkerMapToList(ctx, "decode")
	if len(prefillWorkers) == 0 || len(decodeWorkers) == 0 {
		return "", "", nil
	}
	selectedPrefillWorkerURL, err := scheduler_handler.SelectWorker(ctx, prefillWorkers, message, "prefill")
	if err != nil {
		return "", "", err
	}
	selectedDecodeWorkerURL, err := scheduler_handler.SelectWorker(ctx, decodeWorkers, message, "decode")
	if err != nil {
		return "", "", err
	}
	return selectedPrefillWorkerURL, selectedDecodeWorkerURL, nil
}
