package manager

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"

	"slices"

	"github.com/PaddlePaddle/FastDeploy/router/pkg/logger"
	"github.com/gin-gonic/gin"
	"gopkg.in/yaml.v3"
)

type RegisterConfig struct {
	Instances []InstanceInfo `yaml:"instances"`
}

func GetSplitwise(ctx context.Context) bool {
	if DefaultManager == nil {
		return false
	}
	DefaultManager.mu.RLock()
	defer DefaultManager.mu.RUnlock()
	return DefaultManager.splitwise
}

func GetAllMapServers(ctx context.Context) map[string]*WorkerInfo {
	if DefaultManager == nil {
		return make(map[string]*WorkerInfo)
	}

	DefaultManager.mu.RLock()
	defer DefaultManager.mu.RUnlock()
	allServers := make(map[string]*WorkerInfo)
	for id, workerInfo := range DefaultManager.prefillWorkerMap {
		allServers[id] = workerInfo
	}
	for id, workerInfo := range DefaultManager.decodeWorkerMap {
		allServers[id] = workerInfo
	}
	for id, workerInfo := range DefaultManager.mixedWorkerMap {
		allServers[id] = workerInfo
	}
	return allServers
}

// getWorkerInfo gets WorkerInfo based on URL
func getWorkerInfo(ctx context.Context, url string) *WorkerInfo {
	if DefaultManager == nil {
		return nil
	}
	DefaultManager.mu.RLock()
	defer DefaultManager.mu.RUnlock()

	if w, ok := DefaultManager.prefillWorkerMap[url]; ok {
		return w
	}
	if w, ok := DefaultManager.decodeWorkerMap[url]; ok {
		return w
	}
	if w, ok := DefaultManager.mixedWorkerMap[url]; ok {
		return w
	}
	return nil
}

// BuildDisaggregateInfo builds disaggregate_info structure
func BuildDisaggregateInfo(ctx context.Context, prefillURL, decodeURL string) (map[string]any, error) {
	prefillInfo := getWorkerInfo(ctx, prefillURL)
	decodeInfo := getWorkerInfo(ctx, decodeURL)
	if prefillInfo == nil || decodeInfo == nil {
		return nil, fmt.Errorf("worker instance not found for prefill=%s, decode=%s", prefillURL, decodeURL)
	}

	prefillHost := hostFromURL(prefillInfo.Url)
	decodeHost := hostFromURL(decodeInfo.Url)

	// Check if IPC can be used
	isSameNode := prefillHost != "" && prefillHost == decodeHost
	isSupportIPC := slices.Contains(prefillInfo.TransferProtocol, "ipc") &&
		slices.Contains(decodeInfo.TransferProtocol, "ipc")
	tpPrefill := tpSizeFromWorker(prefillInfo)
	tpDecode := tpSizeFromWorker(decodeInfo)
	isSameTpSize := tpPrefill == tpDecode || tpDecode == 1
	useIPC := isSameNode && isSupportIPC && isSameTpSize

	transferProto := "rdma"
	if useIPC {
		transferProto = "ipc"
	}

	disagg := map[string]any{
		"prefill_ip":             prefillHost,
		"decode_ip":              decodeHost,
		"prefill_connector_port": portStringToInt(Port(prefillInfo.ConnectorPort)),
		"decode_connector_port":  portStringToInt(Port(decodeInfo.ConnectorPort)),
		"decode_device_ids":      []string(decodeInfo.DeviceIDs),
		"decode_rdma_ports":      []string(decodeInfo.RdmaPorts),
		"transfer_protocol":      transferProto,
		"decode_tp_size":         tpDecode,
	}
	return disagg, nil
}

// portStringToInt converts Port (string) to int
func portStringToInt(p Port) int {
	s := string(p)
	if s == "" {
		return 0
	}
	i, err := strconv.Atoi(s)
	if err != nil {
		return 0
	}
	return i
}

// tpSizeFromWorker calculates tp_size (currently no explicit field, uses device_ids count, minimum 1)
func tpSizeFromWorker(w *WorkerInfo) int {
	if w == nil {
		return 0
	}
	if len(w.DeviceIDs) > 0 {
		return len(w.DeviceIDs)
	}
	return 1
}

// hostFromURL extracts host part (without port)
func hostFromURL(raw string) string {
	if raw == "" {
		return ""
	}
	if !strings.HasPrefix(raw, "http://") && !strings.HasPrefix(raw, "https://") {
		raw = "http://" + raw
	}
	u, err := url.Parse(raw)
	if err != nil {
		return ""
	}
	return u.Hostname()
}

func RegisterInstanceCore(ctx context.Context, rawInstance *InstanceInfo) error {
	instance, err := NewInstanceInfo(rawInstance)
	if err != nil {
		return fmt.Errorf("invalid InstanceInfo format:%v", err)
	}

	splitwiseMode := GetSplitwise(ctx)

	instanceRole := instance.Role.EnumValue
	if splitwiseMode && instanceRole == MIXED {
		return fmt.Errorf("splitwise mode only supports PREFILL/DECODE instances")
	}
	if !splitwiseMode && instanceRole != MIXED {
		return fmt.Errorf("only MIXED instances are allowed")
	}

	// Check instance health status
	if !CheckWorkerHealth(ctx, instance.URL()) {
		return fmt.Errorf("service is not healthy")
	}

	allServers := GetAllMapServers(ctx)

	DefaultManager.mu.Lock()
	defer DefaultManager.mu.Unlock()

	workerInfo := &WorkerInfo{
		Url:                   instance.URL(),
		WorkerType:            instance.Role.EnumValue.String(),
		ConnectorPort:         string(instance.ConnectorPort),
		EngineWorkerQueuePort: string(instance.EngineWorkerQueuePort),
		TransferProtocol:      instance.TransferProtocol,
		RdmaPorts:             []string(instance.RDMAPorts),
		DeviceIDs:             []string(instance.DeviceIDs),
		MetricsPort:           string(instance.MetricsPort),
	}

	id := instance.URL()

	if w, exists := allServers[id]; exists {
		wType, err := ParseInstanceRole(w.WorkerType)
		if err == nil {
			switch wType {
			case MIXED:
				delete(DefaultManager.mixedWorkerMap, id)
			case PREFILL:
				delete(DefaultManager.prefillWorkerMap, id)
			case DECODE:
				delete(DefaultManager.decodeWorkerMap, id)
			}
		}
	}

	switch instanceRole {
	case MIXED:
		DefaultManager.mixedWorkerMap[id] = workerInfo
	case PREFILL:
		DefaultManager.prefillWorkerMap[id] = workerInfo
	case DECODE:
		DefaultManager.decodeWorkerMap[id] = workerInfo
	default:
		logger.Warn("Instance %s role is unknown", id)
	}

	return nil
}

func RegisterInstance(c *gin.Context) {
	bodyBytes, err := io.ReadAll(c.Request.Body)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"code": 400,
			"msg":  "Invalid request body",
		})
		return
	}

	var rawInstance InstanceInfo
	err = json.Unmarshal(bodyBytes, &rawInstance)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"code": 400,
			"msg":  fmt.Sprintf("Invalid InstanceInfo JSON format: %v", err),
		})
		return
	}

	if err := RegisterInstanceCore(c.Request.Context(), &rawInstance); err != nil {
		logger.Error("Failed to register instance: %v", err)
		// Return different HTTP status codes based on error type
		if strings.Contains(err.Error(), "not healthy") {
			c.JSON(http.StatusServiceUnavailable, gin.H{
				"code": 503,
				"msg":  err.Error(),
			})
		} else {
			c.JSON(http.StatusBadRequest, gin.H{
				"code": 400,
				"msg":  err.Error(),
			})
		}
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"code": 200,
		"msg":  "Register success",
	})
}

func RegisterInstancesFromConfig(yamlPath string) {
	if yamlPath == "" {
		return
	}
	data, err := os.ReadFile(yamlPath)
	if err != nil {
		logger.Error("Failed to read YAML file %s: %v", yamlPath, err)
		return
	}

	var config RegisterConfig
	if err := yaml.Unmarshal(data, &config); err != nil {
		logger.Error("Failed to unmarshal YAML file %s: %v", yamlPath, err)
		return
	}

	if len(config.Instances) == 0 {
		logger.Info("No instances found in config file %s", yamlPath)
		return
	}

	for i, instanceConfig := range config.Instances {
		if err := RegisterInstanceCore(context.Background(), &instanceConfig); err != nil {
			logger.Error("Failed to register instance from index %d: %v", i, err)
		} else {
			logger.Info("Successfully registered instance from index %d", i)
		}
	}
}

func RegisteredNumber(c *gin.Context) {
	if DefaultManager == nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"code": 400,
			"msg":  "DefaultManager is nil",
		})
		return
	}

	DefaultManager.mu.RLock()
	defer DefaultManager.mu.RUnlock()
	c.JSON(http.StatusOK, gin.H{
		"mixed":   len(DefaultManager.mixedWorkerMap),
		"prefill": len(DefaultManager.prefillWorkerMap),
		"decode":  len(DefaultManager.decodeWorkerMap),
	})
}

func Registered(c *gin.Context) {
	DefaultManager.mu.RLock()
	defer DefaultManager.mu.RUnlock()

	var prefillInstances, decodeInstances, mixedInstances []WorkerInfo
	decodeInstances = make([]WorkerInfo, 0)
	prefillInstances = make([]WorkerInfo, 0)
	mixedInstances = make([]WorkerInfo, 0)
	for _, w := range DefaultManager.prefillWorkerMap {
		prefillInstances = append(prefillInstances, *w)
	}
	for _, w := range DefaultManager.decodeWorkerMap {
		decodeInstances = append(decodeInstances, *w)
	}
	for _, w := range DefaultManager.mixedWorkerMap {
		mixedInstances = append(mixedInstances, *w)
	}
	c.JSON(http.StatusOK, gin.H{
		"code":    http.StatusOK,
		"msg":     "success",
		"decode":  decodeInstances,
		"prefill": prefillInstances,
		"mixed":   mixedInstances,
	})
}
