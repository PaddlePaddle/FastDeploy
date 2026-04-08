package manager

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/PaddlePaddle/FastDeploy/router/internal/config"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
)

func TestGetSplitwise(t *testing.T) {
	t.Run("DefaultManager is nil", func(t *testing.T) {
		originalManager := DefaultManager
		DefaultManager = nil
		defer func() { DefaultManager = originalManager }()

		result := GetSplitwise(context.Background())
		assert.False(t, result)
	})

	t.Run("splitwise mode enabled", func(t *testing.T) {
		Init(&config.Config{Server: config.ServerConfig{Splitwise: true}})
		result := GetSplitwise(context.Background())
		assert.True(t, result)
	})

	t.Run("splitwise mode disabled", func(t *testing.T) {
		Init(&config.Config{Server: config.ServerConfig{Splitwise: false}})
		result := GetSplitwise(context.Background())
		assert.False(t, result)
	})
}

func TestGetAllMapServers(t *testing.T) {
	Init(&config.Config{})
	DefaultManager.prefillWorkerMap = map[string]*WorkerInfo{
		"worker1": {Url: "http://worker1"},
	}
	DefaultManager.decodeWorkerMap = map[string]*WorkerInfo{
		"worker2": {Url: "http://worker2"},
	}
	DefaultManager.mixedWorkerMap = map[string]*WorkerInfo{
		"worker3": {Url: "http://worker3"},
	}

	servers := GetAllMapServers(context.Background())
	assert.Len(t, servers, 3)
	assert.Contains(t, servers, "worker1")
	assert.Contains(t, servers, "worker2")
	assert.Contains(t, servers, "worker3")
}

func TestGetAllMapServers_NilManager(t *testing.T) {
	originalManager := DefaultManager
	DefaultManager = nil
	defer func() { DefaultManager = originalManager }()

	servers := GetAllMapServers(context.Background())
	assert.NotNil(t, servers)
	assert.Len(t, servers, 0)
}

func TestGetWorkerInfo(t *testing.T) {
	Init(&config.Config{})
	DefaultManager.prefillWorkerMap = map[string]*WorkerInfo{
		"http://worker1": {Url: "http://worker1", WorkerType: "prefill"},
	}
	DefaultManager.decodeWorkerMap = map[string]*WorkerInfo{
		"http://worker2": {Url: "http://worker2", WorkerType: "decode"},
	}

	t.Run("find prefill worker", func(t *testing.T) {
		info := getWorkerInfo(context.Background(), "http://worker1")
		assert.NotNil(t, info)
		assert.Equal(t, "prefill", info.WorkerType)
	})

	t.Run("find decode worker", func(t *testing.T) {
		info := getWorkerInfo(context.Background(), "http://worker2")
		assert.NotNil(t, info)
		assert.Equal(t, "decode", info.WorkerType)
	})

	t.Run("worker not found", func(t *testing.T) {
		info := getWorkerInfo(context.Background(), "http://notfound")
		assert.Nil(t, info)
	})
}

func TestBuildDisaggregateInfo(t *testing.T) {
	Init(&config.Config{Server: config.ServerConfig{Splitwise: true}})

	// Setup test workers
	DefaultManager.prefillWorkerMap = map[string]*WorkerInfo{
		"http://127.0.0.1:8000": {
			Url:              "http://127.0.0.1:8000",
			WorkerType:       "prefill",
			ConnectorPort:    "9000",
			TransferProtocol: []string{"rdma"},
			DeviceIDs:        []string{"0", "1"},
			RdmaPorts:        []string{"5000", "5001"},
		},
	}
	DefaultManager.decodeWorkerMap = map[string]*WorkerInfo{
		"http://127.0.0.1:8001": {
			Url:              "http://127.0.0.1:8001",
			WorkerType:       "decode",
			ConnectorPort:    "9001",
			TransferProtocol: []string{"rdma"},
			DeviceIDs:        []string{"0", "1"},
			RdmaPorts:        []string{"5002", "5003"},
		},
	}

	t.Run("successful build", func(t *testing.T) {
		info, err := BuildDisaggregateInfo(context.Background(),
			"http://127.0.0.1:8000", "http://127.0.0.1:8001")
		assert.NoError(t, err)
		assert.NotNil(t, info)
		assert.Equal(t, "127.0.0.1", info["prefill_ip"])
		assert.Equal(t, "127.0.0.1", info["decode_ip"])
		assert.Equal(t, "rdma", info["transfer_protocol"])
		assert.Equal(t, 2, info["decode_tp_size"])
	})

	t.Run("worker not found", func(t *testing.T) {
		_, err := BuildDisaggregateInfo(context.Background(),
			"http://notfound", "http://notfound2")
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "worker instance not found")
	})
}

func TestPortStringToInt(t *testing.T) {
	assert.Equal(t, 8080, portStringToInt(Port("8080")))
	assert.Equal(t, 0, portStringToInt(Port("")))
	assert.Equal(t, 0, portStringToInt(Port("invalid")))
}

func TestTpSizeFromWorker(t *testing.T) {
	t.Run("worker with device IDs", func(t *testing.T) {
		worker := &WorkerInfo{DeviceIDs: []string{"0", "1", "2"}}
		assert.Equal(t, 3, tpSizeFromWorker(worker))
	})

	t.Run("worker without device IDs", func(t *testing.T) {
		worker := &WorkerInfo{DeviceIDs: []string{}}
		assert.Equal(t, 1, tpSizeFromWorker(worker))
	})

	t.Run("nil worker", func(t *testing.T) {
		assert.Equal(t, 0, tpSizeFromWorker(nil))
	})
}

func TestHostFromURL(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{"standard URL", "http://127.0.0.1:8080", "127.0.0.1"},
		{"HTTPS URL", "https://example.com:443", "example.com"},
		{"URL without protocol", "127.0.0.1:8080", "127.0.0.1"},
		{"empty URL", "", ""},
		{"invalid URL", "://invalid", ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := hostFromURL(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestRegisterInstanceCore(t *testing.T) {
	// Setup test server
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer ts.Close()

	Init(&config.Config{Server: config.ServerConfig{Splitwise: false}})

	t.Run("successful registration", func(t *testing.T) {
		instance := &InstanceInfo{
			Role:             Role{EnumValue: MIXED, IsSet: true},
			HostIP:           "127.0.0.1",
			Port:             Port("8080"),
			TransferProtocol: []string{"rdma"},
		}

		err := RegisterInstanceCore(context.Background(), instance)
		assert.NoError(t, err)
	})

	t.Run("invalid instance info", func(t *testing.T) {
		instance := &InstanceInfo{
			Role: Role{EnumValue: MIXED, IsSet: true},
			// Missing required fields
		}

		err := RegisterInstanceCore(context.Background(), instance)
		assert.Error(t, err)
	})

	t.Run("splitwise mode with mixed instance", func(t *testing.T) {
		Init(&config.Config{Server: config.ServerConfig{Splitwise: true}})

		instance := &InstanceInfo{
			Role:             Role{EnumValue: MIXED, IsSet: true},
			HostIP:           "127.0.0.1",
			Port:             Port("8080"),
			TransferProtocol: []string{"rdma"},
		}

		err := RegisterInstanceCore(context.Background(), instance)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "splitwise mode only supports PREFILL/DECODE instances")
	})

	t.Run("non-splitwise mode with prefill instance", func(t *testing.T) {
		Init(&config.Config{Server: config.ServerConfig{Splitwise: false}})

		instance := &InstanceInfo{
			Role:             Role{EnumValue: PREFILL, IsSet: true},
			HostIP:           "127.0.0.1",
			Port:             Port("8080"),
			TransferProtocol: []string{"rdma"},
		}

		err := RegisterInstanceCore(context.Background(), instance)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "only MIXED instances are allowed")
	})

}

func TestRegisterInstance(t *testing.T) {
	// Setup test server
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer ts.Close()

	Init(&config.Config{Server: config.ServerConfig{Splitwise: false}})

	t.Run("successful registration", func(t *testing.T) {
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)

		body := `{"role": "mixed", "host_ip": "127.0.0.1", "port": 8080, "transfer_protocol": ["rdma"]}`
		c.Request = httptest.NewRequest("POST", "/register", bytes.NewBufferString(body))

		RegisterInstance(c)

		assert.Equal(t, http.StatusOK, w.Code)
		assert.Contains(t, w.Body.String(), "Register success")
	})

	t.Run("invalid JSON", func(t *testing.T) {
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)

		c.Request = httptest.NewRequest("POST", "/register", bytes.NewBufferString("invalid json"))

		RegisterInstance(c)

		assert.Equal(t, http.StatusBadRequest, w.Code)
	})

	t.Run("empty body", func(t *testing.T) {
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)

		c.Request = httptest.NewRequest("POST", "/register", bytes.NewBufferString(""))

		RegisterInstance(c)

		assert.Equal(t, http.StatusBadRequest, w.Code)
	})
}

func TestRegisterInstancesFromConfig(t *testing.T) {
	Init(&config.Config{Server: config.ServerConfig{Splitwise: false}})

	// Create a temporary YAML file
	tmpDir := t.TempDir()
	yamlPath := filepath.Join(tmpDir, "test_config.yaml")

	yamlContent := `
instances:
  - role: mixed
    host_ip: 127.0.0.1
    port: 8080
    transfer_protocol:
      - rdma
`
	err := os.WriteFile(yamlPath, []byte(yamlContent), 0644)
	assert.NoError(t, err)

	// This should not panic
	RegisterInstancesFromConfig(yamlPath)
}

func TestRegisterInstancesFromConfig_InvalidPath(t *testing.T) {
	// Should not panic with invalid path
	RegisterInstancesFromConfig("/nonexistent/path.yaml")
}

func TestRegisterInstancesFromConfig_InvalidYAML(t *testing.T) {
	tmpDir := t.TempDir()
	yamlPath := filepath.Join(tmpDir, "invalid.yaml")

	err := os.WriteFile(yamlPath, []byte("invalid: yaml: content:"), 0644)
	assert.NoError(t, err)

	// Should not panic with invalid YAML
	RegisterInstancesFromConfig(yamlPath)
}

func TestRegisterInstancesFromConfig_EmptyFile(t *testing.T) {
	tmpDir := t.TempDir()
	yamlPath := filepath.Join(tmpDir, "empty.yaml")

	err := os.WriteFile(yamlPath, []byte("instances: []"), 0644)
	assert.NoError(t, err)

	// Should not panic with empty instances
	RegisterInstancesFromConfig(yamlPath)
}

func TestRegisteredNumber(t *testing.T) {
	t.Run("DefaultManager is nil", func(t *testing.T) {
		originalManager := DefaultManager
		DefaultManager = nil
		defer func() { DefaultManager = originalManager }()

		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Request = httptest.NewRequest("GET", "/registered/number", nil)

		RegisteredNumber(c)

		assert.Equal(t, http.StatusBadRequest, w.Code)
		assert.Contains(t, w.Body.String(), "DefaultManager is nil")
	})

	t.Run("successful query", func(t *testing.T) {
		Init(&config.Config{})
		DefaultManager.prefillWorkerMap = map[string]*WorkerInfo{
			"worker1": {Url: "http://worker1"},
		}
		DefaultManager.decodeWorkerMap = map[string]*WorkerInfo{
			"worker2": {Url: "http://worker2"},
		}

		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Request = httptest.NewRequest("GET", "/registered/number", nil)

		RegisteredNumber(c)

		assert.Equal(t, http.StatusOK, w.Code)
		assert.Contains(t, w.Body.String(), `"prefill":1`)
		assert.Contains(t, w.Body.String(), `"decode":1`)
	})
}

func TestRegistered(t *testing.T) {
	Init(&config.Config{})
	DefaultManager.prefillWorkerMap = map[string]*WorkerInfo{
		"worker1": {Url: "http://worker1", WorkerType: "prefill"},
	}
	DefaultManager.decodeWorkerMap = map[string]*WorkerInfo{
		"worker2": {Url: "http://worker2", WorkerType: "decode"},
	}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest("GET", "/registered", nil)

	Registered(c)

	assert.Equal(t, http.StatusOK, w.Code)
	assert.Contains(t, w.Body.String(), `"decode"`)
	assert.Contains(t, w.Body.String(), `"prefill"`)
}
