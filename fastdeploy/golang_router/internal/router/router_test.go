package router

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/PaddlePaddle/FastDeploy/router/internal/config"
	"github.com/PaddlePaddle/FastDeploy/router/pkg/logger"
	"github.com/stretchr/testify/assert"
)

func init() {
	// Initialize logger for all tests
	logger.Init("info", "stdout")
}

func TestMiddlewareSetup(t *testing.T) {
	cfg := &config.Config{
		Server: config.ServerConfig{
			Mode: "test",
		},
	}

	router := New(cfg)

	// Test CORS middleware
	t.Run("CORS headers", func(t *testing.T) {
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/", nil)
		router.ServeHTTP(w, req)

		assert.Equal(t, "*", w.Header().Get("Access-Control-Allow-Origin"))
		assert.Equal(t, "true", w.Header().Get("Access-Control-Allow-Credentials"))
	})

	// Test panic recovery - this test is now in middleware_test.go
}
