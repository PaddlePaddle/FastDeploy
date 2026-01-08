package middleware

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/PaddlePaddle/FastDeploy/router/pkg/logger"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
)

func init() {
	// Initialize logger to avoid nil pointer dereference in recovery middleware
	logger.Init("info", "stdout")
}

func TestLoggerMiddleware(t *testing.T) {
	router := gin.New()
	router.Use(Logger())

	router.GET("/test", func(c *gin.Context) {
		c.String(200, "OK")
	})

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/test", nil)
	router.ServeHTTP(w, req)

	assert.Equal(t, 200, w.Code)
}

func TestRecoveryMiddleware(t *testing.T) {
	router := gin.New()
	router.Use(Recovery())

	router.GET("/panic", func(c *gin.Context) {
		panic("test panic")
	})

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/panic", nil)
	router.ServeHTTP(w, req)

	assert.Equal(t, 500, w.Code)
	// The response should contain the error message
	assert.Contains(t, w.Body.String(), "Internal server error")
}
