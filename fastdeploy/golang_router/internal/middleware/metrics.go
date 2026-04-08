package middleware

import (
	"strconv"
	"time"

	"github.com/PaddlePaddle/FastDeploy/router/pkg/metrics"
	"github.com/gin-gonic/gin"
)

// Metrics provides middleware for collecting HTTP request metrics
func Metrics() gin.HandlerFunc {
	return func(c *gin.Context) {
		path := c.Request.URL.Path
		method := c.Request.Method

		// Time before request processing starts
		start := time.Now()

		c.Next() // Process the request

		// Collect response time statistics after request processing completes
		duration := time.Since(start)
		status := strconv.Itoa(c.Writer.Status())

		// Collect metrics information
		metrics.TotalRequests.WithLabelValues(method, path, status).Inc()                 // Increment request count
		metrics.RequestDuration.WithLabelValues(method, path).Observe(duration.Seconds()) // Record request response time
	}
}
