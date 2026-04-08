package middleware

import (
	"github.com/PaddlePaddle/FastDeploy/router/pkg/logger"
	"github.com/gin-gonic/gin"
)

// Logger logger middleware
func Logger() gin.HandlerFunc {
	return gin.LoggerWithFormatter(func(param gin.LogFormatterParams) string {
		logger.Info("[%s] %s %s %d %s %s",
			param.Method,
			param.Path,
			param.Request.Proto,
			param.StatusCode,
			param.Latency,
			param.ClientIP,
		)
		return ""
	})
}

// Recovery recovery middleware
func Recovery() gin.HandlerFunc {
	return gin.CustomRecovery(func(c *gin.Context, recovered interface{}) {
		logger.Error("Panic recovered: %v", recovered)
		c.JSON(500, gin.H{
			"code": 500,
			"msg":  "Internal server error",
		})
		c.Abort()
	})
}
