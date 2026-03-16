package logger

import (
	"log"
	"os"
	"sync"
	"context"
)

var (
	infoLogger  *log.Logger
	errorLogger *log.Logger
	warnLogger  *log.Logger
	debugLogger *log.Logger
	level       string
	once        sync.Once
	logFile     *os.File
)

type contextKey string
const RequestIDKey contextKey = "request_id"

// Init initialize logger
func Init(logLevel, output string) {
	once.Do(func() {
		level = logLevel

		flags := log.LstdFlags | log.Lshortfile

		if output == "file" {
			// Check if logs directory exists
			if _, err := os.Stat("logs"); os.IsNotExist(err) {
				if err := os.MkdirAll("logs", 0755); err != nil {
					log.Fatalln("Failed to create logs directory:", err)
				}
			}
			logFile, err := os.OpenFile("logs/router.log", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
			if err != nil {
				log.Fatalln("Failed to open log file:", err)
			}
			infoLogger = log.New(logFile, "[INFO] ", flags)
			errorLogger = log.New(logFile, "[ERROR] ", flags)
			warnLogger = log.New(logFile, "[WARN] ", flags)
			debugLogger = log.New(logFile, "[DEBUG] ", flags)
		} else {
			infoLogger = log.New(os.Stdout, "[INFO] ", flags)
			errorLogger = log.New(os.Stderr, "[ERROR] ", flags)
			warnLogger = log.New(os.Stdout, "[WARN] ", flags)
			debugLogger = log.New(os.Stdout, "[DEBUG] ", flags)
		}
	})
}

func CloseLogFile() {
	if logFile != nil {
		logFile.Close()
	}
}

func contextPrefix(ctx context.Context) string {
	if ctx == nil {
		return ""
	}
	if rid, ok := ctx.Value(RequestIDKey).(string); ok && rid != "" {
		return "[request_id:" + rid + "] "
	}
	return ""
}

// Info logs informational messages
func Info(ctx context.Context, format string, v ...interface{}) {
	if level == "debug" || level == "info" {
		prefix := contextPrefix(ctx)
		infoLogger.Printf(prefix+format, v...)
	}
}

// Error logs error messages
func Error(ctx context.Context, format string, v ...interface{}) {
	prefix := contextPrefix(ctx)
	errorLogger.Printf(prefix+format, v...)
}

// Warn logs warning messages
func Warn(ctx context.Context, format string, v ...interface{}) {
	if level == "debug" || level == "info" || level == "warn" {
		prefix := contextPrefix(ctx)
		warnLogger.Printf(prefix+format, v...)
	}
}

// Debug logs debug messages
func Debug(ctx context.Context, format string, v ...interface{}) {
	if level == "debug" {
		prefix := contextPrefix(ctx)
		debugLogger.Printf(prefix+format, v...)
	}
}
