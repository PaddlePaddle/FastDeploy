package logger

import (
	"log"
	"os"
	"sync"
)

var (
	infoLogger  *log.Logger
	errorLogger *log.Logger
	warnLogger  *log.Logger
	debugLogger *log.Logger
	level       string
	once        sync.Once
)

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
			file, err := os.OpenFile("logs/router.log", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
			if err != nil {
				log.Fatalln("Failed to open log file:", err)
			}
			defer file.Close()
			infoLogger = log.New(file, "[INFO] ", flags)
			errorLogger = log.New(file, "[ERROR] ", flags)
			warnLogger = log.New(file, "[WARN] ", flags)
			debugLogger = log.New(file, "[DEBUG] ", flags)
		} else {
			infoLogger = log.New(os.Stdout, "[INFO] ", flags)
			errorLogger = log.New(os.Stderr, "[ERROR] ", flags)
			warnLogger = log.New(os.Stdout, "[WARN] ", flags)
			debugLogger = log.New(os.Stdout, "[DEBUG] ", flags)
		}
	})
}

// Info logs informational messages
func Info(format string, v ...interface{}) {
	if level == "debug" || level == "info" {
		infoLogger.Printf(format, v...)
	}
}

// Error logs error messages
func Error(format string, v ...interface{}) {
	errorLogger.Printf(format, v...)
}

// Warn logs warning messages
func Warn(format string, v ...interface{}) {
	if level == "debug" || level == "info" || level == "warn" {
		warnLogger.Printf(format, v...)
	}
}

// Debug logs debug messages
func Debug(format string, v ...interface{}) {
	if level == "debug" {
		debugLogger.Printf(format, v...)
	}
}
