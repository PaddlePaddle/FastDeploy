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
	logFile     *os.File
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
