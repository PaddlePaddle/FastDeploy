package logger

import (
	"bytes"
	"os"
	"strings"
	"testing"
)

func TestLoggerInit(t *testing.T) {
	t.Run("stdout output", func(t *testing.T) {
		Init("debug", "stdout")

		if infoLogger == nil || errorLogger == nil || warnLogger == nil || debugLogger == nil {
			t.Error("Loggers should be initialized")
		}
	})

	t.Run("file output", func(t *testing.T) {
		// Clean up existing log file and directory
		_ = os.RemoveAll("logs")
		_ = os.MkdirAll("logs", 0755)
		defer os.RemoveAll("logs")

		Init("debug", "file")

		if _, err := os.Stat("logs/router.log"); os.IsNotExist(err) {
			t.Error("Log file should be created")
		}
	})
}

func TestLogLevels(t *testing.T) {
	tests := []struct {
		name     string
		level    string
		expected map[string]bool
	}{
		{"debug level", "debug", map[string]bool{
			"debug": true,
			"info":  true,
			"warn":  true,
			"error": true,
		}},
		{"info level", "info", map[string]bool{
			"debug": false,
			"info":  true,
			"warn":  true,
			"error": true,
		}},
		{"warn level", "warn", map[string]bool{
			"debug": false,
			"info":  false,
			"warn":  true,
			"error": true,
		}},
		{"error level", "error", map[string]bool{
			"debug": false,
			"info":  false,
			"warn":  false,
			"error": true,
		}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Initialize logger with test level
			Init(tt.level, "stdout")

			// Capture output for each level separately
			testLevel := func(logFunc func(string, ...interface{}), message string) bool {
				var buf bytes.Buffer
				oldOutput := infoLogger.Writer()

				infoLogger.SetOutput(&buf)
				errorLogger.SetOutput(&buf)
				warnLogger.SetOutput(&buf)
				debugLogger.SetOutput(&buf)

				logFunc(message)

				infoLogger.SetOutput(oldOutput)
				errorLogger.SetOutput(oldOutput)
				warnLogger.SetOutput(oldOutput)
				debugLogger.SetOutput(oldOutput)

				return strings.Contains(buf.String(), message)
			}

			debugPrinted := testLevel(Debug, "debug message")
			infoPrinted := testLevel(Info, "info message")
			warnPrinted := testLevel(Warn, "warn message")
			errorPrinted := testLevel(Error, "error message")

			// Check expected behavior
			if tt.expected["debug"] != debugPrinted {
				t.Errorf("Debug log: expected %v, got %v", tt.expected["debug"], debugPrinted)
			}
			if tt.expected["info"] != infoPrinted {
				t.Errorf("Info log: expected %v, got %v", tt.expected["info"], infoPrinted)
			}
			if tt.expected["warn"] != warnPrinted {
				t.Errorf("Warn log: expected %v, got %v", tt.expected["warn"], warnPrinted)
			}
			if tt.expected["error"] != errorPrinted {
				t.Errorf("Error log: expected %v, got %v", tt.expected["error"], errorPrinted)
			}
		})
	}
}

func TestLogFunctions(t *testing.T) {
	var buf bytes.Buffer
	Init("debug", "stdout")

	// Redirect output
	oldOutput := infoLogger.Writer()
	defer func() { infoLogger.SetOutput(oldOutput) }()
	infoLogger.SetOutput(&buf)

	Info("test %s", "message")
	if !strings.Contains(buf.String(), "test message") {
		t.Error("Info log should contain the message")
	}

	// Similar tests for Error, Warn, Debug...
}
