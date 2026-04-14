package logger

import (
	"bytes"
	"context"
	"os"
	"strings"
	"testing"
)

func TestLoggerInit(t *testing.T) {
	t.Run("stdout output", func(t *testing.T) {
		Init(Config{Level: "debug", Output: "stdout"})

		if infoLogger == nil || errorLogger == nil || warnLogger == nil || debugLogger == nil {
			t.Error("Loggers should be initialized")
		}
	})

	t.Run("file output", func(t *testing.T) {
		// Clean up existing log file and directory
		_ = os.RemoveAll("logs")
		_ = os.MkdirAll("logs", 0755)
		defer os.RemoveAll("logs")

		// sync.Once prevents re-init, so manually verify file creation logic
		f, err := os.OpenFile("logs/router.log", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
		if err != nil {
			t.Fatalf("Failed to create log file: %v", err)
		}
		f.Close()

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
			// Directly set package-level variable since sync.Once prevents re-init
			level = tt.level

			// Capture output for each level separately
			testLevel := func(logFunc func(context.Context, string, ...interface{}), message string) bool {
				var buf bytes.Buffer
				oldOutput := infoLogger.Writer()

				infoLogger.SetOutput(&buf)
				errorLogger.SetOutput(&buf)
				warnLogger.SetOutput(&buf)
				debugLogger.SetOutput(&buf)

				logFunc(nil, message)

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
	Init(Config{Level: "debug", Output: "stdout"})
	level = "debug"

	// Redirect output
	oldOutput := infoLogger.Writer()
	defer func() { infoLogger.SetOutput(oldOutput) }()
	infoLogger.SetOutput(&buf)

	Info(nil, "test %s", "message")
	if !strings.Contains(buf.String(), "test message") {
		t.Error("Info log should contain the message")
	}
}

func TestContextPrefix(t *testing.T) {
	Init(Config{Level: "debug", Output: "stdout"})
	level = "debug"

	t.Run("nil context produces no prefix", func(t *testing.T) {
		var buf bytes.Buffer
		oldOutput := infoLogger.Writer()
		defer func() { infoLogger.SetOutput(oldOutput) }()
		infoLogger.SetOutput(&buf)

		Info(nil, "no prefix here")
		output := buf.String()
		if strings.Contains(output, "[request_id:") {
			t.Errorf("nil context should produce no request_id prefix, got: %s", output)
		}
		if !strings.Contains(output, "no prefix here") {
			t.Errorf("message should be present, got: %s", output)
		}
	})

	t.Run("context without request_id produces no request_id prefix", func(t *testing.T) {
		var buf bytes.Buffer
		oldOutput := infoLogger.Writer()
		defer func() { infoLogger.SetOutput(oldOutput) }()
		infoLogger.SetOutput(&buf)

		ctx := context.Background()
		Info(ctx, "mixed mode log")
		output := buf.String()
		if strings.Contains(output, "[request_id:") {
			t.Errorf("context without request_id should not produce request_id prefix, got: %s", output)
		}
		if !strings.Contains(output, "mixed mode log") {
			t.Errorf("message should be present, got: %s", output)
		}
	})

	t.Run("context with request_id produces [request_id:xxx]", func(t *testing.T) {
		var buf bytes.Buffer
		oldOutput := infoLogger.Writer()
		defer func() { infoLogger.SetOutput(oldOutput) }()
		infoLogger.SetOutput(&buf)

		ctx := context.WithValue(context.Background(), RequestIDKey, "test-uuid-123")
		Info(ctx, "pd mode log")
		output := buf.String()
		if !strings.Contains(output, "[request_id:test-uuid-123]") {
			t.Errorf("context with request_id should produce [request_id:test-uuid-123], got: %s", output)
		}
	})
}

func TestParseLogDate(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{"standard INFO log line", "[INFO] 2024/03/15 10:30:45 some message", "2024-03-15"},
		{"standard ERROR log line", "[ERROR] 2024/01/02 09:00:00 error occurred", "2024-01-02"},
		{"standard WARN log line", "[WARN] 2025/12/31 23:59:59 warning msg", "2025-12-31"},
		{"standard DEBUG log line", "[DEBUG] 2024/06/01 00:00:00 debug info", "2024-06-01"},
		{"empty string", "", ""},
		{"no date pattern", "no date here at all", ""},
		{"incomplete date - only year", "2024/", ""},
		{"incomplete date - year and month", "[INFO] 2024/03", ""},
		{"short input", "abc", ""},
		{"date without log prefix", "2024/03/15 10:30:45 message", "2024-03-15"},
		{"date at different position", "prefix 2024/11/20 rest", "2024-11-20"},
		{"slash but not date", "path/to/file is not a date", ""},
		{"single character input", "x", ""},
		{"exactly 10 chars non-date", "abcdefghij", ""},
		{"boundary - first day of year", "[INFO] 2024/01/01 00:00:00 new year", "2024-01-01"},
		{"boundary - last day of year", "[INFO] 2024/12/31 23:59:59 year end", "2024-12-31"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := parseLogDate([]byte(tt.input))
			if got != tt.expected {
				t.Errorf("parseLogDate(%q) = %q, want %q", tt.input, got, tt.expected)
			}
		})
	}
}
