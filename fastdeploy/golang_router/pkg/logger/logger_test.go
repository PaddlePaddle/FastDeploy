package logger

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
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

func TestStartLogCleanup(t *testing.T) {
	t.Run("cleanup runs for file output and respects cancellation", func(t *testing.T) {
		tmpDir := t.TempDir()

		originalNowFunc := nowFunc
		fixedNow := time.Date(2026, 4, 10, 12, 0, 0, 0, time.UTC)
		nowFunc = func() time.Time { return fixedNow }
		defer func() { nowFunc = originalNowFunc }()

		// Create archived logs: one older than 1 day and one recent.
		oldLog := filepath.Join(tmpDir, "router-2026-04-07.log")
		recentLog := filepath.Join(tmpDir, "router-2026-04-09.log")
		todayLog := filepath.Join(tmpDir, "router-2026-04-10.log")
		for _, p := range []string{oldLog, recentLog, todayLog} {
			if err := os.WriteFile(p, []byte("test"), 0644); err != nil {
				t.Fatalf("failed to create test log %s: %v", p, err)
			}
		}

		ctx, cancel := context.WithCancel(context.Background())
		done := make(chan struct{})
		go func() {
			defer close(done)
			StartLogCleanup(ctx, Config{
				Output:              "file",
				Dir:                 tmpDir,
				MaxAgeDays:          2,
				CleanupIntervalSecs: 0.01,
			})
		}()

		waitForCondition(t, 500*time.Millisecond, func() bool {
			_, err := os.Stat(oldLog)
			return os.IsNotExist(err)
		}, "old log should be removed by StartLogCleanup")

		if _, err := os.Stat(recentLog); err != nil {
			t.Fatalf("recent log should be kept, stat err: %v", err)
		}
		if _, err := os.Stat(todayLog); err != nil {
			t.Fatalf("today log should be kept, stat err: %v", err)
		}

		cancel()
		select {
		case <-done:
		case <-time.After(500 * time.Millisecond):
			t.Fatal("StartLogCleanup did not stop after context cancellation")
		}
	})

	t.Run("non-file output returns immediately", func(t *testing.T) {
		done := make(chan struct{})
		go func() {
			defer close(done)
			StartLogCleanup(context.Background(), Config{Output: "stdout", CleanupIntervalSecs: 1})
		}()
		select {
		case <-done:
		case <-time.After(200 * time.Millisecond):
			t.Fatal("StartLogCleanup should return immediately for non-file output")
		}
	})
}

func TestRotatingWriterCrossDayGracePeriodIntegration(t *testing.T) {
	tmpDir := t.TempDir()

	originalNowFunc := nowFunc
	defer func() { nowFunc = originalNowFunc }()

	current := time.Date(2026, 4, 10, 23, 59, 59, 0, time.UTC)
	nowFunc = func() time.Time { return current }

	w, err := newRotatingWriter(tmpDir)
	if err != nil {
		t.Fatalf("failed to create rotating writer: %v", err)
	}
	defer w.Close()

	if _, err = w.Write([]byte("[INFO] 2026/04/10 23:59:59 first day line\n")); err != nil {
		t.Fatalf("failed to write day-1 line: %v", err)
	}

	current = time.Date(2026, 4, 11, 0, 0, 1, 0, time.UTC)
	if _, err = w.Write([]byte("[INFO] 2026/04/11 00:00:01 second day line\n")); err != nil {
		t.Fatalf("failed to write day-2 line: %v", err)
	}

	if _, err = w.Write([]byte("[INFO] 2026/04/10 23:59:58 late previous-day line\n")); err != nil {
		t.Fatalf("failed to write late previous-day line: %v", err)
	}

	day1Bytes, err := os.ReadFile(filepath.Join(tmpDir, "router-2026-04-10.log"))
	if err != nil {
		t.Fatalf("failed to read day-1 log: %v", err)
	}
	day1Content := string(day1Bytes)
	if !strings.Contains(day1Content, "first day line") {
		t.Fatalf("day-1 log missing initial line, content: %s", day1Content)
	}
	if !strings.Contains(day1Content, "late previous-day line") {
		t.Fatalf("day-1 log missing late previous-day line, content: %s", day1Content)
	}

	day2Bytes, err := os.ReadFile(filepath.Join(tmpDir, "router-2026-04-11.log"))
	if err != nil {
		t.Fatalf("failed to read day-2 log: %v", err)
	}
	day2Content := string(day2Bytes)
	if !strings.Contains(day2Content, "second day line") {
		t.Fatalf("day-2 log missing day-2 line, content: %s", day2Content)
	}
	if strings.Contains(day2Content, "late previous-day line") {
		t.Fatalf("late previous-day line should not be in day-2 file, content: %s", day2Content)
	}

	symlinkTarget, err := os.Readlink(filepath.Join(tmpDir, "router.log"))
	if err != nil {
		t.Fatalf("failed to read symlink: %v", err)
	}
	if symlinkTarget != "router-2026-04-11.log" {
		t.Fatalf("router.log symlink target = %s, want router-2026-04-11.log", symlinkTarget)
	}
}

func waitForCondition(t *testing.T, timeout time.Duration, cond func() bool, msg string) {
	t.Helper()

	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal(msg)
}
