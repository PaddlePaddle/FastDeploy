package logger

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

// Config holds logger configuration.
type Config struct {
	Level               string
	Output              string
	MaxAgeDays          int
	MaxTotalSizeMB      int
	CleanupIntervalSecs float64
}

var (
	infoLogger  *log.Logger
	errorLogger *log.Logger
	warnLogger  *log.Logger
	debugLogger *log.Logger
	level       string
	once        sync.Once
	writer      *rotatingWriter // nil when output is stdout
)

// nowFunc is overridable in tests for time-dependent logic.
var nowFunc = time.Now

type contextKey string

const TraceIDKey contextKey = "trace_id"
const ReqIDKey contextKey = "req_id"
const RequestIDKey contextKey = "request_id"
const SessionIDKey contextKey = "session_id"

// gracePeriod is how long we keep the previous day's file open after rotation.
const gracePeriod = 5 * time.Minute

// rotatingWriter implements io.Writer with day-level rotation and dual-file writes.
// Current day's log is written to "router-YYYY-MM-DD.log" and "router.log" is a
// symlink pointing to the current day's file. On day change a new date file is
// created and the symlink is updated. During a short grace period after rotation,
// log lines whose timestamp belongs to the previous day are written to the old file.
type rotatingWriter struct {
	mu          sync.Mutex
	currentFile *os.File  // today's router-<date>.log
	prevFile    *os.File  // previous day's router-<date>.log during grace period (may be nil)
	currentDate string    // "2006-01-02"
	prevDate    string    // previous date during grace period
	graceUntil  time.Time // when to close prevFile
	logDir      string
}

func newRotatingWriter(logDir string) (*rotatingWriter, error) {
	today := nowFunc().Format("2006-01-02")
	datePath := filepath.Join(logDir, "router-"+today+".log")
	symlinkPath := filepath.Join(logDir, "router.log")

	// Migration: if router.log is a regular file (legacy), rename it to the date file.
	if info, err := os.Lstat(symlinkPath); err == nil && info.Mode().IsRegular() {
		os.Rename(symlinkPath, datePath)
	}

	// Open the date file (append mode).
	f, err := os.OpenFile(datePath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
	if err != nil {
		return nil, err
	}

	// Create/update symlink: router.log -> router-<today>.log
	if err := os.Remove(symlinkPath); err != nil && !os.IsNotExist(err) {
		fmt.Fprintf(os.Stderr, "[ERROR] Failed to remove symlink %s: %v\n", symlinkPath, err)
	}
	if err := os.Symlink("router-"+today+".log", symlinkPath); err != nil {
		fmt.Fprintf(os.Stderr, "[ERROR] Failed to create symlink %s: %v\n", symlinkPath, err)
	}

	return &rotatingWriter{
		currentFile: f,
		currentDate: today,
		logDir:      logDir,
	}, nil
}

func (w *rotatingWriter) Write(p []byte) (n int, err error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	today := nowFunc().Format("2006-01-02")

	// Detect day change and rotate.
	if today != w.currentDate {
		w.rotateLocked(today)
	}

	// Close previous file if grace period expired.
	if w.prevFile != nil && nowFunc().After(w.graceUntil) {
		w.prevFile.Close()
		w.prevFile = nil
		w.prevDate = ""
	}

	// During grace period, route log lines to the correct file based on timestamp.
	target := w.currentFile
	if w.prevFile != nil {
		if logDate := parseLogDate(p); logDate == w.prevDate {
			target = w.prevFile
		}
	}

	return target.Write(p)
}

func (w *rotatingWriter) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.prevFile != nil {
		w.prevFile.Close()
		w.prevFile = nil
	}
	if w.currentFile != nil {
		return w.currentFile.Close()
	}
	return nil
}

// rotateLocked performs the actual file rotation. Must be called with w.mu held.
func (w *rotatingWriter) rotateLocked(newDate string) {
	// Close any lingering previous file.
	if w.prevFile != nil {
		w.prevFile.Close()
		w.prevFile = nil
	}

	// Keep the old date file open for grace period writes.
	w.prevFile = w.currentFile
	w.prevDate = w.currentDate
	w.graceUntil = nowFunc().Add(gracePeriod)

	// Open new date file for the new day.
	datePath := filepath.Join(w.logDir, "router-"+newDate+".log")
	f, err := os.OpenFile(datePath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
	if err != nil {
		log.Printf("[ERROR] failed to open new log file %s: %v, keeping current file", datePath, err)
		if w.prevFile != nil {
			w.currentFile = w.prevFile
			w.currentDate = w.prevDate
			w.prevFile = nil
			w.prevDate = ""
		}
		return
	}
	w.currentFile = f
	w.currentDate = newDate

	// Update symlink: router.log -> router-<newDate>.log
	symlinkPath := filepath.Join(w.logDir, "router.log")
	if err := os.Remove(symlinkPath); err != nil && !os.IsNotExist(err) {
		fmt.Fprintf(os.Stderr, "[ERROR] Failed to remove symlink %s: %v\n", symlinkPath, err)
	}
	if err := os.Symlink("router-"+newDate+".log", symlinkPath); err != nil {
		fmt.Fprintf(os.Stderr, "[ERROR] Failed to create symlink %s: %v\n", symlinkPath, err)
	}
}

// parseLogDate extracts the date from a log line produced by log.LstdFlags.
// Format: "[LEVEL] 2006/01/02 15:04:05 ..."
// Returns "2006-01-02" or empty string on parse failure.
func parseLogDate(p []byte) string {
	// Find the date pattern "YYYY/MM/DD" in the log prefix.
	// log.LstdFlags produces: "2006/01/02 15:04:05" after the logger prefix.
	// The prefix is like "[INFO] " (7 chars), so the date starts around index 7.
	s := string(p)
	for i := 0; i+10 <= len(s); i++ {
		c := s[i]
		if c >= '0' && c <= '9' && i+10 <= len(s) && s[i+4] == '/' && s[i+7] == '/' {
			// Found a candidate "YYYY/MM/DD"
			year := s[i : i+4]
			month := s[i+5 : i+7]
			day := s[i+8 : i+10]
			return year + "-" + month + "-" + day
		}
	}
	return ""
}

// Init initializes the logger.
func Init(cfg Config) {
	once.Do(func() {
		level = cfg.Level
		flags := log.LstdFlags | log.Lshortfile

		if cfg.Output == "file" {
			if _, err := os.Stat("logs"); os.IsNotExist(err) {
				if err := os.MkdirAll("logs", 0755); err != nil {
					log.Fatalln("Failed to create logs directory:", err)
				}
			}
			var err error
			writer, err = newRotatingWriter("logs")
			if err != nil {
				log.Fatalln("Failed to create rotating log writer:", err)
			}
			infoLogger = log.New(writer, "[INFO] ", flags)
			errorLogger = log.New(writer, "[ERROR] ", flags)
			warnLogger = log.New(writer, "[WARN] ", flags)
			debugLogger = log.New(writer, "[DEBUG] ", flags)
		} else {
			infoLogger = log.New(os.Stdout, "[INFO] ", flags)
			errorLogger = log.New(os.Stderr, "[ERROR] ", flags)
			warnLogger = log.New(os.Stdout, "[WARN] ", flags)
			debugLogger = log.New(os.Stdout, "[DEBUG] ", flags)
		}
	})
}

// CloseLogFile closes the log file if in file output mode.
func CloseLogFile() {
	if writer != nil {
		writer.Close()
	}
}

// StartLogCleanup runs periodic log cleanup in a background goroutine.
// It deletes archived log files older than MaxAgeDays and trims total log size
// to stay under MaxTotalSizeMB.
func StartLogCleanup(ctx context.Context, cfg Config) {
	if cfg.Output != "file" {
		return
	}
	if cfg.CleanupIntervalSecs <= 0 {
		return
	}

	ticker := time.NewTicker(time.Duration(cfg.CleanupIntervalSecs * float64(time.Second)))
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			cleanupLogs("logs", cfg.MaxAgeDays, cfg.MaxTotalSizeMB)
		}
	}
}

type logFileInfo struct {
	name string
	path string
	date time.Time
	size int64
}

// cleanupLogs removes archived log files based on age and total size limits.
func cleanupLogs(logDir string, maxAgeDays, maxTotalSizeMB int) {
	entries, err := os.ReadDir(logDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "[WARN] Failed to read log directory for cleanup: %v\n", err)
		return
	}

	now := nowFunc()
	today := now.Format("2006-01-02")
	var archives []logFileInfo

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()

		// router.log is now a symlink; skip it.
		if name == "router.log" {
			continue
		}

		// Match archived files: router-YYYY-MM-DD.log
		if !strings.HasPrefix(name, "router-") || !strings.HasSuffix(name, ".log") {
			continue
		}
		dateStr := strings.TrimPrefix(name, "router-")
		dateStr = strings.TrimSuffix(dateStr, ".log")
		fileDate, err := time.Parse("2006-01-02", dateStr)
		if err != nil {
			continue
		}
		// Never delete today's active date file.
		if dateStr == today {
			continue
		}
		info, err := entry.Info()
		if err != nil {
			continue
		}
		archives = append(archives, logFileInfo{
			name: name,
			path: filepath.Join(logDir, name),
			date: fileDate,
			size: info.Size(),
		})
	}

	// Sort by date ascending (oldest first).
	sort.Slice(archives, func(i, j int) bool {
		return archives[i].date.Before(archives[j].date)
	})

	// Phase 1: Age-based cleanup.
	if maxAgeDays > 0 {
		cutoff := now.AddDate(0, 0, -maxAgeDays)
		remaining := archives[:0]
		for _, f := range archives {
			if f.date.Before(cutoff) {
				if err := os.Remove(f.path); err != nil {
					fmt.Fprintf(os.Stderr, "[ERROR] Failed to remove log file %s: %v\n", f.path, err)
				}
			} else {
				remaining = append(remaining, f)
			}
		}
		archives = remaining
	}

	// Phase 2: Size-based cleanup.
	if maxTotalSizeMB > 0 {
		maxBytes := int64(maxTotalSizeMB) * 1024 * 1024
		var totalSize int64
		for _, f := range archives {
			totalSize += f.size
		}
		for len(archives) > 0 && totalSize > maxBytes {
			oldest := archives[0]
			if err := os.Remove(oldest.path); err != nil {
				fmt.Fprintf(os.Stderr, "[ERROR] Failed to remove log file %s: %v\n", oldest.path, err)
			}
			totalSize -= oldest.size
			archives = archives[1:]
		}
	}
}

func contextPrefix(ctx context.Context) string {
	if ctx == nil {
		return ""
	}
	var prefix string
	if tid, ok := ctx.Value(TraceIDKey).(string); ok && tid != "" {
		prefix += "[trace_id:" + tid + "] "
	}
	if reqID, ok := ctx.Value(ReqIDKey).(string); ok && reqID != "" {
		prefix += "[req_id:" + reqID + "] "
	}
	if sid, ok := ctx.Value(SessionIDKey).(string); ok && sid != "" {
		prefix += "[session_id:" + sid + "] "
	}
	if rid, ok := ctx.Value(RequestIDKey).(string); ok && rid != "" {
		prefix += "[request_id:" + rid + "] "
	}
	return prefix
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
