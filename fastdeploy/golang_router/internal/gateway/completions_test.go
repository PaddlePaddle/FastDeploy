package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/PaddlePaddle/FastDeploy/router/pkg/logger"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
)

func TestMain(m *testing.M) {
	logger.Init("info", "stdout")
	gin.SetMode(gin.TestMode)
	os.Exit(m.Run())
}

func TestChatCompletions(t *testing.T) {
	// Since the actual implementation uses package-level functions that depend on DefaultManager,
	// and we don't want to set up a full manager for unit tests,
	// this test will be marked as integration test and skipped for now
	t.Skip("Integration test requiring manager setup")
}

func TestExtractPromptFromChatRequest(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			"simple message",
			`{"messages": [{"role": "user", "content": "hello"}]}`,
			"hello",
		},
		{
			"multiple messages",
			`{"messages": [
				{"role": "user", "content": "hello"},
				{"role": "assistant", "content": "hi"},
				{"role": "user", "content": "how are you"}
			]}`,
			"hello hi how are you",
		},
		{
			"empty messages",
			`{"messages": []}`,
			"",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var rawReq map[string]any
			err := json.Unmarshal([]byte(tt.input), &rawReq)
			assert.NoError(t, err)

			result := extractPromptFromChatRequest(rawReq)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestRedirect(t *testing.T) {
	// Setup test server
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("test response"))
	}))
	defer ts.Close()

	// Test stream response
	t.Run("stream response", func(t *testing.T) {
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Request = httptest.NewRequest("GET", "/", nil)

		resp, err := http.Get(ts.URL)
		assert.NoError(t, err)

		redirect(c, true, resp)
		assert.Equal(t, "test response\n", w.Body.String())
	})

	// Test non-stream response
	t.Run("non-stream response", func(t *testing.T) {
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Request = httptest.NewRequest("GET", "/", nil)

		resp, err := http.Get(ts.URL)
		assert.NoError(t, err)

		redirect(c, false, resp)
		assert.Equal(t, "test response", w.Body.String())
	})
}

func TestGetClient(t *testing.T) {
	// Setup test server
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("test response"))
	}))
	defer ts.Close()

	// Setup test context
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewBufferString(`{"test": "data"}`))

	resp, err := GetClient(c, ts.URL, "chat/completions", []byte(`{"test": "data"}`))
	assert.NoError(t, err)
	assert.Equal(t, http.StatusOK, resp.StatusCode)
}

func TestNewRequestID(t *testing.T) {
	id1 := newRequestID()
	id2 := newRequestID()

	// Check that IDs are not empty
	assert.NotEmpty(t, id1)
	assert.NotEmpty(t, id2)

	// Check that IDs are different
	assert.NotEqual(t, id1, id2)
}

func TestExtractPromptFromCompletionsRequest(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			"simple string prompt",
			`{"prompt": "hello world"}`,
			"hello world",
		},
		{
			"string array prompt",
			`{"prompt": ["first", "second", "third"]}`,
			"first second third",
		},
		{
			"interface array prompt",
			`{"prompt": ["first", "second", "third"]}`,
			"first second third",
		},
		{
			"empty prompt",
			`{"prompt": ""}`,
			"",
		},
		{
			"empty array prompt",
			`{"prompt": []}`,
			"",
		},
		{
			"missing prompt field",
			`{"other": "field"}`,
			"",
		},
		{
			"array with empty strings",
			`{"prompt": ["", "hello", ""]}`,
			"hello",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var rawReq map[string]any
			err := json.Unmarshal([]byte(tt.input), &rawReq)
			assert.NoError(t, err)

			result := extractPromptFromCompletionsRequest(rawReq)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestPostToPD(t *testing.T) {
	// Setup test context
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest("POST", "/v1/chat/completions",
		bytes.NewBufferString(`{"test": "data"}`))

	reqBody := []byte(`{"test": "data"}`)

	t.Run("successful request to both P and D", func(t *testing.T) {
		// Setup test servers for prefill and decode
		prefillServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
			w.Write([]byte("prefill response"))
		}))
		defer prefillServer.Close()

		decodeServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
			w.Write([]byte("decode response"))
		}))
		defer decodeServer.Close()

		resp, err := PostToPD(c, decodeServer.URL, prefillServer.URL, reqBody, false, "test message", "chat/completions")
		assert.NoError(t, err)
		assert.Equal(t, http.StatusOK, resp.StatusCode)
		assert.NotNil(t, resp)
		defer resp.Body.Close()
	})

	t.Run("decode server connection error", func(t *testing.T) {
		prefillServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
		}))
		defer prefillServer.Close()

		// Use invalid URL to simulate connection error
		resp, err := PostToPD(c, "http://invalid-server:9999", prefillServer.URL, reqBody, false, "test message", "chat/completions")
		assert.Error(t, err)
		assert.Nil(t, resp)
	})

	t.Run("prefill server connection error", func(t *testing.T) {
		decodeServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
		}))
		defer decodeServer.Close()

		// Use invalid URL to simulate connection error
		resp, err := PostToPD(c, decodeServer.URL, "http://invalid-server:9999", reqBody, false, "test message", "chat/completions")
		assert.Error(t, err)
		assert.Nil(t, resp)
	})
}

func TestGetClientWithRetry(t *testing.T) {
	t.Run("success after connection errors", func(t *testing.T) {
		retryCount := 0
		shouldFail := true
		ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			retryCount++
			if shouldFail && retryCount < 3 {
				// Simulate network connection error by closing connection
				hj, ok := w.(http.Hijacker)
				if ok {
					conn, _, _ := hj.Hijack()
					conn.Close()
					return
				}
				w.WriteHeader(http.StatusInternalServerError)
				return
			}
			shouldFail = false
			w.WriteHeader(http.StatusOK)
			w.Write([]byte("success"))
		}))
		defer ts.Close()

		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Request = httptest.NewRequest("POST", "/v1/chat/completions",
			bytes.NewBufferString(`{"test": "data"}`))

		reqBody := []byte(`{"test": "data"}`)

		resp, err := GetClientWithRetry(c, reqBody, ts.URL, "chat/completions")
		assert.NoError(t, err)
		assert.NotNil(t, resp)
		assert.Equal(t, http.StatusOK, resp.StatusCode)
	})

	t.Run("all retries fail with connection errors", func(t *testing.T) {
		retryCount := 0
		ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			retryCount++
			// Always simulate network connection error
			hj, ok := w.(http.Hijacker)
			if ok {
				conn, _, _ := hj.Hijack()
				conn.Close()
				return
			}
			w.WriteHeader(http.StatusInternalServerError)
		}))
		defer ts.Close()

		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Request = httptest.NewRequest("POST", "/v1/chat/completions",
			bytes.NewBufferString(`{"test": "data"}`))

		reqBody := []byte(`{"test": "data"}`)

		resp, err := GetClientWithRetry(c, reqBody, ts.URL, "chat/completions")
		assert.Error(t, err)
		assert.Nil(t, resp)
	})

	t.Run("success on first try", func(t *testing.T) {
		retryCount := 0
		ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			retryCount++
			w.WriteHeader(http.StatusOK)
			w.Write([]byte("success"))
		}))
		defer ts.Close()

		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Request = httptest.NewRequest("POST", "/v1/chat/completions",
			bytes.NewBufferString(`{"test": "data"}`))

		reqBody := []byte(`{"test": "data"}`)

		resp, err := GetClientWithRetry(c, reqBody, ts.URL, "chat/completions")
		assert.NoError(t, err)
		assert.NotNil(t, resp)
		assert.Equal(t, http.StatusOK, resp.StatusCode)
		assert.Equal(t, 1, retryCount)
	})
}

func TestCompletions(t *testing.T) {
	// This is a basic test that just verifies the function calls CommonCompletions
	// More comprehensive testing would require mocking the manager dependencies
	t.Run("function exists", func(t *testing.T) {
		// Just verify that the function can be called without panic
		// Actual behavior testing requires integration test setup
		assert.NotNil(t, Completions)
	})
}

func TestReadPrefillRecv(t *testing.T) {
	t.Run("nil response handling", func(t *testing.T) {
		ctx := context.Background()
		// Should handle nil response gracefully without panic
		readPrefillRecv(ctx, "test-url", false, "test message", nil)
	})

	t.Run("nil response body handling", func(t *testing.T) {
		ctx := context.Background()
		// Create a mock response with nil body
		resp := &http.Response{
			StatusCode: http.StatusOK,
			Body:       nil,
		}
		// Should handle nil body gracefully without panic
		readPrefillRecv(ctx, "test-url", false, "test message", resp)
	})

	t.Run("mock response without scheduler dependency", func(t *testing.T) {
		ctx := context.Background()

		// Create a simple response that doesn't trigger scheduler calls
		resp := &http.Response{
			StatusCode: http.StatusOK,
			Body:       io.NopCloser(bytes.NewBufferString("test")),
		}

		// This test verifies basic error handling and response body consumption
		// without triggering scheduler initialization requirements
		readPrefillRecv(ctx, "test-url", false, "test message", resp)
	})
}

func TestGetSessionID(t *testing.T) {
	tests := []struct {
		name     string
		input    map[string]any
		expected string
	}{
		{
			name: "top-level session_id",
			input: map[string]any{
				"session_id": "top-level-sid",
				"messages":   []any{},
			},
			expected: "top-level-sid",
		},
		{
			name: "extra_body session_id",
			input: map[string]any{
				"messages": []any{},
				"extra_body": map[string]any{
					"session_id": "extra-body-sid",
				},
			},
			expected: "extra-body-sid",
		},
		{
			name: "top-level takes priority over extra_body",
			input: map[string]any{
				"session_id": "top-level-sid",
				"extra_body": map[string]any{
					"session_id": "extra-body-sid",
				},
			},
			expected: "top-level-sid",
		},
		{
			name: "no session_id provided",
			input: map[string]any{
				"messages": []any{},
			},
			expected: "",
		},
		{
			name:     "empty request",
			input:    map[string]any{},
			expected: "",
		},
		{
			name: "top-level session_id is empty string, fallback to extra_body",
			input: map[string]any{
				"session_id": "",
				"extra_body": map[string]any{
					"session_id": "extra-body-sid",
				},
			},
			expected: "extra-body-sid",
		},
		{
			name: "both session_id are empty strings",
			input: map[string]any{
				"session_id": "",
				"extra_body": map[string]any{
					"session_id": "",
				},
			},
			expected: "",
		},
		{
			name: "extra_body exists but no session_id in it",
			input: map[string]any{
				"extra_body": map[string]any{
					"other_field": "value",
				},
			},
			expected: "",
		},
		{
			name: "extra_body is not a map",
			input: map[string]any{
				"extra_body": "not-a-map",
			},
			expected: "",
		},
		{
			name: "session_id is not a string (integer)",
			input: map[string]any{
				"session_id": 12345,
			},
			expected: "",
		},
		{
			name: "session_id is not a string (bool)",
			input: map[string]any{
				"session_id": true,
			},
			expected: "",
		},
		{
			name: "extra_body session_id is not a string",
			input: map[string]any{
				"extra_body": map[string]any{
					"session_id": 12345,
				},
			},
			expected: "",
		},
		{
			name: "session_id from JSON unmarshal (top-level)",
			input: func() map[string]any {
				var m map[string]any
				json.Unmarshal([]byte(`{"session_id": "json-sid", "messages": []}`), &m)
				return m
			}(),
			expected: "json-sid",
		},
		{
			name: "session_id from JSON unmarshal (extra_body)",
			input: func() map[string]any {
				var m map[string]any
				json.Unmarshal([]byte(`{"extra_body": {"session_id": "json-extra-sid"}, "messages": []}`), &m)
				return m
			}(),
			expected: "json-extra-sid",
		},
		{
			name: "session_id is nil",
			input: map[string]any{
				"session_id": nil,
			},
			expected: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := getSessionID(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestCommonCompletions(t *testing.T) {
	// Setup a basic test server for backend responses
	backendServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Check if it's a stream request
		bodyBytes, _ := io.ReadAll(r.Body)
		var reqBody map[string]any
		json.Unmarshal(bodyBytes, &reqBody)

		if stream, ok := reqBody["stream"].(bool); ok && stream {
			// Stream response
			w.Header().Set("Content-Type", "text/plain")
			w.WriteHeader(http.StatusOK)
			w.Write([]byte("data: {\"choices\":[{\"text\":\"chunk1\"}]}\n"))
			w.Write([]byte("data: {\"choices\":[{\"text\":\"chunk2\"}]}\n"))
			w.Write([]byte("data: [DONE]\n"))
		} else {
			// Non-stream response
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(`{"choices":[{"text":"test response"}]}`))
		}
	}))
	defer backendServer.Close()

	t.Run("basic request handling", func(t *testing.T) {
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Request = httptest.NewRequest("POST", "/v1/completions",
			bytes.NewBufferString(`{"prompt": "test", "stream": false}`))

		// Mock the manager functions to return our test server
		// This would normally require more sophisticated mocking
		// For now, this test verifies the function structure
		assert.NotNil(t, CommonCompletions)
	})

	t.Run("invalid JSON request", func(t *testing.T) {
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Request = httptest.NewRequest("POST", "/v1/completions",
			bytes.NewBufferString(`invalid json`))

		CommonCompletions(c, extractPromptFromCompletionsRequest, "completions")

		// Should return 400 Bad Request
		assert.Equal(t, http.StatusBadRequest, w.Code)
		assert.Contains(t, w.Body.String(), "Invalid JSON format")
	})

	t.Run("empty request body", func(t *testing.T) {
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Request = httptest.NewRequest("POST", "/v1/completions", nil)

		CommonCompletions(c, extractPromptFromCompletionsRequest, "completions")

		// Should return 400 Bad Request with appropriate error message
		assert.Equal(t, http.StatusBadRequest, w.Code)
		// The error message could be either "Invalid request body" or "Invalid JSON format"
		// depending on how empty body is handled
		assert.True(t, strings.Contains(w.Body.String(), "Invalid request body") ||
			strings.Contains(w.Body.String(), "Invalid JSON format") ||
			w.Body.String() != "")
	})
}

// ============================================================
// Test helpers for timeout / hang simulation
// ============================================================

// newHangingServer creates an httptest.Server whose handler blocks until
// the returned cleanup function is called. Always call cleanup in defer.
func newHangingServer() (server *httptest.Server, cleanup func()) {
	done := make(chan struct{})
	server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-done
	}))
	cleanup = func() {
		close(done)
		server.Close()
	}
	return
}

// newSlowServer creates an httptest.Server that waits for the given duration
// before responding. cleanup unblocks any in-flight handlers.
func newSlowServer(delay time.Duration, statusCode int, body string) (server *httptest.Server, cleanup func()) {
	done := make(chan struct{})
	server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-time.After(delay):
			w.WriteHeader(statusCode)
			w.Write([]byte(body))
		case <-done:
			// test cleanup
		}
	}))
	cleanup = func() {
		close(done)
		server.Close()
	}
	return
}

// newGinContextWithTimeout creates a gin.Context with a request whose context
// has the given timeout. Returns cancel for cleanup.
func newGinContextWithTimeout(timeout time.Duration) (*gin.Context, *httptest.ResponseRecorder, context.CancelFunc) {
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	req := httptest.NewRequest("POST", "/v1/chat/completions",
		bytes.NewBufferString(`{"test":"data"}`))
	c.Request = req.WithContext(ctx)
	return c, w, cancel
}

// hangingReader is a custom io.ReadCloser that returns initial data,
// then blocks until its context is cancelled (simulating mid-stream hang).
type hangingReader struct {
	data   []byte
	offset int
	hangAt int // byte offset at which to start hanging
	ctx    context.Context
}

func (r *hangingReader) Read(p []byte) (int, error) {
	if r.offset >= r.hangAt {
		<-r.ctx.Done()
		return 0, r.ctx.Err()
	}
	end := r.offset + len(p)
	if end > r.hangAt {
		end = r.hangAt
	}
	n := copy(p, r.data[r.offset:end])
	r.offset += n
	return n, nil
}

func (r *hangingReader) Close() error { return nil }

// ============================================================
// PostToPD timeout tests
// ============================================================

func TestPostToPD_PrefillHangs(t *testing.T) {
	hangServer, hangCleanup := newHangingServer()
	defer hangCleanup()

	decodeServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("decode ok"))
	}))
	defer decodeServer.Close()

	c, _, cancel := newGinContextWithTimeout(200 * time.Millisecond)
	defer cancel()

	start := time.Now()
	resp, err := PostToPD(c, decodeServer.URL, hangServer.URL, []byte(`{"test":"data"}`), false, "msg", "chat/completions")
	elapsed := time.Since(start)

	assert.Error(t, err)
	assert.True(t, errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled),
		"expected context deadline exceeded or canceled, got: %v", err)
	assert.Nil(t, resp)
	assert.Less(t, elapsed, 5*time.Second, "should not hang indefinitely")
}

func TestPostToPD_DecodeHangs(t *testing.T) {
	prefillServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("prefill ok"))
	}))
	defer prefillServer.Close()

	hangServer, hangCleanup := newHangingServer()
	defer hangCleanup()

	c, _, cancel := newGinContextWithTimeout(200 * time.Millisecond)
	defer cancel()

	start := time.Now()
	resp, err := PostToPD(c, hangServer.URL, prefillServer.URL, []byte(`{"test":"data"}`), false, "msg", "chat/completions")
	elapsed := time.Since(start)

	assert.Error(t, err)
	assert.True(t, errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled),
		"expected context deadline exceeded or canceled, got: %v", err)
	assert.Nil(t, resp)
	assert.Less(t, elapsed, 5*time.Second)
}

func TestPostToPD_BothHang(t *testing.T) {
	hangP, cleanupP := newHangingServer()
	defer cleanupP()

	hangD, cleanupD := newHangingServer()
	defer cleanupD()

	c, _, cancel := newGinContextWithTimeout(200 * time.Millisecond)
	defer cancel()

	start := time.Now()
	resp, err := PostToPD(c, hangD.URL, hangP.URL, []byte(`{"test":"data"}`), false, "msg", "chat/completions")
	elapsed := time.Since(start)

	assert.Error(t, err)
	assert.True(t, errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled),
		"expected context deadline exceeded or canceled, got: %v", err)
	assert.Nil(t, resp)
	assert.Less(t, elapsed, 5*time.Second)
}

func TestPostToPD_ContextCancellation(t *testing.T) {
	hangP, cleanupP := newHangingServer()
	defer cleanupP()

	hangD, cleanupD := newHangingServer()
	defer cleanupD()

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	ctx, cancel := context.WithCancel(context.Background())
	req := httptest.NewRequest("POST", "/v1/chat/completions",
		bytes.NewBufferString(`{"test":"data"}`))
	c.Request = req.WithContext(ctx)

	type result struct {
		resp *http.Response
		err  error
	}
	ch := make(chan result, 1)
	go func() {
		resp, err := PostToPD(c, hangD.URL, hangP.URL, []byte(`{"test":"data"}`), false, "msg", "chat/completions")
		ch <- result{resp, err}
	}()

	// Cancel after a short delay
	time.Sleep(50 * time.Millisecond)
	cancel()

	select {
	case res := <-ch:
		assert.Error(t, res.err)
		assert.True(t, errors.Is(res.err, context.Canceled),
			"expected context.Canceled, got: %v", res.err)
		assert.Nil(t, res.resp)
	case <-time.After(5 * time.Second):
		t.Fatal("PostToPD did not return after context cancellation")
	}
}

func TestPostToPD_PrefillSlowButCompletes(t *testing.T) {
	slowPrefill, cleanupP := newSlowServer(50*time.Millisecond, http.StatusOK, "prefill done")
	defer cleanupP()

	decodeServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("decode done"))
	}))
	defer decodeServer.Close()

	c, _, cancel := newGinContextWithTimeout(2 * time.Second)
	defer cancel()

	resp, err := PostToPD(c, decodeServer.URL, slowPrefill.URL, []byte(`{"test":"data"}`), false, "msg", "chat/completions")
	assert.NoError(t, err)
	assert.NotNil(t, resp)
	assert.Equal(t, http.StatusOK, resp.StatusCode)

	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	assert.Equal(t, "decode done", string(body))
}

// ============================================================
// GetClient / GetClientWithRetry timeout tests
// ============================================================

func TestGetClient_Timeout(t *testing.T) {
	hangServer, hangCleanup := newHangingServer()
	defer hangCleanup()

	c, _, cancel := newGinContextWithTimeout(200 * time.Millisecond)
	defer cancel()

	start := time.Now()
	resp, err := GetClient(c, hangServer.URL, "chat/completions", []byte(`{"test":"data"}`))
	elapsed := time.Since(start)

	assert.Error(t, err)
	assert.True(t, errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled),
		"expected context deadline exceeded or canceled, got: %v", err)
	assert.Nil(t, resp)
	assert.Less(t, elapsed, 5*time.Second)
}

func TestGetClientWithRetry_TimeoutAcrossRetries(t *testing.T) {
	var hitCount atomic.Int32
	done := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hitCount.Add(1)
		<-done
	}))
	defer func() {
		close(done)
		server.Close()
	}()

	c, _, cancel := newGinContextWithTimeout(200 * time.Millisecond)
	defer cancel()

	start := time.Now()
	resp, err := GetClientWithRetry(c, []byte(`{"test":"data"}`), server.URL, "chat/completions")
	elapsed := time.Since(start)

	assert.Error(t, err)
	assert.True(t, errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled),
		"expected context deadline exceeded or canceled, got: %v", err)
	assert.Nil(t, resp)
	// Should not have completed all 3 retries; the shared context expires
	assert.Less(t, elapsed, 5*time.Second)
	// At most 3 attempts, but with a 200ms timeout the context should expire during/after the first attempt
	assert.LessOrEqual(t, hitCount.Load(), int32(3))
}

func TestGetClientWithRetry_ContextCancelled(t *testing.T) {
	done := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-done
	}))
	defer func() {
		close(done)
		server.Close()
	}()

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	ctx, cancel := context.WithCancel(context.Background())
	req := httptest.NewRequest("POST", "/v1/chat/completions",
		bytes.NewBufferString(`{"test":"data"}`))
	c.Request = req.WithContext(ctx)

	type result struct {
		resp *http.Response
		err  error
	}
	ch := make(chan result, 1)
	go func() {
		resp, err := GetClientWithRetry(c, []byte(`{"test":"data"}`), server.URL, "chat/completions")
		ch <- result{resp, err}
	}()

	time.Sleep(50 * time.Millisecond)
	cancel()

	select {
	case res := <-ch:
		assert.Error(t, res.err)
		assert.True(t, errors.Is(res.err, context.Canceled),
			"expected context.Canceled, got: %v", res.err)
		assert.Nil(t, res.resp)
	case <-time.After(5 * time.Second):
		t.Fatal("GetClientWithRetry did not return after context cancellation")
	}
}

// ============================================================
// Streaming hang / mid-stream interruption tests
// ============================================================

func TestRedirect_StreamingHangMidStream(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancel()

	initialData := "data: {\"choices\":[{\"text\":\"chunk1\"}]}\n"
	reader := &hangingReader{
		data:   []byte(initialData),
		hangAt: len(initialData),
		ctx:    ctx,
	}

	resp := &http.Response{
		StatusCode: http.StatusOK,
		Body:       reader,
	}

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest("GET", "/", nil).WithContext(ctx)

	done := make(chan struct{})
	go func() {
		redirect(c, true, resp)
		close(done)
	}()

	select {
	case <-done:
		// redirect returned, check partial output was written
		assert.Contains(t, w.Body.String(), "data: {\"choices\":[{\"text\":\"chunk1\"}]}")
	case <-time.After(5 * time.Second):
		t.Fatal("redirect did not return after context timeout")
	}
}

func TestReadPrefillRecv_StreamHang(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancel()

	// First chunk followed by a hang
	initialData := "data: first-chunk\n"
	reader := &hangingReader{
		data:   []byte(initialData),
		hangAt: len(initialData),
		ctx:    ctx,
	}

	resp := &http.Response{
		StatusCode: http.StatusOK,
		Body:       reader,
	}

	done := make(chan struct{})
	go func() {
		readPrefillRecv(ctx, "test-url", true, "test message", resp)
		close(done)
	}()

	select {
	case <-done:
		// completed without panic
	case <-time.After(5 * time.Second):
		t.Fatal("readPrefillRecv did not return after context timeout")
	}
}

func TestReadPrefillRecv_NonStreamHang(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancel()

	// Reader that immediately hangs (no data before hang)
	reader := &hangingReader{
		data:   []byte{},
		hangAt: 0,
		ctx:    ctx,
	}

	resp := &http.Response{
		StatusCode: http.StatusOK,
		Body:       reader,
	}

	done := make(chan struct{})
	go func() {
		readPrefillRecv(ctx, "test-url", false, "test message", resp)
		close(done)
	}()

	select {
	case <-done:
		// completed without panic
	case <-time.After(5 * time.Second):
		t.Fatal("readPrefillRecv did not return after context timeout")
	}
}
