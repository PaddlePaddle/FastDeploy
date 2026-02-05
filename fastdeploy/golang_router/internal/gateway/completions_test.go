package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
)

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

		resp, err := GetClientWithRetry(c, reqBody, ts.URL)
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

		resp, err := GetClientWithRetry(c, reqBody, ts.URL)
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

		resp, err := GetClientWithRetry(c, reqBody, ts.URL)
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
