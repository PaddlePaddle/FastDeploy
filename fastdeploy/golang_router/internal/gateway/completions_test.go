package gateway

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
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

func TestPostToPD(t *testing.T) {
	// Setup test servers
	prefillTS := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer prefillTS.Close()

	decodeTS := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"response": "test"}`))
	}))
	defer decodeTS.Close()

	// Setup test context
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewBufferString(`{"test": "data"}`))

	resp, err := PostToPD(c, decodeTS.URL, prefillTS.URL, []byte(`{"test": "data"}`))
	assert.NoError(t, err)
	assert.Equal(t, http.StatusOK, resp.StatusCode)
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
