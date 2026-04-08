package handler

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"
)

func TestNewHTTPTokenizer(t *testing.T) {
	tests := []struct {
		name     string
		rawURL   string
		timeout  time.Duration
		expected string
		wantNil  bool
	}{
		{
			name:    "empty URL should return nil",
			rawURL:  "",
			timeout: time.Second,
			wantNil: true,
		},
		{
			name:     "URL without scheme should add http://",
			rawURL:   "example.com",
			timeout:  time.Second,
			expected: "http://example.com/tokenize",
		},
		{
			name:     "URL with http scheme should keep it",
			rawURL:   "http://example.com",
			timeout:  time.Second,
			expected: "http://example.com/tokenize",
		},
		{
			name:     "URL with https scheme should keep it",
			rawURL:   "https://example.com",
			timeout:  time.Second,
			expected: "https://example.com/tokenize",
		},
		{
			name:     "URL with path should keep it",
			rawURL:   "http://example.com/api",
			timeout:  time.Second,
			expected: "http://example.com/api",
		},
		{
			name:     "URL with root path should replace with /tokenize",
			rawURL:   "http://example.com/",
			timeout:  time.Second,
			expected: "http://example.com/tokenize",
		},
		{
			name:     "URL with empty path should add /tokenize",
			rawURL:   "http://example.com",
			timeout:  time.Second,
			expected: "http://example.com/tokenize",
		},
		{
			name:     "invalid URL should still work with added scheme",
			rawURL:   "example.com:invalid:port",
			timeout:  time.Second,
			expected: "http://example.com:invalid:port",
		},
		{
			name:     "zero timeout should use default 2s",
			rawURL:   "example.com",
			timeout:  0,
			expected: "http://example.com/tokenize",
		},
		{
			name:     "negative timeout should use default 2s",
			rawURL:   "example.com",
			timeout:  -time.Second,
			expected: "http://example.com/tokenize",
		},
		{
			name:     "URL with port and path should be preserved",
			rawURL:   "http://example.com:8080/v1",
			timeout:  time.Second,
			expected: "http://example.com:8080/v1",
		},
		{
			name:     "URL with query parameters should be preserved",
			rawURL:   "https://example.com?token=abc",
			timeout:  time.Second,
			expected: "https://example.com/tokenize?token=abc",
		},
		{
			name:     "URL with fragment should be preserved",
			rawURL:   "http://example.com#section",
			timeout:  time.Second,
			expected: "http://example.com/tokenize#section",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := NewHTTPTokenizer(tt.rawURL, tt.timeout)

			if tt.wantNil {
				if client != nil {
					t.Errorf("Expected nil client for empty URL, got %v", client)
				}
				return
			}

			if client == nil {
				t.Fatal("Expected non-nil client, got nil")
			}

			httpClient, ok := client.(*httpTokenizer)
			if !ok {
				t.Fatalf("Expected *httpTokenizer, got %T", client)
			}

			if httpClient.url != tt.expected {
				t.Errorf("Expected URL %q, got %q", tt.expected, httpClient.url)
			}

			expectedTimeout := tt.timeout
			if expectedTimeout <= 0 {
				expectedTimeout = 2 * time.Second
			}
			if httpClient.timeout != expectedTimeout {
				t.Errorf("Expected timeout %v, got %v", expectedTimeout, httpClient.timeout)
			}
		})
	}
}

func TestNewHTTPTokenizer_URLParsing(t *testing.T) {
	// Test that URL parsing preserves all components
	rawURL := "https://user:pass@example.com:8080/path?query=value#fragment"
	client := NewHTTPTokenizer(rawURL, time.Second)

	if client == nil {
		t.Fatal("Expected non-nil client, got nil")
	}

	httpClient := client.(*httpTokenizer)
	parsed, err := url.Parse(httpClient.url)
	if err != nil {
		t.Fatalf("Failed to parse client URL: %v", err)
	}

	if parsed.Scheme != "https" {
		t.Errorf("Expected scheme https, got %q", parsed.Scheme)
	}
	if parsed.Host != "example.com:8080" {
		t.Errorf("Expected host example.com:8080, got %q", parsed.Host)
	}
	if parsed.Path != "/path" {
		t.Errorf("Expected path /path, got %q", parsed.Path)
	}
	if parsed.RawQuery != "query=value" {
		t.Errorf("Expected query query=value, got %q", parsed.RawQuery)
	}
	if parsed.Fragment != "fragment" {
		t.Errorf("Expected fragment fragment, got %q", parsed.Fragment)
	}
	if parsed.User.String() != "user:pass" {
		t.Errorf("Expected user user:pass, got %q", parsed.User)
	}
}

func TestNewHTTPTokenizer_ImplementsInterface(t *testing.T) {
	client := NewHTTPTokenizer("example.com", time.Second)

	if client == nil {
		t.Fatal("Expected non-nil client")
	}

	// Verify that the returned client implements the TokenizerClient interface
	var _ TokenizerClient = client

	// Test type assertion
	_, ok := client.(TokenizerClient)
	if !ok {
		t.Error("Returned client does not implement TokenizerClient interface")
	}
}

func TestHTTPTokenizer_Tokenize(t *testing.T) {
	t.Run("nil tokenizer client", func(t *testing.T) {
		var tokenizer *httpTokenizer = nil
		_, err := tokenizer.Tokenize(context.Background(), "test prompt")
		if err == nil {
			t.Error("Expected error for nil tokenizer client")
		}
		if err.Error() != "tokenizer client is nil" {
			t.Errorf("Expected 'tokenizer client is nil', got '%v'", err.Error())
		}
	})

	t.Run("successful tokenization", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// httptest server uses root path by default
			if r.URL.Path != "/" {
				t.Errorf("Expected path /, got %s", r.URL.Path)
			}
			if r.Method != "POST" {
				t.Errorf("Expected POST method, got %s", r.Method)
			}
			if r.Header.Get("Content-Type") != "application/json" {
				t.Errorf("Expected Content-Type application/json, got %s", r.Header.Get("Content-Type"))
			}

			// Verify request body
			var req tokenizerHTTPReq
			err := json.NewDecoder(r.Body).Decode(&req)
			if err != nil {
				t.Fatalf("Failed to decode request: %v", err)
			}
			if req.Text != "test prompt" {
				t.Errorf("Expected Text 'test prompt', got '%s'", req.Text)
			}
			if req.Prompt != "test prompt" {
				t.Errorf("Expected Prompt 'test prompt', got '%s'", req.Prompt)
			}
			if req.Message != "test prompt" {
				t.Errorf("Expected Message 'test prompt', got '%s'", req.Message)
			}

			// Send successful response
			response := map[string]interface{}{
				"input_ids": []int{1, 2, 3, 4, 5},
			}
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			json.NewEncoder(w).Encode(response)
		}))
		defer server.Close()

		tokenizer := &httpTokenizer{
			url:     server.URL,
			timeout: 2 * time.Second,
		}

		tokens, err := tokenizer.Tokenize(context.Background(), "test prompt")
		if err != nil {
			t.Fatalf("Tokenize failed: %v", err)
		}
		expectedTokens := []int{1, 2, 3, 4, 5}
		if len(tokens) != len(expectedTokens) {
			t.Errorf("Expected %d tokens, got %d", len(expectedTokens), len(tokens))
		}
		for i, token := range tokens {
			if token != expectedTokens[i] {
				t.Errorf("Token at index %d: expected %d, got %d", i, expectedTokens[i], token)
			}
		}
	})

	t.Run("http request creation failure", func(t *testing.T) {
		tokenizer := &httpTokenizer{
			url:     "://invalid-url", // Invalid URL to force http.NewRequest to fail
			timeout: 2 * time.Second,
		}

		_, err := tokenizer.Tokenize(context.Background(), "test prompt")
		if err == nil {
			t.Error("Expected error for invalid URL")
		}
		if !strings.Contains(err.Error(), "build tokenizer request failed") {
			t.Errorf("Expected error to contain 'build tokenizer request failed', got '%v'", err.Error())
		}
	})

	t.Run("http request timeout", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			time.Sleep(100 * time.Millisecond) // Simulate slow response
			w.WriteHeader(http.StatusOK)
			json.NewEncoder(w).Encode(map[string]interface{}{"input_ids": []int{1}})
		}))
		defer server.Close()

		tokenizer := &httpTokenizer{
			url:     server.URL,
			timeout: 10 * time.Millisecond, // Very short timeout
		}

		_, err := tokenizer.Tokenize(context.Background(), "test prompt")
		if err == nil {
			t.Error("Expected error for timeout")
		}
		if !strings.Contains(err.Error(), "tokenizer request failed") {
			t.Errorf("Expected error to contain 'tokenizer request failed', got '%v'", err.Error())
		}
	})

	t.Run("http connection failure", func(t *testing.T) {
		tokenizer := &httpTokenizer{
			url:     "http://invalid-server:9999", // Non-existent server
			timeout: 1 * time.Second,
		}

		_, err := tokenizer.Tokenize(context.Background(), "test prompt")
		if err == nil {
			t.Error("Expected error for connection failure")
		}
		if !strings.Contains(err.Error(), "tokenizer request failed") {
			t.Errorf("Expected error to contain 'tokenizer request failed', got '%v'", err.Error())
		}
	})

	t.Run("http response status error", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusInternalServerError)
			w.Write([]byte("internal server error"))
		}))
		defer server.Close()

		tokenizer := &httpTokenizer{
			url:     server.URL,
			timeout: 2 * time.Second,
		}

		_, err := tokenizer.Tokenize(context.Background(), "test prompt")
		if err == nil {
			t.Error("Expected error for status code 500")
		}
		if !strings.Contains(err.Error(), "tokenizer response status 500") {
			t.Errorf("Expected error to contain 'tokenizer response status 500', got '%v'", err.Error())
		}
		if !strings.Contains(err.Error(), "internal server error") {
			t.Errorf("Expected error to contain response body, got '%v'", err.Error())
		}
	})

	t.Run("invalid json response", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			w.Write([]byte("invalid json content"))
		}))
		defer server.Close()

		tokenizer := &httpTokenizer{
			url:     server.URL,
			timeout: 2 * time.Second,
		}

		_, err := tokenizer.Tokenize(context.Background(), "test prompt")
		if err == nil {
			t.Error("Expected error for invalid JSON")
		}
		if !strings.Contains(err.Error(), "parse tokenizer response failed") {
			t.Errorf("Expected error to contain 'parse tokenizer response failed', got '%v'", err.Error())
		}
	})

	t.Run("empty tokens in response", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			response := map[string]interface{}{
				"input_ids": []int{},
			}
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			json.NewEncoder(w).Encode(response)
		}))
		defer server.Close()

		tokenizer := &httpTokenizer{
			url:     server.URL,
			timeout: 2 * time.Second,
		}

		_, err := tokenizer.Tokenize(context.Background(), "test prompt")
		if err == nil {
			t.Error("Expected error for empty tokens")
		}
		if !strings.Contains(err.Error(), "tokenizer response missing tokens") {
			t.Errorf("Expected error to contain 'tokenizer response missing tokens', got '%v'", err.Error())
		}
	})

	t.Run("missing input_ids field", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			response := map[string]interface{}{
				"tokens": []int{1, 2, 3}, // Wrong field name
			}
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			json.NewEncoder(w).Encode(response)
		}))
		defer server.Close()

		tokenizer := &httpTokenizer{
			url:     server.URL,
			timeout: 2 * time.Second,
		}

		_, err := tokenizer.Tokenize(context.Background(), "test prompt")
		if err == nil {
			t.Error("Expected error for missing input_ids")
		}
		if !strings.Contains(err.Error(), "tokenizer response missing tokens") {
			t.Errorf("Expected error to contain 'tokenizer response missing tokens', got '%v'", err.Error())
		}
	})

	t.Run("context cancellation", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			time.Sleep(100 * time.Millisecond) // Simulate slow processing
			w.WriteHeader(http.StatusOK)
			json.NewEncoder(w).Encode(map[string]interface{}{"input_ids": []int{1}})
		}))
		defer server.Close()

		tokenizer := &httpTokenizer{
			url:     server.URL,
			timeout: 2 * time.Second,
		}

		ctx, cancel := context.WithCancel(context.Background())
		cancel() // Cancel immediately

		_, err := tokenizer.Tokenize(ctx, "test prompt")
		if err == nil {
			t.Error("Expected error for cancelled context")
		}
		// Context cancellation should cause request to fail
		if !strings.Contains(err.Error(), "tokenizer request failed") {
			t.Errorf("Expected error to contain 'tokenizer request failed', got '%v'", err.Error())
		}
	})

	t.Run("empty prompt", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var req tokenizerHTTPReq
			err := json.NewDecoder(r.Body).Decode(&req)
			if err != nil {
				t.Fatalf("Failed to decode request: %v", err)
			}
			if req.Text != "" {
				t.Errorf("Expected empty Text, got '%s'", req.Text)
			}
			if req.Prompt != "" {
				t.Errorf("Expected empty Prompt, got '%s'", req.Prompt)
			}
			if req.Message != "" {
				t.Errorf("Expected empty Message, got '%s'", req.Message)
			}

			response := map[string]interface{}{
				"input_ids": []int{},
			}
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			json.NewEncoder(w).Encode(response)
		}))
		defer server.Close()

		tokenizer := &httpTokenizer{
			url:     server.URL,
			timeout: 2 * time.Second,
		}

		_, err := tokenizer.Tokenize(context.Background(), "")
		if err == nil {
			t.Error("Expected error for empty prompt")
		}
		if !strings.Contains(err.Error(), "tokenizer response missing tokens") {
			t.Errorf("Expected error to contain 'tokenizer response missing tokens', got '%v'", err.Error())
		}
	})

	t.Run("very long prompt", func(t *testing.T) {
		longPrompt := string(make([]byte, 10000)) // 10KB prompt

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var req tokenizerHTTPReq
			err := json.NewDecoder(r.Body).Decode(&req)
			if err != nil {
				t.Fatalf("Failed to decode request: %v", err)
			}
			if req.Text != longPrompt {
				t.Error("Request text does not match long prompt")
			}

			response := map[string]interface{}{
				"input_ids": []int{1, 2, 3},
			}
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			json.NewEncoder(w).Encode(response)
		}))
		defer server.Close()

		tokenizer := &httpTokenizer{
			url:     server.URL,
			timeout: 2 * time.Second,
		}

		tokens, err := tokenizer.Tokenize(context.Background(), longPrompt)
		if err != nil {
			t.Fatalf("Tokenize failed for long prompt: %v", err)
		}
		expectedTokens := []int{1, 2, 3}
		if len(tokens) != len(expectedTokens) {
			t.Errorf("Expected %d tokens, got %d", len(expectedTokens), len(tokens))
		}
		for i, token := range tokens {
			if token != expectedTokens[i] {
				t.Errorf("Token at index %d: expected %d, got %d", i, expectedTokens[i], token)
			}
		}
	})

	t.Run("special characters in prompt", func(t *testing.T) {
		specialPrompt := "Hello, 世界! 🚀 Test with emoji and unicode: ñáéíóú"

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var req tokenizerHTTPReq
			err := json.NewDecoder(r.Body).Decode(&req)
			if err != nil {
				t.Fatalf("Failed to decode request: %v", err)
			}
			if req.Text != specialPrompt {
				t.Errorf("Expected Text '%s', got '%s'", specialPrompt, req.Text)
			}

			response := map[string]interface{}{
				"input_ids": []int{10, 20, 30, 40},
			}
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			json.NewEncoder(w).Encode(response)
		}))
		defer server.Close()

		tokenizer := &httpTokenizer{
			url:     server.URL,
			timeout: 2 * time.Second,
		}

		tokens, err := tokenizer.Tokenize(context.Background(), specialPrompt)
		if err != nil {
			t.Fatalf("Tokenize failed for special characters: %v", err)
		}
		expectedTokens := []int{10, 20, 30, 40}
		if len(tokens) != len(expectedTokens) {
			t.Errorf("Expected %d tokens, got %d", len(expectedTokens), len(tokens))
		}
		for i, token := range tokens {
			if token != expectedTokens[i] {
				t.Errorf("Token at index %d: expected %d, got %d", i, expectedTokens[i], token)
			}
		}
	})
}

func TestParseTokensFromBody(t *testing.T) {
	tests := []struct {
		name     string
		input    []byte
		expected []int
		err      error
	}{
		{
			name:     "valid input with tokens",
			input:    []byte(`{"input_ids": [1, 2, 3]}`),
			expected: []int{1, 2, 3},
			err:      nil,
		},
		{
			name:     "empty input_ids array",
			input:    []byte(`{"input_ids": []}`),
			expected: nil,
			err:      errors.New("tokenizer response missing tokens"),
		},
		{
			name:     "missing input_ids field",
			input:    []byte(`{"other_field": "value"}`),
			expected: nil,
			err:      errors.New("tokenizer response missing tokens"),
		},
		{
			name:     "invalid JSON format",
			input:    []byte(`invalid json`),
			expected: nil,
			err:      errors.New("tokenizer response missing tokens"),
		},
		{
			name:     "empty body",
			input:    []byte(``),
			expected: nil,
			err:      errors.New("tokenizer response missing tokens"),
		},
		{
			name:     "large array of tokens",
			input:    []byte(`{"input_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}`),
			expected: []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
			err:      nil,
		},
		{
			name:     "null input_ids",
			input:    []byte(`{"input_ids": null}`),
			expected: nil,
			err:      errors.New("tokenizer response missing tokens"),
		},
		{
			name:     "non-array input_ids",
			input:    []byte(`{"input_ids": "not an array"}`),
			expected: nil,
			err:      errors.New("tokenizer response missing tokens"),
		},
		{
			name:     "malformed array",
			input:    []byte(`{"input_ids": [1, "two", 3]}`),
			expected: nil,
			err:      errors.New("tokenizer response missing tokens"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseTokensFromBody(tt.input)

			// Check if error is expected
			if (err != nil && tt.err == nil) || (err == nil && tt.err != nil) {
				t.Errorf("parseTokensFromBody() error = %v, wantErr %v", err, tt.err)
				return
			}
			if err != nil && tt.err != nil && err.Error() != tt.err.Error() {
				t.Errorf("parseTokensFromBody() error message = %v, want %v", err.Error(), tt.err.Error())
				return
			}

			// Compare actual and expected results
			if len(got) != len(tt.expected) {
				t.Errorf("parseTokensFromBody() = %v, want %v", got, tt.expected)
				return
			}
			for i := range got {
				if got[i] != tt.expected[i] {
					t.Errorf("parseTokensFromBody() = %v, want %v", got, tt.expected)
					return
				}
			}
		})
	}
}
