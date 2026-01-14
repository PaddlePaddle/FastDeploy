package handler

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	urlpkg "net/url"
	"strings"
	"time"
)

// Abstract remote tokenizer interface
type TokenizerClient interface {
	Tokenize(ctx context.Context, prompt string) ([]int, error)
}

// Implement HTTP /tokenize call
type httpTokenizer struct {
	url     string
	timeout time.Duration
}

// Return HTTP-based tokenizer client
func NewHTTPTokenizer(rawURL string, timeout time.Duration) TokenizerClient {
	if rawURL == "" {
		return nil
	}
	if !strings.HasPrefix(rawURL, "http://") && !strings.HasPrefix(rawURL, "https://") {
		rawURL = "http://" + rawURL
	}
	parsed, err := urlpkg.Parse(rawURL)
	if err == nil {
		if parsed.Path == "" || parsed.Path == "/" {
			parsed.Path = "/tokenize"
		}
		rawURL = parsed.String()
	}
	if timeout <= 0 {
		timeout = 2 * time.Second
	}
	return &httpTokenizer{
		url:     rawURL,
		timeout: timeout,
	}
}

type tokenizerHTTPReq struct {
	Text    string `json:"text,omitempty"`
	Prompt  string `json:"prompt,omitempty"`
	Message string `json:"message,omitempty"`
}

func (c *httpTokenizer) Tokenize(ctx context.Context, prompt string) ([]int, error) {
	if c == nil {
		return nil, errors.New("tokenizer client is nil")
	}

	payload := tokenizerHTTPReq{Text: prompt, Prompt: prompt, Message: prompt}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal tokenizer request failed: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("build tokenizer request failed: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: c.timeout}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("tokenizer request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read tokenizer response failed: %w", err)
	}
	if resp.StatusCode >= 300 {
		return nil, fmt.Errorf("tokenizer response status %d: %s", resp.StatusCode, string(respBody))
	}

	tokens, err := parseTokensFromBody(respBody)
	if err != nil {
		return nil, fmt.Errorf("parse tokenizer response failed: %w", err)
	}
	return tokens, nil
}

// Parse tokens from body JSON {"input_ids": []}
func parseTokensFromBody(body []byte) ([]int, error) {
	var input struct {
		InputIDs []int `json:"input_ids"`
	}
	if err := json.Unmarshal(body, &input); err == nil {
		if len(input.InputIDs) > 0 {
			return input.InputIDs, nil
		}
	}

	return nil, errors.New("tokenizer response missing tokens")
}
