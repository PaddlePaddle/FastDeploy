package gateway

import (
	"bufio"
	"bytes"
	"context"
	crand "crypto/rand"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/PaddlePaddle/FastDeploy/router/internal/manager"
	scheduler_handler "github.com/PaddlePaddle/FastDeploy/router/internal/scheduler/handler"
	"github.com/PaddlePaddle/FastDeploy/router/pkg/logger"
	"github.com/PaddlePaddle/FastDeploy/router/pkg/metrics"
	"github.com/gin-gonic/gin"
	"github.com/valyala/bytebufferpool"
)

const maxCapacity = 10 * 1024 * 1024 // 10MB

// newRequestID generates UUIDv4 style request_id
func newRequestID() string {
	b := make([]byte, 16)
	if _, err := crand.Read(b); err == nil {
		// Set version and variant bits, compliant with RFC 4122
		b[6] = (b[6] & 0x0f) | 0x40
		b[8] = (b[8] & 0x3f) | 0x80
		return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:16])
	}
	return fmt.Sprintf("%d-%d", time.Now().UnixNano(), rand.Int63())
}

type PromptExtractor func(rawReq map[string]any) string

// extractPromptFromChatRequest extracts text prompt from OpenAI ChatCompletions style request
func extractPromptFromChatRequest(rawReq map[string]any) string {
	messagesVal, ok := rawReq["messages"]
	if !ok {
		return ""
	}

	messages, ok := messagesVal.([]any)
	if !ok {
		return ""
	}

	var builder strings.Builder

	appendText := func(s string) {
		s = strings.TrimSpace(s)
		if s == "" {
			return
		}
		if builder.Len() > 0 {
			builder.WriteByte(' ')
		}
		builder.WriteString(s)
	}

	for _, msg := range messages {
		msgMap, ok := msg.(map[string]any)
		if !ok {
			continue
		}
		content, ok := msgMap["content"]
		if !ok {
			continue
		}

		switch v := content.(type) {
		case string:
			appendText(v)
		case []any:
			for _, item := range v {
				itemMap, ok := item.(map[string]any)
				if !ok {
					continue
				}
				itemType, _ := itemMap["type"].(string)
				if itemType != "text" {
					continue
				}
				if textVal, ok := itemMap["text"].(string); ok {
					appendText(textVal)
				}
			}
		default:
			// Other structures are ignored for now
		}
	}

	return builder.String()
}

func extractPromptFromCompletionsRequest(rawReq map[string]any) string {
	promptVal, ok := rawReq["prompt"]
	if !ok {
		return ""
	}

	var builder strings.Builder

	appendText := func(s string) {
		s = strings.TrimSpace(s)
		if s == "" {
			return
		}
		if builder.Len() > 0 {
			builder.WriteByte(' ')
		}
		builder.WriteString(s)
	}

	switch v := promptVal.(type) {

	case string:
		appendText(v)

	case []string:
		for _, s := range v {
			appendText(s)
		}

	case []any:
		for _, item := range v {
			if s, ok := item.(string); ok {
				appendText(s)
			}
		}

	default:
		// Other structures are ignored for now
	}

	return builder.String()
}

// PostToPD sends requests to both Prefill and Decode instances, only returns Decode node response
func PostToPD(c *gin.Context, decodeURL, prefillURL string, reqBody []byte, isStream bool, message string, completionEndpoint string) (*http.Response, error) {
	ctx := c.Request.Context()

	decodeEndpoint := fmt.Sprintf("%s/v1/%s", decodeURL, completionEndpoint)
	prefillEndpoint := fmt.Sprintf("%s/v1/%s", prefillURL, completionEndpoint)

	// Construct two requests
	decodeReq, err := http.NewRequestWithContext(ctx, "POST", decodeEndpoint, bytes.NewReader(reqBody))
	if err != nil {
		return nil, err
	}
	prefillReq, err := http.NewRequestWithContext(ctx, "POST", prefillEndpoint, bytes.NewReader(reqBody))
	if err != nil {
		return nil, err
	}

	// Copy request headers
	for k, v := range c.Request.Header {
		if k != "Content-Length" {
			decodeReq.Header[k] = v
			prefillReq.Header[k] = v
		}
	}

	client := &http.Client{}

	type respResult struct {
		resp *http.Response
		err  error
	}

	prefillCh := make(chan respResult, 1)
	decodeCh := make(chan respResult, 1)

	// Concurrently send requests to P/D
	go func() {
		resp, err := client.Do(prefillReq)
		prefillCh <- respResult{resp: resp, err: err}
	}()

	go func() {
		resp, err := client.Do(decodeReq)
		decodeCh <- respResult{resp: resp, err: err}
	}()

	prefillRes := <-prefillCh
	decodeRes := <-decodeCh

	// Prioritize returning Decode errors
	if decodeRes.err != nil {
		if prefillRes.resp != nil {
			prefillRes.resp.Body.Close()
		}
		return nil, decodeRes.err
	}
	if prefillRes.err != nil {
		// Prefill errors are also considered failures to avoid inconsistent behavior
		if decodeRes.resp != nil {
			decodeRes.resp.Body.Close()
		}
		return nil, prefillRes.err
	}

	if prefillRes.resp != nil {
		go readPrefillRecv(ctx, prefillURL, isStream, message, prefillRes.resp)
	}

	return decodeRes.resp, nil
}

func readPrefillRecv(ctx context.Context, url string, isStream bool, message string, backendResp *http.Response) {
	if backendResp == nil || backendResp.Body == nil {
		return
	}
	defer backendResp.Body.Close()

	if isStream {
		buffer := bytebufferpool.Get()
		buffer.Reset()
		defer bytebufferpool.Put(buffer)

		scanner := bufio.NewScanner(backendResp.Body)
		scanner.Buffer(buffer.B, maxCapacity)

		released := false
		defer func() {
			// Fallback to ensure release
			if !released {
				scheduler_handler.Release(ctx, url)
				scheduler_handler.ReleasePrefillTokens(ctx, url, message)
				logger.Debug("[prefill] release in defer (fallback) url=%s", url)
			}
		}()

		for scanner.Scan() {
			_ = scanner.Text()

			// First read that returns data
			if !released {
				scheduler_handler.Release(ctx, url)
				scheduler_handler.ReleasePrefillTokens(ctx, url, message)
				released = true

				logger.Debug("[prefill] first chunk received, release scheduler url=%s", url)
			}
		}

		if err := scanner.Err(); err != nil {
			logger.Debug("[prefill] scanner error: %v", err)
		}
	} else {
		_, err := io.Copy(io.Discard, backendResp.Body)
		if err != nil {
			logger.Debug("[prefill] copy error: %v", err)
		}
	}
}

// ChatCompletions implements request forwarding to actual large model inference service
func ChatCompletions(c *gin.Context) {
	completionEndpoint := "chat/completions"
	CommonCompletions(c, extractPromptFromChatRequest, completionEndpoint)
}

func Completions(c *gin.Context) {
	completionEndpoint := "completions"
	CommonCompletions(c, extractPromptFromCompletionsRequest, completionEndpoint)
}

func CommonCompletions(c *gin.Context, extractor PromptExtractor, completionEndpoint string) {
	ctx := c.Request.Context()

	bodyBytes, err := io.ReadAll(c.Request.Body)
	if err != nil {
		c.Writer.WriteHeader(http.StatusBadRequest)
		c.Writer.Write([]byte(`{"error": "Invalid request body"}`))
		return
	}

	var rawReq map[string]any
	if err := json.Unmarshal(bodyBytes, &rawReq); err != nil {
		c.Writer.WriteHeader(http.StatusBadRequest)
		c.Writer.Write([]byte(`{"error": "Invalid JSON format"}`))
		return
	}

	isSplitwise := manager.GetSplitwise(ctx)

	var (
		destURL         string
		releaseTargets  []string
		requestBodyData []byte
		prefillURL      string
		decodeURL       string
		message         string
	)

	if isSplitwise {
		// PD mode: select instances for Prefill/Decode separately
		message = extractor(rawReq)

		prefillURL, decodeURL, err = manager.SelectWorkerPair(ctx, message)
		if err != nil {
			c.Writer.WriteHeader(http.StatusBadGateway)
			c.Writer.Write([]byte(`{"error": "Failed to select worker pair"}`))
			return
		}
		if prefillURL == "" || decodeURL == "" {
			c.Writer.WriteHeader(http.StatusServiceUnavailable)
			c.Writer.Write([]byte(`{"error": "No available prefill/decode workers"}`))
			return
		}

		// Construct disaggregate_info to ensure selected P/D work in pairs within FastDeploy
		disagg, err := manager.BuildDisaggregateInfo(ctx, prefillURL, decodeURL)
		if err != nil {
			c.Writer.WriteHeader(http.StatusInternalServerError)
			c.Writer.Write([]byte(`{"error": "Failed to build disaggregate_info"}`))
			return
		}

		rawReq["disaggregate_info"] = disagg

		// If user didn't provide request_id, generate one
		if _, ok := rawReq["request_id"]; !ok {
			rawReq["request_id"] = newRequestID()
		}

		// Re-encode request body and send to P and D
		requestBodyData, err = json.Marshal(rawReq)
		if err != nil {
			c.Writer.WriteHeader(http.StatusInternalServerError)
			c.Writer.Write([]byte(`{"error": "Failed to encode modified request"}`))
			return
		}

		destURL = decodeURL
		releaseTargets = []string{decodeURL}

		// Expose scheduling results to caller for debugging/validating scheduling strategy
		c.Writer.Header().Set("X-Router-Prefill-URL", prefillURL)
		c.Writer.Header().Set("X-Router-Decode-URL", decodeURL)
	} else {
		// Non-PD mode: use Mixed instance
		dest, err := manager.SelectWorker(ctx, "")
		if err != nil {
			c.Writer.WriteHeader(http.StatusBadGateway)
			c.Writer.Write([]byte(`{"error": "Failed to select worker"}`))
			return
		}
		destURL = dest
		releaseTargets = []string{destURL}
		requestBodyData = bodyBytes
	}

	// Maintain request_num count for related instances (Inc done in SelectWorker, Release here)
	defer func() {
		for _, url := range releaseTargets {
			scheduler_handler.Release(ctx, url)
		}
	}()

	isStream := false
	if v, ok := rawReq["stream"]; ok {
		stream, ok := v.(bool)
		if ok && stream {
			isStream = true
		}
	}

	// Send request
	var backendResp *http.Response
	if isSplitwise {
		backendResp, err = PostToPD(c, decodeURL, prefillURL, requestBodyData, isStream, message, completionEndpoint)
	} else {
		backendResp, err = GetClientWithRetry(c, requestBodyData, destURL)
	}

	if err != nil {
		c.Writer.WriteHeader(http.StatusBadGateway)
		c.Writer.Write([]byte(`{"error": "Failed to connect to backend service"}`))
		return
	}
	defer backendResp.Body.Close()

	if isSplitwise {
		metrics.InferenceRequests.WithLabelValues("", prefillURL, decodeURL, strconv.Itoa(backendResp.StatusCode)).Inc()
	} else {
		metrics.InferenceRequests.WithLabelValues(destURL, "", "", strconv.Itoa(backendResp.StatusCode)).Inc()
	}
	// Copy response headers
	for k, v := range backendResp.Header {
		if k != "Content-Length" { // Remove Content-Length header
			c.Writer.Header()[k] = v
		}
	}
	//c.Writer.Header().Set("Transfer-Encoding", "chunked") // Set chunked transfer
	if backendResp.StatusCode == http.StatusOK {
		c.Writer.WriteHeader(backendResp.StatusCode)
	}

	redirect(c, isStream, backendResp)
}

func redirect(c *gin.Context, isStream bool, backendResp *http.Response) {
	// Forward response body
	if isStream {
		// Stream response, use buffer pool to avoid frequent buffer creation/destruction
		buffer := bytebufferpool.Get()
		buffer.Reset()
		defer bytebufferpool.Put(buffer)
		scanner := bufio.NewScanner(backendResp.Body)
		scanner.Buffer(buffer.B, maxCapacity) // Key: reset buffer

		for scanner.Scan() {
			line := scanner.Text()
			c.Writer.Write([]byte(line + "\n"))
			c.Writer.Flush()
		}

		if err := scanner.Err(); err != nil {
			logger.Error("scanner error: %v", err)
		}
	} else {
		// Compatible with non-stream response
		io.Copy(c.Writer, backendResp.Body)
	}
}

// GetClientWithRetry adds retry
func GetClientWithRetry(c *gin.Context, bodyBytes []byte, destUrl string) (
	backendResp *http.Response, err error) {
	// Five retries
	maxRetry := 3
	for i := 0; i < maxRetry; i++ {
		// If creating request fails, it's network connection error, check if selected node is elastic resource, if so, delete it
		backendResp, err = GetClient(c, destUrl, "chat/completions", bodyBytes)
		if err == nil { // Return latest bucketsize
			return backendResp, nil
		}
	}
	return nil, err
}

func GetClient(c *gin.Context, address, api string, reqBody []byte) (*http.Response, error) {
	backendURL := fmt.Sprintf("%s/v1/%s", address, api)

	backendReq, err := http.NewRequestWithContext(
		c.Request.Context(),
		"POST",
		backendURL,
		bytes.NewReader(reqBody),
	)
	if err != nil {
		return nil, err
	}
	// Copy request headers
	for k, v := range c.Request.Header {
		if k != "Content-Length" { // Remove Content-Length header
			backendReq.Header[k] = v
		}
	}

	client := &http.Client{}
	backendResp, err := client.Do(backendReq)

	if err != nil {
		return nil, err
	}

	return backendResp, nil
}
