package handler

import (
	"context"
	"encoding/binary"
	"errors"
	"hash/fnv"
	"math"
	"math/rand"
	"sync"
	"time"

	"github.com/PaddlePaddle/FastDeploy/router/pkg/logger"
)

type prefillCacheStrategy struct {
	absThreshold      float64
	relThreshold      float64
	hitRatioWeight    float64
	loadBalanceWeight float64
	cache             *radixPrefixCache
	tokenizer         TokenizerClient
}

type schedulerConfigSnapshot struct {
	balanceAbsThreshold float64
	balanceRelThreshold float64
	hitRatioWeight      float64
	loadBalanceWeight   float64
	cacheBlockSize      int
	tokenizerURL        string
	tokenizerTimeout    time.Duration
}

// newPrefillCacheStrategy initializes cache-aware strategy config
func newPrefillCacheStrategy(cfg *schedulerConfigSnapshot) *prefillCacheStrategy {
	return &prefillCacheStrategy{
		absThreshold:      cfg.balanceAbsThreshold,
		relThreshold:      cfg.balanceRelThreshold,
		hitRatioWeight:    cfg.hitRatioWeight,
		loadBalanceWeight: cfg.loadBalanceWeight,
		cache:             newRadixPrefixCache(cfg.cacheBlockSize),
		tokenizer:         NewHTTPTokenizer(cfg.tokenizerURL, cfg.tokenizerTimeout),
	}
}

// CacheAwarePrefillSelectWorker fallbacks to min tokens on extreme imbalance; otherwise scores by hit rate and load
func CacheAwarePrefillSelectWorker(ctx context.Context, workers []string, message string) (string, error) {
	if len(workers) == 0 {
		return "", nil
	}
	if DefaultScheduler == nil || DefaultScheduler.prefillCache == nil {
		return ProcessTokensSelectWorker(ctx, workers, message)
	}

	strategy := DefaultScheduler.prefillCache

	// 1) Fetch node load; fallback to min tokens on extreme imbalance
	loads := strategy.getRunningRequests(ctx, workers)
	if strategy.isLoadImbalanced(loads) {
		return ProcessTokensSelectWorker(ctx, workers, message)
	}

	// 2）tokenize
	tokens, err := strategy.tokenize(ctx, message)
	if err != nil || len(tokens) == 0 {
		if err != nil {
			logger.Warn("cache-aware prefill: tokenizer failed, fallback to process_tokens: %v", err)
		}
		return ProcessTokensSelectWorker(ctx, workers, message)
	}

	// 3) Compute prefix tree hit rate
	hitRatios := strategy.cache.Match(tokens, toWorkerSet(workers))
	logger.Debug("cache-aware prefill: hashes=%d workers=%d load=%v hit=%v", len(strategy.cache.hasher.prefixHashes(tokens)), len(workers), loads, hitRatios)

	// 4) Compute weighted score from hit rate and load
	selected := strategy.chooseByScore(ctx, workers, loads, hitRatios)

	// 5) Record prefix
	strategy.cache.Record(tokens, selected)
	logger.Debug("cache-aware prefill: selected=%s", selected)
	return selected, nil
}

// tokenize calls remote tokenizer service
func (p *prefillCacheStrategy) tokenize(ctx context.Context, message string) ([]int, error) {
	if message == "" {
		return nil, errors.New("empty prompt for tokenizer")
	}
	if p.tokenizer == nil {
		// Fallback to character-based tokenization
		return charsToTokens(message), nil
	}
	tokens, err := p.tokenizer.Tokenize(ctx, message)
	if err != nil {
		logger.Warn("cache-aware prefill: tokenizer failed, fallback to char tokens: %v", err)
		return charsToTokens(message), nil
	}
	logger.Debug("cache-aware prefill: tokenizer tokens=%v", tokens)
	return tokens, nil
}

// isLoadImbalanced determines if load is imbalanced
func (p *prefillCacheStrategy) isLoadImbalanced(loads map[string]uint64) bool {
	if len(loads) < 2 {
		return false
	}

	maxLoad := uint64(0)
	minLoad := uint64(math.MaxUint64)
	for _, v := range loads {
		if v > maxLoad {
			maxLoad = v
		}
		if v < minLoad {
			minLoad = v
		}
	}

	if maxLoad == minLoad {
		return false
	}

	diff := float64(maxLoad - minLoad)
	relative := diff / float64(maxLoad)

	return diff > p.absThreshold && relative > p.relThreshold
}

// chooseByScore selects worker by cache hit rate and load
func (p *prefillCacheStrategy) chooseByScore(ctx context.Context, workers []string, loads map[string]uint64, hitRatios map[string]int) string {
	if len(workers) == 0 {
		return ""
	}

	// TODO: reuse maxLoad from isLoadImbalanced
	var maxLoad uint64
	for _, w := range workers {
		if v := loads[w]; v > maxLoad {
			maxLoad = v
		}
	}

	bestScore := math.MaxFloat64
	selected := ""

	for _, w := range workers {
		hit := float64(hitRatios[w])
		loadRatio := 0.0
		if maxLoad > 0 {
			loadRatio = float64(loads[w]) / float64(maxLoad)
		}

		score := (100.0-hit)/100*p.hitRatioWeight + loadRatio*p.loadBalanceWeight
		logger.Debug("cache-aware score: worker=%s hit=%.1f loadRatio=%.3f score=%.3f", w, hit, loadRatio, score)

		if score < bestScore {
			bestScore = score
			selected = w
			continue
		}

		// Tie-breaker: prefer lower token load if scores are equal
		if score == bestScore && selected != "" {
			selectedTokens := GetOrCreateTokenCounter(ctx, selected).Get()
			currentTokens := GetOrCreateTokenCounter(ctx, w).Get()
			if currentTokens < selectedTokens {
				selected = w
			}
		}
	}

	return selected
}

// getRunningRequests retrieves running request metrics
func (p *prefillCacheStrategy) getRunningRequests(ctx context.Context, workers []string) map[string]uint64 {
	result := make(map[string]uint64, len(workers))
	if DefaultScheduler == nil || DefaultScheduler.managerAPI == nil {
		return result
	}

	for _, w := range workers {
		running, _, _ := DefaultScheduler.managerAPI.GetMetrics(ctx, w)
		result[w] = uint64(running)
	}
	return result
}

// Track prefix hits using a radix tree keyed by block hash
type radixPrefixCache struct {
	mu               sync.RWMutex
	root             *radixNode
	hasher           *blockHasher
	evictionDuration time.Duration
	maxNodes         int
	nodeCount        int
	allNodes         map[*radixNode]struct{}
}

type radixNode struct {
	key        []uint64
	children   map[uint64]*radixNode
	parent     *radixNode
	workers    map[string]time.Time
	lastAccess time.Time
	contextLen int
}

// newRadixPrefixCache initializes radix prefix cache with eviction and capacity control
func newRadixPrefixCache(blockSize int) *radixPrefixCache {
	if blockSize <= 0 {
		blockSize = 64
	}
	const defaultEvictionDuration = 5 * time.Minute
	const defaultMaxNodes = 200000
	root := &radixNode{
		key:        nil,
		children:   make(map[uint64]*radixNode),
		contextLen: 0,
	}
	cache := &radixPrefixCache{
		root:             root,
		hasher:           newBlockHasher(blockSize),
		evictionDuration: defaultEvictionDuration,
		maxNodes:         defaultMaxNodes,
		nodeCount:        1, // root
		allNodes:         map[*radixNode]struct{}{root: {}},
	}
	go cache.evictionWorker(cache.evictionDuration / 2)
	return cache
}

// Match returns prefix hit rate per candidate worker (0–100)
func (c *radixPrefixCache) Match(tokens []int, allowed map[string]struct{}) map[string]int {
	result := make(map[string]int)
	hashes := c.hasher.prefixHashes(tokens)
	if len(hashes) == 0 {
		return result
	}

	c.mu.RLock()
	node, matched := c.matchPrefixHelper(c.root, hashes)
	length := matched
	logger.Debug("radix match: hashes=%d matched_len=%d node_children=%d", len(hashes), matched, len(node.children))
	for n := node; n != nil; n = n.parent {
		ratio := 0
		if len(hashes) > 0 {
			ratio = length * 100 / len(hashes)
		}
		for w := range n.workers {
			if allowed != nil {
				if _, ok := allowed[w]; !ok {
					continue
				}
			}
			if ratio > result[w] {
				result[w] = ratio
			}
		}
		if len(result) > 0 {
			break
		}
		if n.parent != nil {
			length = n.parent.contextLen
		}
	}
	c.mu.RUnlock()
	return result
}

// Record inserts block-hash prefix into radix tree and tags worker
func (c *radixPrefixCache) Record(tokens []int, worker string) {
	if worker == "" {
		return
	}
	hashes := c.hasher.prefixHashes(tokens)
	if len(hashes) == 0 {
		return
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	node := c.insertHelper(c.root, hashes)
	now := time.Now()
	for n := node; n != nil; n = n.parent {
		if n.workers == nil {
			n.workers = make(map[string]time.Time)
		}
		n.workers[worker] = now
	}
	logger.Debug("radix record: worker=%s hashes=%d node_depth=%d", worker, len(hashes), node.contextLen)
}

// evictionWorker periodically evicts inactive nodes
func (c *radixPrefixCache) evictionWorker(interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		<-ticker.C
		c.evictExpired()
	}
}

func (c *radixPrefixCache) evictExpired() {
	c.mu.Lock()
	defer c.mu.Unlock()
	now := time.Now()
	removed := 0
	for childKey, child := range c.root.children {
		removed += c.evictSubtreeIfExpired(c.root, childKey, child, now)
	}
	if removed > 0 {
		logger.Debug("radix eviction: removed=%d nodeCount=%d", removed, c.nodeCount)
	}
}

// evictSubtreeIfExpired evicts expired nodes and subtrees, returns count of removed nodes
func (c *radixPrefixCache) evictSubtreeIfExpired(parent *radixNode, childKey uint64, node *radixNode, now time.Time) int {
	// Process child nodes first
	removed := 0
	for k, child := range node.children {
		removed += c.evictSubtreeIfExpired(node, k, child, now)
	}

	// Do not delete root node
	if parent == nil {
		return removed
	}

	if now.Sub(node.lastAccess) <= c.evictionDuration {
		return removed
	}

	// Delete expired node and its subtree
	if parent != nil {
		delete(parent.children, childKey)
	}
	removedSubtree := c.countSubtree(node)
	c.nodeCount -= removedSubtree
	if c.nodeCount < 1 {
		c.nodeCount = 1 // At least include root
	}
	c.removeSubtreeFromAll(node)
	return removed + removedSubtree
}

// countSubtree counts nodes in subtree rooted at node
func (c *radixPrefixCache) countSubtree(node *radixNode) int {
	count := 1
	for _, child := range node.children {
		count += c.countSubtree(child)
	}
	return count
}

// removeSubtreeFromAll removes subtree references from allNodes
func (c *radixPrefixCache) removeSubtreeFromAll(node *radixNode) {
	if node == nil {
		return
	}
	delete(c.allNodes, node)
	for _, child := range node.children {
		c.removeSubtreeFromAll(child)
	}
	// Release references for GC
	node.children = nil
	node.parent = nil
	node.workers = nil
}

// matchPrefixHelper finds longest common prefix node in radix tree
func (c *radixPrefixCache) matchPrefixHelper(node *radixNode, hashes []uint64) (*radixNode, int) {
	if len(hashes) == 0 {
		return node, node.contextLen
	}

	if child, ok := node.children[hashes[0]]; ok {
		prefixLen := matchUint64Len(child.key, hashes)
		if prefixLen > 0 {
			if prefixLen == len(child.key) {
				if prefixLen == len(hashes) {
					return child, child.contextLen
				}
				if deeperNode, deeperMatched := c.matchPrefixHelper(child, hashes[prefixLen:]); deeperNode != nil && deeperMatched > 0 {
					return deeperNode, deeperMatched
				}
				return child, child.contextLen
			}
			return child, node.contextLen + prefixLen
		}
	}
	return node, node.contextLen
}

// insertHelper inserts hash sequence into radix tree, splits nodes if needed
func (c *radixPrefixCache) insertHelper(node *radixNode, key []uint64) *radixNode {
	node.lastAccess = time.Now()

	if len(key) == 0 {
		return node
	}

	if child, ok := node.children[key[0]]; ok {
		prefixLen := matchUint64Len(child.key, key)

		if prefixLen == len(child.key) {
			if prefixLen == len(key) {
				child.lastAccess = time.Now()
				return child
			}
			return c.insertHelper(child, key[prefixLen:])
		}

		// Partial match, split required
		newNode := c.splitNode(node, child, prefixLen)
		if prefixLen == len(key) {
			return newNode
		}
		return c.insertHelper(newNode, key[prefixLen:])
	}

	// No matching child, create new node and add to children
	newNode := newRadixNode(node, key)
	node.children[key[0]] = newNode
	c.nodeCount++
	c.allNodes[newNode] = struct{}{}
	c.maybeEvictLocked()
	return newNode
}

func (c *radixPrefixCache) splitNode(parent *radixNode, child *radixNode, prefixLen int) *radixNode {
	commonKey := append([]uint64{}, child.key[:prefixLen]...)

	newNode := newRadixNode(parent, commonKey)
	parent.children[commonKey[0]] = newNode

	// Adjust atomic node
	child.key = append([]uint64{}, child.key[prefixLen:]...)
	child.parent = newNode
	child.contextLen = newNode.contextLen + len(child.key)

	if len(child.key) > 0 {
		newNode.children[child.key[0]] = child
	}
	return newNode
}

// maybeEvictLocked checks node count under write lock and evicts expired nodes if over capacity
func (c *radixPrefixCache) maybeEvictLocked() {
	if c.maxNodes <= 0 || c.nodeCount <= c.maxNodes {
		return
	}
	c.evictExpired()
	// TODO: implement stronger eviction if still over capacity (e.g., evict oldest by lastAccess)
}

// newRadixNode creates radix tree node and computes context length
func newRadixNode(parent *radixNode, key []uint64) *radixNode {
	n := &radixNode{
		key:        append([]uint64{}, key...),
		children:   make(map[uint64]*radixNode),
		parent:     parent,
		lastAccess: time.Now(),
	}
	if parent != nil {
		n.contextLen = parent.contextLen + len(key)
	} else {
		n.contextLen = len(key)
	}
	return n
}

type blockHasher struct {
	blockSize int
	seed      uint64
}

// newBlockHasher creates and initializes a new block hasher
func newBlockHasher(blockSize int) *blockHasher {
	if blockSize <= 0 {
		blockSize = 64
	}
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	return &blockHasher{
		blockSize: blockSize,
		seed:      r.Uint64(),
	}
}

// prefixHashes generates parent-chain hash sequence by block
func (h *blockHasher) prefixHashes(tokens []int) []uint64 {
	if h.blockSize <= 0 || len(tokens) < h.blockSize {
		return nil
	}
	blockCount := len(tokens) / h.blockSize
	hashes := make([]uint64, 0, blockCount)
	parent := h.seed
	buf := make([]byte, 8)

	for i := 0; i+h.blockSize <= len(tokens); i += h.blockSize {
		hasher := fnv.New64a()
		binary.LittleEndian.PutUint64(buf, parent)
		_, _ = hasher.Write(buf)

		for _, token := range tokens[i : i+h.blockSize] {
			binary.LittleEndian.PutUint64(buf, uint64(token))
			_, _ = hasher.Write(buf)
		}
		current := hasher.Sum64()
		hashes = append(hashes, current)
		parent = current
	}
	return hashes
}

func matchUint64Len(a, b []uint64) int {
	minLen := len(a)
	if len(b) < minLen {
		minLen = len(b)
	}
	i := 0
	for i < minLen && a[i] == b[i] {
		i++
	}
	return i
}

func charsToTokens(message string) []int {
	tokens := make([]int, 0, len(message))
	for _, r := range message {
		tokens = append(tokens, int(r))
	}
	return tokens
}

func toWorkerSet(workers []string) map[string]struct{} {
	set := make(map[string]struct{}, len(workers))
	for _, w := range workers {
		set[w] = struct{}{}
	}
	return set
}
