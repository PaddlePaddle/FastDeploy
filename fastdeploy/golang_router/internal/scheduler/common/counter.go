package common

import (
	"sync/atomic"
)

type Counter struct {
	count atomic.Uint64
}

func (c *Counter) Inc() {
	c.count.Add(1)
}

func (c *Counter) Dec() {
	c.count.Add(^uint64(0))
}

func (c *Counter) Get() uint64 {
	return c.count.Load()
}

// TokenCounter records the number of tokens currently being processed by each P instance
type TokenCounter struct {
	tokens atomic.Uint64
}

func (c *TokenCounter) Add(n uint64) {
	c.tokens.Add(n)
}

func (c *TokenCounter) Get() uint64 {
	return c.tokens.Load()
}

func (c *TokenCounter) Sub(n uint64) {
	if n == 0 {
		return
	}
	for {
		old := c.tokens.Load()
		if old == 0 {
			return
		}
		var newVal uint64
		if old <= n {
			newVal = 0
		} else {
			newVal = old - n
		}
		if c.tokens.CompareAndSwap(old, newVal) {
			return
		}
	}
}
