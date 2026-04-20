## 2024-04-20 - Memoizing Hardware and Spec lookups
**Learning:** Checking `paddle.device.cuda.get_device_properties()` and `importlib.util.find_spec("flashinfer")` inside utility functions like `get_sm_version()` and `has_flashinfer()` that are called frequently causes significant overhead, taking ~5ms per 10k calls without caching vs ~0.015ms with caching.
**Action:** Use `@functools.lru_cache` and `@cache` for functions that query hardware features or module specifications iteratively during model execution.
