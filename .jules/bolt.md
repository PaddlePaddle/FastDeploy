## 2026-04-19 - Unnecessary dtype conversions in hot paths
**Learning:** In PaddlePaddle, calling `.astype(dtype)` creates a new tensor and dispatches a kernel even when the tensor is already of the target dtype, which can slow down hot paths like RMSNorm.
**Action:** Add explicit conditional checks (`if tensor.dtype != target_dtype`) before calling `.astype` in frequently executed methods to save memory allocations and kernel dispatch overheads.
