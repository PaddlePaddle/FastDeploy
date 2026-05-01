## 2026-05-01 - [Optimize Entropy Calculation list pop]
**Learning:** In the Python backend codebase, using `pop(0)` sequentially in loops on large lists causes a measurable O(n^2) performance bottleneck, especially in areas like logits/entropy processing (`fastdeploy/model_executor/entropy_utils.py`). It took 0.09s vs 0.004s for 32k tokens.
**Action:** Replace `pop(0)` in loops with slice extension (`extend(lst[idx:idx+n])`) or index-based access where possible.
