## 2026-02-09 - Dataclass Serialization Performance
**Learning:** `dataclasses.asdict` performs a deep copy and recursive conversion which is significantly slower than manual iteration over `__slots__` for simple flat dataclasses or those with known structure. For `RequestMetrics`, manual serialization was ~26% faster than `asdict`.
**Action:** When optimizing serialization of high-frequency dataclasses (especially those with `slots=True`), consider manual dictionary construction instead of `asdict`, but be careful to handle nested dataclasses correctly.
