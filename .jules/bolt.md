
## 2025-05-18 - Faster Dataclass Serialization
**Learning:** For performance-critical dataclass serialization, especially in objects like `RequestMetrics` that use `slots=True`, `dataclasses.asdict()` is noticeably slow. It does unnecessary deep copying and reflection under the hood. Iterating over `__slots__` and manually building the dictionary (and specifically calling `asdict` only on nested unslotted dataclasses like `SpeculateMetrics`) yields a ~2x performance speedup.
**Action:** Replace `asdict(self)` with manual `__slots__` iteration in frequently serialized dataclasses to improve serialization throughput.
