## 2025-04-01 - Fast Serialization for Metrics
**Learning:** `dataclasses.asdict()` relies on recursive deep cloning internally, making it extremely slow for high-frequency operations like serializing metrics per request/token. Shallow iterating over `__dataclass_fields__` directly avoids this overhead.
**Action:** Replace `asdict()` with a custom field iteration method (falling back appropriately) in hot paths like metrics classes (`RequestMetrics`, `SpeculateMetrics`).
