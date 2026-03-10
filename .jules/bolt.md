
## 2024-05-24 - [Optimize RequestMetrics.to_dict serialization]
**Learning:** `dataclasses.asdict` does deep copies, which incurs significant overhead for frequently serialized dataclasses like `RequestMetrics` in a hotpath like the API server.
**Action:** Use manual `__slots__` iteration with `getattr` for faster serialization when the dataclass structure is mostly primitives, while only falling back to `asdict` for nested dataclasses lacking slots (like `SpeculateMetrics`).
