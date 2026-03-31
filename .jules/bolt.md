
## 2025-02-23 - Avoid dataclasses.asdict in Hot Paths
**Learning:** `dataclasses.asdict` does recursive deepcopy internally and is incredibly slow for large dataclasses or objects instantiated frequently. In FastDeploy, it was used in `RequestMetrics.to_dict()`, creating significant overhead.
**Action:** When defining `to_dict()` or custom serialization methods for fast/frequent dataclasses, avoid `asdict`. Instead, iterate through `self.__dataclass_fields__` with `getattr` and do shallow copying for basic types (`int`, `float`, `str`, `bool`, `type(None)`). For nested dataclasses, ensure they also implement their own `to_dict()` method to skip the `asdict` recursive penalty.
