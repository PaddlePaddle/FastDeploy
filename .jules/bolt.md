## 2024-05-18 - Avoid List Comprehensions inside `sum()` and `all()`
**Learning:** Found several places where list comprehensions were evaluated fully in memory before being passed to `sum()` or `all()`. For `all()`, this completely defeats the short-circuiting behavior.
**Action:** Replaced `sum([...])` and `all([...])` with generator expressions `sum(...)` and `all(...)` to reduce memory allocations and enable short-circuiting.
