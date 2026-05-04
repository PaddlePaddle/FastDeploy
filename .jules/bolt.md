## 2024-05-04 - Inefficient list.pop(0) operations
**Learning:** Found multiple usages of `list.pop(0)` in the codebase, particularly in `fastdeploy/model_executor/entropy_utils.py` . `pop(0)` on a list is an O(N) operation because it requires shifting all subsequent elements. For queues or iterating over elements sequentially, using an index or `collections.deque` is much faster.
**Action:** Replace O(N) `pop(0)` operations with O(1) index tracking or `collections.deque` where appropriate to optimize execution performance.
