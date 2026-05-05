## 2025-05-05 - Replacing O(N^2) list pop(0) in fastdeploy/model_executor/entropy_utils.py
**Learning:** Found O(N^2) list operations where `entropy.pop(0)` was called inside nested loops over batch size and sequence lengths.
**Action:** Replace `pop(0)` within a loop with an index tracker and bulk addition via `.extend(list[start:end])`.
