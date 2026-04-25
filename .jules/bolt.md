## 2025-03-01 - O(N^2) String operations

**Learning:** String manipulation inside a loop (like `prefix += s1[i]`) causes O(N^2) behavior due to string immutability in Python, especially problematic since we process JSON objects where string values can be large.
**Action:** Always prefer string slicing `s1[:i]` or Python builtins like `.startswith()`, `.endswith()`, and `.replace()` instead of manual character-by-character loops.
