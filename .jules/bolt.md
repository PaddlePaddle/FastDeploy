## 2025-05-18 - FastDeploy get_tensor PySafeSlice parsing
**Learning:** Checking for `PySafeSlice` by calling `str(type(input))` and looking for a substring (`"PySafeSlice" in str(type(input))`) creates unneeded string formatting overhead in the extremely hot path of `get_tensor` logic since it's frequently called across weight loading and other utils.
**Action:** Replace `str(type(input))` with `type(input).__name__ == "PySafeSlice"` to speed it up.
