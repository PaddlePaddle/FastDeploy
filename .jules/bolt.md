## 2024-04-13 - [Fast numpy array padding]
**Learning:** In the processor, when padding variable length list sequences into a batch (`pad_batch_data`), using intermediate python lists with concatenation before a final `np.array()` wrapper causes severe $O(N \times \text{max\_len})$ overhead.
**Action:** Always prefer `np.full` to pre-allocate an array with the `pad_id`, and write the variable length elements directly to array slices (`array[i, :length] = inst`).
