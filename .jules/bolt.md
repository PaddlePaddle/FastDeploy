## 2024-05-18 - [Fix Sequential I/O Wait in `monitor_instance_health`]
**Learning:** Found a loop intending to `gather` tasks concurrently (had `# gather all tasks concurrently` comment) but was actually awaiting them sequentially `for inst, coro in all_tasks: resp = await coro`. This blocks execution for O(n) duration instead of O(1).
**Action:** Replace `for i, c in enumerate(all_tasks): await c` with `results = await asyncio.gather(*[coro for _, coro in all_tasks], return_exceptions=True)` when executing independent network I/O requests to run them truly concurrently.
