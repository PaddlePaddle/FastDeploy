## [Feature] Support entropy computation in FD gpu_model_runner

### Summary

在 FastDeploy 的 `gpu_model_runner` 路径下支持 entropy 计算，使 `--enable-entropy` 功能同时兼容 FD Runner 和 ernie5_model_runner 两条执行路径。

### 改动内容

**entropy_utils.py**:
- 新增 `_log_entropy()` 公共函数，统一 entropy 输出格式（含 `all_values` 完整序列）
- 新增 `calculate_logits_entropy_fd()`: FD Runner non-MTP 路径的 entropy 计算
- 新增 `speculate_calculate_logits_entropy_fd()`: FD Runner MTP (speculative decoding) 路径的 entropy 计算
- 新增 `flush_entropy_on_stop()`: 处理 `unified_update_model_status` 后因 max_dec_len 截断而新增的 stop 请求
- 重构原有 ernie5_runner 路径使用 `_log_entropy()` 避免重复代码

**pre_and_post_process.py**:
- 根据 `EB5_ENABLE_FD_RUNNER` 环境变量自动路由到对应的 entropy 计算函数
- MTP 路径中在 `unified_update_model_status` 之后调用 `flush_entropy_on_stop`，确保 max_dec_len 截断的请求 entropy 不丢失

**gpu_model_runner.py**:
- 修正 `seq_lens_this_time` 切片为 `[:batch_size]`，确保 FD Runner 路径下 `real_bsz` 计算正确

### FD Runner vs Ernie5 Runner 差异

| 项目 | FD Runner | Ernie5 Runner |
|------|-----------|---------------|
| logits 布局 | 每 batch slot 一行 `[bsz, vocab]` | token 级展开 `[total_tokens, vocab]` |
| MTP 执行顺序 | entropy → update_status → flush | update_status → entropy(含 flush) |
| temperature 索引 | 直接 `temperature[i]` | 需要 `repeat_interleave` 展开 |

### 验证结果

- 并发 6 客户端 + max-num-seqs 4 服务端，MTP/Non-MTP 共 16 条请求全部正确输出 entropy
- FD Runner 与 Ernie5 Runner 首次请求 entropy 前几步完全一致，后续差异属于 FP8+CUDA Graph 固有非确定性
- `flush_entropy_on_stop` 修复了 max_dec_len 截断时 entropy 丢失的问题
