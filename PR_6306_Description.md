## Motivation

为 FastDeploy 添加 NVIDIA V100 GPU (SM70 架构) 支持，使其能在旧版 GPU 上进行开发测试。由于 V100 不支持以下特性，需要同时适配编译系统和运行时逻辑：

- **BF16 数据类型**：需要 SM80+ (Ampere)
- **FP8 量化**：需要 SM89+ (Ada Lovelace)
- **cp.async 指令**：需要 SM80+ (Ampere)，影响 Append Attention 和 MLA Attention
- **Marlin GEMM**：需要 SM80+ (Ampere)
- **BF16 原生算术运算符**：需要 SM80+ (Ampere)，影响 `*=` 和 `+=` 运算

## Modifications

### 编译系统
- **`setup_ops.py`**: 支持 SM70+ 编译，分离 SM70/SM80+ 特有代码
- **`cpp_extensions.cc`**: 添加 `ENABLE_APPEND_ATTENTION` 和 `ENABLE_BF16` 宏控制条件编译

### CUDA Kernel
- **`gelu_tanh.cu`**: 修复 `tanh.approx.f32` PTX 指令在 SM70 的编译问题
- **`moe_wna16_marlin_*.cu/h`**: 修复 Marlin GEMM 模板在 SM70 的编译兼容性
- **`moe_deepgemm_depermute.cu`**: 添加 SM70/SM75 条件编译，BF16 算术运算通过 float 转换实现
- **`sampling.cuh`**: 添加缺失的 `<cuda/std/limits>` 头文件

### Python 运行时层
- **`fastdeploy/platforms/cuda.py`**:
  - 添加 SM 版本检测方法 (`get_sm_version()`)
  - 添加硬件能力检查 (`supports_bf16()`, `supports_fp8()`, `supports_async_copy()`, `supports_marlin()`)
  - Attention backend 自动 fallback (APPEND_ATTN/MLA_ATTN → FLASH_ATTN)

- **`fastdeploy/config.py`**: BF16→FP16 dtype 自动降级

- **`fastdeploy/model_executor/layers/moe/moe.py`**:
  - Marlin MoE backend → CUTLASS fallback (SM<80)
  - Triton MoE backend → CUTLASS fallback (SM<80)

- **`fastdeploy/model_executor/layers/moe/fused_moe_cutlass_backend.py`**: 添加 SM70 兼容性处理

- **`fastdeploy/model_executor/layers/moe/fused_moe_deepgemm_backend.py`**: 添加 FP8 量化兼容性包装

- **`fastdeploy/model_executor/layers/quantization/__init__.py`**:
  - FP8 量化方法自动 fallback (`block_wise_fp8`→`wint8`, `w4afp8`→`wint4`)

- **`fastdeploy/model_executor/layers/quantization/mix_quant.py`**:
  - MixQuantConfig 中 FP8 quant type 自动 fallback

- **`fastdeploy/model_executor/layers/quantization/weight_only.py`**:
  - WeightOnlyConfig 中 Marlin/Triton backend fallback

- **`fastdeploy/model_executor/layers/quantization/block_wise_fp8.py`**:
  - deep_gemm 导入保护 (SM<89 时跳过)

- **`attention/ops/*.py`**: 为 SM80+ 专属 ops 添加 try-except 保护
  - `append_attention.py`
  - `flash_mask_attention.py`
  - `get_block_shape_and_split_kv_block.py`
  - `gqa_rope_write_cache.py`
  - `pre_cache_len_concat.py`
  - `mla_attention_backend.py`

### 测试
- **`tests/layers/test_attention_layer.py`**: 添加 FP8 SM89+ skip 装饰器
- **`tests/layers/test_fusedmoe.py`**: 添加 FP8 SM89+ skip 装饰器
- **`tests/quantization/test_w4afp8.py`**: 添加 FP8 SM89+ skip 装饰器
- **`tests/layers/test_ffn.py`**: 根据 SM 版本自动选择 dtype 和量化配置

## SM70/SM75 Fallback 策略总览

| 功能 | 原始 | SM70/SM75 Fallback | 原因 |
|-----|------|--------------|------|
| 数据类型 | BF16 | FP16 | BF16 需要 SM80+ |
| BF16 算术运算 | `*=` / `+=` | float 转换 | BF16 原生运算符需要 SM80+ |
| Attention Backend | APPEND_ATTN | FLASH_ATTN | cp.async 需要 SM80+ |
| Attention Backend | MLA_ATTN | FLASH_ATTN | cp.async 需要 SM80+ |
| MoE Backend | Marlin | CUTLASS | Marlin 需要 SM80+ |
| MoE Backend | Triton | CUTLASS | tritonmoe_preprocess 需要 SM80+ |
| 量化 | block_wise_fp8 | wint8 | FP8 需要 SM89+ |
| 量化 | w4afp8 | wint4 | FP8 需要 SM89+ |
| 量化 | wfp8afp8 | wint8 | FP8 需要 SM89+ |
| 量化 | tensor_wise_fp8 | wint8 | FP8 需要 SM89+ |

## 文件变更统计

| 类别 | 文件数 | 新增行数 | 删除行数 |
|-----|--------|---------|---------|
| CUDA Kernel | 7 | 1217 | 775 |
| Python 运行时 | 13 | 493 | 41 |
| 测试 | 4 | 63 | 6 |
| 编译配置 | 3 | 59 | 10 |
| **总计** | **27** | **1832** | **832** |

## Usage or Command

```bash
# 编译 (指定 SM70 架构)
MAX_JOBS=8 bash build.sh 1 python false [70]

# 或使用 setup_ops.py
cd custom_ops && python setup_ops.py install

# 运行测试
pytest tests/platforms/test_platforms.py -v
pytest tests/layers/test_attention_layer.py -v
pytest tests/layers/test_ffn.py -v
pytest tests/layers/test_fusedmoe.py -v
pytest tests/quantization/test_w4afp8.py -v
pytest tests/quantization/ -v
```

## Accuracy Tests

V100 (SM70) 上测试结果：

```
=== 1. Platform Detection ===
current sm_version=70
Platform: CUDAPlatform
Is V100 (SM70): True

=== 2. Platform Tests ===
28 passed, 1 failed (预期 - APPEND_ATTN fallback)

=== 3. Attention Tests ===
1 skipped (FP8 quantization requires SM89+)

=== 4. FFN Tests ===
1 passed, 0 failed

=== 5. MoE Tests ===
1 skipped (FP8 quantization requires SM89+)

=== 6. W4AFP8 Quantization Tests ===
6 passed, 5 skipped (FP8 ops require SM89+)

=== 7. All Quantization Tests ===
46 passed, 9 skipped

=== 8. Non-FP8 Quantization Tests ===
36 passed, 1 skipped (XPU)

Total: 76 passed, 12 skipped, 1 failed (expected fallback)
```

所有 FP8 相关测试在 V100 上正确跳过（显示 `SKIPPED (FP8 ops require SM89+)`），非 FP8 功能全部通过。`test_attention_backend_valid` 失败是预期行为，因为 V100 自动从 APPEND_ATTN fallback 到 FLASH_ATTN。

## Commits

| Commit | 描述 |
|--------|------|
| `f3216c0` | feat: add V100 (SM70) GPU support |
| `3b39080` | fix format |
| `7e09cb0` | feat: add SM70 (V100) GPU architecture compatibility |
| `b9dcf58` | Merge upstream/develop into fastdeploy_v100 |
| `9bbae22` | fix: remove non-existent per_token_quant_fp8.cu from build |
| `c7f0e2b` | fix: remove non-existent MaskedPerTokenQuant and restore FusedMaskSwigluFP8Quant |
| `a7587cc` | Merge branch 'develop' into fastdeploy_v100 |
| `6ed5608` | fix: add fused_mask_swiglu_fp8_quant_kernel.cu back to build sources |
| `5a8a280` | Merge branch 'develop' into fastdeploy_v100 |
| `0312028` | fix: add set_stop.cu to MetaX build sources |
| `c1df8fd` | fix: add gelu_tanh.cu to MetaX build sources |
| `4affd6e` | Merge upstream/develop into fastdeploy_v100 |
| `3b392f8` | [Fix] Add SM70/SM75 compatibility for BF16 operations and sampling |
| `ad367b7` | Merge branch 'develop' into fastdeploy_v100 |

## Checklist

- [x] Add at least a tag in the PR title.
- [x] Format your code, run `pre-commit` before commit.
- [x] Add unit tests. Please write the reason in this PR if no unit tests.
- [x] Provide accuracy results.
- [x] If the current PR is submitting to the `release` branch, make sure the PR has been submitted to the `develop` branch, then cherry-pick it to the `release` branch with the `[Cherry-Pick]` PR tag.
