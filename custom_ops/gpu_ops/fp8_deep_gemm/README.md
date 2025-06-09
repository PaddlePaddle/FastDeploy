# DeepGEMM

DeepGEMM 安装流程

## Installation

首先安装自定义算子，确保cutlass已经`git clone`到[custom_ops/third_party/cutlass](../../third_party/cutlass)

安装deep_gemm:

```bash
# Make symbolic links for third-party (CUTLASS and CuTe) include directories
python setup.py develop

# Add the project path to PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH

# or install directly
python setup.py install
```

### Test

```bash
# Test all GEMM implements (normal, contiguous-grouped and masked-grouped)
python tests/test_core.py
```
