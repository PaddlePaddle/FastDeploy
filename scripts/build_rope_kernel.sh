#!/usr/bin/env bash
# 独立编译 RoPE kernel，不影响系统环境
# 用法: bash scripts/build_rope_kernel.sh

set -e

echo "=========================================="
echo "RoPE Kernel 编译脚本 (Metax MACA)"
echo "=========================================="

# 保存原始环境变量
ORIG_PATH=$PATH
ORIG_LD_LIBRARY_PATH=$LD_LIBRARY_PATH

# 设置 Metax 编译环境 (仅在当前 shell)
export MACA_PATH=/opt/maca

# 初始化 cu-bridge (如果不存在)
if [ ! -d ${HOME}/cu-bridge ]; then
    echo "[INFO] 初始化 cu-bridge..."
    ${MACA_PATH}/tools/cu-bridge/tools/pre_make
fi

# 设置编译环境变量
export CUCC_PATH=/opt/maca/tools/cu-bridge
export CUCC_CMAKE_ENTRY=2
export CUDA_PATH=${HOME}/cu-bridge/CUDA_DIR
export PATH=${CUDA_PATH}/bin:${MACA_PATH}/mxgpu_llvm/bin:${MACA_PATH}/bin:${CUCC_PATH}/tools:${CUCC_PATH}/bin:${ORIG_PATH}
export LD_LIBRARY_PATH=${CUDA_PATH}/lib64:${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${ORIG_LD_LIBRARY_PATH}

echo "[INFO] 编译环境已设置"
echo "  CUDA_PATH: $CUDA_PATH"
echo "  CUCC_PATH: $CUCC_PATH"

# 源文件和输出路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_FILE="${PROJECT_ROOT}/custom_ops/metax_ops/apply_rope_qkv.cu"
OUTPUT_FILE="${PROJECT_ROOT}/fastdeploy/model_executor/models/paddleocr_vl/apply_rope_qkv_pd_.so"

echo "[INFO] 源文件: $SRC_FILE"
echo "[INFO] 输出文件: $OUTPUT_FILE"

# 备份原有的 .so (如果存在)
if [ -f "$OUTPUT_FILE" ]; then
    BACKUP_FILE="${OUTPUT_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
    echo "[INFO] 备份原有 .so: $BACKUP_FILE"
    cp "$OUTPUT_FILE" "$BACKUP_FILE"
fi

# 使用 paddle 的 cpp_extension 编译
echo "[INFO] 开始编译..."
cd "$PROJECT_ROOT"

# 设置 Metax 库的链接标志
export LDFLAGS="-L/opt/maca/lib -lruntime_cu -lmcruntime -lmctlassEx -lmccompiler"

python -c "
import os
import sys
import paddle
from paddle.utils.cpp_extension import load

# 设置 Metax 环境变量
os.environ['PADDLE_CUSTOM_DEVICE'] = 'metax_gpu'

source_files = ['${SRC_FILE}']
extra_include = ['${PROJECT_ROOT}/custom_ops/gpu_ops']

try:
    op = load(
        name='apply_rope_qkv_pd',
        sources=source_files,
        extra_include_paths=extra_include + ['${PROJECT_ROOT}/custom_ops/third_party/nlohmann_json/include'],
        extra_cxx_cflags=['-DPADDLE_WITH_CUSTOM_DEVICE_METAX_GPU', '-DPADDLE_DEV'],
        extra_cuda_cflags=['-DPADDLE_WITH_CUSTOM_DEVICE_METAX_GPU', '-DPADDLE_DEV'],
        extra_ldflags=['-L/opt/maca/lib', '-lruntime_cu', '-lmcruntime', '-lmctlassEx', '-lmccompiler'],
        verbose=False
    )
    print('[SUCCESS] 编译成功')
    print(f'编译产物: {op}')
except Exception as e:
    print(f'[ERROR] 编译失败: {e}')
    sys.exit(1)
"

# 找到编译产物并复制到目标位置
BUILD_DIR="${HOME}/.cache/paddle_extensions/apply_rope_qkv_pd"

SO_FILE=$(find "$BUILD_DIR" -name "apply_rope_qkv_pd*.so" -type f 2>/dev/null | head -1)

if [ -z "$SO_FILE" ]; then
    echo "[ERROR] 找不到编译产物 .so 文件"
    exit 1
fi

echo "[INFO] 找到编译产物: $SO_FILE"
echo "[INFO] 复制到: $OUTPUT_FILE"
cp "$SO_FILE" "$OUTPUT_FILE"

# 验证
if [ -f "$OUTPUT_FILE" ]; then
    echo "[SUCCESS] RoPE kernel 编译完成"
    ls -lh "$OUTPUT_FILE"
else
    echo "[ERROR] 复制失败"
    exit 1
fi

# 恢复环境变量 (虽然脚本结束会自动恢复，但显式恢复更清晰)
export PATH=$ORIG_PATH
export LD_LIBRARY_PATH=$ORIG_LD_LIBRARY_PATH

echo "=========================================="
echo "编译完成！"
echo "=========================================="
