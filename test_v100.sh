#!/bin/bash
# ============================================================
# FastDeploy V100 测试脚本（带日志）
# ============================================================

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="test_v100_${TIMESTAMP}.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG_FILE
}

log "=========================================="
log "FastDeploy V100 Test Started"
log "Log file: $LOG_FILE"
log "=========================================="

# 1. Platform 检测验证
log "=== 1. Platform Detection ==="
START_TIME=$(date +%s)
python -c "
from fastdeploy.model_executor.layers.utils import get_sm_version
from fastdeploy.platforms import current_platform
print(f'Platform: {current_platform}')
print(f'SM Version: {get_sm_version()}')
print(f'Is V100 (SM70): {get_sm_version() == 70}')
" 2>&1 | tee -a $LOG_FILE
END_TIME=$(date +%s)
log "Platform Detection completed in $((END_TIME - START_TIME)) seconds"

# 2. Platform Tests
log "=== 2. Platform Tests ==="
START_TIME=$(date +%s)
python -m pytest tests/platforms/test_platforms.py -v 2>&1 | tee -a $LOG_FILE || true
END_TIME=$(date +%s)
log "Platform Tests completed in $((END_TIME - START_TIME)) seconds"

# 3. Attention Tests
log "=== 3. Attention Tests ==="
START_TIME=$(date +%s)
python -m pytest tests/layers/test_attention_layer.py -v 2>&1 | tee -a $LOG_FILE || true
END_TIME=$(date +%s)
log "Attention Tests completed in $((END_TIME - START_TIME)) seconds"

# 4. FFN Tests
log "=== 4. FFN Tests ==="
START_TIME=$(date +%s)
python -m pytest tests/layers/test_ffn.py -v 2>&1 | tee -a $LOG_FILE || true
END_TIME=$(date +%s)
log "FFN Tests completed in $((END_TIME - START_TIME)) seconds"

# 5. MoE Tests
log "=== 5. MoE Tests ==="
START_TIME=$(date +%s)
python -m pytest tests/layers/test_fusedmoe.py -v 2>&1 | tee -a $LOG_FILE || true
END_TIME=$(date +%s)
log "MoE Tests completed in $((END_TIME - START_TIME)) seconds"

# 6. W4AFP8 Quantization Tests
log "=== 6. W4AFP8 Quantization Tests ==="
START_TIME=$(date +%s)
python -m pytest tests/quantization/test_w4afp8.py -v 2>&1 | tee -a $LOG_FILE || true
END_TIME=$(date +%s)
log "W4AFP8 Tests completed in $((END_TIME - START_TIME)) seconds"

# 7. All Quantization Tests
log "=== 7. All Quantization Tests ==="
START_TIME=$(date +%s)
python -m pytest tests/quantization/ -v 2>&1 | tee -a $LOG_FILE || true
END_TIME=$(date +%s)
log "All Quantization Tests completed in $((END_TIME - START_TIME)) seconds"

log "=========================================="
log "Test Completed"
log "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
log "Full log saved to: $LOG_FILE"
log "=========================================="
