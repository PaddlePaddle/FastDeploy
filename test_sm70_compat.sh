#!/bin/bash
# SM70 (V100) Compatibility Test Script
# Usage: bash test_sm70_compat.sh

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="test_sm70_compat_${TIMESTAMP}.log"

echo "=== SM70 Compatibility Test Log ===" | tee $LOG_FILE
echo "Timestamp: $(date)" | tee -a $LOG_FILE
echo "Log file: $LOG_FILE" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# 1. Platform Detection Test
echo "=== 1. Platform Detection Test ===" | tee -a $LOG_FILE
python -c "
from fastdeploy.model_executor.layers.utils import get_sm_version
print(f'current sm_version={get_sm_version()}')
" 2>&1 | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# 2. Basic Inference Test
echo "=== 2. Basic Inference Test ===" | tee -a $LOG_FILE
python -c "
from fastdeploy.platforms import current_platform
from fastdeploy.model_executor.layers.utils import get_sm_version

print(f'Platform: {current_platform}')
print(f'SM Version: {get_sm_version()}')
print(f'Is V100 (SM70): {get_sm_version() == 70}')
" 2>&1 | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# 3. Running Platform Tests
echo "=== 3. Running Platform Tests ===" | tee -a $LOG_FILE
pytest tests/platforms/test_platforms.py -v 2>&1 | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# 4. Running Attention Tests (PR related)
echo "=== 4. Running Attention Tests ===" | tee -a $LOG_FILE
pytest tests/layers/test_attention_layer.py -v 2>&1 | tee -a $LOG_FILE
pytest tests/layers/test_native_paddle_backend.py -v 2>&1 | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# 5. Running FFN Tests (PR related)
echo "=== 5. Running FFN Tests ===" | tee -a $LOG_FILE
pytest tests/layers/test_ffn.py -v 2>&1 | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# 6. Running MoE Tests (PR related)
echo "=== 6. Running MoE Tests ===" | tee -a $LOG_FILE
pytest tests/layers/test_fusedmoe.py -v 2>&1 | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# 7. Running W4AFP8 Quantization Tests (PR related)
echo "=== 7. Running W4AFP8 Quantization Tests ===" | tee -a $LOG_FILE
pytest tests/quantization/test_w4afp8.py -v 2>&1 | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# 8. Running All Quantization Tests
echo "=== 8. Running All Quantization Tests ===" | tee -a $LOG_FILE
pytest tests/quantization/ -v 2>&1 | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# 9. Running Non-FP8 Quantization Tests (V100 supported)
echo "=== 9. Running Non-FP8 Quantization Tests ===" | tee -a $LOG_FILE
pytest tests/quantization/ -v -k "not fp8 and not block_wise" 2>&1 | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

echo "=== Test Complete ===" | tee -a $LOG_FILE
echo "Finished at: $(date)" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# Summary
echo "=== Test Summary ===" | tee -a $LOG_FILE
echo "Tests covered:" | tee -a $LOG_FILE
echo "  - Platform detection (SM version)" | tee -a $LOG_FILE
echo "  - Platform tests" | tee -a $LOG_FILE
echo "  - Attention layer tests (test_attention_layer.py)" | tee -a $LOG_FILE
echo "  - FFN tests (test_ffn.py)" | tee -a $LOG_FILE
echo "  - MoE tests (test_fusedmoe.py)" | tee -a $LOG_FILE
echo "  - W4AFP8 quantization tests (test_w4afp8.py)" | tee -a $LOG_FILE
echo "  - All quantization tests" | tee -a $LOG_FILE
echo "  - Non-FP8 quantization tests (V100 compatible)" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE
echo "Results saved to: $LOG_FILE"
