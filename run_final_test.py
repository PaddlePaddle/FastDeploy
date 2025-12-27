#!/usr/bin/env python3
import os
import sys

# 设置环境变量 - 在导入任何 fastdeploy 模块之前
os.environ["FD_LOG_DIR"] = "/tmp/fastdeploy_logs"
os.environ["MODEL_PATH"] = "/tmp/test_models"
os.environ["FD_ENGINE_QUEUE_PORT"] = "6780"
os.environ["FD_CACHE_QUEUE_PORT"] = "6781"

# 确保日志目录存在
os.makedirs("/tmp/fastdeploy_logs", exist_ok=True)

# 设置路径
sys.path.insert(0, "/home/hj/lzj/FastDeploy")

print("Testing FastDeploy import...")
try:
    import fastdeploy
    print("FastDeploy imported successfully!")

    print("Running TestCommonEngineAdditionalCoverage...")
    from tests.engine.test_common_engine import TestCommonEngineAdditionalCoverage

    import unittest
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestCommonEngineAdditionalCoverage)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print(f"\n=== Final Results ===")
    print(f"Total tests: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    success_rate = result.testsRun - len(result.failures) - len(result.errors)
    print(f"Success rate: {success_rate}/{result.testsRun}")

    if success_rate == result.testsRun:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
