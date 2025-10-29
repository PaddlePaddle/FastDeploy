"""
Comprehensive test suite for splitwise_scheduler module
"""
import unittest
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from test_splitwise_scheduler_config import TestSplitWiseSchedulerConfig
from test_node_info import TestNodeInfo
from test_result_reader import TestResultReader
from test_result_writer import TestResultWriter
from test_api_scheduler import TestAPIScheduler
from test_infer_scheduler import TestInferScheduler
from test_splitwise_scheduler import TestSplitWiseScheduler


def create_test_suite():
    """Create a comprehensive test suite"""
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestSplitWiseSchedulerConfig,
        TestNodeInfo,
        TestResultReader,
        TestResultWriter,
        TestAPIScheduler,
        TestInferScheduler,
        TestSplitWiseScheduler,
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    return suite


def run_tests():
    """Run all tests"""
    suite = create_test_suite()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"测试总结:")
    print(f"运行测试: {result.testsRun}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print(f"跳过: {len(result.skipped)}")
    print(f"{'='*50}")
    
    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\n错误的测试:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
