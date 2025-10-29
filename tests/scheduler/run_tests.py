#!/usr/bin/env python3
"""
Simple test runner for splitwise_scheduler tests
"""
import sys
import os
import unittest

# Add the project root to the Python path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

def run_tests():
    """Run all tests in the scheduler module"""
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"测试总结:")
    print(f"运行测试: {result.testsRun}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print(f"跳过: {len(result.skipped)}")
    print(f"{'='*60}")
    
    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"- {test}")
            print(f"  错误: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\n错误的测试:")
        for test, traceback in result.errors:
            print(f"- {test}")
            print(f"  错误: {traceback.split('Exception:')[-1].strip()}")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    print("开始运行 SplitWise Scheduler 测试套件...")
    print("="*60)
    
    success = run_tests()
    
    if success:
        print("\n✅ 所有测试通过!")
        sys.exit(0)
    else:
        print("\n❌ 部分测试失败!")
        sys.exit(1)
