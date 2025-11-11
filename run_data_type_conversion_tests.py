#!/usr/bin/env python3
"""
Runner script for data type conversion unit tests.
"""

import os
import sys
import unittest

# Add the project root to Python path
sys.path.insert(0, "/data/liujun/learning/paddles/hackthon9th/worktrees/hack9th_no55")


def run_all_tests():
    """Run all data type conversion tests."""

    # Discover and run all test files
    loader = unittest.TestLoader()
    test_dir = "/data/liujun/learning/paddles/hackthon9th/worktrees/hack9th_no55/tests"

    # Load test files
    test_suite = loader.discover(test_dir, pattern="test_data_type_conversion*.py")

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    return result.wasSuccessful()


def run_specific_test(test_file):
    """Run a specific test file."""

    # Import the test module
    module_name = os.path.basename(test_file)[:-3]  # Remove .py extension
    module = __import__(f"tests.{module_name}")

    # Load test suite from module
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(module)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


def main():
    """Main function."""

    if len(sys.argv) > 1:
        # Run specific test file
        test_file = sys.argv[1]
        if os.path.exists(test_file):
            print(f"Running specific test: {test_file}")
            success = run_specific_test(test_file)
        else:
            print(f"Test file not found: {test_file}")
            return 1
    else:
        # Run all data type conversion tests
        print("Running all data type conversion tests...")
        success = run_all_tests()

    if success:
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
