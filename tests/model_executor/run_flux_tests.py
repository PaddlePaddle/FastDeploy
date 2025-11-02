#!/usr/bin/env python
# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Comprehensive test runner for Flux model validation.
Runs all E2E and integration tests and generates a test report.
"""

import sys
import os
import subprocess
import time
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


class Color:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    """Print a formatted header."""
    print(f"\n{Color.BOLD}{Color.BLUE}{'='*70}{Color.RESET}")
    print(f"{Color.BOLD}{Color.BLUE}{text.center(70)}{Color.RESET}")
    print(f"{Color.BOLD}{Color.BLUE}{'='*70}{Color.RESET}\n")


def print_success(text):
    """Print success message."""
    print(f"{Color.GREEN}✅ {text}{Color.RESET}")


def print_error(text):
    """Print error message."""
    print(f"{Color.RED}❌ {text}{Color.RESET}")


def print_warning(text):
    """Print warning message."""
    print(f"{Color.YELLOW}⚠️  {text}{Color.RESET}")


def print_info(text):
    """Print info message."""
    print(f"{Color.BLUE}ℹ️  {text}{Color.RESET}")


def check_dependencies():
    """Check if required dependencies are available."""
    print_header("Checking Dependencies")
    
    dependencies = {
        'paddle': 'PaddlePaddle',
        'numpy': 'NumPy',
        'PIL': 'Pillow'
    }
    
    all_available = True
    
    for module, name in dependencies.items():
        try:
            __import__(module)
            print_success(f"{name} is available")
        except ImportError:
            print_error(f"{name} is NOT available")
            all_available = False
    
    return all_available


def check_model_availability():
    """Check if Flux model is available."""
    print_header("Checking Model Availability")
    
    model_path = os.environ.get('FLUX_MODEL_PATH', None)
    
    if not model_path:
        print_warning("FLUX_MODEL_PATH environment variable not set")
        print_info("Real inference tests will be skipped")
        return False
    
    if not os.path.exists(model_path):
        print_error(f"Model path does not exist: {model_path}")
        return False
    
    # Check for expected subdirectories
    required_dirs = ['transformer', 'text_encoder', 'vae']
    missing_dirs = []
    
    for dir_name in required_dirs:
        dir_path = os.path.join(model_path, dir_name)
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print_error(f"Missing required directories: {', '.join(missing_dirs)}")
        return False
    
    print_success(f"Flux model found at: {model_path}")
    return True


def run_test_file(test_file, description):
    """Run a single test file and return results."""
    print(f"\n{Color.BOLD}Running: {description}{Color.RESET}")
    print(f"File: {test_file}")
    print("-" * 70)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout
        )
        
        elapsed_time = time.time() - start_time
        
        # Parse output
        output = result.stdout + result.stderr
        
        # Count test results
        passed = output.count('PASSED') + output.count('OK')
        failed = output.count('FAILED') + output.count('ERROR')
        skipped = output.count('SKIPPED')
        
        # Determine overall status
        if result.returncode == 0:
            print_success(f"All tests passed ({elapsed_time:.2f}s)")
            status = 'PASSED'
        elif failed > 0:
            print_error(f"Some tests failed ({elapsed_time:.2f}s)")
            status = 'FAILED'
        elif skipped > 0 and passed == 0:
            print_warning(f"All tests skipped ({elapsed_time:.2f}s)")
            status = 'SKIPPED'
        else:
            print_warning(f"Tests completed with warnings ({elapsed_time:.2f}s)")
            status = 'WARNING'
        
        return {
            'file': test_file,
            'description': description,
            'status': status,
            'passed': passed,
            'failed': failed,
            'skipped': skipped,
            'elapsed_time': elapsed_time,
            'output': output
        }
        
    except subprocess.TimeoutExpired:
        print_error("Test timed out (5 minutes)")
        return {
            'file': test_file,
            'description': description,
            'status': 'TIMEOUT',
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'elapsed_time': 300.0,
            'output': 'Test timed out after 5 minutes'
        }
        
    except Exception as e:
        print_error(f"Error running test: {e}")
        return {
            'file': test_file,
            'description': description,
            'status': 'ERROR',
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'elapsed_time': 0.0,
            'output': str(e)
        }


def generate_report(results, output_file='flux_test_report.json'):
    """Generate a comprehensive test report."""
    print_header("Generating Test Report")
    
    # Calculate summary statistics
    total_tests = len(results)
    passed_suites = sum(1 for r in results if r['status'] == 'PASSED')
    failed_suites = sum(1 for r in results if r['status'] == 'FAILED')
    skipped_suites = sum(1 for r in results if r['status'] == 'SKIPPED')
    total_time = sum(r['elapsed_time'] for r in results)
    
    # Create report
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_test_suites': total_tests,
            'passed_suites': passed_suites,
            'failed_suites': failed_suites,
            'skipped_suites': skipped_suites,
            'total_time_seconds': total_time,
        },
        'test_results': results
    }
    
    # Save to file
    report_path = os.path.join(os.path.dirname(__file__), output_file)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print_success(f"Test report saved to: {report_path}")
    
    # Print summary
    print(f"\n{Color.BOLD}Test Summary:{Color.RESET}")
    print(f"  Total test suites: {total_tests}")
    print(f"  {Color.GREEN}Passed: {passed_suites}{Color.RESET}")
    print(f"  {Color.RED}Failed: {failed_suites}{Color.RESET}")
    print(f"  {Color.YELLOW}Skipped: {skipped_suites}{Color.RESET}")
    print(f"  Total time: {total_time:.2f}s")
    
    return report


def main():
    """Main test runner."""
    print_header("Flux Model Test Suite")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check dependencies
    deps_available = check_dependencies()
    if not deps_available:
        print_warning("Some dependencies are missing. Tests may be skipped.")
    
    # Check model availability
    model_available = check_model_availability()
    
    # Define test files
    test_dir = os.path.dirname(__file__)
    tests = [
        {
            'file': os.path.join(test_dir, 'test_flux_e2e.py'),
            'description': 'End-to-End Pipeline Tests',
            'required': 'basic'
        },
        {
            'file': os.path.join(test_dir, 'test_flux_integration_full.py'),
            'description': 'Integration Tests (Weight Loading, Precision, Performance)',
            'required': 'basic'
        },
        {
            'file': os.path.join(test_dir, 'test_flux_real_inference.py'),
            'description': 'Real Model Inference Tests',
            'required': 'model'
        },
    ]
    
    # Run tests
    print_header("Running Tests")
    results = []
    
    for test in tests:
        if test['required'] == 'model' and not model_available:
            print_warning(f"Skipping {test['description']} (model not available)")
            continue
        
        if not os.path.exists(test['file']):
            print_error(f"Test file not found: {test['file']}")
            continue
        
        result = run_test_file(test['file'], test['description'])
        results.append(result)
    
    # Generate report
    report = generate_report(results)
    
    # Final status
    print_header("Final Status")
    
    if report['summary']['failed_suites'] == 0:
        if report['summary']['passed_suites'] > 0:
            print_success("All test suites passed! ✨")
            return 0
        else:
            print_warning("All tests were skipped")
            return 0
    else:
        print_error("Some test suites failed")
        return 1


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nTest run interrupted by user")
        sys.exit(130)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
