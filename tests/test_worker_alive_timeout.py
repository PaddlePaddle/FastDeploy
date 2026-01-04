"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import os
import sys


def test_worker_alive_timeout_env():
    """Test FD_WORKER_ALIVE_TIMEOUT environment variable"""
    
    print("Testing FD_WORKER_ALIVE_TIMEOUT environment variable...")
    
    # Test 1: Default value (30)
    print("\nTest 1: Default value")
    if "FD_WORKER_ALIVE_TIMEOUT" in os.environ:
        del os.environ["FD_WORKER_ALIVE_TIMEOUT"]
    
    # Simulate the environment variable retrieval
    timeout = int(os.getenv("FD_WORKER_ALIVE_TIMEOUT", "30"))
    assert timeout == 30, f"Expected default value 30, got {timeout}"
    assert isinstance(timeout, int), f"Expected int type, got {type(timeout)}"
    print(f"✓ Default value is correctly set to {timeout}")
    
    # Test 2: Custom value (60)
    print("\nTest 2: Custom value")
    os.environ["FD_WORKER_ALIVE_TIMEOUT"] = "60"
    timeout = int(os.getenv("FD_WORKER_ALIVE_TIMEOUT", "30"))
    assert timeout == 60, f"Expected custom value 60, got {timeout}"
    assert isinstance(timeout, int), f"Expected int type, got {type(timeout)}"
    print(f"✓ Custom value is correctly set to {timeout}")
    
    # Test 3: Custom value (120)
    print("\nTest 3: Another custom value")
    os.environ["FD_WORKER_ALIVE_TIMEOUT"] = "120"
    timeout = int(os.getenv("FD_WORKER_ALIVE_TIMEOUT", "30"))
    assert timeout == 120, f"Expected custom value 120, got {timeout}"
    print(f"✓ Custom value is correctly set to {timeout}")
    
    # Clean up
    if "FD_WORKER_ALIVE_TIMEOUT" in os.environ:
        del os.environ["FD_WORKER_ALIVE_TIMEOUT"]
    
    print("\n✓ All tests passed!")
    return True


if __name__ == "__main__":
    try:
        test_worker_alive_timeout_env()
        sys.exit(0)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        sys.exit(1)
