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
import re


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
    
    print("\n✓ All environment variable tests passed!")
    return True


def test_integration_with_serving_files():
    """Test that FD_WORKER_ALIVE_TIMEOUT is properly integrated in serving files"""
    
    print("\nTesting integration with serving_chat.py and serving_completion.py...")
    
    # Get the repository root
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    serving_chat_path = os.path.join(repo_root, "fastdeploy", "entrypoints", "openai", "serving_chat.py")
    serving_completion_path = os.path.join(repo_root, "fastdeploy", "entrypoints", "openai", "serving_completion.py")
    envs_path = os.path.join(repo_root, "fastdeploy", "envs.py")
    
    # Test 1: Check that envs.py has FD_WORKER_ALIVE_TIMEOUT
    print("\nTest 1: Checking envs.py contains FD_WORKER_ALIVE_TIMEOUT")
    with open(envs_path, 'r') as f:
        envs_content = f.read()
    assert 'FD_WORKER_ALIVE_TIMEOUT' in envs_content, "FD_WORKER_ALIVE_TIMEOUT not found in envs.py"
    assert 'lambda: int(os.getenv("FD_WORKER_ALIVE_TIMEOUT"' in envs_content, "FD_WORKER_ALIVE_TIMEOUT definition not correct in envs.py"
    print("✓ envs.py contains FD_WORKER_ALIVE_TIMEOUT")
    
    # Test 2: Check that serving_chat.py imports envs and uses FD_WORKER_ALIVE_TIMEOUT
    print("\nTest 2: Checking serving_chat.py integration")
    with open(serving_chat_path, 'r') as f:
        chat_content = f.read()
    
    assert 'import fastdeploy.envs as envs' in chat_content, "serving_chat.py does not import fastdeploy.envs"
    
    # Check that check_health is called with envs.FD_WORKER_ALIVE_TIMEOUT
    pattern = r'check_health\(time_interval_threashold=envs\.FD_WORKER_ALIVE_TIMEOUT\)'
    matches = re.findall(pattern, chat_content)
    assert len(matches) >= 2, f"Expected at least 2 check_health calls with FD_WORKER_ALIVE_TIMEOUT in serving_chat.py, found {len(matches)}"
    print(f"✓ serving_chat.py imports envs and uses FD_WORKER_ALIVE_TIMEOUT in {len(matches)} check_health calls")
    
    # Test 3: Check that serving_completion.py imports envs and uses FD_WORKER_ALIVE_TIMEOUT
    print("\nTest 3: Checking serving_completion.py integration")
    with open(serving_completion_path, 'r') as f:
        completion_content = f.read()
    
    assert 'import fastdeploy.envs as envs' in completion_content, "serving_completion.py does not import fastdeploy.envs"
    
    # Check that check_health is called with envs.FD_WORKER_ALIVE_TIMEOUT
    matches = re.findall(pattern, completion_content)
    assert len(matches) >= 2, f"Expected at least 2 check_health calls with FD_WORKER_ALIVE_TIMEOUT in serving_completion.py, found {len(matches)}"
    print(f"✓ serving_completion.py imports envs and uses FD_WORKER_ALIVE_TIMEOUT in {len(matches)} check_health calls")
    
    print("\n✓ All integration tests passed!")
    return True


if __name__ == "__main__":
    try:
        test_worker_alive_timeout_env()
        test_integration_with_serving_files()
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        sys.exit(0)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
