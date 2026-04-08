[简体中文](../zh/usage/fastdeploy_unit_test_guide.md)

# FastDeploy Unit Test Specification
1. Test Naming Conventions
   - Test files must start with test_.
   - Test methods are recommended to follow the format test_func_behavior or test_func for clarity and readability.

2. Directory Structure
   - All unit tests should be placed under the test/ directory and organized into subdirectories following the FastDeploy module structure for easier maintenance and navigation.Reference: [vllm](https://github.com/vllm-project/vllm/tree/main/tests)

3. Coverage Scope,Each key module/class/function must be covered, including:

   - Normal Path: Standard input and main workflow.
   - Boundary Input: Empty values, extremely large/small values, and boundary conditions.
   - Abnormal Input: Invalid parameters, incorrect formats, etc. Ensure that exception handling matches expectations (e.g., check exception content).

4. Case Writing & Execution

   - Test cases should support one-click execution in both CI and local environments via pytest or unittest.
   - Use explicit assert statements to validate module behavior or return values; avoid relying solely on printed outputs.
   - Each test case should maintain strong atomicity, focusing on a single behavior and avoiding mixing multiple functions in one test.
   - Tests must be independent of execution order and global state, ensuring complete decoupling between cases.
   - Custom operators must have C++-level unit tests, or forward/backward tests based on Paddle single-layer networks. Refer to Xiaoguang’s documentation on custom operator development toolkit and non-official operator development specification (trial).

5. WebServer-related Tests

   - Avoid starting a WebServer directly in unit tests unless absolutely necessary for end-to-end validation.
   - Prefer using mock for network requests and module interactions to achieve more stable and controllable atomic-level testing.
   - If HTTP requests must be tested:
   - QA will provide a port injection specification (ports are written into environment variables, which test cases can directly load). This spec and scripts are expected to be ready by next Wednesday.
   - Test cases should read ports from environment variables instead of hardcoding them, ensuring reusability in CI multi-instance concurrency and easier local debugging.

<img width="500" height="240" alt="Image" src="https://github.com/user-attachments/assets/2b447cdf-b709-4f30-98f6-e680aef02c6f" />
