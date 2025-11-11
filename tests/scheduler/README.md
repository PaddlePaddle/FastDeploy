# DP Scheduler Unit Tests

This directory contains unit tests for the `fastdeploy/scheduler/dp_scheduler.py` module.

## Test Files

### `test_dp_scheduler_simple.py` (Recommended)
Simplified unit tests that don't require complex imports. These tests provide comprehensive coverage of the DP scheduler functionality including:

- **DPLocalScheduler functionality:**
  - Initialization with different configurations
  - Request lifecycle management
  - Response handling and routing
  - Resource-based request scheduling
  - Recycling of expired/completed requests
  - Splitwise role handling (prefill vs decode)

- **DPScheduler functionality:**
  - Multi-threaded request/response processing
  - Integration with multiprocessing queues
  - Request validation (dp_rank requirement)
  - Delegation to internal scheduler

- **Edge cases and error handling:**
  - Resource constraint scenarios
  - Timeout behavior
  - Thread-safe concurrent operations
  - Malformed request handling

### `test_dp_scheduler.py`
Full-featured unit tests that attempt to import the actual FastDeploy modules. These tests provide more detailed testing but require a proper FastDeploy installation.

## Running Tests

### Simple Tests (Works without installation)
```bash
python tests/scheduler/test_dp_scheduler_simple.py
```

### Full Tests (Requires FastDeploy installation)
```bash
# If FastDeploy is properly installed:
python tests/scheduler/test_dp_scheduler.py

# If using standalone mode (no installation):
FD_TEST_MODE=standalone python tests/scheduler/test_dp_scheduler.py
```

### Using pytest (if available)
```bash
pytest tests/scheduler/ -v
```

## Test Coverage

The unit tests cover the following key aspects of the DP Scheduler:

### 1. Request Management
- Adding requests to the scheduler queue
- Request ID tracking and management
- Request expiration and cleanup (TTL)
- Request prioritization based on availability

### 2. Response Handling
- Processing finished request responses
- Response routing to appropriate queues
- Response aggregation and batching
- Logging of completed requests

### 3. Resource Management
- Block allocation and calculation
- Token limit enforcement
- Batch size optimization
- Memory usage tracking

### 4. Multi-threading Support
- Concurrent request processing
- Thread-safe operations with mutexes
- Background thread management
- Queue-based communication

### 5. Splitwise Role Support
- Prefill role behavior (default)
- Decode role behavior
- Role-specific request recycling
- Resource allocation based on role

### 6. Error Handling
- Invalid request detection
- Missing attribute validation
- Resource constraint handling
- Timeout management

## Architecture Testing

The tests validate the following architectural patterns:

### DPLocalScheduler
- Extends `LocalScheduler` with DP-specific functionality
- Manages request lifecycle with TTL support
- Handles response aggregation and logging
- Supports both prefill and decode roles

### DPScheduler
- Wraps `DPLocalScheduler` with threading support
- Manages inter-process communication via queues
- Coordinates request distribution across multiple workers
- Provides clean interface for DP operations

## Test Methodologies

### Mocking Strategy
- Uses `unittest.mock` for dependency injection
- Simulates complex object interactions
- Avoids heavy dependencies on external modules

### Concurrency Testing
- Tests thread-safe operations with multiple threads
- Validates mutex and condition variable usage
- Ensures proper synchronization

### Edge Case Coverage
- Tests with boundary conditions (empty queues, max limits)
- Validates error paths and exception handling
- Tests timeout and resource exhaustion scenarios

## Development Guidelines

When adding new tests:

1. **Follow the existing pattern**: Use descriptive test method names
2. **Use mocking**: Avoid heavy dependencies where possible
3. **Test both success and failure paths**: Ensure comprehensive coverage
4. **Include edge cases**: Test boundary conditions and error scenarios
5. **Document complex scenarios**: Add comments for non-obvious test logic
6. **Use the simple test file**: Prefer `test_dp_scheduler_simple.py` for new tests

## Integration Notes

These tests are designed to work with:
- Python 3.7+
- Standard library (unittest, threading, multiprocessing)
- No external dependencies required for simple tests

The tests follow the project's testing conventions and are compatible with the CI/CD pipeline.
