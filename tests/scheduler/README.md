# Scheduler Tests

This directory contains unit tests for the FastDeploy scheduler components.

## Local Scheduler Tests

The `test_local_scheduler.py` file contains comprehensive unit tests for the `LocalScheduler` class located in `fastdeploy/scheduler/local_scheduler.py`.

### Test Coverage

The tests cover the following functionality:

#### Core Operations
- **Initialization**: Testing scheduler initialization with various configurations
- **Reset**: Testing scheduler reset functionality
- **Request Management**: Adding, retrieving, and checking requests
- **Response Management**: Adding and retrieving responses
- **Block Calculation**: Testing resource allocation calculations

#### Resource Management
- **Max Size Limits**: Testing behavior when reaching maximum request limits
- **Unlimited Size**: Testing behavior with unlimited size (max_size=0)
- **TTL (Time-to-Live)**: Testing request expiration and cleanup
- **Chunked Prefill**: Testing chunked prefill processing for long requests

#### Error Handling
- **Duplicate Requests**: Testing handling of duplicate request IDs
- **Expired Responses**: Testing behavior with responses for non-existent requests
- **Resource Constraints**: Testing behavior with insufficient resources
- **Edge Cases**: Testing empty inputs and boundary conditions

#### Thread Safety
- **Concurrent Operations**: Testing basic thread safety of scheduler operations

#### Logging
- **Request Logging**: Testing that appropriate log messages are generated
- **Response Logging**: Testing finished response logging
- **Reset Logging**: Testing reset operation logging

### Running Tests

#### Standalone Mode (for local development)
```bash
# Run in standalone mode (automatically falls back if direct import fails)
python tests/scheduler/test_local_scheduler.py

# Or explicitly set standalone mode
FD_TEST_MODE=standalone python tests/scheduler/test_local_scheduler.py
```

#### Normal Mode (for CI/CD with full installation)
```bash
# Run in normal mode (requires fastdeploy to be properly installed)
FD_TEST_MODE=normal python tests/scheduler/test_local_scheduler.py

# Or run with pytest (if configured)
pytest tests/scheduler/test_local_scheduler.py -v
```

### Test Architecture

The tests use a dual-mode architecture:

1. **Standalone Mode**: Uses mock objects and dynamic imports to test the scheduler without requiring a full fastdeploy installation
2. **Normal Mode**: Uses direct imports when fastdeploy is properly installed (for CI/CD environments)

This allows the tests to run in any environment while maintaining full functionality.

### Mock Objects

In standalone mode, the following mock objects are used:
- `MockRequest`: Simulates request objects with ID and token data
- `MockRequestOutput`: Simulates response objects with request ID and completion status
- `MockScheduledRequest`: Simulates scheduled request wrapper objects
- `MockScheduledResponse`: Simulates scheduled response wrapper objects
- `MockLogger`: Captures log messages for verification

### Test Files

- `test_local_scheduler.py`: Unit tests for LocalScheduler class
- `README.md`: This documentation file

### Adding New Tests

When adding new tests:

1. Follow the existing naming convention (`test_<functionality>`)
2. Use descriptive test method names
3. Test both success and failure cases
4. Include edge cases and boundary conditions
5. Add assertions for logging when appropriate (only in standalone mode)
6. Follow the existing mock patterns for new dependencies

### Test Statistics

The current test suite includes:
- **34 test methods**
- **Complete coverage** of all public LocalScheduler methods
- **Edge case testing** for all major scenarios
- **Thread safety testing** for concurrent operations
- **Logging verification** for all critical operations

All tests pass successfully, providing confidence in the reliability and correctness of the LocalScheduler implementation.