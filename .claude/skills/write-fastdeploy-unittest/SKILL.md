# Writing FastDeploy CI / Unit Tests
This skill covers **how to write and run tests** for FastDeploy. FastDeploy uses pytest for unit testing with automatic coverage collection. Tests are classified into **multi-GPU** (sequential) and **single-GPU** (parallel) categories for efficient CI execution.

---

## Core Rules
1. **Use pytest, or unittest** — FastDeploy uses pytest as the test framework with fixtures for common patterns
2. **Follow test classification rules** — Tests are auto-classified by location and content (see Classification section below)
3. **Choose service startup approach as needed** — `FDRunner` (in `tests/conftest.py`) is a convenience wrapper for common test patterns, not a universal requirement; use `fastdeploy.entrypoints.llm.LLM` directly if it doesn't fit
4. **Isolate logs per test** — Use `FD_LOG_DIR` environment variable (auto-set by coverage_run.sh) to isolate test logs
5. **Clean up resources** — Use context manager or `try/finally` for service teardown
6. **Maintain coverage threshold** — PR changes require 80% diff coverage
7. **Prefer appending to existing test files** — Before creating a new test file, search for existing test files that cover the same module (e.g., `tests/worker/test_gpu_model_runner.py` for `fastdeploy/worker/gpu_model_runner.py`). If found, add new test cases to the existing file rather than creating a duplicate
8. **Code style: black (line-length=119) + isort + flake8** — Do NOT manually wrap lines shorter than 119 chars; black will collapse them and pre-commit will fail. After generating code, verify with `flake8 <file> --max-line-length=119`

---

## Test Classification
FastDeploy's CI script (`scripts/coverage_run.sh`) classifies tests into two categories:

### Multi-GPU Tests (Sequential)
Tests that run sequentially (cannot parallelize):

|Rule|Pattern|Example|
|-|-|-|
|**Distributed tests**|`tests/distributed/test_*.py`|Multi-GPU communication tests|
|**E2E tests**|`tests/e2e/test_*.py`|Full serving integration tests|
|**Model loader tests**|`tests/model_loader/test_*.py`|Tests that allocate multiple GPUs|
|**Content patterns**|File contains any pattern below|Service tests, multi-GPU tests|

Content patterns that trigger multi-GPU (sequential) classification — any match causes the file to run sequentially:

```
tensor_parallel_size.*=[1234]
--tensor-parallel-size.*[1234]
"tensor_parallel_size".*[1234]
CUDA_VISIBLE_DEVICES.*0.*1
paddle.distributed.launch.*--gpus.*0.*1
FD_API_PORT
FLASK_PORT
FD_ENGINE_QUEUE_PORT
FD_METRICS_PORT
FD_CACHE_QUEUE_PORT
FD_ROUTER_PORT
FD_CONNECTOR_PORT
FD_RDMA_PORT
```

> **Important**: Port variables (`FD_API_PORT`, etc.) are the primary trigger for multi-GPU classification — any test involving port usage (offline or online inference) runs sequentially. The `tensor_parallel_size` patterns cover `[1234]` which includes `=1`, so a single-GPU service test (TP=1) that also references a port variable is still classified as multi-GPU.

### Single-GPU Tests (Parallel)
All other tests run in parallel. They are automatically split into 2 shards (one per GPU on the 2-card CI runner).

---

## Test Directory Structure
```
tests/
├── batch_invariant/       # Batch processing invariance tests
├── cache_manager/         # KV cache management tests
├── ci_use/                # Used by a separate CI task (excluded from coverage runs)
├── ci_validation/         # CI validation tests (excluded from coverage runs)
│   ├── server/            # Server functionality tests
│   └── stable_cases/      # Stable/accuracy tests
├── conftest.py            # Global pytest configuration and fixtures
├── cov_pytest.ini         # Pytest config for coverage runs
├── deterministic/         # Determinism/reproducibility tests
├── distributed/           # Distributed communication tests (NCCL, RDMA)
├── e2e/                   # End-to-end serving tests (ERNIE, Qwen, etc.)
├── engine/                # LLM engine tests
├── entrypoints/           # API entrypoint tests
├── input/                 # Input processing/tokenization tests
├── layers/                # Layer/attention tests
├── logger/                # Logging tests
├── metrics/               # Prometheus/metrics tests
├── model_executor/        # Model executor tests
├── model_loader/          # Model loading/caching tests
├── multimodal/            # Multimodal (image/audio) tests
├── operators/             # CUDA/Triton operator tests
├── output/                # Output processing/LogProbs tests
├── platforms/             # Platform-specific tests
├── plugins/               # Plugin system tests
├── pooling/               # Prefix pooling tests
├── quantization/          # Quantization tests (W4A/W8A16/FP8)
├── reasoning/             # ERNIE/PaddleOCR/Qwen reasoning tests
├── router/                # Request routing tests
├── scheduler/             # Request scheduler tests
├── spec_decode/           # Speculative decoding tests
├── trace/                 # Tracing/profiling tests
├── usage/                 # Usage examples as tests
├── utils/                 # Utility tests
├── v1/                    # v1 API tests
├── worker/                # Worker process tests
├── xpu_ci/                # XPU-specific CI tests (excluded from coverage runs)
└── metax_ci/              # MetaX GPU CI tests (excluded from coverage runs)
```
---

## CI Environment & Runner
### Runner Configuration
|Property|Value|
|-|-|
|**Runner label**|`GPU-h1z1-2Cards`|
|**Workflow**|`.github/workflows/_unit_test_coverage.yml`|
|**Timeout**|105 minutes total, 600s per test file|
|**Docker image**|`ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleqa:fastdeploy-ciuse-cuda126-paddle-dev`|
|**GPUs**|2x NVIDIA H20 (dynamic port allocation)|

### Environment Variables
The CI derives port variables from the runner name's last segment (GPU card ID):

|Variable|Formula|Example (DEVICE_PORT=0)|
|-|-|-|
|`FD_API_PORT`|`8088 + DEVICE_PORT * 100`|8088|
|`FD_ENGINE_QUEUE_PORT`|`8058 + DEVICE_PORT * 100`|8058|
|`FD_METRICS_PORT`|`8078 + DEVICE_PORT * 100`|8078|
|`FD_CACHE_QUEUE_PORT`|`8098 + DEVICE_PORT * 100`|8098|
|`FD_ROUTER_PORT`|`8048 + DEVICE_PORT * 100`|8048|
|`FD_CONNECTOR_PORT`|`8038 + DEVICE_PORT * 100`|8038|
|`FD_RDMA_PORT`|`8028 + DEVICE_PORT * 100`|8028|
|`FLASK_PORT`|`8068 + DEVICE_PORT * 100`|8068|
|`MODEL_PATH`|Set to `/ModelData` (read-only mount)|-|
|`FD_LOG_DIR`|Auto-set per test to isolate logs|`unittest_logs/<test_dir>/<test_name>/log`|

> **Local defaults**: When env vars are unset locally, `serving_utils.py` uses different defaults (e.g. `FD_API_PORT=8188`). Always set these variables explicitly when running service tests locally.

> **Port conflict awareness**: The CI machine hosts multiple runners simultaneously (one per GPU card), each with port offsets derived from `DEVICE_PORT`. Any test referencing port variables (`FD_API_PORT`, etc.) is automatically classified as **multi-GPU (sequential)** — even single-GPU tests — which prevents same-runner conflicts. However, **cross-runner conflicts** can still occur if two runners on the same machine execute tests that bind to the same port. Therefore:
> - Always read ports from environment variables — never hardcode port numbers
> - If a test needs an auxiliary port (e.g., a mock HTTP server), use `port=0` to let the OS assign an ephemeral port, avoiding collisions between concurrent runners
> - Tests that reference any `FD_*_PORT` variable are guaranteed sequential within one runner, but may run in parallel across runners on the same host

---

## Writing Strategy by Test Type

FastDeploy tests fall into four distinct patterns based on what they exercise. Choose the pattern that matches the code under test.

---

### Pattern 1 — Pure Logic / Data Structure Tests
**Where**: `engine/`, `scheduler/`, `router/`, `output/`, `reasoning/`, `logger/`, `trace/`, `usage/`, `platforms/`, `quantization/`, `model_executor/` (config classes, utils, tokenizers, non-GPU logic)

These tests validate algorithms, config parsing, data classes, or error messages with no GPU or service dependency.

- Use `unittest.TestCase` (most existing tests) or plain pytest (no base class needed)
- Use `unittest.mock.patch` / `MagicMock` to stub heavy dependencies (zmq, redis, subprocess, requests)
- Use `patch.dict("os.environ", ...)` for environment-variable-driven branches
- Assert error message content with `assertIn(str(ctx.exception))`
- **Config class gotcha**: When testing classes that inherit from base configs (e.g., `PretrainedConfig`), always assert against the *actual runtime value* after instantiation, not the default parameter in the function signature. Parent `__init__` calls may override child-set attributes (e.g., a second `super().__init__()` can reset `pad_token_id` to `None`).

```python
import unittest
from unittest.mock import MagicMock, patch
from fastdeploy.module import TargetClass


class TestTargetClass(unittest.TestCase):
    def test_valid_input(self):
        obj = TargetClass(param="value")
        self.assertEqual(obj.result(), expected)

    def test_invalid_input_raises(self):
        with self.assertRaises(ValueError) as ctx:
            TargetClass(param="bad")
        self.assertIn("expected message fragment", str(ctx.exception))

    @patch("fastdeploy.module.external_dep")
    def test_with_mock(self, mock_dep):
        mock_dep.return_value = MagicMock(data="test")
        result = TargetClass().method()
        self.assertIsNotNone(result)
```

---

### Pattern 2 — GPU Kernel / Numerical Accuracy Tests
**Where**: `layers/`, `operators/`, `batch_invariant/`, `spec_decode/`, `worker/`, `multimodal/`, `model_executor/` (GPU ops, kernels, numerical computations only)

These tests run real GPU kernels and compare against a reference (numpy/paddle naive) implementation.

- Use `paddle.set_device("gpu")` in `setUp` or at module level
- Mark with `@pytest.mark.gpu` so CI skips them on non-GPU machines
- Use `np.testing.assert_allclose(rtol=..., atol=...)` for float tolerance; `np.testing.assert_array_equal` for exact integer results
- Cover multiple shapes/dtypes with `@pytest.mark.parametrize` or nested loops

```python
import numpy as np
import paddle
import pytest
from fastdeploy.model_executor.ops import my_kernel


@pytest.mark.gpu
class TestMyKernel:
    def setup_method(self):
        paddle.set_device("gpu")

    @pytest.mark.parametrize("shape", [(1024, 512), (4096, 256)])
    def test_matches_reference(self, shape):
        x = paddle.randn(shape, dtype="float16")
        ref = naive_numpy_impl(x.numpy())
        out = my_kernel(x).cast("float32").numpy()
        np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-3)
```

---

### Pattern 3 — Offline Inference Tests (Python API, real model)
**Where**: `entrypoints/`, `deterministic/`, `pooling/`, `model_loader/`

These tests load a real model via `LLM(...)` or `ModelRegistry` and verify generation outputs or weight shapes. They require `MODEL_PATH` to be set.

- Use `setUpClass` (unittest) or `@pytest.fixture(scope="module")` (pytest) to load the model once per test file
- Guard initialization with `unittest.SkipTest` or `pytest.skip` when `MODEL_PATH` is absent
- Do **not** import `fastdeploy` at module level — import inside the fixture or `setUpClass` to avoid CUDA initialization before fork
- Use `FD_ENGINE_QUEUE_PORT` / `FD_CACHE_QUEUE_PORT` from environment

```python
import os
import unittest
from e2e.utils.serving_utils import FD_ENGINE_QUEUE_PORT, FD_CACHE_QUEUE_PORT


class TestOfflineInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model = os.path.join(os.getenv("MODEL_PATH", ""), "your-model")
        try:
            from fastdeploy.entrypoints.llm import LLM
            from fastdeploy.engine.sampling_params import SamplingParams
            cls.LLM = LLM
            cls.SamplingParams = SamplingParams
            cls.llm = LLM(
                model=model,
                engine_worker_queue_port=int(FD_ENGINE_QUEUE_PORT),
                cache_queue_port=int(FD_CACHE_QUEUE_PORT),
            )
        except Exception as e:
            raise unittest.SkipTest(f"Model init failed: {e}")

    def test_basic_generation(self):
        outputs = self.llm.generate(["Hello"], self.SamplingParams(max_tokens=32))
        self.assertEqual(len(outputs), 1)
```

---

### Pattern 4 — Online Serving / E2E Tests (subprocess + HTTP)
**Where**: `e2e/`, `distributed/`

These tests start the FastDeploy API server (or a distributed job) as a subprocess and interact over HTTP. See existing files under `tests/e2e/` for full examples.

- Use `@pytest.fixture(scope="session", autouse=True)` to start/stop the server once per file
- Launch with `subprocess.Popen(..., start_new_session=True)` and redirect output to `server.log`
- Poll `is_port_open("127.0.0.1", FD_API_PORT)` up to 10 minutes before declaring startup failure
- Tear down with `os.killpg(process.pid, signal.SIGTERM)` + `clean_ports()`
- Use `requests.post` for HTTP validation; assert `status_code` and response fields
- For distributed tests (`tests/distributed/`): launch via `paddle.distributed.launch` subprocess; assert `returncode == 0`

```python
import os, signal, subprocess, sys, time
import pytest, requests
from e2e.utils.serving_utils import (
    FD_API_PORT, FD_CACHE_QUEUE_PORT, FD_ENGINE_QUEUE_PORT, FD_METRICS_PORT,
    clean_ports, is_port_open,
)


@pytest.fixture(scope="session", autouse=True)
def server():
    clean_ports()
    model_path = os.path.join(os.getenv("MODEL_PATH", "."), "your-model")
    cmd = [
        sys.executable, "-m", "fastdeploy.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(FD_API_PORT),
        "--engine-worker-queue-port", str(FD_ENGINE_QUEUE_PORT),
        "--metrics-port", str(FD_METRICS_PORT),
        "--cache-queue-port", str(FD_CACHE_QUEUE_PORT),
        "--tensor-parallel-size", "1",
        "--max-model-len", "4096",
        "--max-num-seqs", "32",
    ]
    with open("server.log", "w") as log:
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    for _ in range(10 * 60):
        if is_port_open("127.0.0.1", FD_API_PORT):
            break
        time.sleep(1)
    else:
        os.killpg(process.pid, signal.SIGTERM)
        raise RuntimeError(f"Server did not start on port {FD_API_PORT}")
    yield
    os.killpg(process.pid, signal.SIGTERM)
    clean_ports()


@pytest.fixture(scope="session")
def api_url():
    return f"http://0.0.0.0:{FD_API_PORT}/v1/chat/completions"


def test_basic_generation(api_url):
    resp = requests.post(api_url,
                         json={"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 32},
                         headers={"Content-Type": "application/json"})
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"]
```

---

## Port Management
Tests that reference any port variable are automatically classified as multi-GPU (sequential).

### Port Variables
Port variables are read from environment. CI injects them automatically per GPU card; `serving_utils.py` provides fallback defaults when running locally:

```python
from e2e.utils.serving_utils import FD_API_PORT, FD_ENGINE_QUEUE_PORT, FD_METRICS_PORT, FD_CACHE_QUEUE_PORT
```

### clean_ports()
`clean_ports()` (from `e2e.utils.serving_utils`) kills processes on the above ports and cleans unix sockets. Call it manually only for tests that don't launch a server via subprocess.

```python
from e2e.utils.serving_utils import clean_ports

clean_ports()
```

---

## Coverage Requirements
### PR Coverage
- **Threshold**: 80% diff coverage
- **Tool**: `diff-cover` with `--fail-under=80`
- **Output**: `diff_coverage.json` uploaded to BOS

### Coverage Configuration
- **Config**: `scripts/.coveragerc`
- **Data**: `coveragedata/.coverage`
- **Report**: `python_coverage_all.xml`

### Running Coverage Locally
```bash
# Install requirements
pip install -r scripts/unittest_requirement.txt

# Set coverage config
export COVERAGE_FILE=coveragedata/.coverage
export COVERAGE_RCFILE=scripts/.coveragerc

# Run single test with coverage
python -m coverage run -m pytest tests/engine/test_engine.py -vv

# Run with --source to limit coverage scope (must be a directory path, NOT a dotted module name)
python -m coverage run --source=fastdeploy/model_executor/models/paddleocr_vl -m pytest tests/model_executor/test_paddleocr_vl_config.py -vv

# Generate report (with per-line missing info)
coverage combine coveragedata/ || echo "No data to combine"
coverage report -m
```

> **Note**: The `--source` parameter accepts directory paths (e.g., `fastdeploy/engine`) or top-level package names (e.g., `fastdeploy`). It does NOT accept dotted module paths like `fastdeploy.engine.module` or file paths like `fastdeploy/engine/module.py` — these will silently produce no coverage data.
---

## pytest Configuration
### Markers
Tests can be marked with `@pytest.mark.gpu`:

```python
@pytest.mark.gpu
def test_gpu_feature(self):
    """Test that requires GPU."""
    ...
```
The `conftest.py` hook automatically skips GPU-marked tests on non-GPU platforms (detected via `/dev/nvidia[0-9]*`).

### pytest.ini (cov_pytest.ini)
```
[pytest]
addopts =
    --ignore=tests/ci_use
    --ignore=tests/ci_validation
    --ignore=tests/operators/test_fused_moe.py
    --ignore=tests/operators/test_w4afp8_gemm.py
    --ignore=tests/model_loader/test_w4a8_model.py
    --ignore=tests/xpu_ci
    --ignore=tests/metax_ci
    --ignore=tests/e2e/4cards_cases
    --ignore=tests/e2e/golang_router
    --ignore=tests/v1/test_schedule_output.py
    --ignore=tests/graph_optimization/test_cuda_graph_dynamic_subgraph.py
```
---

## Test Execution Flow
### CI Execution (coverage_run.sh)
1. **Collect tests**: `pytest --collect-only` to find all `test_*.py` files
2. **Classify tests**: Separate into `multi_gpu` and `single_gpu` based on rules
3. **Run multi-GPU tests**: Sequentially on GPU 0 and GPU 1
4. **Run single-GPU tests**: Split into 2 shards, run in parallel (1 per GPU)
5. **Combine coverage**: Merge coverage data from all shards
6. **Generate reports**: XML coverage + diff coverage for PRs
7. **Upload results**: Upload to BOS storage
8. **Check threshold**: Fail if diff coverage < 80% (exit code 9)

### Local Execution
```bash
# Run all tests
pytest tests/ -vv

# Run specific test directory
pytest tests/engine/ -vv

# Run with coverage
python -m coverage run -m pytest tests/engine/test_engine.py -vv
python -m coverage report

# Run with timeout (same as CI)
timeout 600 python -m coverage run -m pytest tests/engine/test_engine.py -vv
```

---

## Error Handling & Logging
### Isolated Log Directory
The CI automatically sets `FD_LOG_DIR` per test:

```python
import os

log_dir = os.environ.get("FD_LOG_DIR", "log")
```
### Error Logging
Failed tests automatically capture error logs via `pytest_runtest_makereport` hook in `conftest.py`. Logs are saved to `FD_LOG_DIR/pytest_<case_name>_error.log`.

### Retry on OOM
The CI script automatically retries tests killed by OOM (exit code 137) up to 3 times.

---

## Test File Naming Convention

When generating a test file name from the source file path, follow these rules to ensure the file name is **self-identifying in CI logs and test reports without needing the full path**.

### Core Principle

Test file names must carry enough context to identify the module they test when viewed in isolation (e.g., in `pytest` output, coverage reports, or `grep` results). When in doubt, **add the parent module prefix** — a slightly longer name is always better than an ambiguous one.

### Rules (applied in order)

1. **Generic / short leaf names** (e.g., `audio.py`, `video.py`, `tbo.py`, `storage.py`, `config.py`)
   → **Always** prefix with the parent module (or test directory name): `test_<parent>_<leaf>.py`
   - `fastdeploy/multimodal/audio.py` → `tests/multimodal/test_multimodal_audio.py`
   - `fastdeploy/multimodal/video.py` → `tests/multimodal/test_multimodal_video.py`
   - `fastdeploy/worker/tbo.py` → `tests/worker/test_worker_tbo.py`
   - `fastdeploy/scheduler/storage.py` → `tests/scheduler/test_scheduler_storage.py`
   - `fastdeploy/input/image_processors/qwen3_processor.py` → `tests/input/test_image_qwen3_processor.py`

   > **How to judge "generic"**: If the leaf name could plausibly exist in multiple packages (e.g., `utils.py`, `config.py`, `base.py`, single-word names), it is generic.

2. **Leaf name == parent directory name** (e.g., `file_store/file_store.py`, `mooncake_store/mooncake_store.py`)
   → Use `test_<leaf>.py` directly. The repetition already provides context.
   - `transfer_factory/file_store/file_store.py` → `test_file_store.py`
   - `transfer_factory/mooncake_store/mooncake_store.py` → `test_mooncake_store.py`

3. **Leaf name is already specific and self-descriptive** (multi-word compound names that are unique across the project)
   → Use `test_<leaf>.py` directly. No prefix needed.
   - `transfer_factory/ipc_cache_transfer.py` → `test_ipc_cache_transfer.py`
   - `layers/attention/block_multihead_attn_backend.py` → `test_block_multihead_attn_backend.py`
   - `layers/attention/dsa_attention_backend.py` → `test_dsa_attention_backend.py`

   > **How to judge "self-descriptive"**: The name contains 3+ words or includes the module domain (e.g., `ipc_cache_transfer` clearly belongs to cache transfer).

4. **Leaf name != parent directory and is not fully self-descriptive**
   → Prefix with the parent directory name: `test_<parent>_<leaf>.py`
   - `mooncake_store/attention_store.py` → `test_mooncake_attention_store.py`
   - `input/utils/render_timestamp.py` → `test_input_utils_render_timestamp.py`

5. **Collision check**: Before finalizing the name, verify no existing file in the target `tests/` subdirectory has the same name. If a collision is found, add more path components as prefix until unique.

### Consistency Reference

Check existing test files in the same `tests/` subdirectory and **follow the dominant pattern**:
- `tests/worker/` uses `test_worker_*.py` prefix (e.g., `test_worker_process.py`, `test_worker_eplb.py`)
- `tests/multimodal/` uses `test_multimodal_*.py` prefix (e.g., `test_multimodal_utils.py`, `test_multimodal_audio.py`)
- `tests/model_executor/` does NOT prefix (names are already specific: `test_gpt_oss.py`, `test_entropy_utils.py`)

When a directory has an established prefix pattern, always follow it for new files.

### Placement

- The test file goes into the `tests/` subdirectory matching the **top-level source package** (e.g., source in `fastdeploy/cache_manager/...` → test in `tests/cache_manager/`).
- Only create deeper subdirectories if one already exists in `tests/` (e.g., `tests/cache_manager/v1/`).

---

## GPU Execution Environment & Coverage Expectations

Tests are always executed on GPU machines (2x NVIDIA H20). When writing unit tests, keep in mind:

1. **Hardware-dependent code paths may be unreachable** — Some branches depend on specific hardware conditions that cannot be easily simulated in tests:
   - NUMA topology detection (e.g., `nvidia-smi topo`, `/sys/class/nvidia-gpu/`, `/sys/bus/pci/devices/`)
   - Multi-node / RDMA communication paths
   - Specific GPU architecture features (e.g., SM version checks)
   - Device memory capacity checks that vary per hardware

2. **Coverage gaps from hardware-specific paths are acceptable** — Do not force coverage of code that genuinely requires hardware conditions you cannot mock cleanly. The 80% diff coverage threshold accounts for this; it is fine to leave hardware-gated branches uncovered as long as the logical/mockable portions are well-tested.

3. **Mock what you can, skip what you can't** — For functions that mix logic with hardware access (e.g., `_get_numa_node_for_gpu`), mock the system calls (`subprocess.run`, `os.path.exists`, `glob.glob`, file reads) to test the parsing logic. Don't try to cover paths that are purely pass-through to hardware APIs with no testable logic.

4. **Attribute access patterns** — The `CacheTransferManager` and similar objects may not have all methods (like `start()`/`stop()`) as static attributes. When mocking such methods, use `patch.object(..., create=True)` or mock at the `CacheController` level instead.

---

## Codestyle Check

After writing or modifying test files, run pre-commit to ensure the code passes CI style checks:

```bash
# Install pre-commit if not available (ref: tools/codestyle/pre_commit.sh)
pip install pre-commit==4.2.0 clang-format==13.0.0

# Check code style on the new/modified test files
pre-commit run --files tests/path/to/test_xxx.py
```

Fix any reported issues before declaring the test complete.

---

## Checklist
Before submitting a test:

- [ ] File name follows the naming convention above (leaf-name based with parent prefix when needed for disambiguation)
- [ ] Test class follows `Test<Module>` pattern
- [ ] Test methods follow `test_<scenario>` pattern
- [ ] Uses pytest (or unittest)
- [ ] Located in appropriate `tests/` subdirectory
- [ ] Service tests use a session-scoped fixture to start/stop the server subprocess; see `tests/e2e/` for reference patterns
- [ ] Any test referencing port variables or TP config will run sequentially in CI — this includes `tensor_parallel_size=1`
- [ ] Cleans up resources: server subprocess terminated with `os.killpg` + `clean_ports()` in fixture teardown
- [ ] Has `if __name__ == "__main__": pytest.main(...)` or `unittest.main()` for local execution
- [ ] Does not exceed 600s timeout per test file
- [ ] Maintains or improves coverage (80% diff threshold for PRs)
- [ ] Hardware-dependent paths that cannot be mocked are acceptable coverage gaps
- [ ] Passes `pre-commit run --files` code style check (black, isort, flake8)
