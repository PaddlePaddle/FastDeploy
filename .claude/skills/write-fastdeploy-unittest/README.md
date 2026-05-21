# write-fastdeploy-unittest

English | [简体中文](README_CN.md)

A skill that guides AI agents to write CI-compliant unit tests for the FastDeploy.

## Features

- Automatically selects the appropriate test pattern based on the code under test (pure logic / GPU kernel / offline inference / E2E serving)
- Follows FastDeploy CI classification rules (multi-GPU sequential vs single-GPU parallel)
- Meets the 80% diff coverage PR threshold
- Correctly uses port variables, log isolation, and resource cleanup per CI conventions

## Usage

### Basic — specify a source file

```
Use the write-fastdeploy-unittest skill to add unit tests for fastdeploy/cache_manager/transfer_factory/file_store/file_store.py
```

### From coverage report — paste the line directly

```
Use the write-fastdeploy-unittest skill to add unit tests for:

fastdeploy/model_executor/model_loader/default_loader.py    48  32  14  0  26%  37-38, 42, 46-52, 56-66, 69-97
```

The coverage report format is: `file_path  Stmts  Miss  Branch  BrMiss  Cover%  Missing_lines`. The agent will focus on the uncovered lines and write tests specifically targeting those branches.

### From incremental coverage JSON — PR diff coverage check data

```
Use the write-fastdeploy-unittest skill to add unit tests for:

"fastdeploy/worker/gpu_model_runner.py": {"percent_covered": 0.0, "violation_lines": [1398], "covered_lines": [], "violations": [[1398, null]]}
```

JSON field descriptions:
- `percent_covered`: Incremental line coverage percentage
- `violation_lines`: List of uncovered line numbers (target lines that need tests)
- `covered_lines`: List of already-covered line numbers
- `violations`: Violation details, format `[[line_number, branch_info]]`

The agent will focus on lines in `violation_lines` and write tests specifically targeting those branches.

### Workflow

The agent will automatically:
1. Read the target source file and analyze uncovered lines
2. **Check if a test file already exists** (prefer appending test cases to existing files over creating new ones)
3. Select the appropriate test pattern (Pattern 1-4)
4. Append to existing test file, or generate a new test file in the corresponding `tests/` subdirectory
5. Run tests and verify coverage

## Test Pattern Quick Reference

| Pattern | Use Case | Dependencies |
|---------|----------|--------------|
| 1 — Pure Logic | config, utils, scheduler, router, etc. | No GPU; mock external deps |
| 2 — GPU Kernel | ops, layers, numerical computation | Requires GPU; `@pytest.mark.gpu` |
| 3 — Offline Inference | LLM API, model loading | Requires MODEL_PATH |
| 4 — E2E Serving | End-to-end HTTP serving | subprocess + ports |

## Key Conventions

- Test file naming: `test_<module>.py`
- Test class naming: `Test<Module>`
- Coverage verification: `python -m coverage run --source=<directory> -m pytest <test_file> && coverage report -m`
- The `--source` parameter accepts directory paths (e.g., `fastdeploy/engine`) or top-level package names (e.g., `fastdeploy`). It does NOT accept dotted module paths like `fastdeploy.engine.module` or `.py` file paths.

## Related Files

- [SKILL.md](SKILL.md) — Full skill instruction document
