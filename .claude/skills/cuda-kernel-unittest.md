# Skill: CUDA Kernel Unit Test

Write unit tests for PaddlePaddle CUDA custom ops following a modular 4-layer architecture.

## Trigger

When the user asks to write/create/add unit tests for a CUDA kernel (`.cu` file in `custom_ops/`).

## Steps

1. **Read the CUDA kernel source** to understand: input/output tensors, dtypes, shapes, which tensors are CPU vs GPU, scalar attrs, in-place semantics.
2. **Write the test file** in `tests/operators/test_<kernel_name>.py` following the structure below.

## Test File Structure

```python
import unittest
from typing import Any, Dict
import numpy as np
import paddle

# --- Import ops (bypass fastdeploy.__init__) ---
try:
    import sys, os
    _fd_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _fd_root not in sys.path:
        sys.path.insert(0, _fd_root)
    from fastdeploy.import_ops import import_custom_ops
    _package = "fastdeploy.model_executor.ops.gpu"
    import_custom_ops(_package, ".fastdeploy_ops", globals())
except ImportError as e:
    print(f"Import error: {e}")
    raise

CUDA_PLACE = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() else paddle.CPUPlace()
CPU_PLACE = paddle.CPUPlace()


# ============================================================
# Layer 1: Helpers — tensor creation / kernel invocation / output extraction
# ============================================================

def to_paddle_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Convert numpy dict → paddle tensors. CPU tensors must be explicitly handled."""
    paddle_inputs = {}
    for k, v in inputs.items():
        if isinstance(v, (int, bool, float, str)):
            paddle_inputs[k] = v
        elif k in ("<CPU_TENSOR_NAMES>",):  # <-- tensors the kernel expects on CPU
            paddle_inputs[k] = paddle.to_tensor(v, place=CPU_PLACE)
        elif v is not None:
            paddle_inputs[k] = paddle.to_tensor(v, place=CUDA_PLACE)
        else:
            paddle_inputs[k] = None
    return paddle_inputs

def run_kernel(paddle_inputs, inputs):
    """Call the CUDA kernel with paddle tensors + scalar attrs."""
    kernel_name(
        paddle_inputs["tensor_a"],
        # ... all tensor args ...
        inputs["scalar_attr"],  # scalar attrs from raw dict
    )

def get_outputs(paddle_inputs) -> Dict[str, np.ndarray]:
    """Extract ALL in-place-modified tensors back to numpy."""
    keys = ["tensor_a", "tensor_b", ...]
    return {k: paddle_inputs[k].numpy() for k in keys}


# ============================================================
# Layer 2: Input generation
# ============================================================

def gen_<kernel>_inputs(real_bsz=8, ..., seed=42) -> Dict[str, Any]:
    """Generate randomized test inputs. Returns dict with both numpy arrays and scalar configs."""
    rng = np.random.default_rng(seed)
    # ... generate all numpy arrays with correct dtypes/shapes ...
    return { "tensor_a": ..., "scalar_attr": ..., "real_bsz": real_bsz, ... }


# ============================================================
# Layer 3: Reference implementation (pure Python/NumPy)
# ============================================================

def reference_<kernel>(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Python reference — must match CUDA kernel logic exactly."""
    # Deep-copy all mutable arrays
    tensor_a = inputs["tensor_a"].copy()
    # ... replicate kernel logic ...
    return {"tensor_a": tensor_a, ...}


# ============================================================
# Layer 4a: TEST_CONFIGS — all pure-parameter test scenarios
# ============================================================

TEST_CONFIGS = [
    # Each config is a dict of gen_<kernel>_inputs kwargs + a "name" key.
    # Pure parameter variations go here — do NOT create separate test methods for them.
    #
    # --- basic coverage ---
    {"name": "small_batch",   "real_bsz": 1,  "seed": 42, ...},
    {"name": "large_batch",   "real_bsz": 64, "seed": 42, ...},
    # --- mode / strategy variants ---
    {"name": "mode_a",        "real_bsz": 8,  "mode": "a", "seed": 42, ...},
    {"name": "mode_b",        "real_bsz": 8,  "mode": "b", "seed": 42, ...},
    # --- flags ---
    {"name": "reject_all",    "real_bsz": 8,  "reject_all": True, "seed": 42, ...},
    {"name": "accept_all",    "real_bsz": 8,  "accept_all": True, "seed": 42, ...},
    # --- edge cases ---
    {"name": "min_batch",     "real_bsz": 1,  "max_tokens": 1, "seed": 42, ...},
]


# ============================================================
# Layer 4b: Test suite
# ============================================================

class Test<KernelName>(unittest.TestCase):

    # ------ shared helpers ------

    def _run_and_get(self, inputs):
        paddle_inputs = to_paddle_inputs(inputs)
        run_kernel(paddle_inputs, inputs)
        return get_outputs(paddle_inputs)

    def _check_all_outputs(self, inputs, outputs):
        """Compare ALL output tensors against reference + sanity checks."""
        ref = reference_<kernel>(inputs)
        all_keys = ["tensor_a", "tensor_b", ...]
        for key in all_keys:
            np.testing.assert_array_equal(
                outputs[key], ref[key], err_msg=f"{key} mismatch"
            )
        # Add domain-specific sanity checks here

    def _run_full_test(self, config):
        inputs = gen_<kernel>_inputs(**config)
        outputs = self._run_and_get(inputs)
        self._check_all_outputs(inputs, outputs)
        return outputs

    # ------ test cases ------

    def test_configs(self):
        """Run all TEST_CONFIGS via subTest (one subTest per config)."""
        for cfg in TEST_CONFIGS:
            with self.subTest(name=cfg["name"]):
                test_cfg = {k: v for k, v in cfg.items() if k != "name"}
                self._run_full_test(test_cfg)

    # Only keep separate test methods for scenarios that need tensor overrides:
    def test_special_scenario(self):
        """Scenarios that need manual tensor setup beyond gen_inputs params."""
        inputs = gen_<kernel>_inputs(real_bsz=2, seed=42)
        inputs["some_tensor"][0, 2] = special_value  # override specific tensor
        outputs = self._run_and_get(inputs)
        self._check_all_outputs(inputs, outputs)

if __name__ == "__main__":
    unittest.main()
```

## Key Rules

1. **CPU vs GPU tensors**: Read the CUDA kernel `.cu` file carefully. If a tensor is `copy_to(place, false)` inside the host function, it's a CPU tensor input — must use `CPU_PLACE` in `to_paddle_inputs`.
2. **`_check_all_outputs` checks ALL tensors**: Every in-place-modified output tensor must be compared against reference. Never scatter `assertEqual`/`assertTrue` across individual test methods — all checks go through `_check_all_outputs`.
3. **Stochastic kernels**: If the kernel uses `curand` (e.g., top-p sampling), compare only deterministic positions. Skip the last sampled token in `compare_results`. Note: `curand_states` in reference should be sized to `max_step_tokens` (position count), not `bsz` (batch count).
4. **TEST_CONFIGS for pure-parameter scenarios**: Any test that only differs by `gen_inputs` parameters belongs in `TEST_CONFIGS`, not a separate `test_*` method. Only create separate methods when you need to **override specific tensor values** after generation.
5. **Test cases are thin**: Each `test_*` method should be 3-15 lines. It either calls `_run_full_test(config)` or does `gen → override → _run_and_get → _check_all_outputs`.
6. **No `fastdeploy.__init__`**: Import ops via `import_custom_ops` directly to avoid heavy dependency chain.
7. **Padding slots**: Kernel may have `max_bsz > real_bsz`. Reference impl must handle padding slots the same way as the kernel (typically no-op or stop_count++).
