import argparse
import importlib
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class _FakePlace:
    def __init__(self, is_gpu: bool = False) -> None:
        self._is_gpu = is_gpu

    def is_gpu_place(self) -> bool:
        return self._is_gpu


class _FakeTensor:
    __array_priority__ = 1000

    def __init__(self, array: Any, dtype: Optional[str] = None, place: Optional[_FakePlace] = None) -> None:
        if isinstance(array, _FakeTensor):
            array = array.array
        if dtype is not None:
            self.array = np.array(array, dtype=dtype)
        else:
            self.array = np.array(array)
        self.place = place or _FakePlace(False)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"_FakeTensor({self.array!r})"

    def __len__(self) -> int:
        return len(self.array)

    @property
    def dtype(self):  # pragma: no cover - compatibility helper
        return self.array.dtype

    @property
    def shape(self):
        return self.array.shape

    def numpy(self):
        return self.array

    def tolist(self):
        return self.array.tolist()

    def item(self):
        return self.array.item()

    def astype(self, dtype: str) -> "_FakeTensor":
        return _FakeTensor(self.array.astype(dtype), place=self.place)

    def unsqueeze(self, axis: int) -> "_FakeTensor":
        return _FakeTensor(np.expand_dims(self.array, axis=axis), place=self.place)

    def split(self, lengths: List[int]):
        outputs = []
        start = 0
        for length in lengths:
            outputs.append(_FakeTensor(self.array[start : start + length], place=self.place))
            start += length
        return outputs

    def cuda(self) -> "_FakeTensor":
        return _FakeTensor(self.array.copy(), place=_FakePlace(True))

    def __getitem__(self, item):
        if isinstance(item, _FakeTensor):
            item = item.array
        result = self.array.__getitem__(item)
        if isinstance(result, np.ndarray):
            return _FakeTensor(result, place=self.place)
        return result

    def __setitem__(self, key, value):
        if isinstance(value, _FakeTensor):
            value = value.array
        self.array.__setitem__(key, value)

    def __iter__(self):
        if self.array.ndim == 1:
            for value in self.array:
                yield value.item() if hasattr(value, "item") else value
        else:
            for row in self.array:
                yield _FakeTensor(row, place=self.place)

    def _binary_op(self, other: Any, op):
        other_array = other.array if isinstance(other, _FakeTensor) else other
        return _FakeTensor(op(self.array, other_array), place=self.place)

    def __add__(self, other):
        return self._binary_op(other, np.add)

    def __radd__(self, other):
        return self._binary_op(other, np.add)

    def __sub__(self, other):
        return self._binary_op(other, np.subtract)

    def __rsub__(self, other):
        return _FakeTensor(other, place=self.place)._binary_op(self, np.subtract)

    def __truediv__(self, other):
        return self._binary_op(other, np.divide)

    def __mul__(self, other):  # pragma: no cover - completeness helper
        return self._binary_op(other, np.multiply)

    def __eq__(self, other):
        other_array = other.array if isinstance(other, _FakeTensor) else other
        return self.array == other_array


class _FakeLayer:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):  # pragma: no cover - interface requirement
        raise NotImplementedError


class _Identity(_FakeLayer):
    def forward(self, x):
        return x


class _Sigmoid(_FakeLayer):
    def forward(self, x):  # pragma: no cover - unused
        arr = x.array if isinstance(x, _FakeTensor) else np.array(x)
        return _FakeTensor(1 / (1 + np.exp(-arr)))


class _Softmax(_FakeLayer):
    def __init__(self, axis: int = -1):
        self.axis = axis

    def forward(self, x):  # pragma: no cover - unused helper
        arr = x.array if isinstance(x, _FakeTensor) else np.array(x)
        shifted = arr - arr.max(axis=self.axis, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=self.axis, keepdims=True)
        return _FakeTensor(probs)


class _LambdaLayer(_FakeLayer):
    def __init__(self, fn):
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


def _build_paddle_module() -> types.ModuleType:
    paddle = types.ModuleType("paddle")
    paddle.Tensor = _FakeTensor

    def to_tensor(data, dtype=None):
        return _FakeTensor(data, dtype=dtype)

    def zeros(shape, dtype="float32"):
        return _FakeTensor(np.zeros(shape, dtype=dtype))

    def cumsum(tensor, axis=0, out=None):
        result = _FakeTensor(np.cumsum(tensor.array, axis=axis), place=tensor.place)
        if out is not None:
            out[:] = result
            return out
        return result

    def stack(tensors):
        arrays = [tensor.array if isinstance(tensor, _FakeTensor) else tensor for tensor in tensors]
        return _FakeTensor(np.stack(arrays))

    def all(tensor):
        if isinstance(tensor, _FakeTensor):
            value = tensor.array
        else:
            value = np.array(tensor)
        return _FakeTensor(np.array(value.all(), dtype=bool))

    paddle.to_tensor = to_tensor
    paddle.zeros = zeros
    paddle.cumsum = cumsum
    paddle.stack = stack
    paddle.all = all

    nn_module = types.ModuleType("paddle.nn")

    class Layer(_FakeLayer):
        pass

    nn_module.Layer = Layer
    nn_module.Identity = _Identity
    nn_module.Sigmoid = _Sigmoid
    nn_module.Softmax = _Softmax

    functional = types.ModuleType("paddle.nn.functional")

    def _ensure_array(x):
        return x.array if isinstance(x, _FakeTensor) else np.array(x)

    def sigmoid(x):
        arr = _ensure_array(x)
        return _FakeTensor(1 / (1 + np.exp(-arr)))

    def softmax(x, axis=-1):
        arr = _ensure_array(x)
        shifted = arr - arr.max(axis=axis, keepdims=True)
        exp = np.exp(shifted)
        return _FakeTensor(exp / exp.sum(axis=axis, keepdims=True))

    def normalize(x, p=2, axis=-1):
        arr = _ensure_array(x).astype(np.float32)
        norm = np.linalg.norm(arr, ord=p, axis=axis, keepdims=True)
        norm[norm == 0] = 1
        return _FakeTensor(arr / norm)

    functional.sigmoid = sigmoid
    functional.softmax = softmax
    functional.normalize = normalize

    paddle.nn = nn_module
    paddle.nn.functional = functional
    paddle._FakeTensor = _FakeTensor
    paddle._FakePlace = _FakePlace
    paddle.F = functional
    return paddle


def _build_config_module() -> types.ModuleType:
    module = types.ModuleType("fastdeploy.config")

    class PoolerConfig:
        def __init__(self, pooling_type: Optional[str] = None):
            self.pooling_type = pooling_type

    class ModelConfig:
        def __init__(self, pooler_config: Optional[PoolerConfig] = None, num_labels: int = 0):
            self.pooler_config = pooler_config
            self.num_labels = num_labels

    class FDConfig:
        def __init__(self):
            self.model_config = ModelConfig(num_labels=1)

    module.FDConfig = FDConfig
    module.ModelConfig = ModelConfig
    module.PoolerConfig = PoolerConfig
    return module


def _build_pooling_params_module() -> types.ModuleType:
    module = types.ModuleType("fastdeploy.engine.pooling_params")

    class PoolingParams:
        def __init__(
            self,
            task: Optional[str] = None,
            dimensions: Optional[int] = None,
            normalize: Optional[bool] = None,
            softmax: Optional[bool] = None,
            step_tag_id: Optional[int] = None,
            returned_token_ids: Optional[List[int]] = None,
            requires_token_ids: bool = False,
        ) -> None:
            self.task = task
            self.dimensions = dimensions
            self.normalize = normalize
            self.softmax = softmax
            self.step_tag_id = step_tag_id
            self.returned_token_ids = returned_token_ids
            self.requires_token_ids = requires_token_ids
            self.pooling_params = [self]

    module.PoolingParams = PoolingParams
    return module


def _build_metadata_module() -> types.ModuleType:
    module = types.ModuleType("fastdeploy.model_executor.layers.pool.metadata")

    class PoolingCursor:
        def __init__(
            self,
            index: List[int],
            first_token_indices_gpu: _FakeTensor,
            last_token_indices_gpu: _FakeTensor,
            prompt_lens_cpu: _FakeTensor,
            num_scheduled_tokens_cpu: _FakeTensor,
        ) -> None:
            self.index = index
            self.first_token_indices_gpu = first_token_indices_gpu
            self.last_token_indices_gpu = last_token_indices_gpu
            self.prompt_lens_cpu = prompt_lens_cpu
            self.num_scheduled_tokens_cpu = num_scheduled_tokens_cpu

        def __getitem__(self, indices: slice):
            return PoolingCursor(
                index=self.index[indices],
                first_token_indices_gpu=self.first_token_indices_gpu[indices],
                last_token_indices_gpu=self.last_token_indices_gpu[indices],
                prompt_lens_cpu=self.prompt_lens_cpu[indices],
                num_scheduled_tokens_cpu=self.num_scheduled_tokens_cpu[indices],
            )

        def is_partial_prefill(self):
            prompt = self.prompt_lens_cpu.array
            scheduled = self.num_scheduled_tokens_cpu.array
            return not np.array_equal(prompt, scheduled)

    class PoolingMetadata:
        def __init__(
            self,
            prompt_lens: _FakeTensor,
            prompt_token_ids: Optional[_FakeTensor],
            pooling_params: List[Any],
            pooling_cursor: Optional[PoolingCursor] = None,
        ) -> None:
            self.prompt_lens = prompt_lens
            self.prompt_token_ids = prompt_token_ids
            self.pooling_params = pooling_params
            self.pooling_cursor = pooling_cursor

        def __getitem__(self, indices: slice):
            return PoolingMetadata(
                prompt_lens=self.prompt_lens[indices],
                prompt_token_ids=None if self.prompt_token_ids is None else self.prompt_token_ids[indices],
                pooling_params=self.pooling_params[indices],
                pooling_cursor=None if self.pooling_cursor is None else self.pooling_cursor[indices],
            )

        def __len__(self):
            return len(self.pooling_params)

    module.PoolingCursor = PoolingCursor
    module.PoolingMetadata = PoolingMetadata
    return module


def _build_output_module() -> types.ModuleType:
    module = types.ModuleType("fastdeploy.output.pooler")

    class PoolingSequenceGroupOutput(list):
        pass

    class PoolerOutput(list):
        pass

    module.PoolingSequenceGroupOutput = PoolingSequenceGroupOutput
    module.PoolerOutput = PoolerOutput
    return module


def _build_utils_module() -> types.ModuleType:
    module = types.ModuleType("fastdeploy.utils")

    class _Logger:
        def __init__(self):
            self.messages: List[tuple[str, str]] = []

        def warning(self, msg):
            self.messages.append(("warning", msg))

    def get_logger(_name, _filename):
        return _Logger()

    module.get_logger = get_logger
    return module


def _build_tasks_module() -> types.ModuleType:
    module = types.ModuleType("fastdeploy.engine.tasks")
    PoolingTask = str
    module.PoolingTask = PoolingTask
    return module


def _import_pooler_module():
    saved: Dict[str, Optional[types.ModuleType]] = {}

    def _install(name: str, module: types.ModuleType):
        saved[name] = sys.modules.get(name)
        sys.modules[name] = module

    for name, builder in [
        ("paddle", _build_paddle_module),
        ("fastdeploy.config", _build_config_module),
        ("fastdeploy.engine.pooling_params", _build_pooling_params_module),
        ("fastdeploy.model_executor.layers.pool.metadata", _build_metadata_module),
        ("fastdeploy.output.pooler", _build_output_module),
        ("fastdeploy.utils", _build_utils_module),
        ("fastdeploy.engine.tasks", _build_tasks_module),
    ]:
        module = builder()
        _install(name, module)
        if name == "paddle":
            _install("paddle.nn", module.nn)
            _install("paddle.nn.functional", module.nn.functional)

    fastdeploy_pkg = types.ModuleType("fastdeploy")
    fastdeploy_pkg.__path__ = []
    _install("fastdeploy", fastdeploy_pkg)
    for pkg_name in ("fastdeploy.model_executor", "fastdeploy.model_executor.layers"):
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = []
        _install(pkg_name, pkg)

    spec = importlib.util.spec_from_file_location(
        "fastdeploy.model_executor.layers.pooler",
        PROJECT_ROOT / "fastdeploy" / "model_executor" / "layers" / "pooler.py",
    )
    assert spec and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    def _cleanup():
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
        for name in [
            "fastdeploy.model_executor.layers.pooler",
            "fastdeploy.model_executor.layers",
            "fastdeploy.model_executor",
            "fastdeploy",
        ]:
            sys.modules.pop(name, None)

    return module, _cleanup


_POOLER_MODULE, _CLEANUP = _import_pooler_module()
unittest.addModuleCleanup(_CLEANUP)


def _make_cursor(prompt_lens: List[int], device: str = "cpu", num_tokens: Optional[List[int]] = None):
    paddle = sys.modules["paddle"]
    metadata = sys.modules["fastdeploy.model_executor.layers.pool.metadata"]
    if num_tokens is None:
        num_tokens = prompt_lens
    starts = [0]
    for length in num_tokens[:-1]:
        starts.append(starts[-1] + length)
    first_indices = paddle.to_tensor(starts)
    last_indices = paddle.to_tensor([s + l - 1 for s, l in zip(starts, num_tokens)])
    if device == "gpu":
        first_indices = first_indices.cuda()
        last_indices = last_indices.cuda()
    return metadata.PoolingCursor(
        index=list(range(len(prompt_lens))),
        first_token_indices_gpu=first_indices,
        last_token_indices_gpu=last_indices,
        prompt_lens_cpu=paddle.to_tensor(prompt_lens),
        num_scheduled_tokens_cpu=paddle.to_tensor(num_tokens),
    )


def _make_metadata(
    prompt_lens: List[int],
    pooling_params: List[Any],
    token_ids: Optional[np.ndarray] = None,
    device: str = "cpu",
    num_tokens: Optional[List[int]] = None,
):
    paddle = sys.modules["paddle"]
    metadata_mod = sys.modules["fastdeploy.model_executor.layers.pool.metadata"]
    prompt_tensor = paddle.to_tensor(prompt_lens)
    token_tensor = None if token_ids is None else paddle.to_tensor(token_ids)
    cursor = _make_cursor(prompt_lens, device=device, num_tokens=num_tokens)
    return metadata_mod.PoolingMetadata(
        prompt_lens=prompt_tensor,
        prompt_token_ids=token_tensor,
        pooling_params=pooling_params,
        pooling_cursor=cursor,
    )


class PoolerHelpersTest(unittest.TestCase):
    def setUp(self):
        self.pooler_module = _POOLER_MODULE
        self.paddle = sys.modules["paddle"]
        params_mod = sys.modules["fastdeploy.engine.pooling_params"]
        self.PoolingParams = params_mod.PoolingParams
        config_mod = sys.modules["fastdeploy.config"]
        self.PoolerConfig = config_mod.PoolerConfig

    def test_resolved_pooling_config(self):
        cfg = self.PoolerConfig(pooling_type="MEAN")
        pooler = self.pooler_module.Pooler.for_embed(cfg)
        self.assertIsInstance(pooler.head, self.pooler_module.EmbeddingPoolerHead)
        encode_pooler = self.pooler_module.Pooler.for_encode(self.PoolerConfig(pooling_type="STEP"))
        self.assertIsInstance(encode_pooler, self.pooler_module.StepPooler)

    def test_get_tasks_and_prompt_token_ids(self):
        params = [self.PoolingParams(task="encode"), self.PoolingParams(task="embed")]
        metadata = _make_metadata([2, 2], params, token_ids=np.array([[1, 2], [3, 4]]))
        self.assertEqual(self.pooler_module.get_tasks(metadata), ["encode", "embed"])
        tokens = self.pooler_module.get_prompt_token_ids(metadata)
        self.assertEqual([t.tolist() for t in tokens], [[1, 2], [3, 4]])

    def test_pooling_params_update(self):
        params = self.PoolingParams()
        update = self.pooler_module.PoolingParamsUpdate(requires_token_ids=True)
        update.apply(params)
        self.assertTrue(params.requires_token_ids)


class PoolerActivationTest(unittest.TestCase):
    def setUp(self):
        self.pooler_module = _POOLER_MODULE
        self.paddle = sys.modules["paddle"]

    def test_activation_wraps_identity_and_lambda(self):
        nn = sys.modules["paddle"].nn
        identity = self.pooler_module.PoolerActivation.wraps(nn.Identity())
        self.assertIsInstance(identity, self.pooler_module.PoolerIdentity)
        sigmoid = self.pooler_module.PoolerActivation.wraps(nn.Sigmoid())
        self.assertIsInstance(sigmoid, self.pooler_module.PoolerClassify)
        layer = self.pooler_module.PoolerActivation.wraps(_LambdaLayer(lambda tensor: tensor.astype("float32")))
        result = layer.forward(self.paddle.to_tensor([[1, 2], [3, 4]]))
        self.assertEqual(result.dtype, np.float32)

    def test_pooler_classify_branches(self):
        sigmoid_head = self.pooler_module.PoolerClassify(static_num_labels=True)
        logits = self.paddle.to_tensor([[0.0]])
        self.assertLess(sigmoid_head.forward(logits).array[0][0], 1)

        softmax_head = self.pooler_module.PoolerClassify(static_num_labels=False)
        logits = self.paddle.to_tensor([[0.0, 1.0]])
        probs = softmax_head.forward(logits)
        self.assertAlmostEqual(float(probs.array.sum()), 1.0)


class PoolerHeadTest(unittest.TestCase):
    def setUp(self):
        self.pooler_module = _POOLER_MODULE
        params_mod = sys.modules["fastdeploy.engine.pooling_params"]
        self.PoolingParams = params_mod.PoolingParams

    def test_embedding_head_with_projection_and_matryoshka(self):
        head = self.pooler_module.EmbeddingPoolerHead()

        class Doubler(sys.modules["paddle"].nn.Layer):
            def forward(self, x):
                return x + x

        head.projector = Doubler()
        pooling_params = [
            self.PoolingParams(dimensions=2, normalize=True),
            self.PoolingParams(dimensions=None, normalize=False),
        ]
        metadata = _make_metadata([2, 2], pooling_params)
        pooled = [
            sys.modules["paddle"].to_tensor([[1.0, 2.0, 3.0]]),
            sys.modules["paddle"].to_tensor([[4.0, 5.0, 6.0]]),
        ]
        output = head.forward(pooled, metadata)
        self.assertIsInstance(output, list)
        self.assertEqual(output[0].shape[-1], 2)
        self.assertTrue(np.allclose(np.linalg.norm(output[0].array, axis=-1), 1))

    def test_reward_head_softmax_flags(self):
        head = self.pooler_module.RewardPoolerHead()
        pooling_params = [self.PoolingParams(softmax=True), self.PoolingParams(softmax=False)]
        metadata = _make_metadata([1, 1], pooling_params)
        logits = [sys.modules["paddle"].to_tensor([[0.0, 1.0]]), sys.modules["paddle"].to_tensor([[0.5, 0.5]])]
        outputs = head.forward(logits, metadata)
        self.assertTrue(np.allclose(outputs[0].array.sum(), 1.0))
        self.assertTrue(np.allclose(outputs[1].array, logits[1].array))


class PoolingMethodTest(unittest.TestCase):
    def setUp(self):
        self.pooler_module = _POOLER_MODULE
        params_mod = sys.modules["fastdeploy.engine.pooling_params"]
        self.params = [params_mod.PoolingParams(task="encode"), params_mod.PoolingParams(task="encode")]
        self.metadata = _make_metadata([2, 2], self.params)
        self.hidden_states = sys.modules["paddle"].to_tensor(
            [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            dtype="float32",
        )

    def test_last_and_cls_pool(self):
        pooling_cursor = self.metadata.pooling_cursor
        last_pool = self.pooler_module.LastPool()
        output = last_pool.forward(self.hidden_states, self.metadata)
        self.assertEqual(len(output.array), len(pooling_cursor.index))

        cls_pool = self.pooler_module.CLSPool()
        cls_output = cls_pool.forward(self.hidden_states, self.metadata)
        self.assertEqual(cls_output.array.shape[0], len(pooling_cursor.index))

    def test_all_pool_requires_full_prefill(self):
        all_pool = self.pooler_module.AllPool()
        # full prefill works
        output = all_pool.forward(self.hidden_states, self.metadata)
        self.assertEqual(len(output), len(self.params))
        # partial prefill raises
        partial_cursor = _make_cursor([2, 2], num_tokens=[1, 2])
        partial_metadata = _make_metadata([2, 2], self.params)
        partial_metadata.pooling_cursor = partial_cursor
        with self.assertRaises(AssertionError):
            all_pool.forward(self.hidden_states, partial_metadata)

    def test_mean_pool_gpu_branch(self):
        mean_pool = self.pooler_module.MeanPool()
        gpu_states = _FakeTensor(self.hidden_states.array, place=_FakePlace(True))
        metadata = _make_metadata([2, 2], self.params, device="gpu")
        output = mean_pool.forward(gpu_states, metadata)
        self.assertTrue(np.allclose(output.array, np.array([[0.5, 0.5], [2.5, 2.5]])))


class StepAndSimplePoolerTest(unittest.TestCase):
    def setUp(self):
        self.pooler_module = _POOLER_MODULE
        params_mod = sys.modules["fastdeploy.engine.pooling_params"]
        self.PoolingParams = params_mod.PoolingParams

    def test_step_pooler_filters_returned_ids_and_steps(self):
        pooling_params = [
            self.PoolingParams(task="encode", step_tag_id=3, returned_token_ids=[0, 2]),
            self.PoolingParams(task="encode", returned_token_ids=[]),
        ]
        token_ids = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        metadata = _make_metadata([4, 4], pooling_params, token_ids=token_ids)
        pooler = self.pooler_module.StepPooler()
        states = sys.modules["paddle"].to_tensor(np.arange(32).reshape(8, 4))
        output = pooler.forward(states, metadata)
        self.assertEqual(len(output), 2)
        self.assertTrue(all(isinstance(vec, _FakeTensor) for vec in output))
        updates = pooler.get_pooling_updates("encode")
        self.assertTrue(updates.requires_token_ids)

    def test_simple_pooler_embed_and_encode(self):
        config_mod = sys.modules["fastdeploy.config"]
        resolved = self.pooler_module.ResolvedPoolingConfig(
            task="embed", pooling_type=self.pooler_module.PoolingType.ALL
        )
        simple = self.pooler_module.SimplePooler.from_config(resolved, config_mod.ModelConfig())
        self.assertIsInstance(simple.head, self.pooler_module.EmbeddingPoolerHead)
        metadata = _make_metadata([2], [self.PoolingParams(task="embed")])
        hidden = sys.modules["paddle"].to_tensor([[1.0, 2.0]])
        result = simple.forward(hidden, metadata)
        self.assertEqual(len(result.array), 1)


class DispatchPoolerTest(unittest.TestCase):
    def setUp(self):
        self.pooler_module = _POOLER_MODULE
        params_mod = sys.modules["fastdeploy.engine.pooling_params"]
        self.PoolingParams = params_mod.PoolingParams
        self.metadata = _make_metadata([1, 1], [self.PoolingParams(task="encode"), self.PoolingParams(task="encode")])

    def test_dispatch_forward_routes_tasks(self):
        pooling = self.pooler_module.LastPool()
        head = self.pooler_module.RewardPoolerHead()
        pooler = self.pooler_module.SimplePooler(pooling, head)
        dispatch = self.pooler_module.DispatchPooler({"encode": pooler})
        hidden = sys.modules["paddle"].to_tensor([[1.0, 2.0], [3.0, 4.0]])
        output = dispatch.forward(hidden, self.metadata)
        self.assertIsInstance(output, list)

    def test_dispatch_rejects_unknown_task(self):
        pooling = self.pooler_module.LastPool()
        head = self.pooler_module.RewardPoolerHead()
        pooler = self.pooler_module.SimplePooler(pooling, head)
        dispatch = self.pooler_module.DispatchPooler({"encode": pooler})
        bad_metadata = _make_metadata([1], [self.PoolingParams(task="score")])
        with self.assertRaises(ValueError):
            dispatch.forward(sys.modules["paddle"].to_tensor([[0.0]]), bad_metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--print-coverage-command", action="store_true")
    known, remaining = parser.parse_known_args()
    if known.print_coverage_command:
        print("python -m coverage run -m unittest tests.model_executor.test_pooler")
        print("python -m coverage report -m --include='fastdeploy/model_executor/layers/pooler.py'")
    unittest.main(argv=[sys.argv[0]] + remaining)
